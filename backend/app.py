import asyncio
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import openslide
import torch
import tensorrt as trt
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from torchvision import transforms
from dotenv import load_dotenv

from backend.wsi import get_patch_coordinates

APP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(APP_ROOT / ".env")


def configured_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} in {APP_ROOT / '.env'}")
    return Path(value).expanduser()


STATIC_DIR = APP_ROOT / "frontend"
ENGINE_PATH = configured_path("LUNG_ENGINE_PATH")
DEFAULT_SLIDE = os.environ.get("EXAMPLE_SLIDE_PATH", "/mnt/sdb/UHN/batch3/UHN_5510.Slide10.svs")
LUNG_PATCH_SIZE = 512
LYMPH_PATCH_SIZE = 1024
YOLO_IMAGE_SIZE = 640
LYMPH_BATCH_SIZE = 8
YOLO_WEIGHTS = configured_path("LYMPH_YOLO_WEIGHTS")
PATHWIZ_ROOT = configured_path("PATHWIZ_ROOT")
NUM_CLASSES = 3
CLASS_NAMES = ("Cancer", "Granuloma", "Necrosis")
CLASS_COLORS = ("#ff4f9a", "#41c9ef", "#ffb14e")
MEAN = (0.5642, 0.5026, 0.6960)
STD = (0.2724, 0.2838, 0.2167)
LOGIT_THRESHOLD = 0.5
LYMPH_DETECTION_THRESHOLD = 40
MAX_CELLS_PER_EVENT = 1024
UPDATE_INTERVAL_SECONDS = 0.1
logger = logging.getLogger("pathology-demo")


class RunRequest(BaseModel):
    slide_path: str
    tissue_type: Literal["lung", "lymph_node"] = "lung"


@dataclass
class SlideJob:
    id: str
    slide_path: str
    tissue_type: str
    patch_size: int
    batch_size: int
    thumbnail: bytes
    slide_width: int
    slide_height: int
    covered_width: int
    covered_height: int
    grid_columns: int
    grid_rows: int
    coords: list[tuple[int, int]]
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_cells: list[list[float | int]] = field(default_factory=list)
    processed: int = 0
    class_counts: list[int] = field(default_factory=lambda: [0] * NUM_CLASSES)
    sufficient_regions: int = 0
    top5_counts: list[int] = field(default_factory=list)
    top5_average: float = 0.0
    cursor_x: float | None = None
    cursor_y: float | None = None
    started_at: float | None = None
    elapsed: float = 0.0
    done: bool = False
    error: str | None = None


class TensorRTSession:
    def __init__(self, engine_path: Path):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device_index = next(
            (
                i for i in range(torch.cuda.device_count())
                if "RTX 6000 Ada" in torch.cuda.get_device_name(i)
            ),
            None,
        )
        if device_index is None:
            visible = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            raise RuntimeError(f"RTX 6000 Ada not found among visible CUDA devices: {visible}")
        torch.cuda.set_device(device_index)
        self.device = torch.device(f"cuda:{device_index}")
        self.gpu_name = torch.cuda.get_device_name(device_index)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with engine_path.open("rb") as engine_file:
            self.engine = self.runtime.deserialize_cuda_engine(engine_file.read())
        if self.engine is None:
            raise RuntimeError(f"TensorRT failed to deserialize {engine_path}")

        inputs = []
        outputs = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            (inputs if mode == trt.TensorIOMode.INPUT else outputs).append(name)
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"Expected one input and one output, got {inputs=} {outputs=}")
        self.input_name = inputs[0]
        self.output_name = outputs[0]
        if self.engine.get_tensor_dtype(self.input_name) != trt.DataType.FLOAT:
            raise RuntimeError("This app expects a float32 TensorRT input")
        if self.engine.get_tensor_dtype(self.output_name) != trt.DataType.FLOAT:
            raise RuntimeError("This app expects a float32 TensorRT output")

        shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.profile_shapes = []
        for profile_index in range(self.engine.num_optimization_profiles):
            self.profile_shapes.append(
                self.engine.get_tensor_profile_shape(self.input_name, profile_index)
            )
        if shape[0] > 0:
            self.batch_size = shape[0]
        elif self.profile_shapes:
            minimum, optimum, maximum = self.profile_shapes[0]
            self.batch_size = minimum[0] if minimum[0] == maximum[0] else optimum[0]
        else:
            raise RuntimeError("Engine input has dynamic batch but no optimization profile")
        if shape[1:] != (3, LUNG_PATCH_SIZE, LUNG_PATCH_SIZE):
            raise RuntimeError(f"Expected input (B,3,512,512), engine has {shape}")
        if self.batch_size <= 0:
            raise RuntimeError(f"Invalid engine batch size: {self.batch_size}")

        self.stream = torch.cuda.Stream(device=self.device)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT failed to create an execution context")
        if not self.context.set_input_shape(
            self.input_name, (self.batch_size, 3, LUNG_PATCH_SIZE, LUNG_PATCH_SIZE)
        ):
            raise RuntimeError("TensorRT rejected the engine's profiled input shape")
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        if len(output_shape) != 2 or output_shape[0] != self.batch_size or output_shape[1] != NUM_CLASSES:
            raise RuntimeError(f"Expected output (B,3), engine context reports {output_shape}")

        input_shape = (self.batch_size, 3, LUNG_PATCH_SIZE, LUNG_PATCH_SIZE)
        self.host_input = torch.empty(input_shape, dtype=torch.float32, pin_memory=True)
        self.device_input = torch.empty(input_shape, dtype=torch.float32, device=self.device)
        self.device_output = torch.empty(output_shape, dtype=torch.float32, device=self.device)
        self.host_output = torch.empty(output_shape, dtype=torch.float32, pin_memory=True)
        self.context.set_tensor_address(self.input_name, self.device_input.data_ptr())
        self.context.set_tensor_address(self.output_name, self.device_output.data_ptr())

        self.host_input.zero_()
        for _ in range(3):
            with torch.cuda.stream(self.stream):
                self.device_input.copy_(self.host_input, non_blocking=True)
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT warmup enqueue failed")
        self.stream.synchronize()

        print("TensorRT engine loaded", flush=True)
        print(f"GPU: {self.gpu_name}", flush=True)
        print(f"Input: {self.input_name} {shape} {self.engine.get_tensor_dtype(self.input_name)}", flush=True)
        print(f"Output: {self.output_name} {output_shape} {self.engine.get_tensor_dtype(self.output_name)}", flush=True)
        print(f"Optimization profiles: {self.profile_shapes}", flush=True)
        print(f"Effective fixed batch: {self.batch_size}", flush=True)
        print("Model warmup complete", flush=True)

    def infer(self, images: list[torch.Tensor]) -> np.ndarray:
        count = len(images)
        if count < 1 or count > self.batch_size:
            raise ValueError(f"Expected 1..{self.batch_size} images, got {count}")
        self.host_input.zero_()
        self.host_input[:count].copy_(torch.stack(images))
        with torch.cuda.stream(self.stream):
            self.device_input.copy_(self.host_input, non_blocking=True)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT inference enqueue failed")
            self.host_output.copy_(self.device_output, non_blocking=True)
        self.stream.synchronize()
        return self.host_output[:count].numpy().copy()


class LymphocyteDetector:
    def __init__(self, device: torch.device):
        if not YOLO_WEIGHTS.is_file():
            raise RuntimeError(f"Lymphocyte detector weights not found: {YOLO_WEIGHTS}")
        if not PATHWIZ_ROOT.is_dir():
            raise RuntimeError(f"PathWiz source directory not found: {PATHWIZ_ROOT}")
        sys.path.insert(0, str(PATHWIZ_ROOT))
        from yolo_utils.models.common import DetectMultiBackend
        from yolo_utils.utils.general import check_img_size, non_max_suppression

        self.device = device
        self.batch_size = LYMPH_BATCH_SIZE
        self.non_max_suppression = non_max_suppression
        self.confidence = 0.5
        self.iou = 0.45
        self.max_detections = 1000
        self.image_size = YOLO_IMAGE_SIZE
        self.model = DetectMultiBackend(
            str(YOLO_WEIGHTS), device=device, dnn=False, fp16=True
        )
        self.image_size = check_img_size(self.image_size, s=self.model.stride)
        self.model.eval()
        self.model.warmup(imgsz=(LYMPH_BATCH_SIZE, 3, self.image_size, self.image_size))
        names = self.model.names
        self.model_name = ", ".join(str(value) for value in names.values()) if isinstance(names, dict) else ", ".join(map(str, names))
        print(f"Lymphocyte YOLO model loaded: {YOLO_WEIGHTS}", flush=True)
        print(f"YOLO class: {self.model_name}; input crop={LYMPH_PATCH_SIZE}, network={self.image_size}, FP16", flush=True)

    @torch.inference_mode()
    def infer(self, images: list[torch.Tensor]) -> list[int]:
        batch = torch.stack(images).to(self.device, non_blocking=True)
        batch = batch.half().div_(255.0)
        prediction = self.model(batch, augment=False, visualize=False)
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        detections = self.non_max_suppression(
            prediction.float(),
            conf_thres=self.confidence,
            iou_thres=self.iou,
            max_det=self.max_detections,
        )
        return [len(det) if det is not None else 0 for det in detections]


class JobStore:
    def __init__(self):
        self.jobs: dict[str, SlideJob] = {}
        self.active_id: str | None = None
        self.lock = threading.Lock()

    def add(self, job: SlideJob):
        with self.lock:
            if self.active_id:
                active = self.jobs.get(self.active_id)
                if active and not active.done:
                    raise HTTPException(409, "An inference is already running")
            self.jobs[job.id] = job
            self.active_id = job.id
            finished = [key for key, value in self.jobs.items() if value.done and key != job.id]
            for key in finished[:-2]:
                self.jobs.pop(key, None)

    def get(self, job_id: str) -> SlideJob:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Inference job not found")
        return job


jobs = JobStore()


def prepare_slide(slide_path: str, patch_size: int):
    slide = openslide.OpenSlide(slide_path)
    try:
        slide_width, slide_height = slide.level_dimensions[0]
        coords = get_patch_coordinates(slide, patch_size, level=0)
        if not coords:
            raise ValueError(f"Slide is smaller than one {patch_size}x{patch_size} patch")
        grid_columns = (slide_width - patch_size) // patch_size + 1
        grid_rows = (slide_height - patch_size) // patch_size + 1
        covered_width = grid_columns * patch_size
        covered_height = grid_rows * patch_size
        thumbnail = slide.get_thumbnail((2200, 2200)).convert("RGB")
        crop_width = max(1, round(thumbnail.width * covered_width / slide_width))
        crop_height = max(1, round(thumbnail.height * covered_height / slide_height))
        thumbnail = thumbnail.crop((0, 0, crop_width, crop_height))
        output = io.BytesIO()
        thumbnail.save(output, format="JPEG", quality=91, optimize=True)
        return (
            slide_width,
            slide_height,
            covered_width,
            covered_height,
            grid_columns,
            grid_rows,
            coords,
            output.getvalue(),
        )
    finally:
        slide.close()


def update_lung_job(job: SlideJob, coords: list[tuple[int, int]], logits: np.ndarray):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    positives = logits > LOGIT_THRESHOLD
    cells = []
    for (x, y), probs in zip(coords, probabilities):
        cell_index = (y // job.patch_size) * job.grid_columns + (x // job.patch_size)
        cells.append([cell_index, round(float(probs[0]), 4), round(float(probs[1]), 4), round(float(probs[2]), 4)])
    with job.lock:
        job.processed += len(coords)
        for class_index in range(NUM_CLASSES):
            job.class_counts[class_index] += int(positives[:, class_index].sum())
        job.pending_cells.extend(cells)
        last_x, last_y = coords[-1]
        job.cursor_x = (last_x + job.patch_size / 2) / job.covered_width
        job.cursor_y = (last_y + job.patch_size / 2) / job.covered_height
        job.elapsed = time.perf_counter() - (job.started_at or time.perf_counter())


def update_lymph_job(job: SlideJob, coords: list[tuple[int, int]], detection_counts: list[int]):
    cells = []
    for (x, y), count in zip(coords, detection_counts):
        cell_index = (y // job.patch_size) * job.grid_columns + (x // job.patch_size)
        cells.append([cell_index, int(count)])
    with job.lock:
        job.processed += len(coords)
        job.sufficient_regions += sum(count > LYMPH_DETECTION_THRESHOLD for count in detection_counts)
        for count in detection_counts:
            if len(job.top5_counts) < 5:
                job.top5_counts.append(int(count))
                job.top5_counts.sort(reverse=True)
            elif count > job.top5_counts[-1]:
                job.top5_counts[-1] = int(count)
                job.top5_counts.sort(reverse=True)
        job.top5_average = sum(job.top5_counts) / len(job.top5_counts) if job.top5_counts else 0.0
        job.pending_cells.extend(cells)
        last_x, last_y = coords[-1]
        job.cursor_x = (last_x + job.patch_size / 2) / job.covered_width
        job.cursor_y = (last_y + job.patch_size / 2) / job.covered_height
        job.elapsed = time.perf_counter() - (job.started_at or time.perf_counter())


def run_job(job: SlideJob, trt_session: TensorRTSession, yolo_session: LymphocyteDetector):
    slide = None
    lung_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    session = trt_session if job.tissue_type == "lung" else yolo_session
    try:
        with job.lock:
            job.started_at = time.perf_counter()
        slide = openslide.OpenSlide(job.slide_path)

        def read_and_transform(coord):
            x, y = coord
            region = slide.read_region((x, y), 0, (job.patch_size, job.patch_size)).convert("RGB")
            if job.tissue_type == "lung":
                return lung_transform(region), coord
            image = np.asarray(region)
            resized = cv2.resize(image, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
            chw = np.ascontiguousarray(resized.transpose(2, 0, 1))
            return torch.from_numpy(chw), coord

        batch_images: list[torch.Tensor] = []
        batch_coords: list[tuple[int, int]] = []
        chunk_size = max(session.batch_size * 4, 32)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for start in range(0, len(job.coords), chunk_size):
                coord_chunk = job.coords[start:start + chunk_size]
                for tensor, coord in executor.map(read_and_transform, coord_chunk):
                    batch_images.append(tensor)
                    batch_coords.append(coord)
                    if len(batch_images) == session.batch_size:
                        if job.tissue_type == "lung":
                            predictions = trt_session.infer(batch_images)
                            update_lung_job(job, batch_coords, predictions)
                        else:
                            detections = yolo_session.infer(batch_images)
                            update_lymph_job(job, batch_coords, detections)
                        batch_images.clear()
                        batch_coords.clear()
            if batch_images:
                if job.tissue_type == "lung":
                    predictions = trt_session.infer(batch_images)
                    update_lung_job(job, batch_coords, predictions)
                else:
                    detections = yolo_session.infer(batch_images)
                    update_lymph_job(job, batch_coords, detections)

        slide.close()
        slide = None
        with job.lock:
            job.elapsed = time.perf_counter() - (job.started_at or time.perf_counter())
            job.done = True
    except Exception as exc:
        logger.exception("Inference failed for slide %s", job.slide_path)
        with job.lock:
            job.elapsed = time.perf_counter() - (job.started_at or time.perf_counter())
            job.error = str(exc)
            job.done = True
    finally:
        if slide is not None:
            slide.close()


def progress_payload(job: SlideJob, cells: list[list[float | int]]):
    with job.lock:
        processed = job.processed
        elapsed = job.elapsed
        done = job.done
        return {
            "tissue_type": job.tissue_type,
            "processed_regions": processed,
            "total_regions": len(job.coords),
            "progress": processed / max(len(job.coords), 1),
            "cursor_x": job.cursor_x,
            "cursor_y": job.cursor_y,
            "class_counts": job.class_counts.copy(),
            "sufficient_regions": job.sufficient_regions,
            "top5_average": job.top5_average,
            "regions_per_second": processed / elapsed if elapsed > 0 else 0.0,
            "elapsed_seconds": elapsed,
            "cells": cells,
            "done": done,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ENGINE_PATH.is_file():
        raise RuntimeError(f"TensorRT engine not found: {ENGINE_PATH}")
    app.state.trt_session = TensorRTSession(ENGINE_PATH)
    app.state.lymph_session = LymphocyteDetector(app.state.trt_session.device)
    print("Both inference models are ready", flush=True)
    print("Server ready", flush=True)
    yield


app = FastAPI(title="Pathology Inference Viewer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    session: TensorRTSession = app.state.trt_session
    detector: LymphocyteDetector = app.state.lymph_session
    return {
        "ready": True,
        "gpu": session.gpu_name,
        "example_slide": DEFAULT_SLIDE,
        "models": {"lung": "TensorRT ResNeXt50", "lymph_node": detector.model_name},
        "batch_size": session.batch_size,
        "lymph_batch_size": detector.batch_size,
        "input_name": session.input_name,
        "output_name": session.output_name,
        "classes": CLASS_NAMES,
        "precision": "FP16",
    }


@app.post("/api/inference")
async def start_inference(request: RunRequest):
    slide_path = str(Path(request.slide_path).expanduser())
    if not Path(slide_path).is_file():
        raise HTTPException(404, f"Slide file not found: {slide_path}")
    if request.tissue_type == "lung":
        patch_size = LUNG_PATCH_SIZE
        batch_size = app.state.trt_session.batch_size
    else:
        patch_size = LYMPH_PATCH_SIZE
        batch_size = app.state.lymph_session.batch_size
    try:
        prepared = await asyncio.to_thread(prepare_slide, slide_path, patch_size)
    except Exception as exc:
        raise HTTPException(400, f"Could not open slide: {exc}") from exc

    width, height, covered_width, covered_height, columns, rows, coords, thumbnail = prepared
    job = SlideJob(
        id=uuid.uuid4().hex,
        slide_path=slide_path,
        tissue_type=request.tissue_type,
        patch_size=patch_size,
        batch_size=batch_size,
        thumbnail=thumbnail,
        slide_width=width,
        slide_height=height,
        covered_width=covered_width,
        covered_height=covered_height,
        grid_columns=columns,
        grid_rows=rows,
        coords=coords,
    )
    jobs.add(job)
    thread = threading.Thread(
        target=run_job,
        args=(job, app.state.trt_session, app.state.lymph_session),
        name=f"wsi-inference-{job.id[:8]}",
        daemon=True,
    )
    thread.start()
    return {
        "job_id": job.id,
        "tissue_type": job.tissue_type,
        "thumbnail_url": f"/api/jobs/{job.id}/thumbnail",
        "slide_width": width,
        "slide_height": height,
        "display_width": covered_width,
        "display_height": covered_height,
        "grid_columns": columns,
        "grid_rows": rows,
        "total_regions": len(coords),
        "patch_size": patch_size,
        "model_input_size": LUNG_PATCH_SIZE if job.tissue_type == "lung" else YOLO_IMAGE_SIZE,
        "batch_size": batch_size,
        "class_names": CLASS_NAMES if job.tissue_type == "lung" else ("Lymphocyte",),
    }


@app.get("/api/jobs/{job_id}/thumbnail")
def thumbnail(job_id: str):
    job = jobs.get(job_id)
    return Response(content=job.thumbnail, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    job = jobs.get(job_id)

    async def event_stream():
        yield ": connected\n\n"
        last_processed = -1
        while True:
            with job.lock:
                cells = job.pending_cells[:MAX_CELLS_PER_EVENT]
                del job.pending_cells[:len(cells)]
                processed = job.processed
                done = job.done
                error = job.error
                pending_left = bool(job.pending_cells)
            if cells or processed != last_processed:
                last_processed = processed
                yield f"event: progress\ndata: {json.dumps(progress_payload(job, cells), separators=(',', ':'))}\n\n"
            if done and not pending_left:
                if error:
                    yield f"event: job-error\ndata: {json.dumps({'message': error})}\n\n"
                else:
                    payload = progress_payload(job, [])
                    yield f"event: complete\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
                break
            await asyncio.sleep(UPDATE_INTERVAL_SECONDS)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
