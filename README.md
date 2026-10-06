# ROSEDemo

Standalone local visualization app for whole-slide lung and lymph-node inference. The application and frontend are separate from the `InferenceEngine` build/benchmark repository.

## Configure models

The local model locations are read from `.env`. The provided `.env` points to the batch-4 TensorRT lung engine, the LymphAI YOLO weights, the PathWiz source, and the default UHN slide. For another machine, edit `.env` or copy `.env.example` and set those paths. `.env` is git-ignored.

## Start

After cloning, copy `.env.example` to `.env` and set the model paths for your machine. From the ROSEDemo directory, run:

```bash
conda activate RubyMLClassic
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Wait for the TensorRT engine and YOLO model warmups and `Server ready`, then open <http://127.0.0.1:8000/>. Select a tissue type before running. Stop the server with `Ctrl+C`.

## Lung mode

Uses the fixed-batch-4 TensorRT ResNeXt50 engine with two real regions submitted per inference update (the remaining engine slots are padded), non-overlapping 512×512 level-0 regions, and the checkpoint's Cancer / Granuloma / Necrosis normalization. Region class counts use the checkpoint evaluation's logit threshold `> 0.5`.

## Lymph Node mode

Uses PathWiz `DetectMultiBackend` with FP16 and the YOLO weights from `.env`, with inference batches of two regions. It crops 1024×1024 at OpenSlide level 0, resizes each crop to 640×640 using linear interpolation, and runs NMS with confidence `0.5`, IoU `0.45`, and max 1,000 detections. Regions with more than 40 detections are colored green and counted as lymphocyte-sufficient. Slide adequacy is based on whether the top-five region average exceeds 40.

Both models are loaded and warmed once at startup. Patches are scanned left-to-right on every row from x=0 in a non-overlapping grid; no tissue mask is currently applied. The frontend uses a single canvas for the heatmap and batched SSE progress updates.
