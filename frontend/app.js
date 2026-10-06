const pathInput = document.getElementById("slidePath");
const tissueSelect = document.getElementById("tissueType");
const runButton = document.getElementById("runButton");
const readyDot = document.getElementById("readyDot");
const engineState = document.getElementById("engineState");
const gpuLabel = document.getElementById("gpuLabel");
const originalViewport = document.getElementById("originalViewport");
const mapViewport = document.getElementById("mapViewport");
const slideImage = document.getElementById("slideImage");
const heatmap = document.getElementById("heatmap");
const heatContext = heatmap.getContext("2d", { alpha: true });
const originalTarget = document.getElementById("originalTarget");
const mapTarget = document.getElementById("mapTarget");
const targetContexts = [originalTarget, mapTarget].map((canvas) => canvas.getContext("2d"));
const errorMessage = document.getElementById("errorMessage");
const classRows = document.getElementById("classRows");
const lungReadout = document.getElementById("lungReadout");
const lymphReadout = document.getElementById("lymphReadout");
const lymphSufficientRegions = document.getElementById("lymphSufficientRegions");
const lymphTop5Average = document.getElementById("lymphTop5Average");
const progressFill = document.getElementById("progressFill");
const scanPulse = document.getElementById("scanPulse");
const scanLabel = document.getElementById("scanLabel");
const progressPercent = document.getElementById("progressPercent");
const regionsValue = document.getElementById("regionsValue");
const elapsedValue = document.getElementById("elapsedValue");
const throughputValue = document.getElementById("throughputValue");
const etaValue = document.getElementById("etaValue");
const slideAssessment = document.getElementById("slideAssessment");
const adequacyValue = document.getElementById("adequacyValue");
const assessmentRule = document.getElementById("assessmentRule");
const assessmentResultLabel = document.getElementById("assessmentResultLabel");
const etiologyRow = document.getElementById("etiologyRow");
const etiologyValue = document.getElementById("etiologyValue");
const slideDimensions = document.getElementById("slideDimensions");
const mapState = document.getElementById("mapState");
const originalEmpty = document.getElementById("originalEmpty");
const mapEmpty = document.getElementById("mapEmpty");
const originalCaption = document.getElementById("originalCaption");

const classColors = ["#ff4f9a", "#41c9ef", "#ffb14e"];
const logitThresholdProbability = 1 / (1 + Math.exp(-0.5));
let source = null;
let gridColumns = 0;
let gridRows = 0;
let currentTarget = null;
let playback = { active: false, cells: [], head: 0, position: 0, available: 0, finished: false, rowMs: 100, onDone: null };
let currentTissueType = "";
let modelsReady = false;
let inferenceRunning = false;
let lastFrame = 0;

function setError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function setStatus(label, mode = "idle") {
  scanLabel.textContent = label;
  scanPulse.classList.toggle("running", mode === "running");
  scanPulse.classList.toggle("complete", mode === "complete");
}

function resizeTargetCanvases() {
  const ratio = window.devicePixelRatio || 1;
  for (const canvas of [originalTarget, mapTarget]) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) continue;
    const width = Math.round(rect.width * ratio);
    const height = Math.round(rect.height * ratio);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    canvas.getContext("2d").setTransform(ratio, 0, 0, ratio, 0, 0);
  }
}

function drawReticle(context, canvas, position, time) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height || !position) return;
  const x = position.x * rect.width;
  const y = position.y * rect.height;
  const pulse = 7 + Math.sin(time / 230) * 1.2;
  context.save();
  context.clearRect(0, 0, rect.width, rect.height);
  context.lineWidth = 0.7;
  context.strokeStyle = "rgba(232, 236, 235, .35)";
  context.beginPath();
  context.moveTo(x, 0);
  context.lineTo(x, rect.height);
  context.moveTo(0, y);
  context.lineTo(rect.width, y);
  context.stroke();
  context.shadowColor = "rgba(255,255,255,.45)";
  context.shadowBlur = 8;
  context.strokeStyle = "rgba(247,248,244,.9)";
  context.lineWidth = 1;
  context.beginPath();
  context.arc(x, y, pulse, 0, Math.PI * 2);
  context.stroke();
  context.fillStyle = "rgba(255,255,255,.88)";
  context.beginPath();
  context.arc(x, y, 1.4, 0, Math.PI * 2);
  context.fill();
  context.restore();
}

function revealCells(limit) {
  let end = playback.head;
  while (end < playback.cells.length && playback.cells[end][0] < limit) end += 1;
  if (end === playback.head) return;
  drawCells(playback.cells.slice(playback.head, end), currentTissueType);
  playback.head = end;
}

function animate(time) {
  const elapsed = Math.min(time - lastFrame, 48);
  lastFrame = time;
  if (playback.active && gridRows) {
    playback.position = Math.min(playback.position + elapsed / playback.rowMs, playback.available);
    const row = Math.min(Math.floor(playback.position), gridRows - 1);
    const sweep = Math.max(0, Math.min(1, (playback.position - row - 0.2) / 0.6));
    currentTarget = { x: sweep, y: (row + 0.5) / gridRows };
    revealCells(row * gridColumns + Math.floor(sweep * gridColumns));
    if (playback.finished && playback.position >= gridRows && playback.head >= playback.cells.length) {
      playback.active = false;
      const done = playback.onDone;
      playback.onDone = null;
      if (done) done();
    }
  }
  resizeTargetCanvases();
  drawReticle(targetContexts[0], originalTarget, currentTarget, time);
  drawReticle(targetContexts[1], mapTarget, currentTarget, time);
  requestAnimationFrame(animate);
}

function resetVisualization(meta) {
  currentTissueType = meta.tissue_type;
  showTissueMode(currentTissueType);
  gridColumns = meta.grid_columns;
  gridRows = meta.grid_rows;
  const aspect = meta.display_width / meta.display_height;
  document.documentElement.style.setProperty("--slide-ratio", String(aspect));
  heatmap.width = gridColumns;
  heatmap.height = gridRows;
  heatContext.clearRect(0, 0, gridColumns, gridRows);
  heatContext.fillStyle = "#080a0d";
  heatContext.fillRect(0, 0, gridColumns, gridRows);
  slideImage.src = `${meta.thumbnail_url}?t=${Date.now()}`;
  slideImage.hidden = false;
  originalEmpty.hidden = true;
  mapEmpty.hidden = true;
  originalCaption.textContent = "WHOLE-SLIDE OVERVIEW";
  slideDimensions.textContent = `${meta.slide_width.toLocaleString()} × ${meta.slide_height.toLocaleString()} PX`;
  mapState.textContent = `${gridColumns} × ${gridRows} REGIONS`;
  document.getElementById("batchLabel").textContent = meta.tissue_type === "lung"
    ? `${meta.batch_size} / ${meta.engine_batch_size}`
    : "—";
  document.getElementById("lymphBatchLabel").textContent = meta.batch_size;
  lymphSufficientRegions.textContent = "0";
  lymphTop5Average.textContent = "—";
  regionsValue.innerHTML = `0 <small>/ ${meta.total_regions.toLocaleString()}</small>`;
  progressFill.style.width = "0%";
  progressPercent.textContent = "0%";
  elapsedValue.innerHTML = "0.0 <small>SEC</small>";
  throughputValue.innerHTML = "— <small>REGIONS / SEC</small>";
  etaValue.textContent = "—";
  slideAssessment.hidden = true;
  etiologyRow.hidden = true;
  playback = {
    active: true,
    cells: [],
    head: 0,
    position: 0,
    available: 0,
    finished: false,
    rowMs: Math.max(100, Math.min(250, 6000 / gridRows)),
    onDone: null,
  };
  currentTarget = { x: 0, y: 0.5 / gridRows };
}

function updateClassRows(names, counts) {
  const maximum = Math.max(1, ...counts);
  const rows = classRows.querySelectorAll(".class-row");
  names.forEach((name, index) => {
    const row = rows[index];
    if (!row) return;
    row.querySelector(".class-name").lastChild.textContent = name.toUpperCase();
    row.querySelector(".class-count").textContent = counts[index].toLocaleString();
    row.querySelector(".class-fill").style.width = `${(counts[index] / maximum) * 100}%`;
  });
}

function showTissueMode(tissueType) {
  const lung = tissueType === "lung";
  const lymph = tissueType === "lymph_node";
  lungReadout.hidden = !lung;
  lymphReadout.hidden = !lymph;
  lymphSufficientRegions.textContent = "0";
  lymphTop5Average.textContent = "—";
  updateClassRows(["Cancer", "Granuloma", "Necrosis"], [0, 0, 0]);
  slideAssessment.hidden = true;
  runButton.querySelector("span:first-child").textContent = lung
    ? "RUN LUNG INFERENCE"
    : lymph ? "RUN LYMPH NODE DETECTION" : "SELECT TISSUE TYPE";
  mapState.textContent = lung ? "LUNG CLASS MAP" : lymph ? "LYMPHOCYTE DETECTION MAP" : "SELECT TISSUE TYPE";
  runButton.disabled = !modelsReady || !tissueType || inferenceRunning;
}

function releaseRunControls() {
  inferenceRunning = false;
  tissueSelect.disabled = false;
  runButton.disabled = !modelsReady || !tissueSelect.value;
}

function formatEta(seconds) {
  const value = Math.max(0, Math.ceil(seconds));
  if (value < 60) return `${value}s`;
  const minutes = Math.floor(value / 60);
  const remainder = String(value % 60).padStart(2, "0");
  return `${minutes}m ${remainder}s`;
}

function updateAssessment(result, names, tissueType) {
  let adequate;
  let observedResult = "";
  if (tissueType === "lung") {
    const counts = result.class_counts || [0, 0, 0];
    const qualifying = counts.map((count) => count > 50);
    adequate = qualifying.some(Boolean);
    assessmentRule.textContent = "> 50 POSITIVE REGIONS";
    assessmentResultLabel.textContent = "OBSERVED ETIOLOGY";
    if (adequate) {
      const largestQualifyingCount = Math.max(...counts.filter((_, index) => qualifying[index]));
      observedResult = names
        .filter((_, index) => qualifying[index] && counts[index] === largestQualifyingCount)
        .join(" / ");
    }
  } else {
    adequate = Number(result.top5_average || 0) > 40;
    assessmentRule.textContent = "TOP-5 AVERAGE > 40 DETECTIONS";
    assessmentResultLabel.textContent = "LYMPH NODE STATUS";
    observedResult = "LYMPHOCYTE SUFFICIENT";
  }
  slideAssessment.hidden = false;
  adequacyValue.textContent = adequate ? "ADEQUATE" : "INADEQUATE";
  adequacyValue.className = `assessment-value ${adequate ? "adequate" : "inadequate"}`;
  etiologyRow.hidden = !adequate;
  if (adequate) etiologyValue.textContent = observedResult;
}

function colorForProbabilities(probabilities) {
  let strongest = 0;
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[strongest]) strongest = i;
  }
  const confidence = probabilities[strongest];
  if (confidence < logitThresholdProbability) return "rgba(126, 145, 157, .18)";
  const alpha = Math.min(.93, .28 + (confidence - logitThresholdProbability) * 1.55);
  const hex = classColors[strongest].slice(1);
  const red = parseInt(hex.slice(0, 2), 16);
  const green = parseInt(hex.slice(2, 4), 16);
  const blue = parseInt(hex.slice(4, 6), 16);
  return `rgba(${red},${green},${blue},${alpha})`;
}

function drawCells(cells, tissueType) {
  for (const cell of cells) {
    const [index, first, second, third] = cell;
    const x = index % gridColumns;
    const y = Math.floor(index / gridColumns);
    if (tissueType === "lymph_node") {
      heatContext.fillStyle = first > 40 ? "rgba(46,230,120,.88)" : "rgba(126,145,157,.15)";
    } else {
      heatContext.fillStyle = colorForProbabilities([first, second, third]);
    }
    heatContext.fillRect(x, y, 1, 1);
  }
}

function onProgress(data, names) {
  if (data.cells && data.cells.length) playback.cells.push(...data.cells);
  playback.available = Math.max(playback.available, Math.min(gridRows, data.processed_regions / gridColumns));
  const ratio = Math.max(0, Math.min(1, data.progress || 0));
  progressFill.style.width = `${ratio * 100}%`;
  progressPercent.textContent = `${(ratio * 100).toFixed(1)}%`;
  regionsValue.innerHTML = `${data.processed_regions.toLocaleString()} <small>/ ${data.total_regions.toLocaleString()}</small>`;
  elapsedValue.innerHTML = `${(data.elapsed_seconds || 0).toFixed(1)} <small>SEC</small>`;
  throughputValue.innerHTML = `${Math.round(data.regions_per_second || 0).toLocaleString()} <small>REGIONS / SEC</small>`;
  const remaining = data.total_regions - data.processed_regions;
  etaValue.textContent = data.regions_per_second > 0 && remaining > 0
    ? `~${formatEta(remaining / data.regions_per_second)}`
    : remaining <= 0 && data.processed_regions > 0 ? "0s" : "Estimating";
  if (data.tissue_type === "lung") {
    updateClassRows(names, data.class_counts || [0, 0, 0]);
  } else {
    lymphSufficientRegions.textContent = (data.sufficient_regions || 0).toLocaleString();
    lymphTop5Average.textContent = Number(data.top5_average || 0).toFixed(1);
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error(await response.text());
    const status = await response.json();
    pathInput.value = status.example_slide || pathInput.value;
    engineState.textContent = "MODELS READY";
    gpuLabel.textContent = status.gpu.toUpperCase();
    readyDot.classList.remove("offline");
    modelsReady = true;
    showTissueMode(tissueSelect.value);
    document.getElementById("batchLabel").textContent = `${status.inference_batch_size} / ${status.batch_size}`;
    document.getElementById("lymphBatchLabel").textContent = status.lymph_batch_size;
  } catch (error) {
    modelsReady = false;
    engineState.textContent = "MODELS UNAVAILABLE";
    readyDot.classList.add("offline");
    tissueSelect.disabled = true;
    runButton.disabled = true;
    setError(`The inference service is not ready: ${error.message}`);
  }
}

async function runInference() {
  if (source) {
    source.close();
    source = null;
  }
  const tissueType = tissueSelect.value;
  if (!tissueType) {
    setError("Choose a tissue type before starting inference.");
    return;
  }
  setError("");
  inferenceRunning = true;
  runButton.disabled = true;
  tissueSelect.disabled = true;
  setStatus(tissueType === "lung" ? "READING LUNG SLIDE" : "READING LYMPH NODE SLIDE", "running");
  mapState.textContent = "PREPARING MAP";
  try {
    const response = await fetch("/api/inference", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ slide_path: pathInput.value.trim(), tissue_type: tissueType }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Request failed (${response.status})`);
    }
    const meta = await response.json();
    resetVisualization(meta);
    const statusResponse = await fetch("/api/status");
    const status = await statusResponse.json();
    const names = status.classes;
    if (tissueType === "lung") updateClassRows(names, [0, 0, 0]);
    setStatus(tissueType === "lung" ? "SCANNING LUNG" : "SCANNING LYMPH NODE", "running");
    mapState.textContent = tissueType === "lung" ? "LUNG CLASS MAP" : "LYMPHOCYTE DETECTION MAP";
    source = new EventSource(`/api/jobs/${meta.job_id}/events`);
    source.addEventListener("progress", (event) => {
      onProgress(JSON.parse(event.data), names);
    });
    source.addEventListener("complete", (event) => {
      const result = JSON.parse(event.data);
      onProgress(result, names);
      progressFill.style.width = "100%";
      progressPercent.textContent = "100.0%";
      regionsValue.innerHTML = `${result.total_regions.toLocaleString()} <small>/ ${result.total_regions.toLocaleString()}</small>`;
      playback.finished = true;
      playback.onDone = () => {
        updateAssessment(result, names, tissueType);
        setStatus("INFERENCE COMPLETE", "complete");
        mapState.textContent = "INFERENCE COMPLETE";
        releaseRunControls();
      };
      source.close();
      source = null;
    });
    source.addEventListener("job-error", (event) => {
      const detail = JSON.parse(event.data);
      setStatus("INFERENCE STOPPED");
      mapState.textContent = "INFERENCE ERROR";
      setError(detail.message || "Inference failed.");
      playback.active = false;
      playback.onDone = null;
      releaseRunControls();
      source.close();
      source = null;
    });
    source.onerror = () => {
      if (source && source.readyState === EventSource.CLOSED) {
        setStatus("CONNECTION CLOSED");
        setError("The progress stream closed unexpectedly. You can run the slide again.");
        releaseRunControls();
        source.close();
        source = null;
      }
    };
  } catch (error) {
    setStatus("STANDING BY");
    mapState.textContent = "NO PREDICTIONS";
    setError(error.message);
    releaseRunControls();
  }
}

tissueSelect.addEventListener("change", () => showTissueMode(tissueSelect.value));
runButton.addEventListener("click", runInference);
pathInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !runButton.disabled) runInference();
});
window.addEventListener("resize", resizeTargetCanvases);
loadStatus();
requestAnimationFrame(animate);
