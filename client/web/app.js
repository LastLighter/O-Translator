"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const els = {
  stage: $("#stage"),
  dropState: $("#dropState"),
  previewState: $("#previewState"),
  baseImage: $("#baseImage"),
  originalImage: $("#originalImage"),
  compareLayer: $("#compareLayer"),
  compareHandle: $("#compareHandle"),
  compareControl: $("#compareControl"),
  compareRange: $("#compareRange"),
  previewGrid: $(".preview-grid"),
  fileInput: $("#fileInput"),
  chooseBtn: $("#chooseBtn"),
  demoBtn: $("#demoBtn"),
  resetBtn: $("#resetBtn"),
  runBtn: $("#runBtn"),
  sourceLang: $("#sourceLang"),
  targetLang: $("#targetLang"),
  readingOrder: $("#readingOrder"),
  modeGroup: $("#modeGroup"),
  dilatePx: $("#dilatePx"),
  dilateValue: $("#dilateValue"),
  fontScale: $("#fontScale"),
  fontScaleValue: $("#fontScaleValue"),
  translateToggle: $("#translateToggle"),
  processStep: $("#processStep"),
  processDetail: $("#processDetail"),
  progressBar: $("#progressBar"),
  progressValue: $("#progressValue"),
  processingState: $("#processingState"),
  healthBadge: $("#healthBadge"),
  healthText: $("#healthText"),
  providerNotice: $("#providerNotice"),
  providerNoticeText: $("#providerNoticeText"),
  canvasMeta: $("#canvasMeta"),
  resultBar: $("#resultBar"),
  engineStats: $("#engineStats"),
  downloadBtn: $("#downloadBtn"),
  copyBtn: $("#copyBtn"),
  warningsBar: $("#warningsBar"),
  translationsCard: $("#translationsCard"),
  translationList: $("#translationList"),
  lineCount: $("#lineCount"),
  timeline: $("#timeline"),
  viewTabs: $("#viewTabs"),
  toast: $("#toast"),
};

const state = {
  file: null,
  objectUrl: null,
  selectedWidth: 0,
  selectedHeight: 0,
  mode: "auto",
  jobId: null,
  result: null,
  health: null,
  busy: false,
  view: "result",
  pollFailures: 0,
};

const VIEW_URLS = {
  result: () => state.result?.image_url,
  clean: () => state.result?.clean_url,
  mask: () => state.result?.mask_url,
  original: () => state.result?.original_url,
};

const STAGE_LABELS = {
  queued: ["已进入队列", "正在等待工作线程"],
  decoding: ["正在读取截图", "解码、校正方向并缩放"],
  ocr: ["正在识别文字", "PaddleOCR 正在检测文本框"],
  masking: ["正在生成蒙版", "膨胀文字区域，准备修复"],
  processing: ["并行处理中", "LaMa / OpenCV 修复背景，同时请求 DeepSeek"],
  inpainting: ["背景修复完成", "正在等待翻译结果"],
  translation: ["上下文翻译完成", "正在准备排版"],
  rendering: ["正在自适应排版", "匹配原框尺寸、颜色与字号"],
  saving: ["正在保存结果", "写入原图、无字背景、蒙版与成图"],
  complete: ["处理完成", "译文已叠加到画布"],
  failed: ["处理失败", "请查看下方错误信息"],
  cancelled: ["任务已取消", ""],
};

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.health = health;
    applyHealth(health);
  } catch (error) {
    els.healthBadge.className = "health-badge is-error";
    els.healthText.textContent = "服务未连接";
    setProviderNotice("无法连接 O-Translator 服务。请确认 FastAPI 已启动。", "error");
  }
}

function applyHealth(health) {
  const translation = health.engines?.translation || {};
  const ocr = health.engines?.ocr || {};
  const ocrBroken = !ocr.active || ocr.active === "none" || ocr.active === "unavailable";
  els.healthBadge.className = `health-badge ${ocrBroken ? "is-error" : translation.deepseek_configured ? "is-ok" : "is-warn"}`;
  els.healthText.textContent = ocrBroken
    ? "OCR 不可用"
    : translation.deepseek_configured
      ? "DeepSeek 已连接"
      : "离线模式";

  if (translation.deepseek_configured) {
    setProviderNotice(
      `DeepSeek ${escapeText(translation.model || "V3")} 已连接 · 整页上下文批量翻译`,
      "ok",
    );
  } else if (ocrBroken) {
    setProviderNotice(
      "OCR 引擎不可用。请安装 PaddleOCR / paddlepaddle 或 Tesseract 后重启服务。",
      "error",
    );
  } else {
    setProviderNotice(
      "未配置 DEEPSEEK_API_KEY：仍可完成 OCR、背景修复与排版，但会保留原文。",
      "warn",
    );
  }
}

function setProviderNotice(text, tone) {
  els.providerNotice.hidden = false;
  els.providerNotice.classList.toggle("is-ok", tone === "ok");
  els.providerNotice.classList.toggle("is-error", tone === "error");
  els.providerNoticeText.textContent = text;
}

function escapeText(value) {
  return String(value ?? "");
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.classList.toggle("is-error", isError);
  els.toast.classList.add("is-visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => els.toast.classList.remove("is-visible"), 3200);
}

function setMode(mode) {
  state.mode = mode;
  $$("button", els.modeGroup).forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-checked", String(active));
  });
}

function updateSliderLabels() {
  els.dilateValue.value = `${els.dilatePx.value} px`;
  els.fontScaleValue.value = `${Number(els.fontScale.value).toFixed(2)}×`;
}

function updateCanvasMeta() {
  if (!state.file) {
    els.canvasMeta.textContent = "NO IMAGE";
    return;
  }
  const dimensions = state.selectedWidth && state.selectedHeight
    ? `${state.selectedWidth} × ${state.selectedHeight}`
    : "IMAGE";
  els.canvasMeta.textContent = `${dimensions} · ${formatBytes(state.file.size)}`;
}

function setBusy(busy) {
  state.busy = busy;
  els.runBtn.disabled = busy || !state.file;
  els.runBtn.classList.toggle("is-busy", busy);
  els.runBtn.querySelector(".run-btn-label").textContent = busy ? "处理中" : "开始翻译";
  [els.sourceLang, els.targetLang, els.readingOrder, els.dilatePx, els.fontScale, els.translateToggle]
    .forEach((control) => { control.disabled = busy; });
  $$("button", els.modeGroup).forEach((button) => { button.disabled = busy; });
}

function revokeObjectUrl() {
  if (state.objectUrl) {
    URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
  }
}

async function selectFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showToast("请选择 PNG、JPEG、WEBP 或 BMP 图片。", true);
    return;
  }
  const maxBytes = (state.health?.limits?.max_upload_mb || 20) * 1024 * 1024;
  if (file.size > maxBytes) {
    showToast(`图片超过 ${Math.round(maxBytes / 1024 / 1024)} MB 限制。`, true);
    return;
  }

  resetWorkspace({ keepFile: true, silent: true });
  revokeObjectUrl();
  state.file = file;
  state.objectUrl = URL.createObjectURL(file);
  els.baseImage.src = state.objectUrl;
  els.originalImage.src = state.objectUrl;
  els.dropState.hidden = true;
  els.previewState.hidden = false;
  els.resetBtn.hidden = false;
  els.compareLayer.hidden = true;
  els.compareHandle.hidden = true;
  els.compareControl.hidden = true;
  setTabsEnabled(false);
  setView("original");
  updateCanvasMeta();
  setBusy(false);

  try {
    const dimensions = await imageDimensions(state.objectUrl);
    state.selectedWidth = dimensions.width;
    state.selectedHeight = dimensions.height;
    updateCanvasMeta();
  } catch {
    /* Preview decode failure will be reported by the server. */
  }
}

function imageDimensions(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = reject;
    image.src = url;
  });
}

function setTabsEnabled(enabled) {
  $$(".view-tab", els.viewTabs).forEach((tab) => {
    tab.disabled = !enabled && tab.dataset.view !== "original";
  });
}

function setView(view) {
  state.view = view;
  const resultReady = Boolean(state.result);
  let url = VIEW_URLS[view]?.();
  if (!url && view === "original") url = state.objectUrl;
  if (url) {
    els.baseImage.src = state.result
      ? `${url}?v=${encodeURIComponent(state.result.job_id)}`
      : url;
  }
  const compareVisible = view === "result" && resultReady;
  els.compareLayer.hidden = !compareVisible;
  els.compareHandle.hidden = !compareVisible;
  els.compareControl.hidden = !compareVisible;
  if (compareVisible) {
    els.originalImage.src = `${state.result.original_url}?v=${encodeURIComponent(state.result.job_id)}`;
    setSplit(Number(els.compareRange.value));
  }
  $$(".view-tab", els.viewTabs).forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.view === view);
  });
}

function setSplit(percent) {
  const clamped = Math.max(0, Math.min(100, percent));
  els.compareRange.value = String(clamped);
  els.previewGrid.style.setProperty("--split", `${clamped}%`);
}

function resetWorkspace({ keepFile = false, silent = false } = {}) {
  state.jobId = null;
  state.result = null;
  state.pollFailures = 0;
  if (!keepFile) {
    state.file = null;
    state.selectedWidth = 0;
    state.selectedHeight = 0;
    revokeObjectUrl();
    els.fileInput.value = "";
    els.baseImage.removeAttribute("src");
    els.originalImage.removeAttribute("src");
    els.dropState.hidden = false;
    els.previewState.hidden = true;
    els.resetBtn.hidden = true;
    els.canvasMeta.textContent = "NO IMAGE";
  }
  els.processingState.hidden = true;
  els.progressBar.style.width = "0%";
  els.progressValue.textContent = "0%";
  els.resultBar.hidden = true;
  els.warningsBar.hidden = true;
  els.warningsBar.replaceChildren();
  els.translationsCard.hidden = true;
  els.translationList.replaceChildren();
  els.lineCount.textContent = "0 lines";
  els.engineStats.replaceChildren();
  setTimeline("idle");
  setBusy(false);
  setTabsEnabled(false);
  setView("original");
  if (!silent) showToast(keepFile ? "已清空结果，可以重新翻译。" : "画布已清空。");
}

async function runPipeline() {
  if (!state.file || state.busy) return;
  if (!state.health) await loadHealth();

  state.result = null;
  state.jobId = null;
  state.pollFailures = 0;
  setBusy(true);
  setTabsEnabled(false);
  setView("original");
  els.compareLayer.hidden = true;
  els.compareHandle.hidden = true;
  els.compareControl.hidden = true;
  els.processingState.hidden = false;
  els.progressBar.style.width = "3%";
  els.progressValue.textContent = "0%";
  els.processStep.textContent = "正在上传截图";
  els.processDetail.textContent = "创建异步翻译任务…";
  els.resultBar.hidden = true;
  els.warningsBar.hidden = true;
  els.translationsCard.hidden = true;
  setTimeline("start");

  const form = new FormData();
  form.append("file", state.file, state.file.name || "screenshot.png");
  form.append("source_lang", els.sourceLang.value);
  form.append("target_lang", els.targetLang.value);
  form.append("mode", state.mode);
  form.append("reading_order", els.readingOrder.value);
  form.append("dilate_px", els.dilatePx.value);
  form.append("font_scale", els.fontScale.value);
  form.append("translate", String(els.translateToggle.checked));

  try {
    const response = await fetch("/api/jobs", { method: "POST", body: form });
    const payload = await parseResponse(response);
    if (!response.ok) throw new Error(payload.message || payload.detail || "任务创建失败");
    state.jobId = payload.id;
    setTimeline("ocr", "is-active");
    await pollJob(payload.id);
  } catch (error) {
    handleJobError(error.message || "请求失败");
  }
}

async function parseResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { message: text };
  }
}

async function pollJob(jobId) {
  let delay = 650;
  while (state.jobId === jobId) {
    await new Promise((resolve) => setTimeout(resolve, delay));
    try {
      const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      const job = await parseResponse(response);
      if (!response.ok) throw new Error(job.message || `HTTP ${response.status}`);
      state.pollFailures = 0;
      applyJobState(job);
      if (["completed", "failed", "cancelled"].includes(job.status)) {
        if (job.status === "completed") {
          showResult(job);
        } else {
          handleJobError(job.error?.message || job.message || "任务失败");
        }
        return;
      }
      delay = 850;
    } catch (error) {
      state.pollFailures += 1;
      if (state.pollFailures >= 5) {
        handleJobError(error.message || "无法获取任务状态");
        return;
      }
      delay = Math.min(3000, 900 * state.pollFailures);
    }
  }
}

function applyJobState(job) {
  const progress = Number(job.progress || 0);
  const [title, detail] = STAGE_LABELS[job.stage] || ["处理中", job.message || ""];
  els.processStep.textContent = title;
  els.processDetail.textContent = detail || job.message || "";
  els.progressBar.style.width = `${progress}%`;
  els.progressValue.textContent = `${progress}%`;
  updateTimelineForStage(job.stage, job.status);
}

function updateTimelineForStage(stage, status) {
  const steps = ["ocr", "inpaint", "translate", "render"];
  steps.forEach((step) => setTimelineStep(step, ""));
  const active = (step) => setTimelineStep(step, "is-active");
  const done = (...items) => items.forEach((step) => setTimelineStep(step, "is-done"));
  if (status === "failed" || status === "cancelled") {
    setTimelineStep("ocr", "is-error");
    return;
  }
  if (["queued", "decoding"].includes(stage)) {
    active("ocr");
  } else if (stage === "ocr") {
    active("ocr");
  } else if (stage === "masking") {
    done("ocr");
    active("inpaint");
  } else if (stage === "processing") {
    done("ocr");
    active("inpaint");
    active("translate");
  } else if (stage === "inpainting") {
    done("ocr");
    done("inpaint");
    active("translate");
  } else if (stage === "translation") {
    done("ocr");
    done("translate");
    active("inpaint");
  } else if (stage === "rendering") {
    done("ocr", "inpaint", "translate");
    active("render");
  } else if (["saving", "complete"].includes(stage)) {
    done("ocr", "inpaint", "translate", "render");
  }
}

function setTimeline(step, className = "") {
  if (step === "start") {
    $$("li", els.timeline).forEach((item) => { item.className = ""; });
    setTimelineStep("ocr", "is-active");
    return;
  }
  if (step === "idle") {
    $$("li", els.timeline).forEach((item) => { item.className = ""; });
    return;
  }
  setTimelineStep(step, className);
}

function setTimelineStep(step, className) {
  const item = $(`li[data-step="${step}"]`, els.timeline);
  if (!item) return;
  item.classList.remove("is-active", "is-done", "is-error");
  if (className) item.classList.add(className);
}

function showResult(job) {
  const result = job.result;
  if (!result) {
    handleJobError("任务完成但未返回结果。");
    return;
  }
  state.result = result;
  state.busy = false;
  setBusy(false);
  els.processingState.hidden = true;
  setTabsEnabled(true);
  setView("result");
  setSplit(50);
  els.compareLayer.hidden = false;
  els.compareHandle.hidden = false;
  els.compareControl.hidden = false;
  els.resultBar.hidden = false;
  els.downloadBtn.href = `/api/jobs/${job.id}/download`;
  els.canvasMeta.textContent = `${result.width} × ${result.height} · 译文成图`;
  renderEngineStats(result);
  renderWarnings(job.warnings || result.warnings || []);
  renderTranslations(result);
  if (result.empty) {
    showToast("未检测到文字，画布保持原图。");
  } else {
    showToast("翻译与排版完成。");
  }
}

function renderEngineStats(result) {
  els.engineStats.replaceChildren();
  const engines = result.engines || {};
  const add = (label, value) => {
    const chip = document.createElement("span");
    chip.className = "engine-chip";
    const dot = document.createElement("i");
    dot.className = "dot";
    const text = document.createElement("span");
    text.textContent = label;
    const bold = document.createElement("b");
    bold.textContent = String(value);
    chip.append(dot, text, bold);
    els.engineStats.append(chip);
  };
  add("OCR", engines.ocr || "—");
  add("INPAINT", engines.inpainting || "—");
  add("TRANSLATE", result.translation?.provider || "—");
  const timings = result.timings_ms || {};
  if (timings.total_ms) {
    const chip = document.createElement("span");
    chip.className = "engine-chip";
    chip.textContent = `TOTAL ${(timings.total_ms / 1000).toFixed(2)}s`;
    els.engineStats.append(chip);
  }
}

function renderWarnings(warnings) {
  els.warningsBar.replaceChildren();
  els.warningsBar.hidden = warnings.length === 0;
  warnings.forEach((rawMessage) => {
    const message = localizeWarning(rawMessage);
    const item = document.createElement("div");
    item.className = "warning";
    item.textContent = message;
    els.warningsBar.append(item);
  });
}

function localizeWarning(message) {
  const text = String(message || "");
  if (text.includes("DeepSeek is not configured")) {
    return "未配置 DeepSeek API Key，当前保留 OCR 原文。设置 DEEPSEEK_API_KEY 并重启服务即可启用翻译。";
  }
  if (text.includes("LaMa weights were not found")) {
    return "未找到 LaMa 模型，已自动使用 OpenCV Telea 修复背景。运行 scripts/download_models.py 可启用 LaMa。";
  }
  const overflow = text.match(/Text box (\d+) overflowed/);
  if (overflow) {
    return `第 ${Number(overflow[1]) + 1} 个文本框在最小字号下仍然溢出，建议缩短译文或调整排版。`;
  }
  if (text.includes("No text was detected")) {
    return "未检测到文字，画布保持原图。";
  }
  if (text.includes("no text detected")) {
    return "OCR 未检测到文字，已尝试其他可用引擎。";
  }
  if (text.includes("Source and target languages are identical")) {
    return "源语言与目标语言相同，已跳过翻译。";
  }
  if (text.includes("Translation was disabled")) {
    return "本任务已关闭翻译，仅执行 OCR、背景修复与排版。";
  }
  return text;
}

function renderTranslations(result) {
  const regions = Array.isArray(result.regions) ? result.regions : [];
  els.translationsCard.hidden = false;
  els.lineCount.textContent = `${regions.length} lines`;
  els.translationList.replaceChildren();
  if (regions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-lines";
    empty.textContent = "没有检测到文字。可以更换截图，或调整源语言后重试。";
    els.translationList.append(empty);
    return;
  }
  regions.forEach((region, index) => {
    const row = document.createElement("div");
    row.className = "translation-row";

    const counter = document.createElement("span");
    counter.className = "line-index";
    counter.textContent = String(index + 1).padStart(2, "0");

    const source = document.createElement("div");
    source.className = "source-text";
    source.textContent = region.text || "—";

    const arrow = document.createElement("span");
    arrow.className = "line-arrow";
    arrow.textContent = "→";

    const target = document.createElement("div");
    const translated = document.createElement("div");
    translated.className = "translation-text";
    translated.textContent = region.translation || region.text || "—";
    const meta = document.createElement("div");
    meta.className = "line-meta";
    const swatch = document.createElement("i");
    swatch.className = "color-swatch";
    const color = region.text_color || blockTextColor(result, region.id) || [255, 255, 255];
    swatch.style.background = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    const fit = findBlock(result, region.id);
    const details = document.createElement("span");
    details.textContent = fit ? `${fit.font_size}px · ${fit.layout}` : `${Math.round((region.confidence || 0) * 100)}% conf`;
    meta.append(swatch, details);
    if (fit?.overflow) {
      const overflow = document.createElement("span");
      overflow.className = "overflow-tag";
      overflow.textContent = "OVERFLOW";
      meta.append(overflow);
    }
    target.append(translated, meta);
    row.append(counter, source, arrow, target);
    els.translationList.append(row);
  });
}

function findBlock(result, regionId) {
  return (result.blocks || []).find((block) => block.id === regionId);
}

function blockTextColor(result, regionId) {
  return findBlock(result, regionId)?.text_color;
}

function handleJobError(message) {
  state.busy = false;
  setBusy(false);
  els.processingState.hidden = true;
  setTimeline("ocr", "is-error");
  els.warningsBar.hidden = false;
  els.warningsBar.replaceChildren();
  const item = document.createElement("div");
  item.className = "warning is-error";
  item.textContent = message;
  els.warningsBar.append(item);
  showToast(message, true);
  if (!state.result) setView("original");
}

async function loadDemo() {
  try {
    const response = await fetch("/static/assets/demo.png", { cache: "force-cache" });
    if (!response.ok) throw new Error("示例图片缺失");
    const blob = await response.blob();
    const file = new File([blob], "o-translator-demo.png", { type: "image/png" });
    await selectFile(file);
    showToast("示例已载入，点击「开始翻译」体验完整流程。");
  } catch (error) {
    showToast(error.message || "无法载入示例。", true);
  }
}

function bindEvents() {
  els.chooseBtn.addEventListener("click", () => els.fileInput.click());
  els.demoBtn.addEventListener("click", loadDemo);
  els.fileInput.addEventListener("change", () => selectFile(els.fileInput.files?.[0]));
  els.resetBtn.addEventListener("click", () => resetWorkspace());

  els.runBtn.addEventListener("click", runPipeline);
  $$("button", els.modeGroup).forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  els.dilatePx.addEventListener("input", updateSliderLabels);
  els.fontScale.addEventListener("input", updateSliderLabels);

  els.viewTabs.addEventListener("click", (event) => {
    const tab = event.target.closest(".view-tab");
    if (!tab || tab.disabled) return;
    setView(tab.dataset.view);
  });

  els.compareRange.addEventListener("input", () => setSplit(Number(els.compareRange.value)));
  els.previewGrid.addEventListener("pointerdown", (event) => {
    if (!state.result || state.view !== "result" || els.compareLayer.hidden) return;
    const rect = els.previewGrid.getBoundingClientRect();
    const split = ((event.clientX - rect.left) / rect.width) * 100;
    setSplit(split);
  });

  ["dragenter", "dragover"].forEach((name) => {
    els.stage.addEventListener(name, (event) => {
      event.preventDefault();
      els.stage.classList.add("is-dragging");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    els.stage.addEventListener(name, (event) => {
      event.preventDefault();
      if (name === "dragleave" && event.relatedTarget && els.stage.contains(event.relatedTarget)) return;
      els.stage.classList.remove("is-dragging");
    });
  });
  els.stage.addEventListener("drop", (event) => {
    const file = Array.from(event.dataTransfer?.files || []).find((item) => item.type.startsWith("image/"));
    if (file) selectFile(file);
  });

  window.addEventListener("paste", (event) => {
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return;
    const item = Array.from(event.clipboardData?.items || []).find((entry) => entry.type.startsWith("image/"));
    if (item) {
      const file = item.getAsFile();
      if (file) selectFile(file);
    }
  });

  window.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      runPipeline();
    }
    if (event.key === "Escape" && state.file && !state.busy && !els.previewState.hidden) {
      resetWorkspace();
    }
  });

  els.copyBtn.addEventListener("click", async () => {
    if (!state.result) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(state.result, null, 2));
      showToast("结果 JSON 已复制到剪贴板。");
    } catch {
      showToast("浏览器拒绝访问剪贴板。", true);
    }
  });
}

async function init() {
  bindEvents();
  updateSliderLabels();
  setMode("auto");
  setBusy(false);
  await nextFrame();
  loadHealth();
}

document.addEventListener("DOMContentLoaded", init);
