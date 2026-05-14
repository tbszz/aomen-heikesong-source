type Mode = "train" | "eval" | "unknown" | "demo";

type RecorderState = {
  isRecording: boolean;
  buffers: Float32Array[];
  sampleRate: number;
  stream: MediaStream | null;
  audioContext: AudioContext | null;
  sourceNode: MediaStreamAudioSourceNode | null;
  processorNode: ScriptProcessorNode | null;
  gainNode: GainNode | null;
  startedAtMs: number;
};

type PhraseRow = {
  phrase_id: string;
  text: string;
  stage_pack: boolean;
  train: number;
  eval: number;
  corrections: number;
  rejected: number;
  template_count: number;
};

type PhraseResponse = { ok: boolean; phrases: PhraseRow[]; error?: string };

type MatchDebug = {
  engine?: string | null;
  engine_id?: string | null;
  reject_reason?: string | null;
  best_phrase?: string | null;
  best_phrase_id?: string | null;
  second_phrase?: string | null;
  second_phrase_id?: string | null;
  best_dist?: number | null;
  second_dist?: number | null;
  gap?: number | null;
  ratio?: number | null;
};

type DemoResponse = {
  ok: boolean;
  error?: string;
  quality?: Record<string, unknown>;
  raw_audio_url?: string;
  final_text?: string | null;
  decision_source?: string | null;
  reason?: string | null;
  local_debug?: MatchDebug;
  cloud_debug?: Record<string, unknown>;
  normalized_phrase_id?: string | null;
  normalized_phrase_text?: string | null;
  matched_phrase?: string | null;
  tts_audio_url?: string | null;
  audio_match_debug?: MatchDebug;
  engine_id?: string | null;
  top1_score?: number | null;
  score?: number | null;
  margin?: number | null;
  latency_ms?: Record<string, number | null>;
};

type UploadResponse = {
  ok: boolean;
  error?: string;
  quality?: Record<string, unknown>;
  sample?: Record<string, unknown>;
  event?: Record<string, unknown>;
  auto_correction?: CorrectionsFromEvalResponse;
  is_error_event?: boolean;
  next_action?: "none" | "rerecord_truth";
  next_truth_phrase_id?: string | null;
  summary?: EvalSummary;
  phrase_counts?: Record<string, Record<string, number>>;
  rebuild_ok?: boolean | null;
  template_count?: number;
  rebuild_policy?: string;
  rebuild_triggered?: boolean;
  duration_ms?: number | null;
  index_state?: string;
};

type EvalSummary = {
  total: number;
  accepted: number;
  top1: number;
  top2: number;
  top1_rate: number;
  top2_rate: number;
  reject_rate: number;
  failure_reasons: Record<string, number>;
  per_phrase: Record<string, Record<string, number>>;
  confusion_matrix: Record<string, Record<string, number>>;
};

type EvalExportResponse = { ok: boolean; count: number; summary: EvalSummary; events: Record<string, unknown>[] };
type RuntimeConfig = {
  no_reject_mode: boolean;
  selection_mode?: string;
  fallback_margin_threshold?: number;
  fallback_score_threshold?: number;
};
type RuntimeConfigResponse = { ok: boolean; config: RuntimeConfig; error?: string };
type EvalErrorItem = {
  event_id: string;
  truth_phrase_id?: string | null;
  truth_text?: string | null;
  predicted_phrase_id?: string | null;
  predicted_text?: string | null;
  best_phrase_id?: string | null;
  best_text?: string | null;
  score?: number | null;
  margin?: number | null;
  gap?: number | null;
  ratio?: number | null;
  reject_reason?: string | null;
  diagnosis_code?: string | null;
  diagnosis_text?: string | null;
  suggested_fix?: string | null;
  audio_url?: string | null;
  is_error?: boolean;
  created_at?: string | null;
};
type EvalErrorsAnalysis = {
  confusion_pairs?: Array<{
    truth_phrase_id?: string;
    truth_text?: string;
    pred_phrase_id?: string;
    pred_text?: string;
    count?: number;
  }>;
  reason_counts?: Record<string, number>;
  per_phrase_error_rate?: Array<{
    phrase_id?: string;
    phrase_text?: string;
    total?: number;
    errors?: number;
    error_rate?: number;
  }>;
  low_margin_cases?: Array<Record<string, unknown>>;
  accepted_wrong_cases?: Array<Record<string, unknown>>;
};
type EvalErrorsResponse = {
  ok: boolean;
  count: number;
  total_eval_events: number;
  low_margin_threshold: number;
  items: EvalErrorItem[];
  analysis: EvalErrorsAnalysis;
  error?: string;
};
type CorrectionsFromEvalResponse = {
  ok: boolean;
  added: number;
  skipped: Array<Record<string, unknown>>;
  errors: Array<Record<string, unknown>>;
  corrections_count?: number;
  index_state?: string;
  error?: string;
};
type CorrectionsConfirmResponse = {
  ok: boolean;
  moved: number;
  rebuild_ok: boolean | null;
  rebuild_ms?: number | null;
  auto_purify?: {
    disabled_count?: number;
    protected_by_min_active?: number;
  } | null;
  error?: string;
};
type ActiveRerecordTask = {
  truth_phrase_id: string;
  from_eval_event_id: string | null;
  rerecord_batch_id: string;
  required: number;
  completed: number;
};
type PendingCorrectionItem = {
  sample_id: string;
  truth_phrase_id?: string | null;
  created_at?: string | null;
  from_eval_event_id?: string | null;
  rerecord_batch_id?: string | null;
  raw_audio_url?: string | null;
};
type PendingCorrectionsResponse = {
  ok: boolean;
  count: number;
  items: PendingCorrectionItem[];
  error?: string;
};
type UnknownSummary = {
  total: number;
  false_accepts: number;
  correct_rejects: number;
  false_accept_rate: number;
  failure_reasons: Record<string, number>;
  accepted_by_phrase: Record<string, number>;
};
type UnknownExportResponse = { ok: boolean; count: number; summary: UnknownSummary; events: Record<string, unknown>[]; engine_id?: string };

const TARGET_SAMPLE_RATE = 16000;
const MIN_CLIENT_RECORDING_MS = 800;
const TRAIN_TARGET = 15;
const EVAL_TARGET = 15;
const REQUIRED_RERECORD_COUNT = 5;
const LOCKED_PHRASE_IDS = new Set(["p01_he_shui", "p02_chi_fan", "p03_qing_bang_wo", "p04_bu_shu_fu"]);
const LOCKED_PHRASE_TEXTS = new Set(["我想喝水。", "我想吃饭啊。", "请帮我啊。", "我不舒服。", "我想喝水", "我想吃饭啊", "请帮我啊", "我不舒服"]);

const state: RecorderState = {
  isRecording: false,
  buffers: [],
  sampleRate: TARGET_SAMPLE_RATE,
  stream: null,
  audioContext: null,
  sourceNode: null,
  processorNode: null,
  gainNode: null,
  startedAtMs: 0,
};

let currentMode: Mode = "train";
let phrases: PhraseRow[] = [];
let selectedDeviceId = "";
let lastDemoBlob: Blob | null = null;
let lastDemoRawUrl = "";
let lastDemoPredictedId = "";
let runtimeConfig: RuntimeConfig | null = null;
let evalErrorItems: EvalErrorItem[] = [];
const selectedErrorIds = new Set<string>();
let currentErrorFilter: "all" | "accepted_wrong" | "rejected" | "low_margin" = "all";
let pendingRerecordTruthId = "";
let activeRerecordTask: ActiveRerecordTask | null = null;
let currentPendingCorrections: PendingCorrectionItem[] = [];

const byId = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const statusEl = byId<HTMLDivElement>("status");
const holdBtn = byId<HTMLButtonElement>("holdBtn");
const modeTrainBtn = byId<HTMLButtonElement>("modeTrain");
const modeEvalBtn = byId<HTMLButtonElement>("modeEval");
const modeUnknownBtn = byId<HTMLButtonElement>("modeUnknown");
const modeDemoBtn = byId<HTMLButtonElement>("modeDemo");
const trainPanel = byId<HTMLDivElement>("trainPanel");
const evalPanel = byId<HTMLDivElement>("evalPanel");
const unknownPanel = byId<HTMLDivElement>("unknownPanel");
const demoPanel = byId<HTMLDivElement>("demoPanel");
const trainPhraseSelect = byId<HTMLSelectElement>("trainPhraseSelect");
const evalPhraseSelect = byId<HTMLSelectElement>("evalPhraseSelect");
const correctionPhraseSelect = byId<HTMLSelectElement>("correctionPhraseSelect");
const inputDeviceSelect = byId<HTMLSelectElement>("inputDeviceSelect");
const trainProgressEl = byId<HTMLSpanElement>("trainProgress");
const evalProgressEl = byId<HTMLSpanElement>("evalProgress");
const trainResultEl = byId<HTMLDivElement>("trainResult");
const rawAudioEl = byId<HTMLAudioElement>("rawAudio");
const ttsAudioEl = byId<HTMLAudioElement>("ttsAudio");
const demoPredEl = byId<HTMLSpanElement>("demoPred");
const demoSecondEl = byId<HTMLSpanElement>("demoSecond");
const demoScoreEl = byId<HTMLSpanElement>("demoScore");
const demoRejectEl = byId<HTMLSpanElement>("demoReject");
const correctionResultEl = byId<HTMLDivElement>("correctionResult");
const phraseGridEl = byId<HTMLDivElement>("phraseGrid");
const logsEl = byId<HTMLPreElement>("logs");
const auditReportEl = byId<HTMLPreElement>("auditReport");
const evalReportEl = byId<HTMLPreElement>("evalReport");
const runtimeConfigHintEl = byId<HTMLDivElement>("runtimeConfigHint");
const evalErrorFilterEl = byId<HTMLSelectElement>("evalErrorFilter");
const refreshErrorsBtn = byId<HTMLButtonElement>("refreshErrorsBtn");
const addSelectedErrorsBtn = byId<HTMLButtonElement>("addSelectedErrorsBtn");
const confirmSelectedErrorsBtn = byId<HTMLButtonElement>("confirmSelectedErrorsBtn");
const quickConfirmCorrectionsBtn = byId<HTMLButtonElement>("quickConfirmCorrectionsBtn");
const evalWorkbenchSummaryEl = byId<HTMLDivElement>("evalWorkbenchSummary");
const evalAdviceCardEl = byId<HTMLDivElement>("evalAdviceCard");
const evalErrorsBodyEl = byId<HTMLTableSectionElement>("evalErrorsBody");
const evalCorrectionResultEl = byId<HTMLDivElement>("evalCorrectionResult");
const rerecordTaskSummaryEl = byId<HTMLDivElement>("rerecordTaskSummary");
const rerecordPendingBodyEl = byId<HTMLTableSectionElement>("rerecordPendingBody");
const workbenchRecordBtn = byId<HTMLButtonElement>("workbenchRecordBtn");
const workbenchRecordHint = byId<HTMLDivElement>("workbenchRecordHint");
const selectAllErrorsEl = byId<HTMLInputElement>("selectAllErrors");
const evalTotalEl = byId<HTMLDivElement>("evalTotal");
const evalTop1El = byId<HTMLDivElement>("evalTop1");
const evalTop2El = byId<HTMLDivElement>("evalTop2");
const evalRejectEl = byId<HTMLDivElement>("evalReject");
const unknownTotalEl = byId<HTMLDivElement>("unknownTotal");
const unknownFalseAcceptEl = byId<HTMLDivElement>("unknownFalseAccept");
const unknownCorrectRejectEl = byId<HTMLDivElement>("unknownCorrectReject");
const unknownReportEl = byId<HTMLPreElement>("unknownReport");
const resetEvalBtn = byId<HTMLButtonElement>("resetEvalBtn");
const resetUnknownBtn = byId<HTMLButtonElement>("resetUnknownBtn");
const saveCorrectionBtn = byId<HTMLButtonElement>("saveCorrectionBtn");
const confirmCorrectionsBtn = byId<HTMLButtonElement>("confirmCorrectionsBtn");
const refreshBtn = byId<HTMLButtonElement>("refreshBtn");
const rebuildBtn = byId<HTMLButtonElement>("rebuildBtn");
const auditBtn = byId<HTMLButtonElement>("auditBtn");
const archiveBtn = byId<HTMLButtonElement>("archiveBtn");

function pushLog(message: string): void {
  const stamp = new Date().toLocaleTimeString();
  logsEl.textContent = `[${stamp}] ${message}\n${logsEl.textContent || ""}`;
}

function setStatus(text: string, recording = false): void {
  statusEl.textContent = text;
  statusEl.className = `status ${recording ? "recording" : ""}`;
}

function setEvalCorrectionResult(text: string, isError = false): void {
  evalCorrectionResultEl.textContent = text;
  evalCorrectionResultEl.style.color = isError ? "var(--danger)" : "#9fd8ff";
}

function phraseText(id: string | null | undefined): string {
  return phrases.find((p) => p.phrase_id === id)?.text || id || "-";
}

function percent(v: number | undefined): string {
  return `${Math.round((v || 0) * 1000) / 10}%`;
}

function prettyNum(v: number | null | undefined): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "-";
  return v.toFixed(6);
}

function htmlEscape(raw: string): string {
  return raw.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function explainReject(reason: string | null | undefined): string {
  const map: Record<string, string> = {
    none: "命中",
    index_not_ready: "索引未就绪",
    no_templates: "没有训练模板",
    audio_too_short: "录音过短",
    quality_gate_failed: "音频质量未通过",
    distance_too_high: "距离过高",
    separation_too_low: "候选分离度不足",
  };
  return map[reason || "none"] || (reason || "none");
}

function explainQualityFlags(flags: string[]): string {
  if (!flags.length) return "质量校验未通过";
  const m: Record<string, string> = {
    too_short: "录音太短",
    too_long: "录音太长",
    too_quiet: "音量太小",
    mostly_silence: "静音过多",
    no_pcm_samples: "没有有效音频样本",
    not_mono: "不是单声道",
    not_16bit: "不是16位PCM",
    unexpected_sample_rate: "采样率异常",
    duplicate_audio: "重复音频",
  };
  return flags.map((x) => m[x] || x).join("、");
}

function explainSelectionMode(mode: string | null | undefined): string {
  const value = mode || "unknown";
  if (value === "force_top1_no_reject") return "强制 Top1（不拒识）";
  if (value === "reject_then_correct") return "先拒识后纠正";
  if (value === "standard") return "标准门控";
  return value;
}

function setMode(mode: Mode): void {
  currentMode = mode;
  if (mode !== "eval") {
    if (!activeRerecordTask) pendingRerecordTruthId = "";
  }
  modeTrainBtn.classList.toggle("active", mode === "train");
  modeEvalBtn.classList.toggle("active", mode === "eval");
  modeUnknownBtn.classList.toggle("active", mode === "unknown");
  modeDemoBtn.classList.toggle("active", mode === "demo");
  trainPanel.classList.toggle("active", mode === "train");
  evalPanel.classList.toggle("active", mode === "eval");
  unknownPanel.classList.toggle("active", mode === "unknown");
  demoPanel.classList.toggle("active", mode === "demo");
  let text =
    mode === "train"
      ? "训练采集：选择短句后录音"
      : mode === "eval"
      ? "正式评估：请选择评分标签"
      : mode === "unknown"
      ? "负样本测试：请录四句之外声音"
      : "演示推理：直接录音测试";
  if (mode === "eval" && pendingRerecordTruthId) {
    const done = activeRerecordTask?.completed ?? 0;
    const left = Math.max(0, REQUIRED_RERECORD_COUNT - done);
    text = `错题重录模式：请按真值 ${phraseText(pendingRerecordTruthId)} 重录（${done}/${REQUIRED_RERECORD_COUNT}，还需 ${left} 条）`;
  }
  setStatus(text, false);
  updateWorkbenchRerecordControls();
}

function buildRerecordBatchId(): string {
  return `rb_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function updateWorkbenchRerecordControls(): void {
  const hasTask = Boolean(activeRerecordTask);
  workbenchRecordBtn.disabled = !hasTask;
  const done = activeRerecordTask?.completed ?? 0;
  const taskIncomplete = hasTask && done < REQUIRED_RERECORD_COUNT;
  // In rerecord mode, these two actions are intentionally locked to avoid confusion.
  addSelectedErrorsBtn.disabled = Boolean(hasTask);
  confirmSelectedErrorsBtn.disabled = Boolean(taskIncomplete);
  quickConfirmCorrectionsBtn.disabled = Boolean(taskIncomplete);
  if (!hasTask) {
    workbenchRecordBtn.textContent = "在工作台录制重录样本";
    workbenchRecordHint.textContent = "错判后会自动创建重录任务，然后可在这里直接录制，不用切换位置。";
    addSelectedErrorsBtn.title = "";
    confirmSelectedErrorsBtn.title = "";
    quickConfirmCorrectionsBtn.title = "";
    return;
  }
  const left = Math.max(0, REQUIRED_RERECORD_COUNT - done);
  workbenchRecordBtn.textContent = state.isRecording ? "停止并提交本条重录" : `录制重录样本（还需 ${left} 条）`;
  workbenchRecordHint.textContent = `当前任务真值=${phraseText(activeRerecordTask?.truth_phrase_id)}。点击按钮开始，再点击一次结束并提交。也可使用页面顶部大按钮录制。`;
  addSelectedErrorsBtn.title = "重录任务进行中，此按钮暂不可用。";
  if (taskIncomplete) {
    confirmSelectedErrorsBtn.title = `请先完成重录 ${done}/${REQUIRED_RERECORD_COUNT}。`;
    quickConfirmCorrectionsBtn.title = `请先完成重录 ${done}/${REQUIRED_RERECORD_COUNT}。`;
  } else {
    confirmSelectedErrorsBtn.title = "已录满5条，可确认合并纠错。";
    quickConfirmCorrectionsBtn.title = "已录满5条，可确认合并纠错。";
  }
}

function beginRerecordTask(truthId: string, eventId: string | null): void {
  activeRerecordTask = {
    truth_phrase_id: truthId,
    from_eval_event_id: eventId,
    rerecord_batch_id: buildRerecordBatchId(),
    required: REQUIRED_RERECORD_COUNT,
    completed: 0,
  };
  pendingRerecordTruthId = truthId;
  updateWorkbenchRerecordControls();
}

function clearRerecordTask(): void {
  activeRerecordTask = null;
  pendingRerecordTruthId = "";
  currentPendingCorrections = [];
  updateWorkbenchRerecordControls();
  renderRerecordTaskPanel();
}

function renderRerecordTaskPanel(): void {
  rerecordPendingBodyEl.innerHTML = "";
  if (!activeRerecordTask) {
    rerecordTaskSummaryEl.textContent = "当前没有进行中的错题重录任务。";
    return;
  }
  const done = activeRerecordTask.completed;
  const need = activeRerecordTask.required;
  const left = Math.max(0, need - done);
  rerecordTaskSummaryEl.textContent = `真值=${phraseText(activeRerecordTask.truth_phrase_id)}，进度 ${done}/${need}，还需 ${left} 条。`;
  for (const row of currentPendingCorrections) {
    const tr = document.createElement("tr");
    const sampleId = String(row.sample_id || "-");
    const truth = phraseText(row.truth_phrase_id || "");
    const createdAt = String(row.created_at || "-");
    const audio = row.raw_audio_url ? `<audio controls preload="none" src="${htmlEscape(String(row.raw_audio_url))}"></audio>` : "-";
    tr.innerHTML = [
      `<td>${htmlEscape(sampleId)}</td>`,
      `<td>${htmlEscape(truth || "-")}</td>`,
      `<td>${htmlEscape(createdAt)}</td>`,
      `<td>${audio}</td>`,
      `<td><button class="danger" data-delete-pending-sample-id="${htmlEscape(sampleId)}">删除</button></td>`,
    ].join("");
    rerecordPendingBodyEl.appendChild(tr);
  }
}

function updateSelects(): void {
  const render = (select: HTMLSelectElement): void => {
    const current = select.value;
    select.innerHTML = "";
    for (const p of phrases) {
      const opt = document.createElement("option");
      opt.value = p.phrase_id;
      opt.textContent = `${p.text}${p.stage_pack ? "（舞台句）" : ""}`;
      select.appendChild(opt);
    }
    if (current && phrases.some((p) => p.phrase_id === current)) select.value = current;
  };
  render(trainPhraseSelect);
  render(evalPhraseSelect);
  render(correctionPhraseSelect);
}

function renderPhraseGrid(): void {
  phraseGridEl.innerHTML = "";
  for (const p of phrases) {
    const div = document.createElement("div");
    div.className = `phrase-card ${p.stage_pack ? "stage" : ""}`;
    div.innerHTML = `<b>${p.text}</b><br>train ${p.train}/${TRAIN_TARGET} 路 eval ${p.eval}/${EVAL_TARGET}<br>corrections ${p.corrections} 路 rejected ${p.rejected}<br>templates ${p.template_count}`;
    phraseGridEl.appendChild(div);
  }
  const train = phrases.find((p) => p.phrase_id === trainPhraseSelect.value);
  trainProgressEl.textContent = train ? `${train.text}: ${train.train}/${TRAIN_TARGET}` : "-";
  const ev = phrases.find((p) => p.phrase_id === evalPhraseSelect.value);
  evalProgressEl.textContent = ev ? `${ev.text}: ${ev.eval}/${EVAL_TARGET}` : "-";
}

function renderEvalSummary(summary: EvalSummary | null): void {
  if (!summary) {
    evalTotalEl.textContent = "0";
    evalTop1El.textContent = "0%";
    evalTop2El.textContent = "0%";
    evalRejectEl.textContent = "0%";
    evalReportEl.textContent = "暂无评估报告。";
    return;
  }
  evalTotalEl.textContent = String(summary.total || 0);
  evalTop1El.textContent = percent(summary.top1_rate);
  evalTop2El.textContent = percent(summary.top2_rate);
  evalRejectEl.textContent = percent(summary.reject_rate);
  evalReportEl.textContent = JSON.stringify(summary, null, 2);
}

function isLowMargin(item: EvalErrorItem): boolean {
  const margin = typeof item.margin === "number" ? item.margin : typeof item.gap === "number" ? item.gap : null;
  const threshold = typeof runtimeConfig?.fallback_margin_threshold === "number" ? runtimeConfig.fallback_margin_threshold : 0.0025;
  return typeof margin === "number" && Number.isFinite(margin) && margin < threshold;
}

function isAcceptedWrong(item: EvalErrorItem): boolean {
  return (item.reject_reason || "unknown") === "none" && !!item.truth_phrase_id && !!item.predicted_phrase_id && item.truth_phrase_id !== item.predicted_phrase_id;
}

function isRejected(item: EvalErrorItem): boolean {
  return (item.reject_reason || "unknown") !== "none";
}

function filteredErrorItems(): EvalErrorItem[] {
  if (currentErrorFilter === "all") return evalErrorItems;
  if (currentErrorFilter === "accepted_wrong") return evalErrorItems.filter((x) => isAcceptedWrong(x));
  if (currentErrorFilter === "rejected") return evalErrorItems.filter((x) => isRejected(x));
  return evalErrorItems.filter((x) => isLowMargin(x));
}

function renderAdviceCard(analysis: EvalErrorsAnalysis | null): void {
  if (!analysis) {
    evalAdviceCardEl.textContent = "暂无错题建议。";
    return;
  }
  const pairs = Array.isArray(analysis.confusion_pairs) ? analysis.confusion_pairs : [];
  const top = pairs[0];
  const acceptedWrong = Array.isArray(analysis.accepted_wrong_cases) ? analysis.accepted_wrong_cases.length : 0;
  const lowMargin = Array.isArray(analysis.low_margin_cases) ? analysis.low_margin_cases.length : 0;
  if (!top) {
    evalAdviceCardEl.textContent = `本轮无明显混淆对。accepted_wrong=${acceptedWrong}，low_margin=${lowMargin}。建议继续 fresh 评估确认稳定性。`;
    return;
  }
  const truth = top.truth_text || top.truth_phrase_id || "-";
  const pred = top.pred_text || top.pred_phrase_id || "-";
  const count = top.count || 0;
  const recommend = Math.max(3, Math.min(8, count * 2));
  evalAdviceCardEl.textContent = `最严重混淆对：${truth} -> ${pred}（${count} 次）。建议下轮定向补录：真值 ${recommend} 条 + 对照 ${Math.max(2, Math.floor(recommend / 2))} 条。`;
}

function renderErrorTable(): void {
  const rows = filteredErrorItems();
  evalErrorsBodyEl.innerHTML = "";
  for (const item of rows) {
    const tr = document.createElement("tr");
    const id = item.event_id;
    const checked = selectedErrorIds.has(id) ? "checked" : "";
    const truth = item.truth_text || phraseText(item.truth_phrase_id || "");
    const pred = item.predicted_text || item.best_text || phraseText(item.predicted_phrase_id || "");
    const margin = prettyNum(typeof item.margin === "number" ? item.margin : typeof item.gap === "number" ? item.gap : null);
    const reason = item.reject_reason || "unknown";
    const reasonClass = reason === "none" ? "chip" : "chip err";
    const diagnosis = item.diagnosis_text || item.diagnosis_code || "-";
    const suggest = item.suggested_fix || "-";
    const audio = item.audio_url ? `<audio controls preload=\"none\" src=\"${htmlEscape(item.audio_url)}\"></audio>` : "-";
    const rerBtn = item.truth_phrase_id
      ? `<button class="warn" data-rerecord-truth-id="${htmlEscape(String(item.truth_phrase_id))}">重录替换</button>`
      : "-";
    tr.innerHTML = [
      `<td><input type="checkbox" data-event-id="${htmlEscape(id)}" ${checked} /></td>`,
      `<td>${htmlEscape(String(truth || "-"))}</td>`,
      `<td>${htmlEscape(String(pred || "-"))}</td>`,
      `<td>${htmlEscape(String(margin))}</td>`,
      `<td><span class="${reasonClass}">${htmlEscape(explainReject(reason))}</span></td>`,
      `<td>${htmlEscape(String(diagnosis))}</td>`,
      `<td>${htmlEscape(String(suggest))}</td>`,
      `<td>${audio}<div style="margin-top:6px;">${rerBtn}</div></td>`,
    ].join("");
    evalErrorsBodyEl.appendChild(tr);
  }

  const selectedVisible = rows.filter((row) => selectedErrorIds.has(row.event_id)).length;
  evalWorkbenchSummaryEl.textContent = `筛选=${currentErrorFilter}，显示 ${rows.length} 条，已勾选 ${selectedVisible} 条（总勾选 ${selectedErrorIds.size}）。`;
}

function updateRuntimeHint(): void {
  if (!runtimeConfig) {
    runtimeConfigHintEl.textContent = "策略读取失败。";
    return;
  }
  const mode = explainSelectionMode(runtimeConfig.selection_mode || (runtimeConfig.no_reject_mode ? "force_top1_no_reject" : "reject_then_correct"));
  const score = prettyNum(runtimeConfig.fallback_score_threshold ?? null);
  const margin = prettyNum(runtimeConfig.fallback_margin_threshold ?? null);
  runtimeConfigHintEl.textContent = `当前策略：${mode}；no_reject_mode=${String(runtimeConfig.no_reject_mode)}；fallback_score_threshold=${score}；fallback_margin_threshold=${margin}`;
}

function renderUnknownSummary(summary: UnknownSummary | null): void {
  if (!summary) {
    unknownTotalEl.textContent = "0";
    unknownFalseAcceptEl.textContent = "0%";
    unknownCorrectRejectEl.textContent = "0";
    unknownReportEl.textContent = "暂无负样本报告。";
    return;
  }
  unknownTotalEl.textContent = String(summary.total || 0);
  unknownFalseAcceptEl.textContent = percent(summary.false_accept_rate);
  unknownCorrectRejectEl.textContent = String(summary.correct_rejects || 0);
  unknownReportEl.textContent = JSON.stringify(summary, null, 2);
}

function lockToFourPhrases(rows: PhraseRow[]): PhraseRow[] {
  const filtered = rows.filter((row) => LOCKED_PHRASE_IDS.has(row.phrase_id) || LOCKED_PHRASE_TEXTS.has(row.text));
  return filtered.slice(0, 4);
}

async function refreshPhrases(): Promise<void> {
  const res = await fetch("/api/v3/phrases");
  const data = (await res.json()) as PhraseResponse;
  if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
  phrases = lockToFourPhrases(data.phrases || []);
  updateSelects();
  renderPhraseGrid();
}

async function refreshEvalSummary(): Promise<void> {
  const res = await fetch("/api/v3/eval/export");
  if (!res.ok) return;
  const data = (await res.json()) as EvalExportResponse;
  if (data.ok) renderEvalSummary(data.summary);
}

async function refreshRuntimeConfig(): Promise<void> {
  const res = await fetch("/api/v3/runtime/config");
  if (!res.ok) return;
  const data = (await res.json()) as RuntimeConfigResponse;
  if (!data.ok) return;
  runtimeConfig = data.config;
  updateRuntimeHint();
}

async function refreshEvalErrors(): Promise<void> {
  const res = await fetch("/api/v3/eval/errors");
  if (res.status === 404) {
    evalErrorItems = [];
    renderAdviceCard(null);
    renderErrorTable();
    return;
  }
  if (!res.ok) throw new Error(`/api/v3/eval/errors HTTP ${res.status}`);
  const data = (await res.json()) as EvalErrorsResponse;
  if (!data.ok) throw new Error(data.error || "/api/v3/eval/errors failed");
  evalErrorItems = Array.isArray(data.items) ? data.items : [];
  renderAdviceCard(data.analysis || null);
  renderErrorTable();
}

async function refreshRerecordPending(): Promise<void> {
  if (!activeRerecordTask) {
    currentPendingCorrections = [];
    renderRerecordTaskPanel();
    return;
  }
  const params = new URLSearchParams({
    truth_phrase_id: activeRerecordTask.truth_phrase_id,
    rerecord_batch_id: activeRerecordTask.rerecord_batch_id,
  });
  // Tolerate backend route drift/restart race by trying both slash variants.
  let res = await fetch(`/api/v3/corrections/pending?${params.toString()}`);
  if (res.status === 404) {
    res = await fetch(`/api/v3/corrections/pending/?${params.toString()}`);
  }
  if (!res.ok) throw new Error(`pending HTTP ${res.status}`);
  const data = (await res.json()) as PendingCorrectionsResponse;
  if (!data.ok) throw new Error(data.error || "pending corrections api failed");
  currentPendingCorrections = Array.isArray(data.items) ? data.items : [];
  activeRerecordTask.completed = currentPendingCorrections.length;
  renderRerecordTaskPanel();
  updateWorkbenchRerecordControls();
}

async function refreshUnknownSummary(): Promise<void> {
  const res = await fetch("/api/v3/unknown/export");
  if (!res.ok) return;
  const data = (await res.json()) as UnknownExportResponse;
  if (data.ok) renderUnknownSummary(data.summary);
}

function mergeFloat32(buffers: Float32Array[]): Float32Array {
  const total = buffers.reduce((sum, b) => sum + b.length, 0);
  const merged = new Float32Array(total);
  let offset = 0;
  for (const b of buffers) {
    merged.set(b, offset);
    offset += b.length;
  }
  return merged;
}

function resampleLinear(input: Float32Array, inputSampleRate: number, outputSampleRate: number): Float32Array {
  if (inputSampleRate === outputSampleRate) return input;
  const ratio = inputSampleRate / outputSampleRate;
  const outLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const src = i * ratio;
    const left = Math.floor(src);
    const right = Math.min(left + 1, input.length - 1);
    const t = src - left;
    output[i] = input[left] * (1 - t) + input[right] * t;
  }
  return output;
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);
  const writeString = (offset: number, text: string): void => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);
  let offset = 44;
  for (const sample of samples) {
    const s = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

async function startRecording(): Promise<void> {
  if (state.isRecording) return;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: selectedDeviceId ? { exact: selectedDeviceId } : undefined,
        channelCount: 1,
        noiseSuppression: false,
        echoCancellation: false,
        autoGainControl: false,
      },
      video: false,
    });
    state.audioContext = new AudioContext();
    state.sampleRate = state.audioContext.sampleRate;
    state.buffers = [];
    state.sourceNode = state.audioContext.createMediaStreamSource(state.stream);
    state.processorNode = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.gainNode = state.audioContext.createGain();
    state.gainNode.gain.value = 0;
    state.processorNode.onaudioprocess = (event: AudioProcessingEvent) => {
      state.buffers.push(new Float32Array(event.inputBuffer.getChannelData(0)));
    };
    state.sourceNode.connect(state.processorNode);
    state.processorNode.connect(state.gainNode);
    state.gainNode.connect(state.audioContext.destination);
    state.isRecording = true;
    state.startedAtMs = performance.now();
    setStatus("录音中...松开后处理", true);
    pushLog(`录音开始 input_sr=${state.sampleRate}`);
    updateWorkbenchRerecordControls();
  } catch (err) {
    setStatus("无法打开麦克风", false);
    pushLog(`无法打开麦克风：${String(err)}`);
  }
}

async function stopRecording(): Promise<void> {
  if (!state.isRecording) return;
  state.isRecording = false;
  updateWorkbenchRerecordControls();
  try {
    state.processorNode?.disconnect();
    state.sourceNode?.disconnect();
    state.gainNode?.disconnect();
    state.stream?.getTracks().forEach((t) => t.stop());
    await state.audioContext?.close();
  } finally {
    state.processorNode = null;
    state.sourceNode = null;
    state.gainNode = null;
    state.stream = null;
    state.audioContext = null;
  }
  const durationMs = Math.round(performance.now() - state.startedAtMs);
  const merged = mergeFloat32(state.buffers);
  if (durationMs < MIN_CLIENT_RECORDING_MS || merged.length === 0) {
    setStatus("录音太短，请按住久一点", false);
    pushLog(`录音过短 duration=${durationMs}ms`);
    return;
  }
  const wavBlob = encodeWav(resampleLinear(merged, state.sampleRate, TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE);
  rawAudioEl.src = URL.createObjectURL(wavBlob);
  pushLog(`录音结束 duration=${durationMs}ms mode=${currentMode}`);
  try {
    if (currentMode === "train") await uploadTrain(wavBlob);
    if (currentMode === "eval") await uploadEval(wavBlob);
    if (currentMode === "unknown") await uploadUnknown(wavBlob);
    if (currentMode === "demo") await processDemo(wavBlob);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`失败：${msg}`, false);
    pushLog(`处理失败：${msg}`);
  }
  updateWorkbenchRerecordControls();
}

async function uploadTrain(blob: Blob): Promise<void> {
  const phraseId = trainPhraseSelect.value;
  const fd = new FormData();
  fd.append("phrase_id", phraseId);
  fd.append("rebuild_policy", "deferred");
  fd.append("file", blob, "train.wav");
  setStatus("训练样本入库中...", false);
  const res = await fetch("/api/v3/samples/upload", { method: "POST", body: fd });
  const data = (await res.json()) as UploadResponse;
  if (!res.ok || !data.ok) {
    const flags = (data.quality?.quality_flags as string[] | undefined)?.join(", ") || data.error || "upload failed";
    throw new Error(`训练样本被拒收：${flags}`);
  }
  const warnings = (data.quality?.warning_flags as string[] | undefined)?.join(", ");
  const indexState = String(data.index_state || "pending_rebuild");
  const stateText = indexState === "ready" ? "可评估" : indexState === "rebuild_failed" ? "重建失败" : "待重建";
  const rebuildPart = data.rebuild_triggered ? `本次已重建(${data.duration_ms ?? "-"}ms)` : "延迟重建(按批次条件触发)";
  trainResultEl.textContent = `已入库：${phraseText(phraseId)}；状态=${stateText}；${rebuildPart}；模板数=${data.template_count ?? "-"}${warnings ? `；警告=${warnings}` : ""}`;
  setStatus("训练样本已保存", false);
  await refreshPhrases();
}

async function uploadEval(blob: Blob): Promise<void> {
  if (activeRerecordTask) {
    await saveRerecordToCorrections(activeRerecordTask, blob);
    return;
  }
  const truthId = evalPhraseSelect.value;
  const fd = new FormData();
  fd.append("truth_phrase_id", truthId);
  fd.append("file", blob, "eval.wav");
  setStatus("正式评估中：正在判定...", false);
  const res = await fetch("/api/v3/eval/upload", { method: "POST", body: fd });
  const data = (await res.json()) as UploadResponse;
  if (!res.ok || !data.ok) {
    const flags = (data.quality?.quality_flags as string[] | undefined)?.join("、");
    const reason = flags || data.error || "评估上传失败";
    throw new Error(`评估失败：${reason}。请重新录一条清晰语音再试。`);
  }
  renderEvalSummary(data.summary || null);
  await refreshEvalErrors();
  const ev = data.event || {};
  const prediction = String(ev.predicted_text || ev.best_text || "-");
  const predictedText = typeof ev.predicted_text === "string" ? ev.predicted_text : typeof ev.best_text === "string" ? ev.best_text : "";
  const truthText = phraseText(truthId);
  const nextAction = data.next_action || "none";
  const nextTruthId = data.next_truth_phrase_id || truthId;
  const forceTextMismatch = !!predictedText && !!truthText && predictedText !== truthText;
  const isError = Boolean(data.is_error_event) || nextAction === "rerecord_truth" || forceTextMismatch;
  if (isError) {
    const eventId = typeof ev.event_id === "string" ? ev.event_id : null;
    beginRerecordTask(nextTruthId, eventId);
    evalPhraseSelect.value = nextTruthId;
    renderPhraseGrid();
    setMode("eval");
    try {
      await refreshRerecordPending();
    } catch (pendingErr) {
      // Do not block error reroute on pending panel failure.
      pushLog(`pending 刷新失败（已忽略，不影响重录）：${pendingErr instanceof Error ? pendingErr.message : String(pendingErr)}`);
      renderRerecordTaskPanel();
    }
    const msg = `检测到错误：truth=${phraseText(truthId)}，pred=${prediction}。请开始 ${REQUIRED_RERECORD_COUNT} 条强制重录。`;
    setStatus(msg, false);
    setEvalCorrectionResult(`错题重录任务已创建：${phraseText(nextTruthId)}（0/${REQUIRED_RERECORD_COUNT}）`);
    pushLog(msg);
  } else {
    const msg = `评估通过：真值=${phraseText(truthId)}，预测=${prediction}`;
    setStatus(msg, false);
    setEvalCorrectionResult(msg);
    pushLog(`评估通过：truth=${phraseText(truthId)}，pred=${prediction}`);
  }
  await refreshPhrases();
}

async function saveRerecordToCorrections(task: ActiveRerecordTask, blob: Blob): Promise<void> {
  const truthId = task.truth_phrase_id;
  const fd = new FormData();
  fd.append("truth_phrase_id", truthId);
  fd.append("predicted_phrase_id", "");
  fd.append("raw_audio_url", "");
  fd.append("rerecord_batch_id", task.rerecord_batch_id);
  fd.append("from_eval_event_id", task.from_eval_event_id || "");
  fd.append("file", blob, "rerecord_correction.wav");
  setStatus(`错题重录上传中：${phraseText(truthId)}`, false);
  const res = await fetch("/api/v3/corrections/upload", { method: "POST", body: fd });
  const data = (await res.json()) as UploadResponse;
  if (!res.ok || !data.ok) {
    const q = data.quality as { quality_flags?: string[] } | undefined;
    const flags = Array.isArray(q?.quality_flags) ? q!.quality_flags : [];
    const reason = flags.length ? explainQualityFlags(flags) : data.error || `HTTP ${res.status}`;
    throw new Error(`重录样本未入库：${reason}`);
  }
  await refreshRerecordPending();
  const done = task.completed;
  const left = Math.max(0, REQUIRED_RERECORD_COUNT - done);
  if (left > 0) {
    const msg = `重录已保存：${phraseText(truthId)}，当前 ${done}/${REQUIRED_RERECORD_COUNT}，还需 ${left} 条。`;
    setStatus(msg, false);
    setEvalCorrectionResult(msg);
    pushLog(msg);
  } else {
    const msg = `重录任务已完成：${phraseText(truthId)}，${done}/${REQUIRED_RERECORD_COUNT}。请手动点击“确认合并纠错”。`;
    setStatus(msg, false);
    setEvalCorrectionResult(msg);
    pushLog(msg);
  }
  correctionResultEl.textContent = `已完成重录：${phraseText(truthId)}（${done}/${REQUIRED_RERECORD_COUNT}）`;
  await refreshPhrases();
  await refreshEvalErrors();
}

async function uploadUnknown(blob: Blob): Promise<void> {
  const fd = new FormData();
  fd.append("file", blob, "unknown.wav");
  setStatus("负样本测试中：正在分析四句之外声音", false);
  const res = await fetch("/api/v3/unknown/upload", { method: "POST", body: fd });
  const data = (await res.json()) as { ok: boolean; error?: string; quality?: Record<string, unknown>; event?: Record<string, unknown>; summary?: UnknownSummary };
  if (!res.ok || !data.ok) {
    const flags = (data.quality?.quality_flags as string[] | undefined)?.join(", ") || data.error || "unknown upload failed";
    throw new Error(`负样本被拒收：${flags}`);
  }
  renderUnknownSummary(data.summary || null);
  const ev = data.event || {};
  const pred = String(ev.predicted_text || ev.best_text || "-");
  setStatus(`负样本测试完成：Top1=${pred}`, false);
  pushLog(`负样本测试记录：${JSON.stringify(ev)}`);
}

async function processDemo(blob: Blob): Promise<void> {
  lastDemoBlob = blob;
  const fd = new FormData();
  fd.append("file", blob, "demo.wav");
  setStatus("演示推理中（本地优先，低置信时云回退）...", false);
  const res = await fetch("/api/v3/hybrid/process", { method: "POST", body: fd });
  const data = (await res.json()) as DemoResponse;
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  const debug = data.local_debug || data.audio_match_debug || {};
  lastDemoRawUrl = data.raw_audio_url || "";
  lastDemoPredictedId = (debug.best_phrase_id || "") as string;
  demoPredEl.textContent = data.final_text || data.matched_phrase || debug.best_phrase || "-";
  demoSecondEl.textContent = debug.second_phrase || "-";
  const top1Score = typeof data.top1_score === "number" ? data.top1_score : typeof data.score === "number" ? data.score : null;
  demoScoreEl.textContent = prettyNum(top1Score);
  const bestHint = debug.best_phrase ? `；第一候选=${debug.best_phrase}` : "";
  const src = data.decision_source || "local_accept";
  demoRejectEl.textContent = `${explainReject(debug.reject_reason)} 路 ${src}${bestHint}`;
  if (data.tts_audio_url) ttsAudioEl.src = data.tts_audio_url;
  const shown = data.final_text || data.matched_phrase || debug.best_phrase || "-";
  setStatus(`演示输出：${shown} (${src})`, false);
  pushLog(`演示结果：${JSON.stringify({ source: src, reason: data.reason, local: debug, cloud: data.cloud_debug || {} })}`);
}

async function saveCorrection(): Promise<void> {
  if (!lastDemoBlob) throw new Error("没有可保存的演示录音");
  const truthId = correctionPhraseSelect.value;
  const fd = new FormData();
  fd.append("truth_phrase_id", truthId);
  fd.append("predicted_phrase_id", lastDemoPredictedId || "");
  fd.append("raw_audio_url", lastDemoRawUrl || "");
  fd.append("file", lastDemoBlob, "correction.wav");
  const res = await fetch("/api/v3/corrections/upload", { method: "POST", body: fd });
  const data = (await res.json()) as UploadResponse;
  if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
  correctionResultEl.textContent = `已保存纠正样本：${phraseText(truthId)}（未自动合并训练库）`;
  await refreshPhrases();
}

async function confirmCorrections(): Promise<void> {
  if (activeRerecordTask && activeRerecordTask.completed < REQUIRED_RERECORD_COUNT) {
    throw new Error(`当前重录进度 ${activeRerecordTask.completed}/${REQUIRED_RERECORD_COUNT}，未满 ${REQUIRED_RERECORD_COUNT} 条，禁止合并`);
  }
  const staged = await addVisibleErrorEventsToCorrections();
  const res = await fetch("/api/v3/corrections/confirm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) });
  const data = (await res.json()) as CorrectionsConfirmResponse;
  if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
  const rebuildText = data.rebuild_ok === null ? "未触发" : data.rebuild_ok ? "成功" : "失败";
  const disabled = Number(data.auto_purify?.disabled_count || 0);
  const protectedCount = Number(data.auto_purify?.protected_by_min_active || 0);
  const msg = `已合并 ${data.moved} 条纠错（自动入池 added=${staged.added}/${staged.total}，skipped=${staged.skipped}，errors=${staged.errors}），重建=${rebuildText}；净化禁用=${disabled}，最小保留保护=${protectedCount}`;
  correctionResultEl.textContent = msg;
  setEvalCorrectionResult(msg);
  setStatus(msg, false);
  pushLog(`confirm_corrections: ${msg}`);
  if (data.moved > 0 || !activeRerecordTask) clearRerecordTask();
  await refreshPhrases();
  await refreshEvalErrors();
}


async function addSelectedEvalErrorsToCorrections(): Promise<void> {
  if (activeRerecordTask && activeRerecordTask.completed < REQUIRED_RERECORD_COUNT) {
    throw new Error(`当前有重录任务进行中（${activeRerecordTask.completed}/${REQUIRED_RERECORD_COUNT}）。请先录满5条后再操作“加入纠错池”。`);
  }
  const eventIds = Array.from(selectedErrorIds.values());
  if (!eventIds.length) throw new Error("请先勾选错题");
  const res = await fetch("/api/v3/corrections/from_eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event_ids: eventIds }),
  });
  const data = (await res.json()) as CorrectionsFromEvalResponse;
  if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
  const msg = `已加入纠错池：added=${data.added}，skipped=${data.skipped.length}，errors=${data.errors.length}`;
  setStatus(msg, false);
  setEvalCorrectionResult(msg, data.errors.length > 0);
  pushLog(
    `from_eval: ${JSON.stringify({
      added: data.added,
      skipped: data.skipped.length,
      errors: data.errors.length,
      corrections_count: data.corrections_count,
      index_state: data.index_state,
    })}`
  );
  await refreshPhrases();
  await refreshEvalErrors();
}

async function addVisibleErrorEventsToCorrections(): Promise<{ added: number; skipped: number; errors: number; total: number }> {
  const rows = filteredErrorItems();
  const eventIds = rows
    .filter((item) => Boolean(item.event_id) && (item.is_error === true || isAcceptedWrong(item) || isRejected(item)))
    .map((item) => item.event_id);
  if (!eventIds.length) return { added: 0, skipped: 0, errors: 0, total: 0 };
  const res = await fetch("/api/v3/corrections/from_eval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ event_ids: eventIds }),
  });
  const data = (await res.json()) as CorrectionsFromEvalResponse;
  if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return {
    added: Number(data.added || 0),
    skipped: Array.isArray(data.skipped) ? data.skipped.length : 0,
    errors: Array.isArray(data.errors) ? data.errors.length : 0,
    total: eventIds.length,
  };
}

async function deletePendingCorrection(sampleId: string): Promise<void> {
  const res = await fetch(`/api/v3/corrections/pending/${encodeURIComponent(sampleId)}`, { method: "DELETE" });
  const data = (await res.json()) as { ok: boolean; deleted_sample_id?: string; corrections_count?: number; error?: string };
  if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
  await refreshRerecordPending();
  const done = activeRerecordTask?.completed ?? 0;
  const msg = `已删除待合并纠错样本 ${data.deleted_sample_id || sampleId}，当前重录进度 ${done}/${REQUIRED_RERECORD_COUNT}。`;
  setStatus(msg, false);
  setEvalCorrectionResult(msg);
  pushLog(msg);
}

async function refreshInputDevices(): Promise<void> {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const audioInputs = devices.filter((d) => d.kind === "audioinput");
  inputDeviceSelect.innerHTML = "";
  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = "系统默认麦克风";
  inputDeviceSelect.appendChild(auto);
  for (const d of audioInputs) {
    const opt = document.createElement("option");
    opt.value = d.deviceId;
    opt.textContent = d.label || `麦克风 ${inputDeviceSelect.length}`;
    inputDeviceSelect.appendChild(opt);
  }
}

async function rebuildIndex(): Promise<void> {
  const res = await fetch("/api/v3/index/rebuild?engine=engine_v3_personalized_top1", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason: "manual_rebuild_button" }) });
  const data = (await res.json()) as { ok: boolean; template_count?: number; templates_built?: number; error?: string; duration_ms?: number };
  if (!res.ok || !data.ok) throw new Error(data.error || "没有可用训练模板");
  const n = data.templates_built ?? data.template_count ?? 0;
  setStatus(`V3索引已重建：${n} 个模板`, false);
  pushLog(`索引重建完成：templates=${n}, duration_ms=${data.duration_ms ?? "-"}`);
  await refreshPhrases();
}

async function auditData(): Promise<void> {
  const res = await fetch("/api/v3/data/audit");
  const data = (await res.json()) as Record<string, unknown>;
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  auditReportEl.textContent = JSON.stringify(data, null, 2);
  pushLog("审计报告已生成到 data_v3/audit_report_v3.json");
}

async function archiveLegacy(): Promise<void> {
  const res = await fetch("/api/dataset/archive-legacy", { method: "POST" });
  const data = (await res.json()) as { ok: boolean; archive_dir?: string; error?: string };
  if (!res.ok || !data.ok) throw new Error(data.error || `HTTP ${res.status}`);
  pushLog(`旧 data 已复制归档：${data.archive_dir || "nothing"}`);
}

modeTrainBtn.addEventListener("click", () => setMode("train"));
modeEvalBtn.addEventListener("click", () => setMode("eval"));
modeUnknownBtn.addEventListener("click", () => setMode("unknown"));
modeDemoBtn.addEventListener("click", () => setMode("demo"));
trainPhraseSelect.addEventListener("change", renderPhraseGrid);
evalPhraseSelect.addEventListener("change", renderPhraseGrid);
inputDeviceSelect.addEventListener("change", () => {
  selectedDeviceId = inputDeviceSelect.value;
});
holdBtn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  void startRecording();
});
holdBtn.addEventListener("pointerup", (e) => {
  e.preventDefault();
  void stopRecording();
});
holdBtn.addEventListener("pointercancel", () => {
  void stopRecording();
});
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat) {
    e.preventDefault();
    void startRecording();
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    void stopRecording();
  }
});
workbenchRecordBtn.addEventListener("click", () => {
  if (!activeRerecordTask) {
    const msg = "当前没有错题重录任务。请先触发一次错判，系统会自动创建任务。";
    setStatus(msg, false);
    setEvalCorrectionResult(msg, true);
    return;
  }
  if (state.isRecording) {
    void stopRecording();
  } else {
    void startRecording();
  }
});

resetEvalBtn.addEventListener("click", () => {
  void fetch("/api/v3/eval/reset", { method: "POST" })
    .then(() => refreshEvalSummary())
    .then(() => refreshPhrases())
    .then(() => {
      clearRerecordTask();
      setStatus("新一轮评估已开始：请重新录 4×10 条 fresh eval", false);
      setEvalCorrectionResult("评估已重置，错题重录任务已清空。");
      pushLog("旧评估事件已清空。现在录的新样本才是校正后的真实泛化测试。");
    });
});
resetUnknownBtn.addEventListener("click", () => {
  void fetch("/api/v3/unknown/reset", { method: "POST" })
    .then(() => refreshUnknownSummary())
    .then(() => {
      setStatus("负样本测试已清空：请录 20 条四句之外的声音", false);
      pushLog("负样本测试事件已清空。");
    });
});
saveCorrectionBtn.addEventListener("click", () => void saveCorrection().catch((e) => setStatus(`纠正失败：${e instanceof Error ? e.message : String(e)}`, false)));
confirmCorrectionsBtn.addEventListener("click", () =>
  void confirmCorrections().catch((e) => {
    const msg = `合并失败：${e instanceof Error ? e.message : String(e)}`;
    setStatus(msg, false);
    setEvalCorrectionResult(msg, true);
    pushLog(msg);
  })
);
quickConfirmCorrectionsBtn.addEventListener("click", () =>
  void confirmCorrections().catch((e) => {
    const msg = `合并失败：${e instanceof Error ? e.message : String(e)}`;
    setStatus(msg, false);
    setEvalCorrectionResult(msg, true);
    pushLog(msg);
  })
);
refreshBtn.addEventListener("click", () => void refreshPhrases().then(() => refreshEvalSummary()).then(() => refreshUnknownSummary()).then(() => pushLog("V3 数据已刷新")));
refreshBtn.addEventListener("click", () => void refreshRuntimeConfig().then(() => refreshEvalErrors()).catch(() => undefined));
rebuildBtn.addEventListener("click", () => void rebuildIndex().catch((e) => setStatus(`重建失败：${e instanceof Error ? e.message : String(e)}`, false)));
auditBtn.addEventListener("click", () => void auditData().catch((e) => setStatus(`审计失败：${e instanceof Error ? e.message : String(e)}`, false)));
archiveBtn.addEventListener("click", () => void archiveLegacy().catch((e) => setStatus(`归档失败：${e instanceof Error ? e.message : String(e)}`, false)));
refreshErrorsBtn.addEventListener("click", () =>
  void refreshEvalErrors()
    .then(() => {
      const msg = "错题列表已刷新。";
      setStatus(msg, false);
      setEvalCorrectionResult(msg);
    })
    .catch((e) => {
      const msg = `错题刷新失败：${e instanceof Error ? e.message : String(e)}`;
      setStatus(msg, false);
      setEvalCorrectionResult(msg, true);
      pushLog(msg);
    })
);
addSelectedErrorsBtn.addEventListener("click", () =>
  void addSelectedEvalErrorsToCorrections().catch((e) => {
    const msg = `入纠错池失败：${e instanceof Error ? e.message : String(e)}`;
    setStatus(msg, false);
    setEvalCorrectionResult(msg, true);
    pushLog(msg);
  })
);
confirmSelectedErrorsBtn.addEventListener("click", () =>
  void confirmCorrections().catch((e) => {
    const msg = `确认合并失败：${e instanceof Error ? e.message : String(e)}`;
    setStatus(msg, false);
    setEvalCorrectionResult(msg, true);
    pushLog(msg);
  })
);
evalErrorFilterEl.addEventListener("change", () => {
  const value = evalErrorFilterEl.value as "all" | "accepted_wrong" | "rejected" | "low_margin";
  currentErrorFilter = value;
  renderErrorTable();
});
evalErrorsBodyEl.addEventListener("change", (evt) => {
  const target = evt.target as HTMLInputElement | null;
  if (!target || target.tagName !== "INPUT" || target.type !== "checkbox") return;
  const eventId = target.getAttribute("data-event-id");
  if (!eventId) return;
  if (target.checked) selectedErrorIds.add(eventId);
  else selectedErrorIds.delete(eventId);
  renderErrorTable();
});
evalErrorsBodyEl.addEventListener("click", (evt) => {
  const target = evt.target as HTMLElement | null;
  if (!target) return;
  const btn = target.closest("button[data-rerecord-truth-id]") as HTMLButtonElement | null;
  if (!btn) return;
  const truthId = btn.getAttribute("data-rerecord-truth-id");
  if (!truthId) return;
  beginRerecordTask(truthId, null);
  evalPhraseSelect.value = truthId;
  renderPhraseGrid();
  setMode("eval");
  void refreshRerecordPending()
    .then(() => {
      const msg = `错题重录已就绪：请按住录音，按真值 ${phraseText(truthId)} 重录（0/${REQUIRED_RERECORD_COUNT}）`;
      setStatus(msg, false);
      setEvalCorrectionResult(msg);
      pushLog(msg);
    })
    .catch((e) => {
      setStatus(`重录任务初始化失败：${e instanceof Error ? e.message : String(e)}`, false);
      setEvalCorrectionResult(`重录任务初始化失败：${e instanceof Error ? e.message : String(e)}`, true);
    });
});
rerecordPendingBodyEl.addEventListener("click", (evt) => {
  const target = evt.target as HTMLElement | null;
  if (!target) return;
  const btn = target.closest("button[data-delete-pending-sample-id]") as HTMLButtonElement | null;
  if (!btn) return;
  const sampleId = btn.getAttribute("data-delete-pending-sample-id");
  if (!sampleId) return;
  void deletePendingCorrection(sampleId).catch((e) => {
    const msg = e instanceof Error ? e.message : String(e);
    setStatus(`删除待合并样本失败：${msg}`, false);
    setEvalCorrectionResult(`删除待合并样本失败：${msg}`, true);
  });
});
selectAllErrorsEl.addEventListener("change", () => {
  const visible = filteredErrorItems();
  for (const item of visible) {
    if (selectAllErrorsEl.checked) selectedErrorIds.add(item.event_id);
    else selectedErrorIds.delete(item.event_id);
  }
  renderErrorTable();
});

pushLog(`V3 页面已就绪。建议先在训练模式采集4句，每句${TRAIN_TARGET}条，评估每句${EVAL_TARGET}条。`);
setMode("train");
updateWorkbenchRerecordControls();
renderRerecordTaskPanel();
setEvalCorrectionResult("评估纠错结果将在这里显示。");
void (async () => {
  try {
    await refreshPhrases();
  } catch (err) {
    setStatus(`初始化失败：${err instanceof Error ? err.message : String(err)}`, false);
    return;
  }
  try {
    await refreshRuntimeConfig();
    await refreshEvalSummary();
    await refreshEvalErrors();
    await refreshUnknownSummary();
    await refreshInputDevices();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(`初始化部分失败：${msg}`, false);
    setEvalCorrectionResult(`初始化警告：${msg}。核心录音/评估功能不受影响，可继续使用。`, true);
    pushLog(`初始化非关键步骤失败：${msg}`);
  }
})();

navigator.mediaDevices?.addEventListener?.("devicechange", () => {
  void refreshInputDevices();
});


