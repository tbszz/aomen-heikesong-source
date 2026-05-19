import { useEffect, useMemo, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import * as speechCommands from "@tensorflow-models/speech-commands";

type Stage = "record" | "train" | "use";
type DataSplit = "train" | "eval";

type TrainingState = {
  loss: number;
  acc: number;
  valLoss?: number;
  valAcc?: number;
  epoch: number;
  totalEpochs: number;
};

type RecognitionEvent = {
  label: string;
  score: number;
  at: number;
  rejected: boolean;
};

type EvalPhraseMetric = {
  phrase: string;
  total: number;
  correct: number;
  rejected: number;
  accuracy: number;
};

type EvalSummary = {
  total: number;
  correct: number;
  rejected: number;
  accuracy: number;
  rejectRate: number;
  perPhrase: EvalPhraseMetric[];
};

const NOISE_LABEL = "_background_noise_";
const TRAIN_TRANSFER_NAME = "voicebridge-mvp-train";
const EVAL_TRANSFER_NAME = "voicebridge-mvp-eval";

const DEFAULT_PHRASES = [
  "但是工作原因会经常与人打交道",
  "你叫什么名字",
  "你是哪里人",
  "你猜猜我多大",
  "初次见面",
  "多多关照",
  "很高兴认识你",
  "我叫李朋程",
  "我平时在北京上班",
  "我是内蒙古人",
  "我来自内蒙古",
  "朋是朋友的朋",
  "程是过程的程"
];

const SPEAKING_MODES = [
  { id: "normal", label: "正常语调", description: "标准语速和音量" },
  { id: "soft", label: "轻声/低音量", description: "降低音量，接近耳语" },
  { id: "fast", label: "稍快", description: "自然语速但稍快" },
  { id: "slow", label: "稍慢", description: "清晰且稍慢的发音" },
  { id: "fatigued", label: "疲劳/不稳定", description: "虚弱或不稳定的声音状态" }
] as const;

type SpeakingModeId = (typeof SPEAKING_MODES)[number]["id"];

type ModeStat = {
  train: number;
  eval: number;
};

type ModeStats = Record<string, Record<SpeakingModeId, ModeStat>>;

type DatasetMeta = {
  phrases: string[];
  updatedAt: string;
};

const MODE_TARGET_PER_STATE = 10;
const MIN_TOTAL_PER_PHRASE = 30;
const RECOMMENDED_TOTAL_PER_PHRASE = 50;
const MIN_NOISE_SAMPLES = 10;

const DATASET_META_KEY = "voicebridge.dataset.meta.v2";
const MODE_STATS_KEY = "voicebridge.mode.stats.v1";
const EVAL_EXAMPLES_KEY = "voicebridge.eval.examples.v1";
const THRESHOLD_KEY = "voicebridge.threshold.v1";
const TTS_ENABLED_KEY = "voicebridge.tts.enabled.v1";

export function App() {
  const [stage, setStage] = useState<Stage>("record");
  const [selectedPhrase, setSelectedPhrase] = useState(DEFAULT_PHRASES[0]);
  const [selectedMode, setSelectedMode] = useState<SpeakingModeId>("normal");
  const [selectedSplit, setSelectedSplit] = useState<DataSplit>("train");
  const [sampleDurationSec, setSampleDurationSec] = useState(1.2);
  const [collecting, setCollecting] = useState(false);
  const [training, setTraining] = useState(false);
  const [listening, setListening] = useState(false);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("Loading model...");
  const [error, setError] = useState<string | null>(null);
  const [trainCounts, setTrainCounts] = useState<Record<string, number>>({});
  const [evalCounts, setEvalCounts] = useState<Record<string, number>>({});
  const [modeStats, setModeStats] = useState<ModeStats>({});
  const [trainingState, setTrainingState] = useState<TrainingState | null>(null);
  const [events, setEvents] = useState<RecognitionEvent[]>([]);
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null);
  const [threshold, setThreshold] = useState<number>(() => {
    const raw = localStorage.getItem(THRESHOLD_KEY);
    return raw ? Number(raw) : 0.72;
  });
  const [ttsEnabled, setTtsEnabled] = useState<boolean>(() => {
    return localStorage.getItem(TTS_ENABLED_KEY) !== "false";
  });
  const [strictModeCoverage, setStrictModeCoverage] = useState(true);
  const [epochs, setEpochs] = useState(30);
  const [fineTuningEpochs, setFineTuningEpochs] = useState(12);
  const [noiseMixRatio, setNoiseMixRatio] = useState(0.25);
  const [useBackendSvm, setUseBackendSvm] = useState(false);
  const [svmStatus, setSvmStatus] = useState<{loaded: boolean; classes: string[]} | null>(null);

  const trainTransferRef = useRef<speechCommands.TransferSpeechCommandRecognizer | null>(null);
  const evalTransferRef = useRef<speechCommands.TransferSpeechCommandRecognizer | null>(null);

  const phrases = useMemo(() => DEFAULT_PHRASES, []);

  useEffect(() => {
    localStorage.setItem(THRESHOLD_KEY, String(threshold));
  }, [threshold]);

  useEffect(() => {
    localStorage.setItem(TTS_ENABLED_KEY, String(ttsEnabled));
  }, [ttsEnabled]);

  useEffect(() => {
    localStorage.setItem(MODE_STATS_KEY, JSON.stringify(modeStats));
  }, [modeStats]);

  useEffect(() => {
    const current = selectedPhrase;
    if (!phrases.includes(current) && phrases.length > 0) {
      setSelectedPhrase(phrases[0]);
    }
  }, [phrases, selectedPhrase]);

  useEffect(() => {
    setModeStats((prev) => normalizeModeStats(prev, phrases));
  }, [phrases]);

  useEffect(() => {
    let mounted = true;

    const setup = async () => {
      try {
        setLoading(true);
        setError(null);
        setStatus("Initializing speech command base model...");

        const base = speechCommands.create("BROWSER_FFT");
        await base.ensureModelLoaded();

        const trainTransfer = base.createTransfer(TRAIN_TRANSFER_NAME);
        const evalTransfer = base.createTransfer(EVAL_TRANSFER_NAME);

        trainTransferRef.current = trainTransfer;
        evalTransferRef.current = evalTransfer;

        setSelectedPhrase(DEFAULT_PHRASES[0]);
        writeDatasetMeta({ phrases: DEFAULT_PHRASES, updatedAt: new Date().toISOString() });
        const localModeStats = readModeStats();
        setModeStats(normalizeModeStats(localModeStats, DEFAULT_PHRASES));

        const evalSerialized = localStorage.getItem(EVAL_EXAMPLES_KEY);
        if (evalSerialized) {
          try {
            evalTransfer.loadExamples(base64ToArrayBuffer(evalSerialized), true);
            const changed = pruneTransferDatasetToAllowedLabels(evalTransfer, [
              ...DEFAULT_PHRASES,
              NOISE_LABEL
            ]);
            if (changed) {
              persistEvalExamples(evalTransfer);
            }
          } catch (loadErr) {
            console.warn("Failed to load eval examples", loadErr);
          }
        }

        const loadedTrainModel = await loadSavedModel(trainTransfer);
        if (loadedTrainModel) {
          const valid = transferModelLabelsAllowed(trainTransfer.wordLabels(), DEFAULT_PHRASES);
          if (!valid) {
            await speechCommands.deleteSavedTransferModel(TRAIN_TRANSFER_NAME);
            setStatus("Old saved model used removed labels and was deleted. Please retrain with current 4 phrases.");
            setStage("record");
          } else {
            setStatus("Saved transfer model loaded from IndexedDB.");
            setStage("use");
          }
        } else {
          setStatus("Base model ready. Record train/eval samples for transfer learning.");
        }

        // 检查后端SVM状态
        try {
          const res = await fetch("http://127.0.0.1:8766/api/svm/status");
          if (res.ok) {
            const data = await res.json();
            setSvmStatus(data);
          }
        } catch (e) {
          console.warn("SVM状态获取失败:", e);
        }

        if (mounted) {
          setTrainCounts(trainTransfer.countExamples());
          setEvalCounts(evalTransfer.countExamples());
        }
      } catch (err) {
        setError(stringifyError(err));
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    setup();

    return () => {
      mounted = false;
      void stopListening();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const trainSampleTotal = Object.values(trainCounts).reduce((a, b) => a + b, 0);
  const evalSampleTotal = Object.values(evalCounts).reduce((a, b) => a + b, 0);

  async function collect(label: string) {
    const transfer = selectedSplit === "train" ? trainTransferRef.current : evalTransferRef.current;
    if (!transfer || collecting || training || listening) {
      return;
    }

    try {
      setCollecting(true);
      setError(null);
      setStatus(`Recording ${selectedSplit} sample for "${label}" (${selectedMode})...`);

      await transfer.collectExample(label, {
        durationSec: sampleDurationSec,
        includeRawAudio: false
      });

      if (selectedSplit === "train") {
        setTrainCounts(transfer.countExamples());
      } else {
        setEvalCounts(transfer.countExamples());
        persistEvalExamples(transfer);
      }

      if (label !== NOISE_LABEL) {
        setModeStats((prev) => incrementModeStat(prev, label, selectedMode, selectedSplit));
      }

      setStatus(`Sample added to ${selectedSplit} set for "${label}".`);
    } catch (err) {
      setError(stringifyError(err));
    } finally {
      setCollecting(false);
    }
  }

  async function clearTrainExamples() {
    const transfer = trainTransferRef.current;
    if (!transfer || collecting || training || listening) {
      return;
    }

    transfer.clearExamples();
    setTrainCounts({});
    setModeStats((prev) => clearSplitFromModeStats(prev, "train", phrases));
    setStatus("Train examples cleared.");
  }

  async function clearEvalExamples() {
    const transfer = evalTransferRef.current;
    if (!transfer || collecting || training || listening) {
      return;
    }

    transfer.clearExamples();
    setEvalCounts({});
    setModeStats((prev) => clearSplitFromModeStats(prev, "eval", phrases));
    localStorage.removeItem(EVAL_EXAMPLES_KEY);
    setStatus("Eval examples cleared.");
  }

  async function trainModel() {
    const transfer = trainTransferRef.current;
    if (!transfer || training || collecting || listening) {
      return;
    }

    try {
      setTraining(true);
      setError(null);
      setTrainingState(null);
      setEvalSummary(null);

      if (phrases.length !== 4) {
        setError("This phase is intentionally constrained to 4 short sentences. Keep exactly 4 phrases for the MVP.");
        return;
      }

      const localCounts = transfer.countExamples();
      for (const phrase of phrases) {
        const total = localCounts[phrase] ?? 0;
        if (total < MIN_TOTAL_PER_PHRASE) {
          setError(`Phrase "${phrase}" has ${total} train samples. Need at least ${MIN_TOTAL_PER_PHRASE}.`);
          return;
        }
      }

      if ((localCounts[NOISE_LABEL] ?? 0) < MIN_NOISE_SAMPLES) {
        setError(`Need at least ${MIN_NOISE_SAMPLES} train noise samples before training.`);
        return;
      }

      if (strictModeCoverage) {
        for (const phrase of phrases) {
          for (const mode of SPEAKING_MODES) {
            const count = modeStats[phrase]?.[mode.id]?.train ?? 0;
            if (count < MODE_TARGET_PER_STATE) {
              setError(`Phrase "${phrase}" / ${mode.label} has ${count}. Need ${MODE_TARGET_PER_STATE}.`);
              return;
            }
          }
        }
      }

      setStatus("Training transfer model...");

      await transfer.train({
        epochs,
        validationSplit: 0.2,
        fineTuningEpochs,
        augmentByMixingNoiseRatio: noiseMixRatio,
        callback: {
          onEpochEnd: async (epoch: number, logs?: Record<string, number>) => {
            setTrainingState({
              epoch: epoch + 1,
              totalEpochs: epochs,
              loss: logs?.loss ?? NaN,
              acc: logs?.acc ?? logs?.accuracy ?? NaN,
              valLoss: logs?.val_loss,
              valAcc: logs?.val_acc ?? logs?.val_accuracy
            });
          }
        }
      });

      await transfer.save();
      writeDatasetMeta({ phrases, updatedAt: new Date().toISOString() });

      setStatus("Training complete. Transfer model saved to IndexedDB.");
      setStage("use");
    } catch (err) {
      setError(stringifyError(err));
    } finally {
      setTraining(false);
    }
  }

  async function runHoldoutEvaluation() {
    const transfer = trainTransferRef.current;
    const evalTransfer = evalTransferRef.current;

    if (!transfer || !evalTransfer || training || collecting || listening) {
      return;
    }

    try {
      setError(null);
      setStatus("Running holdout evaluation...");

      const labels = transfer.wordLabels();
      const phraseSet = new Set(phrases);
      let total = 0;
      let correct = 0;
      let rejected = 0;

      const phraseMetrics: EvalPhraseMetric[] = [];

      for (const phrase of phrases) {
        const examples = evalTransfer.getExamples(phrase);
        let phraseTotal = 0;
        let phraseCorrect = 0;
        let phraseRejected = 0;

        for (const item of examples) {
          const pred = await predictExample(transfer, item.example.spectrogram, labels);
          const predRejected = pred.score < threshold || pred.label === NOISE_LABEL || !phraseSet.has(pred.label);

          total += 1;
          phraseTotal += 1;

          if (predRejected) {
            rejected += 1;
            phraseRejected += 1;
            continue;
          }

          if (pred.label === phrase) {
            correct += 1;
            phraseCorrect += 1;
          }
        }

        phraseMetrics.push({
          phrase,
          total: phraseTotal,
          correct: phraseCorrect,
          rejected: phraseRejected,
          accuracy: phraseTotal > 0 ? phraseCorrect / phraseTotal : 0
        });
      }

      if (total === 0) {
        setError("No evaluation samples found. Record eval samples first.");
        return;
      }

      const summary: EvalSummary = {
        total,
        correct,
        rejected,
        accuracy: correct / total,
        rejectRate: rejected / total,
        perPhrase: phraseMetrics
      };

      setEvalSummary(summary);
      setStatus("Holdout evaluation complete.");
    } catch (err) {
      setError(stringifyError(err));
    }
  }

  async function toggleListening() {
    if (listening) {
      await stopListening();
      return;
    }

    const transfer = trainTransferRef.current;
    if (!transfer || collecting || training) {
      return;
    }

    try {
      setError(null);
      setStatus("Starting real-time recognition...");

      await transfer.listen(
        async ({ scores }) => {
          const labels = transfer.wordLabels();
          const arr = Array.from(scores as Float32Array);
          const best = pickBest(labels, arr);

          const rejected = best.score < threshold || best.label === NOISE_LABEL;
          const label = rejected ? "(rejected)" : best.label;

          setEvents((prev) => [{
            label,
            score: best.score,
            at: Date.now(),
            rejected
          }, ...prev].slice(0, 20));

          if (!rejected && ttsEnabled) {
            speak(best.label);
          }
        },
        {
          probabilityThreshold: threshold,
          overlapFactor: 0.5,
          invokeCallbackOnNoiseAndUnknown: true
        }
      );

      setListening(true);
      setStatus("Listening for commands...");
    } catch (err) {
      setError(stringifyError(err));
    }
  }

  async function stopListening() {
    const transfer = trainTransferRef.current;
    if (!transfer || !transfer.isListening()) {
      setListening(false);
      return;
    }

    await transfer.stopListening();
    setListening(false);
    setStatus("Recognition stopped.");
  }

  async function svmRecordAndPredict() {
    if (listening || !svmStatus?.loaded) return;

    try {
      setListening(true);
      setStatus("Recording audio...");
      setError(null);

      // 使用浏览器API录制音频
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());

        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const arrayBuffer = await audioBlob.arrayBuffer();
        const audioData = new Float32Array(arrayBuffer);

        setStatus("Predicting with SVM...");

        const formData = new FormData();
        const wavBlob = new Blob([audioData.buffer], { type: 'audio/wav' });
        formData.append('file', wavBlob, 'recording.wav');

        const res = await fetch('http://127.0.0.1:8766/api/svm/predict', {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || 'Prediction failed');
        }

        const result = await res.json();
        const phrase = result.phrase;

        setEvents((prev) => [{
          label: phrase,
          score: 1.0,
          at: Date.now(),
          rejected: false
        }, ...prev].slice(0, 20));

        setStatus(`Recognized: ${phrase}`);

        if (ttsEnabled) {
          speak(phrase);
        }

        setListening(false);
      };

      mediaRecorder.start();

      // 录制2秒
      setTimeout(() => {
        mediaRecorder.stop();
      }, 2000);

    } catch (err) {
      setError(stringifyError(err));
      setListening(false);
    }
  }

  async function resetSavedModel() {
    if (listening) {
      await stopListening();
    }

    try {
      setError(null);
      await speechCommands.deleteSavedTransferModel(TRAIN_TRANSFER_NAME);
      trainTransferRef.current?.clearExamples();
      setTrainCounts({});
      setTrainingState(null);
      setEvents([]);
      setEvalSummary(null);
      setModeStats((prev) => clearSplitFromModeStats(prev, "train", phrases));
      setStage("record");
      setStatus("Saved train model removed from IndexedDB.");
    } catch (err) {
      setError(stringifyError(err));
    }
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <h1>VoiceBridge Browser MVP</h1>
        <p>
          Record - Train - Recognize in browser using TensorFlow.js Speech Commands.
          This version enforces a 4-phrase pilot with train/eval split and state-aware sampling.
        </p>
      </header>

      <section className="toolbar">
        <button className={stage === "record" ? "active" : ""} onClick={() => setStage("record")}>1. 录音</button>
        <button className={stage === "train" ? "active" : ""} onClick={() => setStage("train")}>2. 训练</button>
        <button className={stage === "use" ? "active" : ""} onClick={() => setStage("use")}>3. 使用</button>
      </section>

      <section className="status-card">
        <div><strong>Status:</strong> {loading ? "Loading..." : status}</div>
        <div><strong>训练样本:</strong> {trainSampleTotal}</div>
        <div><strong>评估样本:</strong> {evalSampleTotal}</div>
        {error ? <div className="error"><strong>Error:</strong> {error}</div> : null}
      </section>

      {stage === "record" && (
        <section className="panel">
          <h2>录音页面</h2>
          <p>
            Pilot recommendation: fixed 4 short high-frequency sentences. Train target is {RECOMMENDED_TOTAL_PER_PHRASE}
            per phrase ({MODE_TARGET_PER_STATE} x {SPEAKING_MODES.length} states).
          </p>

          <div className="row">
            <span className="ok">固定四句（已锁定）</span>
            {phrases.map((phrase) => (
              <span key={phrase} className="fixed-tag">{phrase}</span>
            ))}
          </div>

          <div className="row">
            <label>Data split</label>
            <select value={selectedSplit} onChange={(e) => setSelectedSplit(e.target.value as DataSplit)}>
              <option value="train">Train set</option>
              <option value="eval">Eval holdout set</option>
            </select>

            <label>Speaking state</label>
            <select value={selectedMode} onChange={(e) => setSelectedMode(e.target.value as SpeakingModeId)}>
              {SPEAKING_MODES.map((mode) => (
                <option key={mode.id} value={mode.id}>{mode.label}</option>
              ))}
            </select>

            <label>Sample duration (seconds)</label>
            <input
              type="number"
              min={0.8}
              max={3}
              step={0.1}
              value={sampleDurationSec}
              onChange={(e) => setSampleDurationSec(Number(e.target.value))}
            />
          </div>

          <div className="row">
            <label>短语</label>
            <select value={selectedPhrase} onChange={(e) => setSelectedPhrase(e.target.value)}>
              {phrases.map((phrase) => (
                <option key={phrase} value={phrase}>{phrase}</option>
              ))}
            </select>
            <button disabled={collecting || loading} onClick={() => void collect(selectedPhrase)}>
              {collecting ? "Recording..." : `Record ${selectedSplit} sample`}
            </button>
            <button disabled={collecting || loading} onClick={() => void collect(NOISE_LABEL)}>
              Record {selectedSplit} noise
            </button>
          </div>

          <div className="row">
            <button disabled={collecting || loading} onClick={() => void clearTrainExamples()}>Clear train samples</button>
            <button disabled={collecting || loading} onClick={() => void clearEvalExamples()}>Clear eval samples</button>
          </div>

          <div className="mode-guide">
            {SPEAKING_MODES.map((mode) => (
              <div key={mode.id} className="mode-card">
                <strong>{mode.label}</strong>
                <p>{mode.description}</p>
              </div>
            ))}
          </div>

          <div className="counts-grid">
            {phrases.map((phrase) => {
              const trainTotal = trainCounts[phrase] ?? 0;
              const evalTotal = evalCounts[phrase] ?? 0;
              const recommendedReady = trainTotal >= RECOMMENDED_TOTAL_PER_PHRASE;

              return (
                <div key={phrase} className="count-item vertical">
                  <div className="count-header">
                    <span>{phrase}</span>
                    <strong className={recommendedReady ? "ok" : "warn"}>{trainTotal} train</strong>
                  </div>
                  <div className="sub">Eval: {evalTotal}</div>
                  {SPEAKING_MODES.map((mode) => (
                    <div key={`${phrase}-${mode.id}`} className="sub-row">
                      <span>{mode.label}</span>
                      <span>
                        T {modeStats[phrase]?.[mode.id]?.train ?? 0} / E {modeStats[phrase]?.[mode.id]?.eval ?? 0}
                      </span>
                    </div>
                  ))}
                </div>
              );
            })}
            <div className="count-item noise vertical">
              <span>{NOISE_LABEL}</span>
              <strong>Train: {trainCounts[NOISE_LABEL] ?? 0}</strong>
              <strong>Eval: {evalCounts[NOISE_LABEL] ?? 0}</strong>
            </div>
          </div>
        </section>
      )}

      {stage === "train" && (
        <section className="panel">
          <h2>训练页面</h2>
          <p>
            Train only on the train split. Holdout eval split is never mixed into training.
          </p>

          <div className="row">
            <label>
              <input
                type="checkbox"
                checked={strictModeCoverage}
                onChange={(e) => setStrictModeCoverage(e.target.checked)}
              />
              Enforce {MODE_TARGET_PER_STATE} samples per phrase per speaking state
            </label>
          </div>

          <div className="row">
            <label>Epochs</label>
            <input type="number" min={10} max={80} value={epochs} onChange={(e) => setEpochs(Number(e.target.value))} />

            <label>Fine-tuning epochs</label>
            <input
              type="number"
              min={0}
              max={40}
              value={fineTuningEpochs}
              onChange={(e) => setFineTuningEpochs(Number(e.target.value))}
            />

            <label>Noise mix ratio (augmentation)</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.05}
              value={noiseMixRatio}
              onChange={(e) => setNoiseMixRatio(Number(e.target.value))}
            />
          </div>

          <p className="note">
            Implemented augmentation path uses built-in noise mixing (`augmentByMixingNoiseRatio`).
            Volume/speed/fatigue variation is handled at collection time via speaking-state buckets.
          </p>

          <button className="primary" disabled={training || collecting || loading} onClick={() => void trainModel()}>
            {training ? "Training..." : "Start transfer training"}
          </button>

          {trainingState && (
            <div className="metrics">
              <div>Epoch: {trainingState.epoch} / {trainingState.totalEpochs}</div>
              <div>Loss: {fmt(trainingState.loss)}</div>
              <div>Accuracy: {fmt(trainingState.acc)}</div>
              <div>Val Loss: {fmt(trainingState.valLoss)}</div>
              <div>Val Acc: {fmt(trainingState.valAcc)}</div>
            </div>
          )}
        </section>
      )}

      {stage === "use" && (
        <section className="panel">
          <h2>使用与评估页面</h2>
          <div className="row">
            <label>Recognition mode:</label>
            <label>
              <input
                type="radio"
                name="recogMode"
                checked={!useBackendSvm}
                onChange={() => setUseBackendSvm(false)}
              />
              Browser TF.js
            </label>
            <label>
              <input
                type="radio"
                name="recogMode"
                checked={useBackendSvm}
                onChange={() => setUseBackendSvm(true)}
                disabled={!svmStatus?.loaded}
              />
              Backend SVM {svmStatus?.loaded ? "(OK)" : "(Loading...)"}
            </label>
          </div>

          {!useBackendSvm && (
          <div className="row">
            <label>概率阈值</label>
            <input
              type="range"
              min={0.4}
              max={0.95}
              step={0.01}
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
            <strong>{threshold.toFixed(2)}</strong>
          </div>
          )}

          <div className="row">
            <label>
              <input
                type="checkbox"
                checked={ttsEnabled}
                onChange={(e) => setTtsEnabled(e.target.checked)}
              />
              启用浏览器TTS语音输出
            </label>

            {useBackendSvm ? (
            <div className="row">
              <button
                className={listening ? "danger" : "primary"}
                onClick={() => void svmRecordAndPredict()}
                disabled={listening}
              >
                {listening ? "Recording..." : "Record & Predict (SVM)"}
              </button>
              <span style={{fontSize: '12px', color: '#666'}}>
                Available phrases: {svmStatus?.classes?.slice(0,3).join(', ')}...
              </span>
            </div>
          ) : (
            <>
            <button className={listening ? "danger" : "primary"} onClick={() => void toggleListening()}>
              {listening ? "Stop recognition" : "Start recognition"}
            </button>
            <button onClick={() => void runHoldoutEvaluation()}>Run holdout eval</button>
            <button onClick={() => void resetSavedModel()}>Delete saved model</button>
            </>
          )}
          </div>

          {evalSummary ? (
            <div className="eval-card">
              <h3>Holdout Evaluation</h3>
              <div className="metrics">
                <div>Total: {evalSummary.total}</div>
                <div>Correct: {evalSummary.correct}</div>
                <div>Rejected: {evalSummary.rejected}</div>
                <div>Accuracy: {(evalSummary.accuracy * 100).toFixed(1)}%</div>
                <div>Reject rate: {(evalSummary.rejectRate * 100).toFixed(1)}%</div>
              </div>
              <table>
                <thead>
                  <tr>
                    <th>Phrase</th>
                    <th>Total</th>
                    <th>Correct</th>
                    <th>Rejected</th>
                    <th>Accuracy</th>
                  </tr>
                </thead>
                <tbody>
                  {evalSummary.perPhrase.map((metric) => (
                    <tr key={metric.phrase}>
                      <td>{metric.phrase}</td>
                      <td>{metric.total}</td>
                      <td>{metric.correct}</td>
                      <td>{metric.rejected}</td>
                      <td>{(metric.accuracy * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}

          <div className="events">
            {events.length === 0 ? (
              <p>No recognition events yet.</p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Result</th>
                    <th>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev) => (
                    <tr key={`${ev.at}-${ev.label}-${ev.score.toFixed(4)}`}>
                      <td>{new Date(ev.at).toLocaleTimeString()}</td>
                      <td className={ev.rejected ? "rejected" : "accepted"}>{ev.label}</td>
                      <td>{ev.score.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>
      )}
    </main>
  );
}

async function predictExample(
  transfer: speechCommands.TransferSpeechCommandRecognizer,
  spectrogram: speechCommands.SpectrogramData,
  labels: string[]
): Promise<{ label: string; score: number }> {
  const frameCount = Math.floor(spectrogram.data.length / spectrogram.frameSize);
  const input = tf.tensor4d(spectrogram.data, [1, frameCount, spectrogram.frameSize, 1]);
  const output = await transfer.recognize(input);
  input.dispose();

  const scores = Array.from(output.scores as Float32Array);
  return pickBest(labels, scores);
}

function pickBest(labels: string[], scores: number[]): { label: string; score: number } {
  let bestLabel = labels[0] ?? "";
  let bestScore = scores[0] ?? 0;

  for (let i = 1; i < scores.length; i += 1) {
    if (scores[i] > bestScore) {
      bestScore = scores[i];
      bestLabel = labels[i] ?? bestLabel;
    }
  }

  return { label: bestLabel, score: bestScore };
}

function fmt(v: number | undefined): string {
  if (typeof v !== "number" || Number.isNaN(v)) {
    return "-";
  }
  return v.toFixed(4);
}

function stringifyError(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err);
}

function writeDatasetMeta(meta: DatasetMeta): void {
  localStorage.setItem(DATASET_META_KEY, JSON.stringify(meta));
}

function readDatasetMeta(): DatasetMeta | null {
  const raw = localStorage.getItem(DATASET_META_KEY);
  if (!raw) {
    return null;
  }

  try {
    return JSON.parse(raw) as DatasetMeta;
  } catch {
    return null;
  }
}

function readModeStats(): ModeStats {
  const raw = localStorage.getItem(MODE_STATS_KEY);
  if (!raw) {
    return {};
  }

  try {
    return JSON.parse(raw) as ModeStats;
  } catch {
    return {};
  }
}

function normalizeModeStats(stats: ModeStats, phrases: string[]): ModeStats {
  const out: ModeStats = {};

  for (const phrase of phrases) {
    out[phrase] = {} as Record<SpeakingModeId, ModeStat>;
    for (const mode of SPEAKING_MODES) {
      const current = stats[phrase]?.[mode.id];
      out[phrase][mode.id] = {
        train: current?.train ?? 0,
        eval: current?.eval ?? 0
      };
    }
  }

  return out;
}

function transferModelLabelsAllowed(labels: string[], allowedPhrases: string[]): boolean {
  const allowed = new Set([NOISE_LABEL, ...allowedPhrases]);
  for (const label of labels) {
    if (!allowed.has(label)) {
      return false;
    }
  }
  return true;
}

function pruneTransferDatasetToAllowedLabels(
  transfer: speechCommands.TransferSpeechCommandRecognizer,
  allowedLabels: string[]
): boolean {
  const allowed = new Set(allowedLabels);
  const metaLabels = transfer.getMetadata().wordLabels ?? [];
  let changed = false;

  for (const label of metaLabels) {
    if (allowed.has(label)) {
      continue;
    }
    const examples = transfer.getExamples(label);
    for (const item of examples) {
      transfer.removeExample(item.uid);
      changed = true;
    }
  }

  return changed;
}

function incrementModeStat(
  stats: ModeStats,
  phrase: string,
  mode: SpeakingModeId,
  split: DataSplit
): ModeStats {
  const next = { ...stats };
  const phraseStats = next[phrase] ? { ...next[phrase] } : ({} as Record<SpeakingModeId, ModeStat>);

  for (const m of SPEAKING_MODES) {
    if (!phraseStats[m.id]) {
      phraseStats[m.id] = { train: 0, eval: 0 };
    }
  }

  const old = phraseStats[mode];
  phraseStats[mode] = {
    ...old,
    [split]: old[split] + 1
  };

  next[phrase] = phraseStats;
  return next;
}

function clearSplitFromModeStats(stats: ModeStats, split: DataSplit, phrases: string[]): ModeStats {
  const next: ModeStats = {};

  for (const phrase of phrases) {
    next[phrase] = {} as Record<SpeakingModeId, ModeStat>;
    for (const mode of SPEAKING_MODES) {
      const old = stats[phrase]?.[mode.id] ?? { train: 0, eval: 0 };
      next[phrase][mode.id] = {
        ...old,
        [split]: 0
      };
    }
  }

  return next;
}

function persistEvalExamples(transfer: speechCommands.TransferSpeechCommandRecognizer): void {
  try {
    if (transfer.isDatasetEmpty()) {
      localStorage.removeItem(EVAL_EXAMPLES_KEY);
      return;
    }

    const serialized = transfer.serializeExamples();
    localStorage.setItem(EVAL_EXAMPLES_KEY, arrayBufferToBase64(serialized));
  } catch (err) {
    console.warn("Failed to persist eval examples", err);
  }
}

async function loadSavedModel(
  transfer: speechCommands.TransferSpeechCommandRecognizer
): Promise<boolean> {
  const names = await speechCommands.listSavedTransferModels();
  if (!names.includes(TRAIN_TRANSFER_NAME)) {
    return false;
  }

  await transfer.load();
  return true;
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";

  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }

  return btoa(binary);
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

function speak(text: string): void {
  if (!("speechSynthesis" in window)) {
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

