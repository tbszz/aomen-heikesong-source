import { useEffect, useMemo, useRef, useState } from "react";

type AppMode = "train" | "eval" | "unknown" | "demo";

type Phrase = {
  phrase_id: string;
  text: string;
  train: number;
  eval: number;
  corrections: number;
  rejected: number;
};

type EvalSummary = {
  total: number;
  accepted: number;
  top1: number;
  top2: number;
  top1_rate: number;
  top2_rate: number;
  reject_rate: number;
};

type EvalError = {
  id: string;
  truth_phrase_id: string;
  truth_text: string;
  pred_phrase_id: string;
  pred_text: string;
  score: number;
  margin: number;
  is_rerecord: boolean;
  is_corrected: boolean;
  created_at: string;
};

type UnknownSummary = {
  total: number;
  false_accepts: number;
  correct_rejects: number;
  false_accept_rate: number;
};

type DemoResult = {
  final_text: string;
  final_phrase_id: string;
  matched_phrase: string;
  score: number;
  pred_phrase_id: string;
  pred_text: string;
  reject_reason: string;
  tts_audio_url?: string;
  source: string;
};

const API_BASE = "/api/v3";

export function App() {
  const [mode, setMode] = useState<AppMode>("train");
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("加载中...");
  const [error, setError] = useState<string | null>(null);

  // 数据
  const [phrases, setPhrases] = useState<Phrase[]>([]);
  const [evalSummary, setEvalSummary] = useState<EvalSummary | null>(null);
  const [evalErrors, setEvalErrors] = useState<EvalError[]>([]);
  const [unknownSummary, setUnknownSummary] = useState<UnknownSummary | null>(null);

  // 录音状态
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

  // 各种选中的值
  const [selectedPhraseId, setSelectedPhraseId] = useState("");
  const [demoResult, setDemoResult] = useState<DemoResult | null>(null);
  const [ttsEnabled, setTtsEnabled] = useState(true);

  // 音频录制相关
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  // 加载短语列表
  useEffect(() => {
    loadPhrases();
    loadEvalSummary();
    loadUnknownSummary();
  }, []);

  async function loadPhrases() {
    try {
      const res = await fetch(`${API_BASE}/phrases`);
      if (res.ok) {
        const data = await res.json();
        const phraseList = data.phrases || [];
        setPhrases(phraseList);
        if (phraseList.length > 0) {
          setSelectedPhraseId(phraseList[0].phrase_id);
        }
      }
    } catch (e) {
      console.warn("加载短语失败:", e);
      setError("无法连接后端，请确保后端已启动");
    } finally {
      setLoading(false);
    }
  }

  async function loadEvalSummary() {
    try {
      const res = await fetch(`${API_BASE}/eval/export`);
      if (res.ok) {
        const data = await res.json();
        setEvalSummary(data.summary || null);
      }
    } catch (e) {
      console.warn("加载评估统计失败:", e);
    }
  }

  async function loadEvalErrors() {
    try {
      const res = await fetch(`${API_BASE}/eval/errors`);
      if (res.ok) {
        const data = await res.json();
        setEvalErrors(data.items || []);
      }
    } catch (e) {
      console.warn("加载错题列表失败:", e);
    }
  }

  async function loadUnknownSummary() {
    try {
      const res = await fetch(`${API_BASE}/unknown/export`);
      if (res.ok) {
        const data = await res.json();
        setUnknownSummary(data.summary || null);
      }
    } catch (e) {
      console.warn("加载负样本统计失败:", e);
    }
  }

  // 录音功能
  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        setAudioBlob(blob);
        stream.getTracks().forEach(t => t.stop());
        setRecording(false);
      };

      mediaRecorder.start();
      setRecording(true);
      setStatus("录音中...");
    } catch (e) {
      setError("无法访问麦克风: " + stringifyError(e));
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
  }

  // 上传训练样本
  async function uploadTrainSample() {
    if (!audioBlob || !selectedPhraseId) return;

    setStatus("上传训练样本...");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("phrase_id", selectedPhraseId);
      formData.append("rebuild_policy", "auto");
      formData.append("file", audioBlob, "recording.wav");

      const res = await fetch(`${API_BASE}/samples/upload`, {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "上传失败");
      }

      const result = await res.json();
      setStatus(`上传成功！样本数: ${result.total_train_samples || "?"}`);
      setAudioBlob(null);
      loadPhrases(); // 刷新短语统计
    } catch (e) {
      setError(stringifyError(e));
    }
  }

  // 上传评估样本
  async function uploadEvalSample() {
    if (!audioBlob || !selectedPhraseId) return;

    setStatus("上传评估样本...");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("truth_phrase_id", selectedPhraseId);
      formData.append("file", audioBlob, "recording.wav");

      const res = await fetch(`${API_BASE}/eval/upload`, {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "上传失败");
      }

      const result = await res.json();
      setStatus(`评估完成: ${result.final_text || "已处理"}`);
      setAudioBlob(null);
      loadEvalSummary(); // 刷新统计
      loadEvalErrors(); // 刷新错题列表
    } catch (e) {
      setError(stringifyError(e));
    }
  }

  // 上传负样本
  async function uploadUnknownSample() {
    if (!audioBlob) return;

    setStatus("上传负样本...");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.wav");

      const res = await fetch(`${API_BASE}/unknown/upload`, {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "上传失败");
      }

      const result = await res.json();
      setStatus(`负样本测试: ${result.summary?.result || "已处理"}`);
      setAudioBlob(null);
      loadUnknownSummary(); // 刷新统计
    } catch (e) {
      setError(stringifyError(e));
    }
  }

  // 演示推理
  async function processDemo() {
    if (!audioBlob) return;

    setStatus("处理中...");
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.wav");

      const res = await fetch(`${API_BASE}/hybrid/process`, {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "处理失败");
      }

      const result = await res.json();
      setDemoResult(result);
      setStatus("处理完成");

      // TTS 播放
      if (ttsEnabled && result.final_text) {
        speak(result.final_text);
      }
      setAudioBlob(null);
    } catch (e) {
      setError(stringifyError(e));
    }
  }

  // 保存纠正样本
  async function saveCorrection(truthPhraseId: string) {
    if (!audioBlob) return;

    try {
      const formData = new FormData();
      formData.append("truth_phrase_id", truthPhraseId);
      formData.append("file", audioBlob, "recording.wav");

      const res = await fetch(`${API_BASE}/corrections/upload`, {
        method: "POST",
        body: formData
      });

      if (res.ok) {
        setStatus("纠正样本已保存");
        setAudioBlob(null);
      }
    } catch (e) {
      setError(stringifyError(e));
    }
  }

  // 确认合并纠正样本
  async function confirmCorrections() {
    try {
      const res = await fetch(`${API_BASE}/corrections/confirm`, {
        method: "POST"
      });
      if (res.ok) {
        setStatus("纠正样本已合并到训练库");
        loadPhrases();
      }
    } catch (e) {
      setError(stringifyError(e));
    }
  }

  // 从错题加入纠错池
  async function addErrorsToCorrectionPool(errorIds: string[]) {
    try {
      const res = await fetch(`${API_BASE}/corrections/from_eval`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ error_ids: errorIds })
      });
      if (res.ok) {
        setStatus("已加入纠错池");
        loadEvalErrors();
      }
    } catch (e) {
      setError(stringifyError(e));
    }
  }

  // 加载错题列表（当进入 eval 模式时）
  useEffect(() => {
    if (mode === "eval") {
      loadEvalErrors();
    }
  }, [mode]);

  // TTS
  function speak(text: string) {
    if (!("speechSynthesis" in window)) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  }

  // 获取当前选中短语文本
  const selectedPhrase = useMemo(() => {
    return phrases.find(p => p.phrase_id === selectedPhraseId);
  }, [phrases, selectedPhraseId]);

  // 选择当前模式的处理函数
  const handleRecordAndProcess = () => {
    switch (mode) {
      case "train":
        uploadTrainSample();
        break;
      case "eval":
        uploadEvalSample();
        break;
      case "unknown":
        uploadUnknownSample();
        break;
      case "demo":
        processDemo();
        break;
    }
  };

  return (
    <main className="app-shell">
      <header className="hero">
        <h1>VoiceBridge</h1>
        <p>语音识别训练与评估系统</p>
      </header>

      {/* Tab 导航 */}
      <nav className="mode-tabs">
        <button className={mode === "train" ? "active" : ""} onClick={() => setMode("train")}>
          1. 训练采集
        </button>
        <button className={mode === "eval" ? "active" : ""} onClick={() => setMode("eval")}>
          2. 正式评估
        </button>
        <button className={mode === "unknown" ? "active" : ""} onClick={() => setMode("unknown")}>
          3. 负样本测试
        </button>
        <button className={mode === "demo" ? "active" : ""} onClick={() => setMode("demo")}>
          4. 演示推理
        </button>
      </nav>

      {/* 状态栏 */}
      <section className="panel">
        <div className="row">
          <strong>状态:</strong> {loading ? "加载中..." : status}
          {error && <span className="error"> {error}</span>}
        </div>
        <div className="row">
          <label>
            <input
              type="checkbox"
              checked={ttsEnabled}
              onChange={(e) => setTtsEnabled(e.target.checked)}
            />
            启用语音反馈(TTS)
          </label>
        </div>
      </section>

      {/* 训练采集面板 */}
      {mode === "train" && (
        <section className="panel">
          <h2>训练采集</h2>
          <p className="note">选择要录制的短句，按住大按钮录音后自动上传到训练库</p>

          <div className="row">
            <label>选择短句:</label>
            <select value={selectedPhraseId} onChange={(e) => setSelectedPhraseId(e.target.value)}>
              {phrases.map(p => (
                <option key={p.phrase_id} value={p.phrase_id}>
                  {p.text}
                </option>
              ))}
            </select>
          </div>

          <button
            className={`record-btn ${recording ? "recording" : ""}`}
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onMouseLeave={recording ? stopRecording : undefined}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
          >
            {recording ? "松开停止录音" : "按住录音"}
          </button>
          <p className="record-hint">按住按钮录音，松开后自动上传</p>

          {audioBlob && (
            <button className="primary" onClick={uploadTrainSample} style={{ marginTop: 12 }}>
              确认上传
            </button>
          )}

          {/* 短语统计 */}
          <h3>采集进度</h3>
          <div className="phrase-grid">
            {phrases.map(p => (
              <div key={p.phrase_id} className="phrase-item">
                <div className="text">{p.text}</div>
                <div className="counts">
                  <span>训练: <span className="num">{p.train}</span></span>
                  <span>评估: <span className="num">{p.eval}</span></span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* 评估面板 */}
      {mode === "eval" && (
        <section className="panel">
          <h2>正式评估</h2>
          <p className="note">选择评分标签，录音后系统会评估识别准确率，并记录错题用于重录</p>

          <div className="row">
            <label>评分标签:</label>
            <select value={selectedPhraseId} onChange={(e) => setSelectedPhraseId(e.target.value)}>
              {phrases.map(p => (
                <option key={p.phrase_id} value={p.phrase_id}>
                  {p.text}
                </option>
              ))}
            </select>
          </div>

          <button
            className={`record-btn ${recording ? "recording" : ""}`}
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onMouseLeave={recording ? stopRecording : undefined}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
          >
            {recording ? "松开停止录音" : "按住录音"}
          </button>
          <p className="record-hint">按住按钮录音，松开后开始评估</p>

          {audioBlob && (
            <button className="primary" onClick={uploadEvalSample} style={{ marginTop: 12 }}>
              确认评估
            </button>
          )}

          {/* 评估统计 */}
          {evalSummary && (
            <>
              <h3>评估统计</h3>
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="value">{evalSummary.total}</div>
                  <div className="label">总数</div>
                </div>
                <div className="stat-card success">
                  <div className="value">{(evalSummary.top1_rate * 100).toFixed(1)}%</div>
                  <div className="label">Top1 准确率</div>
                </div>
                <div className="stat-card success">
                  <div className="value">{(evalSummary.top2_rate * 100).toFixed(1)}%</div>
                  <div className="label">Top2 准确率</div>
                </div>
                <div className="stat-card warning">
                  <div className="value">{(evalSummary.reject_rate * 100).toFixed(1)}%</div>
                  <div className="label">拒识率</div>
                </div>
              </div>
            </>
          )}

          {/* 错题列表 */}
          {evalErrors.length > 0 && (
            <>
              <h3>错题列表 ({evalErrors.length})</h3>
              <table>
                <thead>
                  <tr>
                    <th>真值</th>
                    <th>预测</th>
                    <th>分数</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {evalErrors.slice(0, 10).map(err => (
                    <tr key={err.id}>
                      <td>{err.truth_text}</td>
                      <td className="error">{err.pred_text}</td>
                      <td>{err.score.toFixed(3)}</td>
                      <td>
                        <button onClick={() => addErrorsToCorrectionPool([err.id])}>
                          加入纠错池
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </section>
      )}

      {/* 负样本测试面板 */}
      {mode === "unknown" && (
        <section className="panel">
          <h2>负样本测试</h2>
          <p className="note">录制非目标语句（不要说固定的那几句），测试模型的拒识能力</p>

          <button
            className={`record-btn ${recording ? "recording" : ""}`}
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onMouseLeave={recording ? stopRecording : undefined}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
          >
            {recording ? "松开停止录音" : "按住录音"}
          </button>
          <p className="record-hint">按住按钮录音，松开后上传测试</p>

          {audioBlob && (
            <button className="primary" onClick={uploadUnknownSample} style={{ marginTop: 12 }}>
              确认测试
            </button>
          )}

          {/* 负样本统计 */}
          {unknownSummary && (
            <>
              <h3>负样本统计</h3>
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="value">{unknownSummary.total}</div>
                  <div className="label">总数</div>
                </div>
                <div className="stat-card success">
                  <div className="value">{unknownSummary.correct_rejects}</div>
                  <div className="label">正确拒识</div>
                </div>
                <div className="stat-card danger">
                  <div className="value">{unknownSummary.false_accepts}</div>
                  <div className="label">误接受</div>
                </div>
                <div className="stat-card warning">
                  <div className="value">{(unknownSummary.false_accept_rate * 100).toFixed(1)}%</div>
                  <div className="label">误接受率</div>
                </div>
              </div>
            </>
          )}
        </section>
      )}

      {/* 演示推理面板 */}
      {mode === "demo" && (
        <section className="panel">
          <h2>演示推理</h2>
          <p className="note">实时语音识别演示，识别后会播放语音反馈，识别错误可保存为纠正样本</p>

          <button
            className={`record-btn ${recording ? "recording" : ""}`}
            onMouseDown={startRecording}
            onMouseUp={stopRecording}
            onMouseLeave={recording ? stopRecording : undefined}
            onTouchStart={startRecording}
            onTouchEnd={stopRecording}
          >
            {recording ? "松开停止录音" : "按住录音"}
          </button>
          <p className="record-hint">按住按钮录音，松开后开始识别</p>

          {audioBlob && (
            <button className="primary" onClick={processDemo} style={{ marginTop: 12 }}>
              确认识别
            </button>
          )}

          {/* 识别结果 */}
          {demoResult && (
            <div className="result-box">
              <div className="main-result">{demoResult.final_text || "(无结果)"}</div>
              <div className="sub-result">
                <span>预测: {demoResult.pred_text}</span>
                {demoResult.score > 0 && <span className="score"> 置信度: {(demoResult.score * 100).toFixed(1)}%</span>}
              </div>
              {demoResult.reject_reason && (
                <div className="muted">拒识原因: {demoResult.reject_reason}</div>
              )}
              <div className="muted">来源: {demoResult.source}</div>

              {/* 纠正功能 */}
              <div style={{ marginTop: 16 }}>
                <p className="muted">如果识别错误，请选择正确的结果保存为纠正样本：</p>
                <div className="row" style={{ flexWrap: "wrap" }}>
                  {phrases.slice(0, 4).map(p => (
                    <button
                      key={p.phrase_id}
                      onClick={() => saveCorrection(p.phrase_id)}
                      disabled={!audioBlob}
                    >
                      保存为: {p.text}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* 纠错池操作 */}
          <div style={{ marginTop: 20 }}>
            <button onClick={confirmCorrections}>确认合并纠正样本到训练库</button>
          </div>
        </section>
      )}
    </main>
  );
}

function stringifyError(err: unknown): string {
  if (err instanceof Error) {
    return err.message;
  }
  return String(err);
}