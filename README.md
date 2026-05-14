# VoiceBridge（开源版）

> 中文优先文档（附英文摘要）

VoiceBridge 是一个面向构音障碍/喉麦输入场景的短语级语音桥接系统：
采集短音频 -> 匹配目标短语 -> 输出规范文本，并可选进行 TTS 播放。

## 仓库范围（重要）

本仓库只发布**源代码**，不会上传以下内容：

- 个人或本地训练数据
- 录音音频样本
- 模型产物与索引缓存
- 运行日志与本地状态文件

## 技术栈

- 后端：Python（`web_ui_app.py`）
- 前端：TypeScript（`webui/src/main.ts`）+ 静态页面（`webui/static/index.html` / `webui/static/main.js`）
- 工具脚本：PowerShell + Python（根目录与 `tools/`）

## 快速开始（Windows PowerShell）

1. 初始化环境

```powershell
./setup_env.ps1
```

2. 可选：设置云 ASR Key（仅回退路径使用）

```powershell
$env:SILICONFLOW_API_KEY="your_key_here"
```

3. 启动 Web UI

```powershell
./launch_web_ui.ps1
```

4. 如果前端有改动，重新构建

```powershell
bun install
bun run build:frontend
```

## 隐私与数据规范

- 严禁提交用户音频、训练数据、日志与本地状态。
- 密钥只允许放在环境变量中。
- `.gitignore` 已按开源发布场景配置默认拦截规则。

## English Summary

VoiceBridge is a phrase-level speech bridge for dysarthric / throat-mic input.
This public repository is **source-code only** and intentionally excludes personal data, recordings, runtime logs, and local artifacts.

Quick start:

```powershell
./setup_env.ps1
$env:SILICONFLOW_API_KEY="your_key_here"  # optional
./launch_web_ui.ps1
bun install
bun run build:frontend
```

## License

MIT (see `LICENSE`).
## Browser-only TF.js MVP (React + Bun)

A new browser-only MVP app now exists in `webui-react/`.
It implements transfer learning using `@tensorflow-models/speech-commands` with three pages:

1. Record
2. Train
3. Use (real-time recognition + rejection + browser TTS)

### Run the MVP

```powershell
bun run dev:web-mvp
```

Then open: `http://127.0.0.1:5173`

### Build the MVP

```powershell
bun run build:web-mvp
```

### Notes

- Model is persisted in IndexedDB via `transferRecognizer.save()` and reloaded via `transferRecognizer.load()`.
- Phrase template and UX settings are stored in `localStorage`.
- Existing Python backend flow remains unchanged and can run in parallel.
