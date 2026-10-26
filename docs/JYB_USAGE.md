# 合入 openmic 后的 TTS 运行说明

以下在仓库根目录执行，每条指令一行。使用 **Python 3.11**，无需 GPU 或训练。新增功能均在 `src/openmic/`，统一单稿入口为 `python -m openmic tts`；原 `python -m openmic --topic ...` 和原页面仍保留。

## Windows PowerShell：安装与离线检查

先进入本次准备好的 `checkout` 目录，再执行：

```powershell
$env:PYTHONUTF8='1'
python --version
python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Please use Python 3.11'"
python -m venv .venv-jyb
.\.venv-jyb\Scripts\python.exe -m pip install --upgrade pip
.\.venv-jyb\Scripts\python.exe -m pip install -e . -r requirements-jyb.txt
.\.venv-jyb\Scripts\python.exe -m pip check
.\.venv-jyb\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-jyb\Scripts\python.exe -m compileall -q src tests
.\.venv-jyb\Scripts\python.exe -m openmic tts --help
.\.venv-jyb\Scripts\python.exe -m openmic tts plan --script examples/jyb_demo.txt --out artifacts/jyb/plan.json
.\.venv-jyb\Scripts\python.exe -m openmic tts synth --script examples/jyb_demo.txt --backend tone --conditions baseline timing emotion performance --out artifacts/jyb/offline/shopping
```

这些检查无需密钥，不调用收费接口。tone 只生成测试音，不读中文；不能把 tone 交给真实音频验收器并期待合格。任一步报错先停在该步。

需要打开编辑和生成页面时执行，浏览器访问 `http://127.0.0.1:8512`：

```powershell
.\.venv-jyb\Scripts\python.exe -m streamlit run src/openmic/tts_app.py --server.address 127.0.0.1 --server.port 8512
```

页面占用终端，Ctrl+C 停止。后续命令在同一仓库目录的新终端执行，或先停止页面。

## Windows：真实语音与五份固定输入

本节会调用收费 TTS。先创建本机配置；不覆盖已有 `.env`：

```powershell
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
notepad .env
```

在 `.env` 填写下列字段，其中中文密钥占位符必须替换成实际值；不要将 Key 写进命令或提交：

```dotenv
TTS_API_KEY=你的硅基流动密钥
TTS_BASE_URL=https://api.siliconflow.cn/v1
TTS_MODEL=FunAudioLLM/CosyVoice2-0.5B
TTS_VOICE=FunAudioLLM/CosyVoice2-0.5B:alex
TTS_FORMAT=wav
TTS_SAMPLE_RATE=16000
```

先短稿试听，再生成五主题。`examples/jyb_run_02` 是历史 02 版最终输入，不需要再次请求 LLM；API 随机性使新音频和耗时不同于历史结果。本版仍有分段/情绪造成音色漂移的风险。

```powershell
.\.venv-jyb\Scripts\python.exe -m openmic tts synth --script examples/jyb_demo.txt --backend siliconflow --conditions baseline timing emotion performance --out artifacts/jyb/run_02/outputs/short_demo/shopping
.\.venv-jyb\Scripts\python.exe -m openmic.tts_batch synth --scripts examples/jyb_run_02 --backend siliconflow --conditions baseline timing emotion performance --out artifacts/jyb/run_02/outputs/real
.\.venv-jyb\Scripts\python.exe -m openmic.tts_acceptance --runs artifacts/jyb/run_02/outputs/real --out artifacts/jyb/run_02/outputs/acceptance --require-duration
$env:JYB_RUN_ROOT=(Join-Path (Get-Location).Path 'artifacts/jyb/run_02')
.\.venv-jyb\Scripts\python.exe -m streamlit run src/openmic/tts_results_app.py --server.address 127.0.0.1 --server.port 8513
```

打开 `http://127.0.0.1:8513`。结果页只读已有文件，不调用 API。新生成目录没有历史执行器的 `state.json`、trace 或冷测时，对应区域会显示未提供，不能编造来源和基准结果。历史摘要见 [JYB_RUN_02_RESULTS.md](JYB_RUN_02_RESULTS.md)。

若在本次整理所在的原电脑只想试听旧 02 结果，在仓库根目录运行以下两行；不会复制或上传音频：

```powershell
$env:JYB_RUN_ROOT=(Resolve-Path -LiteralPath '../../jyb_run_20261026_02').Path
.\.venv-jyb\Scripts\python.exe -m streamlit run src/openmic/tts_results_app.py --server.address 127.0.0.1 --server.port 8513
```

上述相对路径仅适用于本次本机目录结构。其他电脑将 `JYB_RUN_ROOT` 指向自己的结果根目录。

## Linux / AutoDL

先提供 Python 3.11，例如在已初始化 conda 的 AutoDL 终端中执行：

```bash
conda create -n openmic-jyb python=3.11 -y
conda activate openmic-jyb
```

进入仓库根目录，逐行运行：

```bash
export PYTHONUTF8=1
python3 --version
python3 -c "import sys; assert sys.version_info[:2] == (3, 11), 'Please use Python 3.11'"
python3 -m venv .venv-jyb
.venv-jyb/bin/python -m pip install --upgrade pip
.venv-jyb/bin/python -m pip install -e . -r requirements-jyb.txt
.venv-jyb/bin/python -m pip check
.venv-jyb/bin/python -m unittest discover -s tests -v
.venv-jyb/bin/python -m compileall -q src tests
.venv-jyb/bin/python -m openmic tts plan --script examples/jyb_demo.txt --out artifacts/jyb/plan.json
.venv-jyb/bin/python -m openmic tts synth --script examples/jyb_demo.txt --backend tone --conditions baseline timing emotion performance --out artifacts/jyb/offline/shopping
```

需要真实语音时继续执行，编辑 `.env` 时字段同上：

```bash
cp -n .env.example .env
nano .env
.venv-jyb/bin/python -m openmic tts synth --script examples/jyb_demo.txt --backend siliconflow --conditions baseline timing emotion performance --out artifacts/jyb/run_02/outputs/short_demo/shopping
.venv-jyb/bin/python -m openmic.tts_batch synth --scripts examples/jyb_run_02 --backend siliconflow --conditions baseline timing emotion performance --out artifacts/jyb/run_02/outputs/real
.venv-jyb/bin/python -m openmic.tts_acceptance --runs artifacts/jyb/run_02/outputs/real --out artifacts/jyb/run_02/outputs/acceptance --require-duration
export JYB_RUN_ROOT="$PWD/artifacts/jyb/run_02"
.venv-jyb/bin/python -m streamlit run src/openmic/tts_results_app.py --server.address 127.0.0.1 --server.port 8513 --server.headless true
```

本机连接 AutoDL 页面时，用实例真实 SSH 地址和端口替换下行中文，再打开 `http://127.0.0.1:18513`：

```powershell
ssh -N -L 18513:127.0.0.1:8513 -p 你的SSH端口 root@你的SSH主机
```

Linux/AutoDL 指令是复现路径，本次没有在真实实例运行；Ubuntu CI 以实际上传后记录为准。

## 可选：新稿、冷测试、ASR 与真人盲评

以下命令同样在仓库根目录运行，需将 `python` 换为你实际环境的解释器：Windows 为 `.\.venv-jyb\Scripts\python.exe`，Linux 为 `.venv-jyb/bin/python`。

重新生成五 Agent 稿还需在 `.env` 填写账号可调用的 `OPENMIC_API_KEY`、`OPENMIC_BASE_URL`、`OPENMIC_MODEL`。生成新稿会调用收费 LLM，且不再是历史 02 同稿输入：

```bash
python -m openmic.tts_batch generate --backend autogen --topics examples/jyb_topics.json --out artifacts/jyb/new_scripts
```

逐份检查 `.txt` 和 `.trace.json`，确认 `approved`、最近 Writer/Coach 全文字词一致和情绪标签有效，再合成。`generate` 会保存未批准稿并提示，退出码不保证批准；`synth` 只自动校验标签，不自动执行 `openmic.tts_review.review_script()`。旧本机执行器的断点状态和定向修稿没有移植为通用工具。

下面冷测只测网购一份，`--no-cache` 必须保留；TTS 和 ASR 均会收费。ASR 默认 SenseVoiceSmall，字符差异率不能当作 MOS：

```bash
python -m openmic tts synth --script examples/jyb_run_02/shopping.txt --backend siliconflow --conditions performance --no-cache --out artifacts/jyb/run_02/outputs/benchmark/shopping
python -m openmic.tts_asr_check --runs artifacts/jyb/run_02/outputs/real --conditions baseline performance --out artifacts/jyb/run_02/outputs/asr/report.json
python -m openmic.tts_evaluate pack --runs artifacts/jyb/run_02/outputs/real --out artifacts/jyb/run_02/outputs/blind --listeners 8 --seed 42
```

`pack` 输出目录必须为空或不存在，重打包用新目录名。只向听众提供 `participant_package`，`investigator` 的条件答案留给研究者。收到真实评分后放入 `artifacts/jyb/run_02/ratings_filled`，再运行：

```bash
python -m openmic.tts_evaluate summarize --ratings artifacts/jyb/run_02/ratings_filled --key artifacts/jyb/run_02/outputs/blind/investigator/key.csv --out artifacts/jyb/run_02/outputs/study_summary
```

空模板不是实验结果。当前没有真人 MOS 或感知质量提升结论；分段回调不等于文本/音频双流式，未训练 latent。
