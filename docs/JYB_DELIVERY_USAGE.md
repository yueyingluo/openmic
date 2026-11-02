# JYB 累计更新：安装、验证与使用

在仓库根目录运行。使用 Python 3.11，CPU 即可；不需要 CUDA、模型训练或 AutoDL。所有命令每行一条。任一步报错先停，不继续后面的合成或提交。

这是新舞台/完整性入口的说明。旧四条件入口继续保留，详见 [JYB_USAGE.md](JYB_USAGE.md)。`openmic tts synth` 仍是旧四条件，不能把它的结果标成完整性修复版。

## 1. Windows PowerShell：安装

```powershell
$env:PYTHONUTF8='1'
python --version
python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Use Python 3.11'"
python -m venv .venv-jyb
.\.venv-jyb\Scripts\python.exe -m pip install --upgrade pip
.\.venv-jyb\Scripts\python.exe -m pip install -e '.[evaluation]' -r requirements-jyb-delivery.txt
.\.venv-jyb\Scripts\python.exe -m pip check
```

不激活虚拟环境，直接调用其 Python，避免 PowerShell 执行策略拦截 activate。`evaluation` 此处是项目的 CPU 依赖组，不是 `evaluation-gpu` / `evaluation-neural`。

## 2. Windows：不调用 API 的验证

```powershell
.\.venv-jyb\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-jyb\Scripts\python.exe -m compileall -q src tests
.\.venv-jyb\Scripts\python.exe -m openmic --backend mock --topic '我的网购经历'
.\.venv-jyb\Scripts\python.exe -m openmic tts plan --script examples/jyb_demo.txt --out artifacts/jyb/check/legacy_plan.json
.\.venv-jyb\Scripts\python.exe -m openmic tts synth --script examples/jyb_demo.txt --backend tone --conditions baseline timing emotion performance --out artifacts/jyb/check/legacy
.\.venv-jyb\Scripts\python.exe -m openmic tts stage plan --script examples/jyb_demo.txt --manual-script --planner rules --out artifacts/jyb/check/stage_plan.json
.\.venv-jyb\Scripts\python.exe -m openmic tts stage synth --script examples/jyb_demo.txt --manual-script --plan artifacts/jyb/check/stage_plan.json --backend tone --profiles stage_prompt stage_course stage_club --out artifacts/jyb/check/stage
.\.venv-jyb\Scripts\python.exe -m openmic tts integrity plan --script examples/jyb_demo.txt --manual-script --stage-plan artifacts/jyb/check/stage_plan.json --out artifacts/jyb/check/integrity_plan.json
.\.venv-jyb\Scripts\python.exe -m openmic tts integrity --help
```

`tone` 只检验执行链，不朗读中文，不能把它当真实语音交给 ASR 验收。完整性接口的真实 `synth` 会同时使用 TTS 和 ASR；完整性离线行为由内存 WAV / 模拟识别的单元测试覆盖，不通过关闭检查来假造成功。

评估组的独立 CPU 测试可另行核对；不修改该组文件：

```powershell
.\.venv-jyb\Scripts\python.exe -m pytest -c evaluation/pytest.ini -q evaluation/tests
```

## 3. Linux：同一套 CPU 环境与验证

Linux 需要已有 Python 3.11 和 venv 支持。若系统 `python3` 不是 3.11，下面显式使用 `python3.11`；无需安装 GPU 依赖。

```bash
python3.11 --version
python3.11 -m venv .venv-jyb
.venv-jyb/bin/python -m pip install --upgrade pip
.venv-jyb/bin/python -m pip install -e '.[evaluation]' -r requirements-jyb-delivery.txt
.venv-jyb/bin/python -m pip check
.venv-jyb/bin/python -m unittest discover -s tests -v
.venv-jyb/bin/python -m compileall -q src tests
.venv-jyb/bin/python -m openmic --backend mock --topic '我的网购经历'
.venv-jyb/bin/python -m openmic tts plan --script examples/jyb_demo.txt --out artifacts/jyb/check/legacy_plan.json
.venv-jyb/bin/python -m openmic tts synth --script examples/jyb_demo.txt --backend tone --conditions baseline timing emotion performance --out artifacts/jyb/check/legacy
.venv-jyb/bin/python -m openmic tts stage plan --script examples/jyb_demo.txt --manual-script --planner rules --out artifacts/jyb/check/stage_plan.json
.venv-jyb/bin/python -m openmic tts stage synth --script examples/jyb_demo.txt --manual-script --plan artifacts/jyb/check/stage_plan.json --backend tone --profiles stage_prompt stage_course stage_club --out artifacts/jyb/check/stage
.venv-jyb/bin/python -m openmic tts integrity plan --script examples/jyb_demo.txt --manual-script --stage-plan artifacts/jyb/check/stage_plan.json --out artifacts/jyb/check/integrity_plan.json
.venv-jyb/bin/python -m openmic tts integrity --help
.venv-jyb/bin/python -m pytest -c evaluation/pytest.ini -q evaluation/tests
```

本次本机验证是 Windows，不能据此声称 Linux 已实际运行。Ubuntu CI 仍由仓库负责。

## 4. 先看已有音频与 slides，不必重新合成

完整交付包的 `local_demo/index.html` 可双击，包含三主题 × 三版本的原音频/整稿匹配音频，以及食堂相同片段。当前独立一页预览是 `local_demo/jyb_slides.html`，团队版 JYB 位于第8页。

仓库试听页面为 `docs/jyb_audio_compare.html`，共享 slides 仍为 `presentation/midterm.html`。`presentation/` 只提交 `midterm.html` 与 `README.md`，当前 slide 不含原第二页的试听链接。生成音频按协作规范不上传，所以仅 clone 仓库时播放器有文本和溯源，没有音频；这不是请求 TTS 的按钮。使用本地交付包的 `tools/copy_demo_media.py --checkout ...` 把已有录音复制到 `docs/media/jyb_compare/`，再单独打开试听页。此目录被忽略，不要强制加入 Git。

若当前就是交付包的 `repository/` 或按上传说明新建的 `upload_checkout/`，其父目录有 `tools/`，可直接执行：

```powershell
.\.venv-jyb\Scripts\python.exe ..\tools\copy_demo_media.py --checkout .
Start-Process (Join-Path (Get-Location).Path 'presentation\midterm.html')
Start-Process (Join-Path (Get-Location).Path 'docs\jyb_audio_compare.html')
```

Linux 对应：

```bash
.venv-jyb/bin/python ../tools/copy_demo_media.py --checkout .
```

普通 GitHub clone 不含交付包的 `../tools/`；可按上一段手工复制媒体目录，或用播放器的文件夹选择按钮加载 `local_demo/audio/`。

本次匹配以三个版本为一组重新计算，不借用上一轮四版本的增益。每整稿固定一个增益，保留原始 WAV；短选段在同一语音片段边界裁出，不是词级对齐。试听不能充当真人盲评或 MOS 数据。

## 5. 可选：把组员的新稿送入新执行层（会收费，不是提交前必跑项）

先把生成端交出的 `script.txt` 和对应 `.trace.json` 放到本地忽略目录。真实稿件不使用 `--manual-script` 绕过来源检查。以下以 `artifacts/jyb/new/script.txt` / `script.trace.json` 为例，文件必须实际存在；本次整理没有代跑这些收费命令。

在本机 `.env` 按原 `.env.example` 填写 Key；TTS 用 `TTS_API_KEY` 等现有配置，ASR 使用同一 TTS 服务配置。不要把 Key 写入命令或文档。未通过 Writer/Coach/QC/标签校验的稿不进入合成。

```powershell
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
notepad .env
.\.venv-jyb\Scripts\python.exe -m openmic tts stage plan --script artifacts/jyb/new/script.txt --trace artifacts/jyb/new/script.trace.json --planner rules --out artifacts/jyb/new/stage_plan.json
.\.venv-jyb\Scripts\python.exe -m openmic tts integrity plan --script artifacts/jyb/new/script.txt --trace artifacts/jyb/new/script.trace.json --stage-plan artifacts/jyb/new/stage_plan.json --out artifacts/jyb/new/integrity_plan.json
.\.venv-jyb\Scripts\python.exe -m openmic tts integrity synth --script artifacts/jyb/new/script.txt --trace artifacts/jyb/new/script.trace.json --stage-plan artifacts/jyb/new/stage_plan.json --assets-root artifacts/jyb/new --out artifacts/jyb/new/integrity
```

Linux 对应真实入口：

```bash
.venv-jyb/bin/python -m openmic tts stage plan --script artifacts/jyb/new/script.txt --trace artifacts/jyb/new/script.trace.json --planner rules --out artifacts/jyb/new/stage_plan.json
.venv-jyb/bin/python -m openmic tts integrity plan --script artifacts/jyb/new/script.txt --trace artifacts/jyb/new/script.trace.json --stage-plan artifacts/jyb/new/stage_plan.json --out artifacts/jyb/new/integrity_plan.json
.venv-jyb/bin/python -m openmic tts integrity synth --script artifacts/jyb/new/script.txt --trace artifacts/jyb/new/script.trace.json --stage-plan artifacts/jyb/new/stage_plan.json --assets-root artifacts/jyb/new --out artifacts/jyb/new/integrity
```

正常输出是 `integrity/accepted.json`、各句候选/进度及完整通过后的 `safe_course/`、`safe_club/`。缺一句就不出新的整稿。遇到账户、授权或服务问题会保留中断状态，恢复后读持久账本，不能删掉失败候选来重置次数。已有输出属于其他计划或参数时会拒绝覆盖，应改用新目录。

规则 planner 完全离线。`--planner llm` 则额外调用模型，只允许给原文加注释；失败或超时不自动伪装成规则成功。默认超时 240 秒，允许范围 30–600 秒。

## 6. 审核与特殊候选

默认不使用历史两条人审或两条正字法豁免。只有给出显式审核文件及对应 SHA、并核验其绑定关系，才可接受该份具体录音；机器报告中的失败仍原样保留。无审核依据时，即使 accepted 条目自己声称 `human_review.passed` 也不能通过。

`--authority PATH` 与 `--authority-sha256 HASH` 必须成对提供；其 SHA 由 `python -c "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('本机审核文件.json').read_bytes()).hexdigest())"` 计算，不能在不审核内容的情况下盲信文件。`--verbatim-rescue` 显式允许第 5 次逐字朗读候选；默认只使用 4 次常规新请求。第 06/07 类标点/重音输入试验另须审核文件逐句列出的授权范围，单独布尔值不能授权。`--voice` 可显式设置请求音色；省略则按 `TTS_VOICE` 等现有配置读取。特殊次数和参数均进入账本，恢复执行不会重置。

相关文件结构与限制以本次随附源码 `tts_integrity_listening.py` 和 `python -m openmic tts integrity --help` 为准；实际审核必须针对该 WAV 进行。历史 06 末尾标点和 07 重音试验只是一次性记录，不是对新文本的默认处理。

网购“我说不对啊，他明明很轻松啊。”的两次核听仍缺两个“啊”，没有通过。不要为了做出四份成品而删除原词、塞静音或复用别人的完整句。确需继续研究时，先区分 ASR 拼写问题与真实漏读，再设计新方案。

审核文件字段、两种试验变换与 Python 接口见 [JYB_INTEGRITY_AUTHORITY.md](JYB_INTEGRITY_AUTHORITY.md)。此文件不含真实录音或可自动套用的历史人审凭据。
