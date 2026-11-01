# JYB 贡献说明：将 02 版 TTS 表演控制合入 openmic

本次在 `csw` 提交 `8b6dadfdaffdb1809fc9892de50da9cede186106` 的基础上，将 `jyb_run_20261026_02` 对应功能整合进已有的 **`src/openmic/`**。不再新增独立 Python 包，也不包含后续音色稳定版。

## 在陈首文工作基础上增加了什么

csw 已完善脚本提取、标签清理、CosyVoice2 请求、WAV Header 修复、清理后空文本检查、有效 WAV 与采样率检查，以及声道、音频时长和请求耗时展示。这些基础能力保留在 `openmic.tts` 和原页面中，不计为 JYB 的新增贡献。

JYB 接手精确停顿、重音/情绪执行和普通朗读/表演版对照。新增执行层复用 `openmic.tts.TTSConfig`、脚本提取和 WAV 工具；所有生产模块直接放入 `src/openmic/`，用 `tts_` 前缀区分职责，避免与原 `workflow.py`、`cli.py` 等文件冲突。

| 新增模块，路径均为 `src/openmic/` 下 | 职责 |
|---|---|
| `tts_plan.py` | 将结构化稿件编译成 `Speech`、`Pause` 事件，校验情绪/语速/标签，处理默认停顿、显式覆盖和可选语气词 |
| `tts_backends.py` | 复用基础 TTS 配置与 WAV 修复，构造中性/情绪请求，严格检查 16 kHz 单声道 16-bit PCM；提供 tone 测试后端 |
| `tts_render.py` | 裁边、短淡入淡出、PCM 静音拼接、片段缓存、逐段回调及指标/时间轴保存 |
| `tts_workflow.py` | 复用原五 Agent 角色与路由，约束 Writer/Coach 的可执行输出协议 |
| `tts_review.py` | 检查结构、字词一致性、标签协议和模型质检信息 |
| `tts_cli.py` | `plan`、`synth`、`generate` 的实现，由统一 `openmic tts` 入口调用 |
| `tts_batch.py` | 五主题批量生成及四条件合成 |
| `tts_app.py` | 稿件编辑、表演计划预览与四条件生成页面 |
| `tts_evaluate.py` | 匿名试听材料、随机播放顺序、空评分表、评分校验及按听众配对汇总 |
| `tts_asr_check.py` | 按音频和参考正文哈希复用转写，计算 ASR 字符差异率，辅助核查正文 |
| `tts_acceptance.py` | WAV/时间轴/静音/同稿条件/实际时长及冷测试元数据验收 |
| `tts_showcase_app.py`、`tts_launch_showcase.py` | 保留 02 版早期结果布局的只读页及启动器，并更新脚本位置 |
| `tts_results_app.py` | 将 02 版实际使用的结果播放器合入包内，通过 `JYB_RUN_ROOT` 读取已有结果，不调用模型 API |

## 这次如何整合目录和入口

先前包中的 `plan.py`、`render.py` 等改为 `openmic/tts_plan.py`、`openmic/tts_render.py` 等；先前包的 `__main__.py` 改为 `openmic/tts_cli.py`。导入、测试 patch 目标、Streamlit 脚本路径和启动器目标一并更新。独立执行目录中的 `tools/viewer.py` 改为 `openmic/tts_results_app.py`，其默认结果根目录按新的文件层级修正。

原 `openmic/cli.py` 增加 `tts` 派发：

```bash
python -m openmic tts plan --script examples/jyb_demo.txt --out artifacts/jyb/plan.json
python -m openmic tts synth --script examples/jyb_demo.txt --backend tone --conditions baseline timing emotion performance --out artifacts/jyb/offline/shopping
```

安装后的控制台入口也可写为 `openmic tts ...`。原有命令仍可运行，例如：

```bash
python -m openmic --backend mock --topic '我的网购经历'
```

派发按首个参数是否为 `tts` 判断，只有进入表演控制功能才导入相关模块。原 `SiliconFlowTTS.synthesize(text, voice=..., speed=...)`、`TTSResult`、原五 Agent 文件和原 `streamlit_app.py` 行为保持。原页面不会自动切换为四条件页；需要表演控制时启动 `tts_app.py`。`pyproject.toml` 不需要增加第二个包或改写原控制台入口。

## 协议与四条件设计

支持 `[INTRO]`、`[SETUP]`、`[PUNCHLINE]`、`[CALLBACK]`、`[PAUSE=...]`、`[EMOTION=...]`、`[SPEED=...]`、`[EMPHASIS]...[/EMPHASIS]` 和 `[FILLER=...]`。未知控制标记会报错，避免当作正文朗读。

16 kHz 下，0.8 秒和 2 秒分别插入 12,800 和 32,000 个零采样点。显式停顿覆盖对应边界默认值，相邻包袱的共享默认边界取较长值。模型原有自然停顿、弱音和呼吸仍可能增加听觉间隔，数字静音不等于音素级强制对齐。

| 条件 | 情绪 | 语速 | 重音 | 停顿 |
|---|---|---|---|---|
| `baseline` | neutral | 1.0 | 去掉 | 普通片段间隔 0.12 秒 |
| `timing` | neutral | 1.0 | 去掉 | 执行计划 |
| `emotion` | 执行标签 | 1.0 | 去掉 | 普通片段间隔 0.12 秒 |
| `performance` | 执行标签 | 执行标签 | 保留 | 执行计划 |

同主题使用相同正文、音色参数、分段和计划；`baseline` 与 `timing` 通过共享缓存复用语音 PCM，隔离静音变量。完整表演同时改变多个因素，不能单独归因于情绪或重音。情绪指令与 `<strong>` 是模型软控制，固定 `voice` 参数不保证各片段听觉音色完全相同。

## 新增的样例、依赖、测试和文档

- `requirements-jyb.txt`、`constraints-jyb.txt`：补充依赖并保留 02 版验证过的关键版本组合，复现环境为 Python 3.11。
- `examples/jyb_demo.txt`、`examples/jyb_topics.json`：短协议示例与固定五主题配置。
- `examples/jyb_showcase/`：早期原创固定稿，不能说成真实五 Agent 输出。
- `examples/jyb_run_02/`：02 版实际采用的五份最终标记稿和哈希清单，便于不重新运行 LLM 的同稿 TTS 实验。
- 原 02 的 10 个 `test_jyb_*.py` 文件迁移到新模块导入；另外将结果播放器的 3 项测试纳入源码测试，并增加统一 CLI/迁移集成回归。历史原 94 项和新整合后的测试总数需分别报告，本次准确结果以运行日志和 CI 为准。
- 本文：说明在 csw 基础上的贡献、整合方式、接口兼容与局限。
- [JYB_USAGE.md](JYB_USAGE.md)：Windows/Linux 的安装、测试、合成、播放器及评测指令。
- [JYB_RUN_02_RESULTS.md](JYB_RUN_02_RESULTS.md)：只记录 02 版历史实测，保留统计边界，不冒充本次迁移后的新测量。

README 增加功能与文档入口；`.gitignore` 排除私有配置、环境、音频和权重；CI 保留 Python 3.9 基础检查，增加 Python 3.11 的整合功能验证。不能将 Python 3.11 的锁定组合直接安装进原 Python 3.9 job。

## 验证与复现

在仓库根目录的 Python 3.11 环境内逐行执行：

```bash
python -m pip install -e . -r requirements-jyb.txt
python -m pip check
python -m unittest discover -s tests -v
python -m compileall -q src tests
python -m openmic tts plan --script examples/jyb_demo.txt --out artifacts/jyb/plan.json
python -m openmic tts synth --script examples/jyb_demo.txt --backend tone --conditions baseline timing emotion performance --out artifacts/jyb/offline/shopping
```

上述步骤无需 Key，不调用真实模型。tone 是测试音，会被真实音频验收器拒绝，不能按真实语音验收流程要求其通过。新环境、Ubuntu CI 和真实 API 的验证结果应各自如实报告，不能由本地离线测试推断。

本次整合后已在 Windows / Python 3.11.7 验证：**102 项测试通过**（原 94 项迁移、3 项结果页测试、5 项统一 CLI 集成测试），依赖检查、编译检查、原 Mock 命令、新计划命令和四条件 tone 合成均通过。测试只读复用原 02 的依赖环境，并通过 `PYTHONPATH` 指向新 `checkout/src`；没有向旧环境安装代码，没有重新调用收费模型，也未声称新建环境或 Ubuntu 已实跑。CLI 集成子进程禁止网络及旧独立包导入，并检查原命令不提前加载新 TTS 依赖。

每个条件输出 `audio.wav`、`metrics.json`、`timeline.json`，主题目录保存 `script.txt`、`plan.json`。结果页读取 `JYB_RUN_ROOT/outputs/real/<主题>/<条件>/` 和 `outputs/short_demo/shopping/<条件>/`。早期 `tts_showcase_app.py` 使用的是 `real_demo_*`、`real_autogen_*`、`real_showcase_*` 布局，两者不要混用。

旧本机 `tools/run.py` 的断点状态、定向修稿和重试历史没有移植为通用执行器。`generate` 会保留未批准稿，退出码不表示质检通过；`synth` 自动检查标签协议，不自动调用 `review_script()`。新生成稿必须先核对 `approved`、最近 Writer/Coach 的正文一致性和标签，再请求收费 TTS。

## 已知局限与贡献边界

用户试听已发现 02 版同一音频内音色漂移，情绪句可能明显换声。此次保留 02 版控制逻辑，只整合代码结构与入口；没有引入后续固定参考音色、连续母带、WSOLA、说话人嵌入或 Whisper 对齐方案。

真人听评尚未完成，没有 MOS 或效果提升结论。ASR 仅辅助核查正文，TTS 冷测试不代表端到端延迟；历史五主题文本生成小于 30 秒为 0/5。没有训练 TTS、没有 learned latent、没有文本/音频双流式，也未在 AutoDL 实例实跑。分段回调只表示完整音频片段就绪。

遵循 [CONTRIBUTING.md](../CONTRIBUTING.md) 和 [PR 模板](../.github/PULL_REQUEST_TEMPLATE.md)：提供测试、接口说明与复现命令，不提交 Key、生成音频、模型权重、环境或执行缓存。分支使用用户指定的 `jyb`，基于 `csw`；PR base 选 `csw`，至少一位组员 Review 后再合并。暂无 Issue 时如实说明分工，不虚构编号。
