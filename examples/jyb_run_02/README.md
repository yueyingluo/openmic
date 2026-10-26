# 02 版正式标记稿

这里的五个 `.txt` 文件逐字节复制自 `jyb_run_20261026_02/outputs/scripts/` 的最终稿，供同稿复现实验。`manifest.json` 记录 SHA-256。它们来自该轮真实五 Agent 流程，正文与最近 Writer/Coach 核对通过；执行阶段曾规范化停顿标记与换行。不是本次整理时重新生成的稿件。

| 文件 | 主题 |
|---|---|
| shopping.txt | 我的网购经历 |
| canteen.txt | 大学食堂选择困难 |
| deadline.txt | 期末作业与拖延症 |
| fitness.txt | 办了健身卡之后 |
| groupwork.txt | 小组合作的日常 |

`../jyb_showcase/` 是前期手工固定测试稿，与本目录的正式五 Agent 稿不同。完整历史轨迹、失败尝试、音频和缓存保留在本机旧实验目录；不随本次代码贡献上传。

从仓库根目录合成这五份稿件（会调用收费 TTS API）：

```bash
python -m openmic.tts_batch synth --scripts examples/jyb_run_02 --backend siliconflow --conditions baseline timing emotion performance --out artifacts/jyb/run_02/outputs/real
```

先按 [运行说明](../../docs/JYB_USAGE.md) 安装并配置本机 `.env`。新合成使用相同输入和条件，服务具有随机性，音频与耗时不会逐字节复现历史结果。历史模型评分不是人工评分，本目录没有 MOS 或真人效果提升结论。
