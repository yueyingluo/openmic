# 中文幽默与端到端声音评估

这一目录评估“自动分数能否反映人的等级与排序”。最新结果与可调用接口见 [V3 报告](V3_REPORT.md) 和 [V3 复现](V3_REPRODUCE.md)：文本冻结确认 187/407 = 45.95%，50% 目标未达到；表达适切性和可懂度分别用对应公开人工标签验证。完整作品的真人幽默盲听仍待完成，公开短句标签不能代替它。历史 [V2 报告](V2_REPORT.md) 保留直接 Omni 评分的负结果。

它是独立研究工具，不改变生成流程、TTS或现有 `QualityDecision.score`。现有流程中的多维0—10质检分，不能直接替换成这里的1—5人工幽默评分或二分类概率。

已完成的首轮实验见 [REPORT.md](REPORT.md)，原文本协议见 [PROTOCOL.md](PROTOCOL.md)，原文本试标说明见 [ANNOTATION.md](ANNOTATION.md)。截至2026-10-27，本评估工具尚未导入新的项目真实盲评，不存在已经验证的项目人类相关系数；公开语料等级及队友用于开发的旧稿评分不等于独立端到端盲测。

## 不用服务器的最小复现

在仓库根目录使用Python 3.9以上版本：

```bash
python -m pip install -e '.[evaluation]'
python -m pytest -c evaluation/pytest.ini -q evaluation/tests
python evaluation/runtime/judge_io.py
python evaluation/runtime/analysis.py \
  --samples evaluation/examples/samples.synthetic.jsonl \
  --ratings evaluation/examples/ratings.synthetic.jsonl \
  --predictions evaluation/examples/predictions.synthetic.jsonl \
  --output artifacts/evaluation-synthetic.json
```

`examples/*.synthetic.jsonl`只用于检验程序，评分者是虚构标记，相关系数不能写进实验结果。上述命令不会下载数据、调用模型或读取个人评分表。`judge_io.py`将12篇自写试标稿转换为待评审请求，输出在被Git忽略的 `runtime/data/`；试标稿的来源与局限写在 `pilot_manifest.json`，它们不是项目生成结果，也不能转为正式测试集。

统计脚本仍可直接执行。V3 另提供安装包中的 `openmic-evaluate` 命令与 `openmic.evaluation.OfflineEvaluator`，需要自己的本地模型 bundle，调用方法见 V3 复现文档。仓库仅提交汇总结果，逐样本模型分数保留在本地，避免影响正在进行的盲评。

## 输入输出交接

每行一个JSON对象；样本和预测按 `sample_id` 对齐。文本组提交自包含的中文段子，保留主题和故事族ID；正式测试必须按主题/故事族划分，不能把同一故事的改写分到开发与测试两边。

| 文件 | 字段 |
| --- | --- |
| 样本JSONL | `sample_id`, `text`, `topic_id`, `story_family_id`, `split`, `formal_test_eligible` |
| 人工JSONL | `sample_id`, `rater_id`, `annotation_round`, `humor_score`（1—5整数或null）, `understood`（yes/partial/no，可选）, `comment`（可选） |
| 预测JSONL | `sample_id`, `model_id`, `score`（有限数值或null）；推理脚本另外保存 `text_sha256` |

缺失或无效预测保留null，不能填0；看懂但不好笑可以给1分，因不理解而无法判断则留空。对同一批样本，先冻结原文、ID、分组和哈希再标注及推理；统计程序按ID连接，调用者必须确认输入来自同一批冻结文本。`sample_id+rater_id+annotation_round`和`sample_id+model_id`不得重复，不能混合多个标注轮次。

真实标注后的分析命令：

```bash
python evaluation/runtime/analysis.py \
  --samples artifacts/samples.jsonl --ratings artifacts/ratings.jsonl \
  --predictions artifacts/predictions.jsonl \
  --split pilot --min-raters 1 --output artifacts/human-alignment.json
```

正式多人测试改为 `--split test --min-raters 3`，样本必须标记 `formal_test_eligible: true`。个人研究用 `personal_test`，不能把单人的偏好推广到一般读者。评分者接触不到来源、生成条件和模型分数；实际姓名、评分记录不提交Git。

输出同时包含每个方法的覆盖率、各自有效样本上的相关性、所有输入方法共同有效集合上的相关性、主题簇bootstrap及模型差值区间、同主题排序准确率、评分者间一致性。包含低覆盖率方法会缩小共同集合，比较指定方法时另存筛选后的预测文件，并明确报告比较集合，不能暗中删掉失败样本。

## 重做源域CPU实验

这是显式下载步骤，约219 MB。先阅读来源与许可记录 `sources.json`；本仓库不再分发CFunSet原文或模型权重。源站未给出明确许可字段的材料，不据此推定可再分发。

```bash
python evaluation/runtime/fetch_inputs.py
python evaluation/runtime/prepare_data.py
python evaluation/runtime/baseline.py
```

`fetch_inputs.py`只下载固定revision的公开数据和BERT词表配置，验证数据SHA256；失败的 `.partial` 不自动覆盖，应检查后再处理。`prepare_data.py`按任务定义提取标签，先去重再按重复连通组分层划分。审计报告和预测写入忽略目录，不覆盖 `results/` 的历史结果。依赖升级可能影响数值或划分；应对照 `results/data_audit.json` 的manifest哈希，变化时作为新实验记录。

## 在自己获准的空闲GPU上运行

GPU依赖单独安装；测试和CPU步骤不要求PyTorch或模型权重。历史GPU版本记录在 `results/environment_versions.json`。新环境的数值复现仍需记录Python、依赖、CUDA和GPU型号。

```bash
python -m pip install -e '.[evaluation,evaluation-gpu]'
cp evaluation/experiment.yaml evaluation/experiment.local.yaml
```

在本地配置中填写 `resources.gpu_uuid`，并在获得机器使用授权后设置 `authorization.server_approved: true`。默认模板不授权任何服务器。用对应UUID设置 `CUDA_VISIBLE_DEVICES`，运行前确认空闲；脚本还会检查是否已有计算进程。不要提交本地配置。模型首次加载会下载固定revision权重，请预留磁盘及网络时间。

```bash
# 在Linux GPU机器上，外层timeout覆盖加载和推理；UUID使用自己的空闲设备。
export CUDA_VISIBLE_DEVICES=GPU-your-approved-id
python evaluation/runtime/judge_io.py
timeout --signal=TERM --kill-after=10s 1800 python evaluation/runtime/gpu_experiment.py \
  --stage bert --seconds 1800 --config evaluation/experiment.local.yaml
timeout --signal=TERM --kill-after=10s 900 python evaluation/runtime/gpu_experiment.py \
  --stage qwen --seconds 900 --config evaluation/experiment.local.yaml
timeout --signal=TERM --kill-after=10s 900 python evaluation/runtime/gpu_experiment.py \
  --stage cfun --seconds 900 --config evaluation/experiment.local.yaml
```

BERT需先完成上面的源域数据准备。三个阶段顺序运行，不用多卡。已有阶段结果时拒绝重跑覆盖，应在独立实验副本中运行。下载时间也在外层上限内，首次加载可能因此超时；可以先在无GPU占用时准备本地Hugging Face缓存，不要因为超时改动revision。

二分类标签概率接受同一格式的任意样本，不要求真实评分：

```bash
timeout --signal=TERM --kill-after=10s 900 python evaluation/runtime/label_likelihood.py \
  --model qwen --samples evaluation/runtime/data/pilot_samples.jsonl \
  --output artifacts/qwen-labels-v1 --config evaluation/experiment.local.yaml
# 用 --model cfun 和新的输出目录运行第二个模型。
```

候选完整标签加结束标记的对数概率求和，不做长度归一化；分差0为固定分类阈值，正好相等判幽默。结果不是1—5分，不是经过校准的人类发笑概率；强制二选一不支持弃权。`label_likelihood.py`是从已执行实验中抽取的便携入口，本次迁移验证了CPU逻辑，没有重新下载/重跑GPU模型；历史结果不是这次提交重新生成的。

未实现/未验证：正式项目生成文本测试、真实人工排序相关性、跨主题×评分者双重bootstrap、生成条件及生成器家族混杂分析、把此评分用于生成闭环后的质量提升。纯格式修复只允许保持原评审语义；本轮发现的9条无效响应不适用，没有伪造修复后分数。
