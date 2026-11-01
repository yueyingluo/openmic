# 多档与声音实验复现

主结果见 [V2_REPORT.md](V2_REPORT.md)，固定计划见 [v2_plan.yaml](v2_plan.yaml)。这是独立研究入口，不会自动替换项目质检器。现有自动声音分数没有通过人工排序验证。

## 环境与资产

CPU审计和统计可用 `pip install -e '.[evaluation]'`。Omni推理另建Python 3.12环境，安装 `evaluation/requirements-omni.txt`。不要同时安装旧的 `evaluation-gpu` extra：它为历史文本实验固定了不同的Torch/Transformers版本。实际版本见 `results/v2/*environment.json`。CUDA 12.8/H200是本次执行环境；其他硬件应单独验证。

以下变量均指自己的本地路径；原始数据不在Git中。先按 `DATASETS_V2.md` 下载固定版本CMHL及CFunSet。Omni、BERT与CEAEval人工包共约31 GB，按27项固定URL及SHA256下载：

```bash
python evaluation/runtime/fetch_v2_assets.py \
  --manifest evaluation/v2_assets.json --output artifacts/v2-assets
```

下载器只获取清单列出的公开资产，不接受访问门槛、不读取私密接口、不执行远程模型代码。保留`.partial`以便续传，全部文件核对SHA256。没有下载166 GB弱标注包。CEAEval仅限其CC-BY-NC-4.0许可范围使用。

## 五级文本

```bash
python evaluation/runtime/graded_text.py \
  --corpus artifacts/e2e_data_audit/cmhl \
  --cfunset evaluation/runtime/data/raw/CFunSet.json \
  --output artifacts/v2-text
```

先复现队友CMHL分组，再排除CFunSet精确重叠组；参考评分保存在`records.private.jsonl`，送给评审的`judge_inputs.jsonl`不含标签。CPU模型在train拟合，只按validation选参数。

仅在获准且空闲的单张GPU上运行，`CUDA_VISIBLE_DEVICES`必须为同一UUID。代码会拒绝与其他计算进程共用该卡。输出目录必须是新的，外层timeout保证加载和推理都受限。

```bash
export CUDA_VISIBLE_DEVICES=GPU-your-approved-id
OMP_NUM_THREADS=4 timeout 7200 python evaluation/runtime/omni_judge.py \
  --model artifacts/v2-assets/Qwen2.5-Omni-7B \
  --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
  --inputs artifacts/v2-text/judge_inputs.jsonl --output artifacts/v2-text-omni \
  --mode text --gpu-uuid "$CUDA_VISIBLE_DEVICES" --batch-size 8 --seconds 7100

OMP_NUM_THREADS=4 timeout 1800 python evaluation/runtime/graded_bert.py \
  --model artifacts/v2-assets/bert --records artifacts/v2-text/records.private.jsonl \
  --output artifacts/v2-bert --gpu-uuid "$CUDA_VISIBLE_DEVICES" --seconds 1700
```

Omni按固定数字token的条件概率求等级期望，记录候选概率质量；不把数值覆盖率称为自由生成格式成功率。BERT是新的五分类训练，输出三个固定种子的等级期望均值，不把旧二分类概率拉伸成1—5。超512 token文本排除并报告，不丢掉结尾笑点后继续打分。

## 实际声音

```bash
python evaluation/runtime/audio_data.py \
  --archive artifacts/v2-assets/CEAEval-Data/manual_annotation.tar.gz \
  --archive-sha256 66dc6001b86cf7fb8f381b9fa7a79fca68596e505de9418d1d3d63f36e4cf8c1 \
  --root artifacts/ceaeval-extracted --output artifacts/v2-audio

OMP_NUM_THREADS=4 timeout 7200 python evaluation/runtime/omni_judge.py \
  --model artifacts/v2-assets/Qwen2.5-Omni-7B \
  --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
  --inputs artifacts/v2-audio/judge_inputs.jsonl --output artifacts/v2-audio-omni \
  --mode audio --gpu-uuid "$CUDA_VISIBLE_DEVICES" --batch-size 4 --seconds 7100
```

读取真实WAV、转单声道并重采样到16 kHz，传给Omni音频处理器。量表为原始0—5表达适切性，与项目1—5幽默量表不同。`text_only`是文字对照；它不是“去掉声音但保持完全同一提示”的严格因果消融。上下文仅为同文件此前台词的末2000字，不是原论文的完整叙事语境。因此这是独立的有限上下文基线，不宣称复现作者官方benchmark成绩。

## 冻结选择与报告

```bash
python evaluation/runtime/graded_analysis.py \
  --references artifacts/v2-audio/references.private.jsonl \
  --predictions artifacts/v2-audio-omni/predictions.private.jsonl \
  --mode audio --lock artifacts/audio-selection.lock.json --select-only

python evaluation/runtime/graded_analysis.py \
  --references artifacts/v2-audio/references.private.jsonl \
  --predictions artifacts/v2-audio-omni/predictions.private.jsonl \
  --mode audio --lock artifacts/audio-selection.lock.json \
  --output artifacts/audio-comparison.json
```

先运行`--select-only`，只读取开发分区选择结果；再按该锁报告测试。文本改为`--mode text`；与CPU/BERT比较时，将参考及各预测过滤到固定`judge_inputs.jsonl`列出的ID，再传入多个`--predictions`文件。不能因看到测试结果重新选择prompt。未覆盖样本不填0；同时报各方法覆盖率和共同集合。按故事/文本重复组做5000次bootstrap，差值使用相同重采样。

## 离线项目链路与人工交接

没有配置生产接口时，可以用本地Omni作为OpenAI-compatible loopback服务运行真实AutoGen工作流；这是另一套明确标注的模型/TTS配置。不会调用付费API。生成前确保`PYTHONPATH=src`指向要验证的项目版本。

```bash
PYTHONPATH=src timeout 7200 python evaluation/runtime/offline_project.py \
  --stage generate --model artifacts/v2-assets/Qwen2.5-Omni-7B \
  --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
  --project-revision YOUR_CHECKED_OUT_COMMIT --output artifacts/project-pilot \
  --gpu-uuid "$CUDA_VISIBLE_DEVICES" --topics 2

timeout 7200 python evaluation/runtime/offline_project.py \
  --stage speak --model artifacts/v2-assets/Qwen2.5-Omni-7B \
  --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
  --output artifacts/project-pilot --gpu-uuid "$CUDA_VISIBLE_DEVICES"

python evaluation/runtime/omni_judge.py \
  --mode project --model artifacts/v2-assets/Qwen2.5-Omni-7B \
  --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 \
  --inputs artifacts/project-pilot/audio_manifest.private.jsonl \
  --output artifacts/project-pilot-judged --gpu-uuid "$CUDA_VISIBLE_DEVICES" --batch-size 1

python evaluation/runtime/blind_audio_pack.py \
  --manifest artifacts/project-pilot/audio_manifest.private.jsonl \
  --output artifacts/listening-pack
```

同一Omni家族生成与评审有自偏好混杂，只用于技术链路验证。语音使用固定公开音色、强制文本token、分段合成及250 ms拼接静音；不是CosyVoice，不声称实现了队友的重音/情绪/精确停顿控制，也不把技术试运行当作正式对照实验。强制文本一致不等于ASR或真人确认所有词均清楚读出。

只交付`listening-pack/`给评分者，旁边的映射和样本JSONL保留在研究侧。评分页面没有模型分数、来源或默认评分，可导出JSON。收到真实评分后才执行：

```bash
python evaluation/runtime/import_audio_ratings.py \
  --export PATH_TO_ACTUAL_RATER_EXPORT.json \
  --mapping artifacts/listening-pack.mapping.private.json \
  --output artifacts/audio-human-ratings.jsonl

python evaluation/runtime/analysis.py \
  --samples artifacts/listening-pack.samples.private.jsonl \
  --ratings artifacts/audio-human-ratings.jsonl \
  --predictions artifacts/project-pilot-judged/predictions.private.jsonl \
  --split pilot --min-raters 1 --output artifacts/audio-human-alignment.json
```

一位评分者只能说明个人偏好；同主题两个条件的极小技术样本不能给出稳定研究结论。正式结论仍需新主题、多位独立评分者、冻结方法后的完整作品盲听。
