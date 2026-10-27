# V3 复现与调用

主结果与未达标项见 [V3_REPORT.md](V3_REPORT.md)。不需要生产 API 配置、密钥或付费服务。原始语料、私人人工评分、逐条预测及权重留在自己的 `artifacts/`；不要提交公共仓库。

## 环境

Python 3.10—3.12。新建独立环境，不能同时安装旧 `evaluation-gpu` extra，它固定了另一组 Torch/Transformers。

```bash
python3.11 -m venv .venv-neural
LC_ALL=en_US.UTF-8 .venv-neural/bin/python -m pip install -e '.[evaluation,evaluation-neural]'
source .venv-neural/bin/activate
```

中文路径下旧版 setuptools 的 editable 安装可能读取 ASCII locale；上面的 UTF-8 locale 在本地安装验证过。`tokenizers` 范围兼容历史 CPU 工具与新 Transformers；神经评估另固定 Torch 2.9.1、Torchvision 0.24.1、Transformers 4.57.1、NumPy 1.26.4、SciPy 1.15.3、scikit-learn 1.7.2、PEFT 0.17.1。Torch/Torchvision 使用[官方对应版本](https://pytorch.org/get-started/previous-versions/)。Omni processor 即使只输入音频也会初始化视觉组件，因此显式安装 Pillow 11.3.0 与 Torchvision。额外固定 PyArrow 19.0.1，避免新版 Arrow 的 NumPy 2 运行时要求与训练环境冲突。

GPU 训练使用 H200/H100 单卡、CUDA 12.8；最终主文本确认和默认接口采用 CPU FP32。GPU BF16仅作精度复核，不再选模型或充当第二次独立测试。

## 资产

先按 [DATASETS_V2.md](DATASETS_V2.md) 获取固定 CMHL 与 CFunSet；原 V2 下载器有逐文件大小与 SHA256 验证。

```bash
python evaluation/runtime/fetch_v2_assets.py --manifest evaluation/v2_assets.json --output artifacts/assets
python evaluation/runtime/fetch_v2_assets.py --manifest evaluation/v3_assets.json --output artifacts/assets
python evaluation/runtime/fetch_v2_assets.py --manifest evaluation/v3_macbert_assets.json --output artifacts/assets
python evaluation/runtime/fetch_v2_assets.py --manifest evaluation/v3_qwen_assets.json --output artifacts/assets
python evaluation/runtime/fetch_v2_assets.py --manifest evaluation/v3_asr_assets.json --output artifacts/assets
```

同一资产不必重复下载。清单固定 revision、大小、哈希；下载器保留 `.partial` 续传，既有完整文件必须通过哈希。`v3_full_work_assets.json` 只是完整 Omni 与 Whisper 所需文件的合并子集，不是额外一套权重。不要把下载结束、文件选中或 rsync 返回成功等同于校验完成；跨机器传输后重新校验全部文件。

完整训练资产需要数十 GB 磁盘，下载命令适合资源充足的实验环境；只调用已导出的文本 bundle 不需要下载其它模型和音频包。

TMHINT-QI 作者发布在公开 Drive。连接器可读元数据，但 2.28 GB 超过其 256 MB raw fetch 上限，实际通过公开下载传输，未使用登录 cookies：

```bash
python -m pip install gdown==5.2.0
gdown --no-cookies 1TMDiz6dnS76hxyeAcCQxeSqqEOH4UDN0 -O artifacts/assets/TMHINTQI.zip
```

校验值在 [v3_intelligibility_assets.json](v3_intelligibility_assets.json)。准备器强制校验 2,278,236,226 字节、SHA256 `cda6581c9fb0ea634b8bac4c6503e772feb080629a5d95eb84f3d1888876912b`，解压完整读取以验证 ZIP CRC，并拒绝越界路径与链接。许可/使用范围以作者原发布为准；CEAEval 为 CC-BY-NC-4.0，本项目实验限课程研究。

## 文本训练与一次确认

先用 V2 工具准备保留等级、排除已知 CFunSet 重合组的记录：

```bash
python evaluation/runtime/graded_text.py --corpus artifacts/cmhl --cfunset artifacts/CFunSet.json --output artifacts/v3-source
```

下列 `CUDA_VISIBLE_DEVICES` 必须换为已获准且实时空闲的一张 GPU UUID。代码拒绝与其他计算进程共用该卡。每阶段用外层超时；输出目录必须是新的。

```bash
export CUDA_VISIBLE_DEVICES=GPU-your-approved-id
export PYTHONPATH=src:evaluation/runtime
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

timeout 1800 python evaluation/runtime/v3_text.py prepare \
  --plan evaluation/v3_plan.json --records artifacts/v3-source/records.private.jsonl \
  --model artifacts/assets/bert --embedding-model artifacts/assets/bge-small-zh-v1.5 \
  --output artifacts/v3-text-data --gpu-uuid "$CUDA_VISIBLE_DEVICES"

timeout 7200 python evaluation/runtime/v3_text.py select \
  --plan evaluation/v3_plan.json --model artifacts/assets/bert \
  --data artifacts/v3-text-data --output artifacts/v3-roberta --gpu-uuid "$CUDA_VISIBLE_DEVICES"

timeout 7200 python evaluation/runtime/v3_text.py select \
  --plan evaluation/v3_macbert_plan.json --model artifacts/assets/chinese-macbert-large \
  --data artifacts/v3-text-data --output artifacts/v3-macbert --gpu-uuid "$CUDA_VISIBLE_DEVICES"

timeout 7200 python evaluation/runtime/v3_qwen.py select \
  --plan evaluation/v3_qwen_plan.json --model artifacts/assets/Qwen3-8B \
  --data artifacts/v3-text-data --output artifacts/v3-qwen --gpu-uuid "$CUDA_VISIBLE_DEVICES" --seconds 7100

python evaluation/runtime/v3_finalize_selection.py \
  --run artifacts/v3-roberta --run artifacts/v3-macbert --run artifacts/v3-qwen \
  --output artifacts/v3-text-final.lock.json
```

`select` 只读取训练/开发标签；Qwen 在另一台机器训练时，确认文件直到最终冻结后才传入该机器。分区与近重复规则由无标签文本特征决定，CPU/GPU 浮点与库版本可能影响重新聚类的边界，因此保存自己的分区文件及哈希并核对计数，不能看到确认成绩后重分组。原始执行的文字性元数据修正单列在 `v3_metadata_correction.json`，未改变操作参数。

原始执行最终选择 RoBERTa-CORN。它的主确认命令是：

```bash
python evaluation/runtime/v3_text.py report --device cpu \
  --plan evaluation/v3_plan.json --data artifacts/v3-text-data --output artifacts/v3-roberta
```

报告器拒绝已存在的 `confirmation.json`，并检查计划、准备记录和确认文件哈希。若重新执行开发实验选出了别的方法，应按自己的冻结结果报告，不能跳过选择过程强行复用本次赢家。Qwen 对应 `v3_qwen.py report`；不得看确认成绩后再换回另一候选。

确认以后只允许固定方法的技术复核，不再调参。主结果 187/407、基线 173/407；预处理支持 407/408，未覆盖一条。不得更换分母或将等级期望四舍五入冒充 argmax 分类。

## 人工表达适切性

```bash
python evaluation/runtime/v3_audio.py prepare --plan evaluation/v3_plan.json \
  --root artifacts/ceaeval-extracted --output artifacts/v3-expression-data

timeout 7200 python evaluation/runtime/v3_audio.py features --plan evaluation/v3_plan.json \
  --model artifacts/assets/Qwen2.5-Omni-7B --embedding-model artifacts/assets/bge-small-zh-v1.5 \
  --data artifacts/v3-expression-data --output artifacts/v3-expression --gpu-uuid "$CUDA_VISIBLE_DEVICES"

python evaluation/runtime/v3_audio.py select --plan evaluation/v3_plan.json \
  --data artifacts/v3-expression-data --output artifacts/v3-expression

python evaluation/runtime/v3_audio.py report --plan evaluation/v3_plan.json \
  --data artifacts/v3-expression-data --output artifacts/v3-expression
```

原人工包的验证/测试故事退出新确认；采用剩余故事的固定划分。波形解码帧数、重采样及编码器帧数均核对完整覆盖。文字对照仅使用台词，模型也未获得完整叙事语境，不能称为作者完整 CEAEval-M 的复现。

## 人工可懂度

```bash
python evaluation/runtime/v3_intelligibility.py prepare --plan evaluation/v3_intelligibility_plan.json \
  --archive artifacts/assets/TMHINTQI.zip --root artifacts/tmhint-extracted --output artifacts/v3-intelligibility-data

timeout 7200 python evaluation/runtime/v3_audio.py features --plan evaluation/v3_intelligibility_plan.json \
  --model artifacts/assets/Qwen2.5-Omni-7B --skip-text-embeddings \
  --data artifacts/v3-intelligibility-data --output artifacts/v3-intelligibility --gpu-uuid "$CUDA_VISIBLE_DEVICES"

python evaluation/runtime/v3_intelligibility.py select --plan evaluation/v3_intelligibility_plan.json \
  --data artifacts/v3-intelligibility-data --output artifacts/v3-intelligibility

python evaluation/runtime/v3_intelligibility.py report --plan evaluation/v3_intelligibility_plan.json \
  --data artifacts/v3-intelligibility-data --output artifacts/v3-intelligibility
```

最终实现采用 float64 拟合高维回归，原 float32 数值警告及确认前修正记录见 `results/v3/intelligibility_numerics.json`。底层句编号跨噪声、增强、说话人保持同组；人类目标来自 `intelligibility_score`，不是音质。没有发布文本的任务不构造假文字对照。

## 导出与统一调用

```bash
python evaluation/runtime/export_v3_bundle.py \
  --text-run artifacts/v3-roberta --text-plan evaluation/v3_plan.json \
  --audio-run artifacts/v3-expression --intelligibility-run artifacts/v3-intelligibility \
  --output artifacts/v3-bundle

openmic-evaluate --bundle artifacts/v3-bundle --verify-bundle \
  --text '待评的完整短文本。' --evidence

openmic-evaluate --bundle artifacts/v3-bundle --audio recording.wav
```

bundle 包含选定三个编码器 checkpoint、对应回归器、必要的冻结音频编码器和配置，不含语料或真人评分。配置与文件清单共同绑定 bundle ID；`--verify-bundle` 完整检查权重哈希。只加载自己生成的 joblib bundle。未导出权重时接口明确返回 `unconfigured`，不会伪造模型评分。

完整稿件、完整实际音频的实验模式需显式提供本地 Omni：

```bash
openmic-evaluate --bundle artifacts/v3-bundle --text-file script.txt --audio performance.wav \
  --experimental-omni-model artifacts/assets/Qwen2.5-Omni-7B \
  --asr-model artifacts/assets/whisper-large-v3-turbo \
  --device cuda --output artifacts/full-work-evaluation.private.json
```

该命令同样先检查 `CUDA_VISIBLE_DEVICES` 指定卡空闲。BERT 超 512 token 不截断；实验 Omni 完整输入上限 30,000 个多模态 token，超限明确失败。声音的整体幽默、表演表达和可懂度分别返回任务、量表、验证范围；不会合成一个声称有效的总分。原生 CEAEval 表达量表 0—5、完整作品实验表达量表 1—5，不作跨任务比较。

长音频可懂度的四秒分窗聚合标为 `experimental_aggregate`，覆盖所有波形帧，未获完整作品真人验证。ASR 独立读完整波形的 25 秒段，计算转写/原稿字错误率；它不能代替真人可懂度标签。Whisper 返回字典形式的底层生成记录，以实际生成步数和 EOS 判断解码上限，避免把库自动删去的 EOS 当作失败。

Python 接口：

```python
from openmic.evaluation import OfflineEvaluator

judge = OfflineEvaluator("artifacts/v3-bundle", device="cpu")
result = judge.evaluate(text="待评的完整短文本。", evidence=True)
```

文本依据可选返回实际删句敏感性，含可复算的原句位置与分数差；不生成虚构笑点解释。输出 `status`、每个维度状态、`failures`、覆盖与版本，对应 [JSON Schema](schemas/evaluation_response.schema.json)。等级概率未做人群置信度校准。某个可选模型失败，不抹去其余已经算出的结果。

固定 ASR 与完整接口核查命令：

```bash
python evaluation/runtime/v3_asr.py --plan evaluation/v3_asr_plan.json \
  --model artifacts/assets/whisper-large-v3-turbo --inputs artifacts/v2-audio/references.private.jsonl \
  --test-only --output artifacts/v3-asr-human --gpu-uuid "$CUDA_VISIBLE_DEVICES"

python evaluation/runtime/v3_asr.py --plan evaluation/v3_asr_plan.json \
  --model artifacts/assets/whisper-large-v3-turbo --inputs artifacts/project-pilot/audio_manifest.private.jsonl \
  --output artifacts/v3-asr-project --gpu-uuid "$CUDA_VISIBLE_DEVICES"

python evaluation/runtime/v3_api_smoke.py --plan evaluation/v3_full_work_plan.json \
  --bundle artifacts/v3-bundle --inputs artifacts/project-pilot/audio_manifest.private.jsonl \
  --omni-model artifacts/assets/Qwen2.5-Omni-7B --asr-model artifacts/assets/whisper-large-v3-turbo \
  --device cuda --gpu-uuid "$CUDA_VISIBLE_DEVICES" --output artifacts/v3-api-integration
```

项目试验材料沿用 V2 的四段固定完整音频；新建生成材料必须另存清单，不能冒充本次输入。接口核查发现运行失败、波形不完整或指定完整评分未覆盖每条时会返回非零退出码，并保留失败记录。

## 检查与集群

```bash
python -m pytest -c evaluation/pytest.ini -q evaluation/tests
PYTHONPATH=src python -m unittest discover -s tests -v
```

测试覆盖 argmax/期望区别、CORN 概率与梯度、主题/近重复隔离、声音尾部覆盖、可懂度与音质列分离、长音频分窗覆盖、超长拒绝、真实删句敏感性、开发冻结防护和 LoRA 分类头保存/载入。完整作品还需运行实际 `v3_api_smoke.py`，合成 fixture 不能证明有效性。

Slurm 可使用 `v3_slurm.py` 包装一次有界阶段。它要求已有单 GPU allocation，先用退出的子进程读取已分配 UUID，再设置同一 UUID给阶段；记录任务、设备及命令。不要在登录节点直接占 GPU，不终止他人进程，不把集群作业与 Codex 周期自动化混为一谈。

项目完整作品的三维真人听评使用 [V3_LISTENING.md](V3_LISTENING.md)：新空白表分别记录幽默、表达和字词可懂度，检查播放覆盖，并有显式导入与分维分析。原 V2 表继续由既有工具处理；两轮不混为独立评分者。只在取得真实评分后导入，原私人工作簿保持原样。
