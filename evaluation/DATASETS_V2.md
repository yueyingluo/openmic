# 分级文本与声音数据：选择及实际核验

核验日期：2026-10-26。文本文件已经下载并逐行统计；音频部分核查了官方说明、仓库文件列表、版本和访问条件，尚未下载或逐条审计音频。来源版本和文件hash见 [e2e_sources.json](e2e_sources.json)，本地实际统计见 [graded_sources_audit.json](results/graded_sources_audit.json)。本仓库不再分发原始语料。

## 选用的数据

| 数据 | 标签与规模 | 在本项目中的作用 | 当前状态 |
| --- | --- | --- | --- |
| [CMHL](https://github.com/SamTseng/Chinese_Humor_MultiLabeled) | 3,365条中文笑话，完整1—5级；另有8类手法、6类意图 | 文本等级评分的主要开发语料 | 已下载、核对三张标注表及重复 |
| [CCL2019 Task 2](https://github.com/DUTIR-Emotion-Group/CCL2019-Chinese-Humor-Computation) | 训练16,670、测试4,171；只有1/3/5三档 | 文本序数评分补充诊断 | 已下载、核对标签和跨划分重复 |
| [CEAEval-D人工子集](https://huggingface.co/datasets/TianRW/CEAEval-Data) | 官方记录7,136片段、91个故事；整体表达适切性0—5，另有韵律/情绪等描述 | 中文声音表达是否契合上下文 | 无访问门槛；人工压缩包约8.00 GB，未下载 |
| [SpeechJudge](https://speechjudge.github.io/) | 官方记录约99K语音对，人工自然度偏好与可懂度标签，含中文 | 自然度/可懂度的备选校准 | 数据仓库需申请访问，未接受条款或下载 |
| 项目实际生成的完整音频 | 拟收集整体好笑程度1—5及同主题偏好 | 最终端到端结论 | 尚待生成与盲听，不由公开数据替代 |

## 文本：完整五档确实存在，但不能直接当作最终金标准

CMHL各档实测数量为 **363 / 867 / 1,313 / 729 / 93**，高分少，训练应按组划分并报告各档误差，不能只优化多数档。三张表按行ID和规范化文本对齐；原始ID有25行重复，不能仅凭ID连接或去重。应使用稳定的文本hash及故事族分组。

[原论文](https://aclanthology.org/2020.lrec-1.168/)说明每条笑话只由一位标注者给标签，且幽默等级与读者反馈的相关性较弱。因此它适合建立五级预测器和比较prompt，但不足以证明“拟合大众幽默感”。手法、意图是类别标签，并不是八个或六个质量分数。仓库声明MIT许可；其收集的第三方文本来源权利未逐一核实。

队友的文本PR已经接入该语料，包含等级回归、近重复分组和可选生成参考。评估侧应复用其固定版本与分组，记录哪些材料已经进入生成参考或prompt开发。原有R1人工反馈也已用于改稿，不再充当全新测试。不要换一个随机种子重分数据来制造独立性。

CCL的标签实测如下，**1/3/5表示三档，不是缺了两档的完整五级任务**：

| 划分 | 1 | 3 | 5 | 总行数 |
| --- | ---: | ---: | ---: | ---: |
| train | 4,408 | 8,624 | 3,638 | 16,670 |
| test | 1,103 | 2,159 | 909 | 4,171 |
| development | 无标签 | 无标签 | 无标签 | 1,041 |

规范化后，原train/test共有30个文本；development的1,041个文本**全部出现在test中**。开发提示应从去重后的train按故事族另划验证集，不要利用development对应的test标签调参后再报告全test成绩。原仓库未发现明确数据许可证；先保留来源与本地研究审计，不打包再分发文本。

## 与CFunSet的重叠

本次对CFunSet全部记录的input/output字符串，按HTML清理、NFKC、大小写统一、繁转简、只保留字母数字进行精确匹配。下表计数是**唯一规范化文本数**，不是语料行数或已证实训练曝光数。

| 语料 | 与CFunSet规范化精确重合 |
| --- | ---: |
| CCL train | 6,779 |
| CCL development | 242 |
| CCL test | 950 |
| CMHL | 41 |

CMHL与CCL train/test分别重合36/8个文本，不能把两个语料简单视为相互独立。没有进行模糊、子串或故事族匹配，以上只是污染下界；删掉这些文本也不能声称完全无污染。CFunModel使用CFunSet训练，因此对应成绩只能按来源相关诊断解读。其他基座模型的预训练曝光未知。

## 声音：哪些标签能支持哪些结论

CEAEval-D来自中文有声书，人工标签针对上下文表达适切性，适合检验音频评审能否识别韵律、情绪与语境的配合。其15个标注字段包括元数据和类别，不能误写成15项MOS分。原始整体分为0—5，可含小数，须保留原量表；与项目1—5幽默评分分开统计。许可为CC-BY-NC-4.0。

固定revision的文件元数据表明：`manual_annotation.tar.gz`为8,002,679,318字节；`weak_annotation.tar.gz`为165,968,823,540字节。优先只准备人工包。弱标注是模型生成的描述，不能混入人工测试。下载后还需核验实际条目、标签缺失、音频存在性、时长与故事映射；按故事分组，若说话人身份可识别，再避免同一说话人跨组。不能直接照README的全量snapshot命令把约166 GB弱标注一起拉下。

SpeechJudge的[官方仓库](https://github.com/AmphionTeam/SpeechJudge)将SpeechJudge-Eval作为 `RMSnow/SpeechJudge-Data` 的 `test` 划分加载。它不是另一份可以假定公开下载的独立仓库；当前需要账户接受数据访问条件，未执行这一操作。许可为CC-BY-NC-4.0。若后续取得访问权限，筛选中文并遵循官方划分；SpeechJudge-GRM自身已在该数据上训练，不能在训练部分验证后宣称泛化。其自然度偏好仍不能证明脱口秀好笑。

本轮未找到经核实、同时满足“中文完整脱口秀音频 + 多档幽默评分 + 独立多人标注”的现成公开数据。不能把未找到写成确定不存在。

## 不作为主数据的候选

- [UR-FUNNY](https://github.com/ROC-HCI/UR-FUNNY)：有文本/声音/视频，但主任务仍是二分类，不满足多档主评价要求。
- [StandUp4AI](https://github.com/Standup4AI/dataset)：有脱口秀与笑声时间信息，但列出的七种语言不含中文；笑声出现也不是独立多档人工评分。
- [SpeechParaling-Bench](https://speechparaling-bench.github.io/)：可作副语言控制能力挑战集；其模型评审结果不能冒充人类幽默等级。

## 复现本次审计

按 `e2e_sources.json` 中 `text_files` 的固定URL下载到一个本地根目录，保持 `file` 相对路径，并核对SHA256。准备CFunSet原始JSON后运行：

```bash
python -m pip install opencc-python-reimplemented==0.1.7
python evaluation/runtime/audit_graded_sources.py \
  --root artifacts/e2e_data_audit \
  --cfunset evaluation/runtime/data/raw/CFunSet.json \
  --simplify \
  --output artifacts/graded-sources-audit.json
```

`--cfunset`应指向自己的真实文件位置；省略它则不检查CFunSet重叠。省略`--simplify`会漏掉繁简对应，不能拿那个计数宣称语料独立。脚本输出汇总与hash，不输出原文或逐条评分，已有输出文件拒绝覆盖。最终执行协议见 [E2E_DESIGN.md](E2E_DESIGN.md)。
