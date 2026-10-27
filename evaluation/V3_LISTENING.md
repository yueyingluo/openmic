# 完整作品分维听评

本流程准备空白听评表，分别收集整体幽默、表达效果、字词可懂度。它不补造真人评分，也不读取原私人评分工作簿。原 V2 表及其已有导出保持原样。

目前四段作品是既有技术试评，只有两个主题，不具备新的正式测试资格。即使收到了评分，也只能报告本次评分者在这四段作品上的探索性一致性。不得据此选生成条件后宣称正式测试有效。

## 给评分者的材料

~~~bash
python evaluation/runtime/v3_listening.py build \
  --manifest artifacts/project-pilot/audio_manifest.private.jsonl \
  --output artifacts/listening-v3
~~~

构建前逐一核对音频 SHA256。给评分者的目录只有 index.html、README.txt 和匿名音频；身份映射与分析样本清单保存在该目录外的 .mapping.private.json、.samples.private.jsonl。表中不显示模型分数、生成条件、原稿或真实样本身份。

评分者打开 index.html，填写匿名编号、独立听完整段。三个维度均为 1—5 档，没有默认分数，可以留空。理解背景与听清字词分开记录：无法理解内容时不能给幽默与表达打分，但仍可判断字词是否清楚。无法判断时留空并可备注原因。

初次打开按浏览器随机打乱呈现顺序，草稿恢复保持自己的顺序；顺序随导出保存。播放覆盖按已播放区间的并集统计，跳转和重复播放不增加重复时长。有分数的条目需达到 95% 播放覆盖并勾选已听完整段。覆盖是技术记录，不能证明真实身份、注意力、理解或独立作答；不能将它当作认证。曾听过内容需勾选既往接触。

页面只在本机保存草稿，导出 JSON 后由评分者自行交回；不会发邮件或上传。不同评分者宜使用独立浏览器资料或自己的设备，避免恢复别人的草稿。不要把表中的测试输入或导出的评分放进公共 Git。

## 导入与一致性分析

只导入评分者明确交回的文件。导入器检查协议、数据集身份、完整样本名单、同一匿名编号、顺序、原生分数、理解情况及播放记录；留空仍为空，不填零或均值。输出使用排他创建，不能覆盖已有文件。

~~~bash
python evaluation/runtime/v3_listening.py import \
  --export artifacts/listener-01-export.json \
  --mapping artifacts/listening-v3.mapping.private.json \
  --output artifacts/listener-01.imported.private.jsonl

python evaluation/runtime/v3_listening.py predictions \
  --results artifacts/v3-api-integration/predictions.private.jsonl \
  --output artifacts/listening-v3.predictions.private.jsonl

python evaluation/runtime/v3_listening.py analyze \
  --samples artifacts/listening-v3.samples.private.jsonl \
  --ratings artifacts/listening-v3.ratings.private.jsonl \
  --predictions artifacts/listening-v3.predictions.private.jsonl \
  --output artifacts/listening-v3.analysis.private.json
~~~

多人导入文件经检查后按行合并到 listening-v3.ratings.private.jsonl；同一人、同一样本、同一轮次的重复会被拒绝。默认每条至少三位有效评分者，不能把重复评分当作不同人。个人偏好试验可显式使用 --min-raters 1，报告会区分个人偏好与实际评分者面板；它不获得一般人群代表性。

分析分别报告三个维度的有效覆盖、Spearman/Kendall、同主题配对、序数 Krippendorff alpha、留一评分者相关性和按主题重采样区间。无真实评分时保留 awaiting_human_ratings_or_predictions，不会输出虚构相关性。只有两个主题时，相关性和区间都不稳定，不能把样本内相关当作泛化证据。既往曝光随私有导出保留，应在报告中说明，不能看模型成绩后选择性删掉评分者。

听评的字词可懂度是主观 1—5 分，不等于 TMHINT-QI 的 0—10 个识别汉字。表达的主观 1—5 分也不等于 CEAEval 原生任务的 0—5 分。这里只比较排序一致性，不计算跨任务绝对误差，不用平均人工评分四舍五入制造五分类准确率，更不替代 V3 文本确认集的 45.95%。

若要正式验证，需要另行冻结新的作品、主题、生成配置、评价器和分析规则，再收集真实且独立的评分。当前三个公开确认集均已查看，不能继续调参后沿用独立测试之名。
