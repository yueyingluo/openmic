# 协作规范

## 分支和 PR

- `main` 始终保持可运行；不要直接向 `main` 推功能代码；
- 分支命名：`feat/<name>`、`fix/<name>`、`docs/<name>`；
- 一个 PR 只解决一个 Issue；至少一位其他成员 Review 后合并；
- 不提交 API Key、模型权重、大型数据集；生成音频仅允许下文列明的课堂展示短片例外。

## 完成定义

一个任务只有同时满足以下条件才算完成：

1. 代码或文档已经进入 PR；
2. 有最小测试或可复现命令；
3. 更新了相关接口说明；
4. Reviewer 能在自己的环境复现；
5. 没有把密钥和大文件提交进 Git。

## 接口优先

模块尚未实现时先提交接口和 Mock，不要让其他成员等待。改变共享输入输出结构时必须在 PR 描述中列出影响范围。


## 课堂展示音频例外

为使同学下载完整仓库后能直接播放 JYB 的期中展示，允许随 slides 提交以下三段已审核的“截止日期”短音频：

- `assets/jyb_midterm_audio/deadline_01_baseline.wav`
- `assets/jyb_midterm_audio/deadline_02_first_performance.wav`
- `assets/jyb_midterm_audio/deadline_03_integrity_repair.wav`

三段分别为普通朗读原版、第一轮完整表演、新紧凑版·完整性修复，均在“不是灵感，是情绪。”后结束，总计 1,661,416 字节。
这是针对以上三个路径的例外，不包含完整生成音频、其他试听结果、模型或数据集。`.gitignore` 仅为这三个 WAV 添加反忽略规则；使用说明见 `presentation/README.md`。
