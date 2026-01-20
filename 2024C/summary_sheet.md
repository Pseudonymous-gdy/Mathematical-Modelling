Summary Sheet — 快速索引（供报告附录使用）

关键文件汇总：
- 原始逐分数据：`2024_Wimbledon_featured_matches.csv`
- 数据字典：`2024_data_dictionary.csv`
- 提取的时序特征：`match_time_series_features.csv`
- 基线模型输出（notebook 单元）: 在 `EDA.ipynb` 中查看“最小基线”单元
- 快速 DBN 摘要：`dbn_quick_summary.csv`
- 精确 DBN 摘要：`dbn_precise_summary.csv`
- 单场隐状态均值（示例）：`match_<match_id>_dbn_Mmean.csv`（见工作区）

重要图表索引（在 `EDA.ipynb`）
- 发球优势条形图（server effect）
- 校准图（baseline calibration plot）
- Swings 可视化（rolling winrate vs cum_diff）
- 反转窗口聚合统计表
- DBN 隐状态曲线与 90% HDI（单场）

运行/复现说明（简要）
1. 打开 `EDA.ipynb`，按顺序运行单元（建议按组运行：数据加载 → 特征抽取 → 基线 → EDA → DBN）。
2. 若要重跑 DBN（更精确）：在 DBN 单元调整 ADVI 迭代数或改为 MCMC（需更高计算资源）。

联系方式/下一步
- 我已在工作区生成 report_outline.md、coach_memo.md、summary_sheet.md、以及各 CSV 文件。
- 需要我把这些内容合并成一个 PDF 报告草稿或准备一份 1–2 页的现场 quick-sheet 吗？