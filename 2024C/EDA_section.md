# EDA 部分：Wimbledon Featured Matches（初步探索性分析）

## 概览（目标）
- 目的：使用逐分（point-by-point）数据检验并可视化比赛中的“momentum / swings”的存在性、强度与触发因素，同时量化并控制发球方优势的混杂影响，为后续建模与教练建议提供证据基础。
- 主要产物：发球优势分析、压力状态下的赢分率比较、score-state 热力图、单场滚动 momentum 曲线与反转触发器对齐分析。
- 代码与数据位置：所有分析代码与图表在笔记本 `2024C/eda_initial_analysis.ipynb` 中实现。

## 数据与预处理（Methods）
- 数据粒度：每行为一个 point，包含 `match_id, set_no, game_no, point_no, server, serve_no, p1_score, p2_score, p1_games, p2_games, p1_sets, p2_sets, point_victor, serve_speed, rally_count, p1_distance_run, p2_distance_run` 等字段。
- 关键派生列：
  - `server_numeric`：统一为 {1, 2}；
  - `is_first_serve`：由 `serve_no` 推断（缺失时以回退值处理）；
  - `server_won_point`：比较 `point_victor` 与 `server_numeric` 得到二值结果；
  - server-relative score：`server_score` / `returner_score`（用于热力图与分数态控制）；
  - `pressure_bucket`：按优先级赋值（Break > Tiebreak > Deuce > Baseline）；
  - 滚动指标 `momentum`：以窗口 15 的 rolling win-rate 差（p1 − p2）作为动量 proxy（min_periods=5）。
- 稳健性与可视化约定：每个分组同时报告样本量 `n` 与 Wilson 置信区间（函数 `wilson_confidence_interval`），并在图中标注 `n=0` 或 `low n` 以区分小样本情形。

## 关键图（建议纳入论文的 3–6 张图）
- 图 1 — 发球优势（Serve Advantage）
  - 内容：按 `server_numeric` × `is_first_serve` 汇总 win_rate、n、Wilson CI。
  - 结论：一发 vs 二发之间胜率差显著，表明发球是强混杂因子；若不控制发球，momentum 指标可能被发球轮转误导。

- 图 2 — 压力状态下的赢分率（Pressure Buckets）
  - 内容：Baseline / Deuce / BreakPoint / Tiebreak 条件下的 server win_rate，分 Player1/Player2 并显示 n。
  - 结论：压力情形显著改变胜率，表明点与点之间非独立同分布（non-iid），在建模时需加入 score-state/pressure 控制。

- 图 3 — Score-State 热力图（Serve–Score Control Map）
  - 内容：以 server 相对分（0/15/30/40/AD）为行、returner 分为列，展示不同分数配对下 server 的 win_rate（分别为一发/二发）。
  - 结论：识别高/低胜率区域与样本稀疏区（空格或 n=0），提示在模型中对稀疏分数态做平滑或分箱处理。

- 图 4 — 单场滚动 momentum（Representative Match）
  - 内容：在代表性比赛中绘制 rolling momentum（p1 rolling win-rate − p2 rolling win-rate），填色显示正/负区间，并标注检测到的 swing 转折点与盘/局边界。
  - 结论：可视化显示持续多分或多局的优势段及方向反转，支持 swings 的存在性（但需统计检验以排除随机性解释）。

## 触发器对齐分析（Trigger-aligned windows）
- 方法：对每个检测到的反转点（swing），截取前后窗口（例如 −10 到 +10 分）；对齐后计算并绘制：
  - `first_serve_pct`（或 `is_first_serve` 的比率）；
  - `serve_speed` 平均与置信区间；
  - `rally_count` 平均；
  - 失误率（UE / DF）与获胜手段分布（winner / ace）；
  - 跑动差（p1_distance − p2_distance）。
- 目的：识别在反转事件前后是否存在一致的物理或技术信号（例如一发命中率下降或 UE 增加），为解释性结论提供证据。

## 检验教练质疑：Is it just random?（Null 模型与 permutation test）
- Null 模型建议：在控制 `serve` 与 `score-state` 的条件下，按经验概率为每个点生成 iid Bernoulli（或在相同 score-state 下重采样真实点结果），生成多次模拟序列并在模拟上运行同样的 swing 检测器。
- 统计量：比较实测的 swing 频率、最长持续时长、平均持续时长等在模拟分布中的分位，计算 p-value。
- 要点：必须在 null 中保留发球序列与 score-state，否则模拟会低估随机性（导致伪显著）。

## 主要 EDA 发现（≥5 条 Insights）
1. 发球优势是最强的混杂因子：一发显著高于二发的赢分概率，若不控制会错误解释为 momentum。 
2. 压力状态（break / deuce / tiebreak）显著改变点胜率，证明点序列非 iid，应在模型中显式控制。 
3. Swings 在可视化层面存在：滚动 momentum 曲线中出现持续段与反转，但许多反转出现在低样本或特定 score-state，需要用 Null 模型检验显著性。 
4. 触发器候选：在若些反转样本中观察到一发命中率下降、发球速度下降或 UE 升高，提示疲劳或心理波动可能促成反转。 
5. 数据缺失限制解释能力：体能追踪（跑动）、发球速度或某些事件（UE）缺失较多，影响触发器分析与泛化能力。

## 局限性与下一步建议（Limitations & Next Steps）
- 样本稀疏：某些 score-pair/pressure bucket 在单场或个别选手上样本过少，应聚合或采用贝叶斯平滑。
- 测量与记录不一致：不同比赛的字段完备度不同，最好在多场合并分析并记录数据质量。
- 建模建议：基于 EDA，优先尝试“隐状态 + 外生输入（serve, score-state, rolling proxies）”的模型（如 HMM / state-space / switching models），并使用上述 Null 模型检验 swing 统计的显著性。
- 泛化检验：在女单、不同场地及不同级别比赛上重复 EDA，量化效果差异与数据差异。

## 可复现性与交付
- 所有数据预处理与图表代码已实现于笔记本 `2024C/eda_initial_analysis.ipynb`（包含用于绘图的 helper 函数与 Wilson CI 计算）。
- 推荐导出：将最终选定的 3–6 张图导出为高分辨 PNG（或 SVG）并在论文中配短 caption；将此 Markdown 存为章节草稿并合并到 ≤25 页报告与 1–2 页教练 memo 中。

---

如果你需要，我可以：
- 1) 将本文件保存为 `2024C/EDA_section.md`（已创建）；
- 2) 同时把 notebook 中对应的关键图导出到 `2024C/artefacts/figures/`；
- 3) 把该段落转换为 LaTeX 格式并生成论文模板片段。

你要我接下来做哪一项？（导出图 / 生成 LaTeX / 直接开始写 ≤25 页报告中的 EDA 部分）