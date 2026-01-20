报告提纲：温网决赛动量（Momentum/Flow）建模与检验

1. 摘要（<=1页）
- 研究问题与动机
- 数据简介（逐分数据字段概览）
- 主要结论（是否发现显著动量、影响因子）

2. 数据与可提取特征（1–2页）
- 数据来源与清洗流程
- 核心字段说明（发球、比分、回合数、发球速度、失误等）
- 特征分类：即时可用 vs 事后解释可用

3. EDA（3–5页，含关键图）
- 发球优势证明（server effect，一发/二发拆分）
- 关键分/比分压力影响（break/deuce/tiebreak/局末）
- Swings 可视化（单场滚动胜率与累积分差）
- 反转触发器对齐（发球速率/一发比/回合数/失误）
- ≥5 条 insight → 导出建模选择

4. 基线模型（1–2页）
- 逻辑回归（server + score_diff）实现细节
- 指标：log-loss、Brier、校准图
- 基线性能与限制讨论

5. DBN 建模（6–8页）
- 模型假设与生成过程（M_t 动量隐状态、转移与观测方程）
- 可选扩展（break point 交互、外生 u_t）
- 推断方法（VI/ADVI，若必要做 MCMC）
- 单场示例结果：M_t 曲线、90% 区间、参数后验（alpha, rho, eta）
- 检验：alpha=0 / rho=0 的结论与 OOS 比较

6. 反转预测与解释（2–3页）
- 反转事件定义（E[M_t] 过零 / 转向概率阈值）
- 案例：至少一场比赛的短期反转预测与触发因素贡献（serve_form, fatigue）
- 可操作建议（教练备忘录要点）

7. 泛化与局限（1–2页）
- 在其他比赛/场地/女单/乒乓的可行性讨论
- 数据缺失项与优先采集建议

8. 结论与建议（<=1页）

9. 附录（代码片段、模型设定、CSV 路径、关键图索引）

关键输出文件（工作区路径）：
- 基线结果：`match_time_series_features.csv`、notebook 中基线单元输出
- DBN 单场 M_t CSV：match_<match_id>_dbn_Mmean.csv
- 快速与精确摘要：`dbn_quick_summary.csv`, `dbn_precise_summary.csv`
- EDA 图：在 `EDA.ipynb` 生成的图像单元

下一步：我会生成 1–2 页教练备忘录草稿并把 summary sheet 与 AI 使用报告一起写入工作区。若有偏好，我将把报告格式输出为 Markdown。