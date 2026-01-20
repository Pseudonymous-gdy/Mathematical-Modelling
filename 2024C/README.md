下面把“你没看过任何聊天历史”的版本从头梳理一遍：我们现在做到了哪里、接下来怎么做（按你要的 5 点组织）。

---

## 1) 题目拆解与任务要求

题目用温网决赛“控制权多次换边、持续很多分甚至很多局的 swings”引出 momentum：它被球员感知，但难以度量，也不清楚哪些事件会创造/改变它。

你要完成的任务是（官方 bullets）：

1. **建模 flow**：随逐分发生，识别“此刻谁更占优、强度多大”，并可视化整场走势；同时注意发球方优势要考虑。
2. **检验教练质疑**：教练认为 swings/连胜只是随机，用你的模型/指标评估。
3. **预测反转 + 找因素 + 给建议**：用至少一场比赛预测 flow 即将从一方转向另一方；指出相关因素，并给备战建议。
4. **外推泛化**：在其他比赛测试效果，分析缺失因素，讨论能否泛化到女单、不同场地甚至乒乓。
5. **交付**：≤25页报告 + 1–2页教练 memo + summary sheet/目录/参考文献；如用AI要附 AI use report。

---

## 2) 数据可提取内容

数据是**逐分（point-by-point）序列**。题面明确字段范式包括：谁发球、谁得分、盘/局/比分、回合数、跑动距离、发球速度与落点、破发点等。

结合题面给的 examples，你至少能提取四大类（你之前总结得很对）：

* **上下文比分/阶段**：`set_no, game_no, point_no, p1_score/p2_score, p1_games/p2_games, p1_sets/p2_sets` 等（题面例子逐列解释了这些如何表示“第几盘第几局第几分、比分多少”）。
* **发球情况**：`server, serve_no, speed_mph, serve_width, serve_depth, return_depth`（题面例子给了一个完整解释）。
* **体能表现**：`p1_distance_run, p2_distance_run, rally_count`（题面例子同样给了定义）。
* **技术事件与得分方式**：winner/ace/double fault/unforced error/net point/break point 等（题面明确“这些列让我们判断这一分怎么赢的/是否破发点”等）。

> 建模时要把特征分成两套：
> **当下预测可用**（开打前/发球后立刻可知：server、比分、阶段、历史滚动统计） vs **事后解释可用**（winner/UE/跑动/回合数属于本分结束后才知道，更适合解释“为什么反转”）。

---

## 3) EDA 要讲的故事（用来揭示建模动机）

EDA 的“主线叙事”建议是：**逐分数据强结构依赖 + 发球优势是强混杂 + 关键分压力导致非平稳 + swings 需要一个动态隐状态来解释**。

你可以按 3–6 张关键图把故事讲圆（summary.md 也建议至少 3–6 张关键 EDA 图、并产出≥5条 insight 指向建模选择）。L1-L3

推荐的 EDA “故事板”：

1. **发球优势（必须先证明是混杂）**

   * 图：server=1 vs server=2 的赢分率差异；分一发/二发。
   * 结论导向：如果不控制 server，你的 momentum 会“学到”发球轮转而不是走势（题面也提醒发球方赢分概率高）。

2. **关键分/比分压力的结构性变化（非 i.i.d.）**

   * 图：break point / deuce / tiebreak 条件下的赢分率变化；盘末局末的变化。
   * 结论导向：点与点不是独立同分布，比分状态会改变胜率（summary.md也强调“强依赖结构”）。

3. **swings 的“存在性可视化”（不等于证明）**

   * 图：某场比赛的“滚动赢分率/滚动优势”曲线；标出盘局边界。
   * 结论导向：我们确实看到“持续很多分/局的优势段 + 方向改变”，这对应题面叙事中的 swings。

4. **潜在触发器线索（解释反转的候选因素）**

   * 图：反转点前后的 serve speed/一发比例、rally_count、跑动差、UE/DF 率变化（用“反转窗口”对齐）。
   * 结论导向：反转可能由发球状态/疲劳/关键分事件驱动，所以需要“隐状态 + 外生输入”的动态模型，而不是静态回归。

---

## 4) DBN 模型（从可提取内容出发，momentum 为标量）

我们把题面 momentum 直接定义成一个**标量隐状态** (M_t)：表示“此刻控制权/净优势强度（带方向）”。这对应题面“优势会持续 many points/games 且会 change of direction”。

### 4.1 变量与特征怎么进模型

* 观测结果：(y_t\in{0,1})：第 (t) 分 P1 是否赢（point_victor）。
* 控制变量（必须）：
  (c_t=)(server_t, score_state_t, importance_t)。

  * server 来自 `server`；score_state 来自盘/局/分与比分列；importance 用 break point / deuce-AD / tiebreak 构造。L12-L27
* 外生输入（解释动量演化）：
  (u_{t-1}=)(serve_form_{t-1}, fatigue_{t-1})

  * serve_form：历史窗口的一发比例、均速、落点稳定性（由 `serve_no/speed_mph/serve_width/serve_depth` 汇总）。
  * fatigue：历史窗口的回合数与跑动（由 `rally_count` 与 distance_run 汇总）。

### 4.2 DBN 的生成模型（最简、易解释、适合变分推断）

**状态演化（momentum 的“惯性 + 被因素驱动”）**
[
M_t \mid M_{t-1},u_{t-1} \sim \mathcal N\big(\rho M_{t-1}+\eta^\top u_{t-1},\ \sigma_M^2\big)
]

* (\rho)：动量持续性（是否“能持续很多分/局”的核心参数）
* (\eta)：发球状态/疲劳是否推动动量变化（解释“为什么会反转”）

**观测模型（当分胜率；控制变量清晰）**
[
y_t \mid M_t,c_t \sim \mathrm{Bernoulli}\Big(\sigma(\beta^\top c_t + \alpha M_t)\Big)
]

* (\beta^\top c_t)：把发球优势与比分压力“控制住”，避免混淆
* (\alpha M_t)：动量对当分胜率的净贡献

**推断方式：变分推断（VI）**
按照 summary.md 的统一模板，就是“隐状态→观测→贝叶斯更新”，并输出不确定性区间。
我们用 (q(M_{1:T})) 近似后验（可先用 mean-field 正态，再升级成相邻相关的结构化 VI），优化 ELBO 得到每一分的 (M_t) 后验均值与置信带。

---

## 5) 接下来应该做什么（按可落地的执行清单）

**Step 0｜确定评估协议（先定死，防止“假提升”）**

* 以“按时间顺序”的 forward / rolling split 做验证（逐分序列强依赖）。

**Step 1｜做 baseline（必须有对照）**

* 最小 baseline：只用 server + score/importance 的逻辑回归预测每分胜率。
* 指标：log loss / Brier + 一张校准图（概率预测必须看校准）。

**Step 2｜完成 EDA 故事板并产出“insight→建模选择映射”**

* 至少 3 张关键图、≥5 条 insight，并标注分别推动了哪些模型设计（控制 server、建隐状态、关键分权重等）。

**Step 3｜实现 DBN + VI（先简后繁）**

* 先做：单隐状态 (M_t)+线性高斯转移+logistic 观测；输出 (M_t) 曲线与置信带（flow 可视化）。
* 再做：关键分交互（让动量在 break point/tiebreak 更“放大”），看预测与解释是否改善。

**Step 4｜“momentum 是否随机”的检验（对应任务 2）**

* Null：(\alpha=0) 或 (\rho=0)（无动量/无持续性）
* Alt：(\alpha\neq0,\rho>0)
* 用 out-of-sample log loss + 后验区间证明（(\rho) 的 credible interval 是否显著大于 0）来回答教练质疑。

**Step 5｜反转预测与 memo（对应任务 3）**

* 定义“反转事件”：(\mathbb E[M_t]) 过零或 (\Pr(M_{t+1}M_t<0)) 超阈值；
* 输出“反转风险概率 + 触发因素贡献”（serve_form、fatigue 在转移方程里的作用）。

**Step 6｜外推与泛化讨论（对应任务 4）**

* 在其他比赛跑一遍，比较哪些比赛 (\rho) 更大/更小、哪些因素更关键，并讨论为何（场地、球员风格、数据缺失等）。

---

如果你愿意，我可以下一步直接把 **“DBN + VI 的训练伪代码（逐分过滤）+ 反转检测定义 + 需要画的 6 张图”**一口气给出来，保证你能按这个路线写进 25 页论文里。