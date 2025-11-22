# 第二问第一阶段建模总结 / Stage I Model Summary for Question 2
显式为

$$
y_{i,t}=\ln p_{i,t}:=\alpha_i+\rho y_{i,t-1}+\beta\ln(1+\text{tariff}_{i,t})
$$

计算 $M_{i,2025}=p_{i,2025}M_{2025}^{tot}$ ，其中 $M_{2025}^{tot}$ 可由自回归得到，意思为各国出口车的贸易总量。

计算 $Q_{i,t}=\frac{M_{i,t}}{s_{i,t}}$ ，其中 $s_{i,t}$ 是国家在本国生产量除以国家实际生产量。

实现优化问题：
<img width="872" height="280" alt="image" src="https://github.com/user-attachments/assets/9d18d407-78f2-478c-a3a0-569186a8f13f" />


# 第三问建模总结
先把关键问题说清楚，然后给你一份可以直接当“第三问—数据与模型准备”写进论文里的框架。

---

## 一、有没有“高 / 中 / 低端芯片”的数据？HTS 怎么对应？

### 1.1 现实情况：没有“现成的高中低端表”，但可以靠 HTS 自己划分

现实世界里关税、贸易统计用的是 **HS/HTS 商品编码**，并不会直接标注 “high-end / mid-end / low-end chips”。
绝大部分芯片归在：

* **HS 8541**：二极管、晶体管等离散器件
* **HS 8542**：电子集成电路（Electronic integrated circuits）

在 8542 下面有你能在各种 HTS/HS 资料里查到的 6 位子目：([外贸在线][1])

* **854231**：Processors and controllers, electronic integrated circuits
* **854232**：Memories, electronic integrated circuits
* **854233**：Amplifiers, electronic integrated circuits
* **854239**：Other electronic integrated circuits
* **854290**：Parts of electronic integrated circuits

**结论 1（直接回答你）：**

> 没有官方“高中低端芯片数据库”，但有明确的 **按产品类型（处理器 / 存储 / 模拟 / 其他）划分的 HS/HTS 代码**。
> 我们需要 **自定义一套“高中低端”映射规则**，然后去 DataWeb/海关数据里按 HS 代码查贸易额和关税。

### 1.2 比赛中可用的高中低映射（建议方案）

结合 SIA/WSTS 对芯片类型的分类（Logic / Memory / Analog / Discrete 等） 和 HS 8541/8542 的结构，可以在论文里明确写出这样的**建模假设**：

* **高端芯片（High-end）**：

  * 逻辑处理器、控制器、高性能存储等 —— 对应 **854231（processors & controllers）+ 部分 854232（memory）**；
  * 同时与美国出口管制重点对象（“advanced computing chips” / ECCN 3A090 等）对应，在出口管制变量里统一作为高端。([Federal Register][2])

* **中端芯片（Mid-end）**：

  * 模拟放大器、通用模拟器件 —— 主对应 **854233（amplifiers）**；
  * 再加上未被“先进计算”规则直接覆盖的一般存储器（854232 的一部分）。

* **低端芯片（Low-end）**：

  * 各类离散器件（**8541**），以及大量 **854239（other ICs）** 中工艺节点偏老、技术门槛较低的产品。

这套划分的**诚实说法**就是：

> “我们基于产品功能和技术敏感性，将 HS 8541/8542 下的集成电路划分为高、中、低三档。
> 这种映射不是官方标准，但与出口管制重点（高性能逻辑与存储）、产业报告对‘先进 / 传统’的划分是一致的。”

这一点在比赛论文里写清楚，就是一个很好的“建模假设 + 数据分层”加分点。

---

## 二、关税、出口管制时间点、贸易额：可以如何查？

### 2.1 关税数据——USITC DataWeb + Section 301 列表

**数据源：USITC DataWeb – U.S. Trade & Tariff Data**

* DataWeb 是官方的美国贸易与关税数据库，可以按 **国家 + HS/HTS + 时间** 查到进口、出口、基准关税和附加关税。([dataweb.usitc.gov][3])
* 比赛题目第 [6] 条参考文献给的就是这个网站，你们是被鼓励用的。

**Section 301 对华加征关税：**

* 从 2018 年起，美国根据 301 调查，对来自中国的部分商品（按 HS 码列出）加征 **7.5%–25% 的附加关税**，覆盖四个清单（List 1–4A），总规模约 3700 亿美元。([Sandler, Travis & Rosenberg, P.A.][4])
* USITC 发布的 **China Tariffs 表** 中列出包括 **8541、8542** 在内的众多 HS 子目，显示对中国商品加征的额外 25% 或更高税率，并在 2024–2025 年提出将部分半导体项目从 25% 提高到 50%。([hts.usitc.gov][5])

> 对于建模来说，你不必用到所有历史清单，只需要**设定一个“有效关税”变量**：
>
> $$
> Tariff_{k,t}^{US;on;CN}
> = \text{基准关税率}*{k,t} + \text{Section301附加关税率}*{k,t},
> $$
> 
> 用 DataWeb 按 HS 8542/8541 + Partner=China + 2020–2025 提取即可。

**DataWeb 操作流程（可以直接写在“数据来源”小节）：**

1. 注册 DataWeb 免费账户（需要 login.gov，一次性注册）。([dataweb.usitc.gov][3])
2. 在查询界面选择：

   * 数据类型：**U.S. Imports / U.S. Exports**；
   * 维度：Commodity = HS/HTS（输入 8542, 8541 或更细的 854231 等），
     Partner = China，Time = 2020–2025（按年或按月）；
   * 添加字段：Import Value、Export Value、Quantity、General Duty Rate、China 301 Additional Duty（如果有）。
3. 导出为 CSV，在本地用 Python / Excel 进行聚合和分组（按你定义的高/中/低端分组）。

### 2.2 出口管制时间点（高端芯片）

高端芯片的出口管制主要来自美国商务部 BIS 的两轮大规则：([Federal Register][2])

1. **2022 年 10 月 7 日规则**（Advanced Computing and Semiconductor Manufacturing Equipment Rule）：

   * 对“先进计算集成电路”和超级计算机相关用途实施更严格的出口管制；
   * 成立新的 ECCN 类别（如 3A090、3B090），专门针对一定性能指标以上的 GPU/加速器和相关制造设备。

2. **2023 年 10 月升级规则**（Advanced Semiconductors Rule）：

   * 完善和收紧 2022 年规则，引入“数据中心标准”“总处理性能（TPP）阈值”等，进一步界定何种 AI 芯片属于“受控高端芯片”。

在模型里，你可以定义一个非常简单但清晰的 **出口管制哑变量**：

* $EC_t^{(1)} = 0$（2022Q3 及以前）， $EC_t^{(1)} = 1$ （2022Q4 起）；
* 如要更精细：在 2023Q4 再定义 $EC_t^{(2)}$ 表示更严的一轮规则。

这两个时间点非常容易在论文里画纵线（time line）+ 在回归中作为事件变量使用。

### 2.3 贸易额数据（中美双边 + 全球对照）

**核心数据：仍然来自 DataWeb。**

* 查询维度：

  * Imports from China（美国自中国进口： $M_{HS, CN→US, t}$ ），
  * Exports to China（美国对中国出口： $X_{HS, US→CN, t}$ ），
  * Exports to World（美国对世界出口： $X_{HS, US→World, t}$ ）。
* 用你上面定义的高中低端 HS 分组，把这些值加总成：

  * $X_{H,US\to CN,t}, X_{M,US\to CN,t}, X_{L,US\to CN,t}$
  * $M_{H,CN\to US,t}, M_{M,CN\to US,t}, M_{L,CN\to US,t}$
  * 以及相应的对世界出口（用于做 share）。

**国内制造强度 & 价格：来自 FRED**

* **半导体工业生产指数**：Industrial Production: Semiconductor and Other Electronic Component (NAICS 3344)([FRED][6])
* **就业人数**：All Employees, Semiconductor and Other Electronic Component Manufacturing (CES3133440001)([FRED][7])
* **生产者价格指数（PPI）**：Producer Price Index by Industry: Semiconductor and Other Electronic Component Manufacturing([FRED][8])

这些时间序列从 1980s 到 2025 都有，你只需要截取 2015–2025，对应题目给的数据时间段即可。

---

## 三、能不能把回归只做在“中美芯片贸易”上？

**可以，且是合理、说得过去的简化。**

理由可以在论文里分三点写：

1. **政策焦点就是“对华高端芯片出口管制 + 对华关税”**

   * 题目在第三问中特别强调：对中国的高端芯片出口管制 + 用关税替代补贴的争论，本质是围绕中国这个“主要竞争对手”。
   * 从国家安全角度，关键是“美国对中国的高端芯片供给”和“中国对美国高端芯片的依赖”。

2. **数据成本考虑：中美双边 + 少量全球指标**

   * 若要完整世界多国回归，数据工作量暴涨；
   * 用 **全球总需求**（来自 SIA/WSTS 的 global sales by segment）作为控制变量，
     再聚焦于 **美国 vs 中国的双边贸易流**，既保留了全球趋势，又大幅降低数据成本。

3. **模型层面：用“中美高端贸易 + 美国总制造强度”就能回答题意**

   * 题目问的是：关税政策对 **美国国内制造** 和 **高 / 中 / 低端芯片贸易** 的影响；
   * 你只要展示：

     * 高端对华出口量 / 占比如何随关税 + 出口管制变化；
     * 美国总半导体生产指数 / 就业是否有显著响应；
   * 就已经足够支撑“经济效率 + 国家安全”的讨论。

在论文里可以明确声明：

> “鉴于题目第三问的政策焦点主要在于美国与中国之间的高端芯片贸易与出口管制，
> 我们将回归分析限缩在中美双边贸易部分，并用全球市场指标作为外生控制变量，以在有限时间与数据获取成本下，
> 捕捉关税与出口管制的主要经济与安全效应。”

---

## 四、完整的“数据流获取 + 模型建立”报告（可以直接改写成论文一节）

下面这一段你可以几乎照搬成 “3.1–3.4 数据与模型准备（Question 3）”。

---

### 4.1 数据需求与分层定义

**目标：**
构建一个以 **中美芯片贸易** 为核心的经验模型，分析 **2020–2025 年** 期间美国关税与高端芯片出口管制对：

* 美国半导体制造强度（产出、就业）；
* 高 / 中 / 低端芯片中美双边进出口结构；

的影响，并据此构造 **经济效率指数 E_t** 和 **国家安全指数 S_t**。

**分层定义：**

* 高端芯片 (k=H)：HS 854231 + 部分 854232（逻辑处理器与高端存储）；
* 中端芯片 (k=M)：HS 854233 + 余下较为通用的 854232；
* 低端芯片 (k=L)：HS 8541 所有 + 854239 其余集成电路。

（在正文里说明这是基于产品功能和管制敏感性的一种合理划分。）

---

### 4.2 数据来源与获取流程

**(1) 中美芯片贸易与关税：USITC DataWeb**([dataweb.usitc.gov][3])

* 网站：DataWeb – U.S. Trade & Tariff Data（题目[6]引用）。

* 对每个 HS 代码 $h\in{8541*,8542*}$ ，提取：

  * 美国自中国进口：Import Value, Quantity（2020–2025, annual 或 quarterly）；
  * 美国对中国出口：Export Value, Quantity；
  * General Duty Rate（基准关税）；
  * Section 301 Additional Duty（对华附加关税）。([hts.usitc.gov][5])

* 将这些 HS 代码按上文高中低分组，加总得到：

  * $X_{k,US\to CN,t}$ 、 $M_{k,CN\to US,t}$ ；
  * 以及对应的加权平均关税率 $Tariff_{k,t}^{US;on;CN}$ 。

**(2) 美国半导体制造与价格：FRED**([FRED][6])

* 工业生产指数 (IP_t)：Industrial Production: Semiconductor and Other Electronic Component（IPG3344S）；
* 就业人数 (EMP_t)：All Employees, Semiconductor and Other Electronic Component Manufacturing（CES3133440001）；
* 生产者价格指数 (PPI_t)：PPI by Industry: Semiconductor and Other Electronic Component Manufacturing。

将月度数据转成年度或季度平均，统一时间尺度。

**(3) 全球需求与产品结构：SIA / WSTS 报告**

* 全球半导体销售额 $G_{total,t}$ ；
* 按 Logic / Memory / Analog / Discrete 的销售占比，构造 $G_{H,t},G_{M,t},G_{L,t}$ ；
* 美国企业全球市占率、地区市场份额，用于国家安全指标的背景。

**(4) 出口管制时间点：BIS 与说明文档**([Federal Register][2])

* 2022-10-07：高级计算与半导体制造设备规则生效；
* 2023-10：Advanced Semiconductors Rule 发布，完善与收紧前述规则；
* 将其编码为 $EC_t^{(1)}, EC_t^{(2)} \in {0,1}$ 的事件变量，用于高端芯片出口回归。

---

### 4.3 变量构造与预处理

1. **贸易变量：**

   * 对每个 $k\in{H,M,L}$ ：
   
$$
X_{k,t} = \sum_{h\in\mathcal{H}*k} X*{h,US\to CN,t},\quad M_{k,t} = \sum_{h\in\mathcal{H}*k} M*{h,CN\to US,t}.
$$
   
   * 对世界出口：

$$
X_{k,t}^{World} = \sum_{h\in\mathcal{H}*k} X*{h,US\to World,t},
$$
     
     并构造对华出口份额

$$
s_{k,t}^{CN} = \frac{X_{k,t}}{X_{k,t}^{World}}.
$$

3. **有效关税：**

   * 对每个 HS 码，构造

$$
Tariff_{h,t}^{US;on;CN} = GeneralDuty_{h,t} + Sec301_{h,t},
$$
     
     再对组内加权平均得到 $Tariff_{k,t}^{US;on;CN}$ 。

4. **美国制造强度：**

   * 设 $Y_t$ 为工业生产指数 IPG3344S 或就业人数 CES3133440001；
   * 如需高中低拆分，可用 SIA/WSTS 的逻辑 / 存储 / 模拟比例进行近似分配（简单乘权重即可）。

5. **经济效率与国家安全指标（为后续 E_t, S_t 做准备）：**

   * 经济效率：

     * 总芯片消费： $I_1 = \sum_k (X_{k,t} + M_{k,t})$ 或美国对芯片的表观消费量；
     * 价格负担： $I_2 = PPI_t$ ；
     * 产业增加值 / 贸易差额： $I_3 = \sum_k (X_{k,t} - M_{k,t})$ 。

   * 国家安全：

     * 高端自给率： $S_1 = \frac{Y_{H,t}}{Y_{H,t} + M_{H,t}}$（可用生产 + 进口近似）；
     * 高端市占率： $S_2 = \frac{X_{H,t}^{World}}{G_{H,t}}$ ；
     * 高端对华出口依赖： $S_3 = \frac{X_{H,t}}{X_{H,t}^{World}}$ （负向）；
     * 高端对华进口依赖： $S_4 = \frac{M_{H,t}}{\sum_k M_{k,t}}$ （负向）。

6. **标准化与权重：**

   * 对 $I_j,S_j$ 做极差或 z-score 标准化；
   * 用 **熵权法** 和 **PCA** 分别求出权重，构造

$$
E_t^{(entropy)}, E_t^{(PCA)},\quad
S_t^{(entropy)}, S_t^{(PCA)}.
$$

---

### 4.4 回归模型设定（只限中美部分）

**(A) 美国半导体制造强度的回归模型**

对总制造强度 $Y_t$（或高端子项 $Y_{H,t}$）设定：

$$
\ln Y_t = \beta_0 + \beta_1 \ln G_{total,t} + \beta_2 PPI_t + \beta_3 Tariff_{H,t}^{US;on;CN} + \beta_4 EC_t^{(1)} + \beta_5 EC_t^{(2)} + \varepsilon_t.
$$

- $\beta_1$ ：全球需求对美国制造的拉动；
- $\beta_2$ ：价格上升对产出的抑制；
- $\beta_3$ ：对华关税对美国高端制造的影响（符号不确定，可讨论“保护 vs 成本”）；
- $\beta_4,\beta_5$ ：出口管制事件对制造强度的影响。

**(B) 高端芯片对华出口份额的 logit 回归**

$$
\text{logit}(s_{H,t}^{CN}) = \ln\frac{s_{H,t}^{CN}}{1 - s_{H,t}^{CN}} = \gamma_0 + \gamma_1 Tariff_{H,t}^{US;on;CN} + \gamma_2 EC_t^{(1)} + \gamma_3 EC_t^{(2)} + \gamma_4 \ln G_{H,t} + u_t.
$$

- 主要看 $\gamma_1,\gamma_2,\gamma_3$ 的符号与显著性，解释“高端对华出口如何随关税与管制收缩”。

**(C) 中 / 低端芯片贸易的对比回归**

同样可以对 $X_{M,t}, X_{L,t}$ 或对应份额做简单 OLS，为论文提供：

* “高端 vs 中低端在关税与管制面前的差异反应”——支撑国家安全的论点：

  * 高端对华出口对管制更敏感，中低端主要受全球需求与一般关税影响。

---

### 4.5 经济效率与国家安全指数的构造与比较

在得到回归结果后，你们可以做一个简单的**情景模拟**：

* 基线情景：维持当前关税与出口管制作水平（参照 2025 前状态）；
* 情景 A：提高高端对华关税 $Tariff_{H,t}^{US;on;CN}$ （近似题目中的“更高递进关税”）；
* 情景 B：进一步收紧出口管制（增加 $EC^{(2)}$ ）；
* 情景 C：减缓关税，仅以出口管制为主。

将这些情景代入回归方程，得到每个情景下的 $Y_t,X_{k,t},M_{k,t}$ ，再计算相应的 $E_t,S_t$ ，画出：

* $E_t$ vs $S_t$ 的散点或轨迹图
* 以及不同情景下 (E,S) 的箱线图或雷达图

就可以写出“**哪一类政策组合在经济效率与国家安全之间更平衡**”的结论。

---

## 五、总结一句话帮你定心

* **HTS 级别的数据是有的**：HS/HTS 8541、8542 及其子目，可以在 DataWeb 中按中美双边提取贸易额和关税；
* **“高中低端”需要你自己基于 HS 代码 + 产业逻辑来划分**，这是可接受而且能拿分的建模假设；
* **完全可以把回归任务限缩在中美芯片贸易上**，配合全球需求和美国制造强度作为背景变量，就足够回答题目第三问，并且控制了数据搜集成本；
* 再加上 **熵权 + PCA 的综合指数 + 简单事件回归 / 情景分析**，第三问整体会非常像一篇“带有真实政策味道的 C 题论文”，亮点不少。

[1]: https://www.foreign-trade.com/reference/hscode.htm?code=8542&utm_source=chatgpt.com "HS Code 8542 - Harmonized System ..."
[2]: https://www.federalregister.gov/documents/2022/10/13/2022-21658/implementation-of-additional-export-controls-certain-advanced-computing-and-semiconductor?utm_source=chatgpt.com "Implementation of Additional Export Controls: Certain ..."
[3]: https://dataweb.usitc.gov/?utm_source=chatgpt.com "DataWeb: U.S. Trade & Tariff Data"
[4]: https://www.strtrade.com/trade-news-resources/tariff-actions-resources/section-301-tariffs-on-china?utm_source=chatgpt.com "Section 301 Tariffs on China"
[5]: https://hts.usitc.gov/reststop/file?filename=China+Tariffs&release=currentRelease&utm_source=chatgpt.com "China Tariffs (Last Updated March 6, 2025) Additional ..."
[6]: https://fred.stlouisfed.org/series/IPG3344S?utm_source=chatgpt.com "Industrial Production: Manufacturing: Durable Goods ... - FRED"
[7]: https://fred.stlouisfed.org/series/CES3133440001?utm_source=chatgpt.com "All Employees, Semiconductor and Other Electronic ... - FRED"
[8]: https://fred.stlouisfed.org/series/PCU33443344?utm_source=chatgpt.com "Producer Price Index by Industry: Semiconductor and ... - FRED"
