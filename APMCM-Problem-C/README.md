## Model Summary 1 (English)

### 1. Basic structure: weight × price

For each year $t$ and exporter $i \in \{\mathrm{US}, \mathrm{AR}, \mathrm{BR}\}$, we model physical trade weight as

$$
\text{Weight}_{i,t} = \text{Weight}_{\text{total},t} \cdot p_{i,t},
$$

where $p_{i,t}$ is the share of country $i$ in the total imported weight.  
Trade value is then

$$
\text{Value}_{i,t} = \text{Price}_{i,t} \cdot \text{Weight}_{i,t}.
$$

---

### 2. Time-series block: autoregression

Both the total weight and country-specific prices are modeled via autoregressive (AR) time-series models over 2015–2025:

- $\ln \text{Weight}_{\text{total},t}$
- $\ln \text{Price}_{i,t}$

Each is regressed on its own lags (and possibly common macro controls).  
These AR equations capture the temporal evolution of the overall import scale and of price levels.

---

### 3. Share block $p_i$: gravity-style log-linear regression (US as benchmark)

We take the US as the reference country and describe the market share $p_{i,t}$ using a gravity-type log-linear model.

For $i \in \{\mathrm{AR}, \mathrm{BR}\}$ we write

$$
\ln \frac{p_{i,t}}{p_{\mathrm{US},t}}
= \alpha_i + \beta_1 \ln\big(\text{US total tariffs on China}_t\big) + \beta_2 \ln\Bigg(\frac{\text{final production}_i}{\text{final production}_{\mathrm{US}} + \text{final production}_{\mathrm{BR}} + \text{final production}_{\mathrm{AR}}}\Bigg)_t + \beta_3 \ln\big(\text{China's soybean tariff on US}_t\big) + \beta_4 \varepsilon_{i,t}.
$$

Here:

- $\alpha_i$ is a country-specific bias (fixed effect);
- $\ln(\text{US total tariffs on China}_t)$ captures the overall US tariff policy against China;
- the production term is the log of the production ratio of country $i$ among the three exporters;
- $\ln(\text{China's soybean tariff on US}_t)$ is China’s retaliatory tariff on US soybeans;
- $\varepsilon_{i,t}$ is the elasticity term (see Section 5).

This corresponds to a gravity equation written in terms of **log share ratios** rather than log trade volumes.

---

### 4. Bias terms and identification

Originally, each country has its own bias term.  
However, when we move to the log-ratio form

$$
\ln \frac{p_{i,t}}{p_{\mathrm{US},t}},
$$

only **relative** biases are identified.  
We can set the US bias to zero and estimate only two intercepts $\alpha_{\mathrm{AR}}$ and $\alpha_{\mathrm{BR}}$ instead of three.  
This is standard in multinomial logit / gravity estimation where one category is treated as the benchmark.

---

### 5. Elasticity term $\dfrac{\partial \text{Weight}_i}{\partial \text{Price}_i}$

The term $\dfrac{\partial \text{Weight}_i}{\partial \text{Price}_i}$, represents the **elasticity** of import demand:

$$
\varepsilon_{i,t} = \dfrac{\partial \text{Weight}_i}{\partial \text{Price}_i},
$$

which measures how sensitive China’s imported quantity from country $i$ is to changes in that country’s export price.  
Economically, this is a **trade (import demand) elasticity**, describing the balance / bargaining power between importer and exporter: a higher absolute elasticity means that a small price increase leads to a large reduction in import quantity.

---

## 模型总结（中文）

### 1. 基本结构：数量 × 价格

对每一年 $t$，以及出口国 $i \in \{\text{美国}, \text{阿根廷}, \text{巴西}\}$，先对物理进口量建模：

$$
\text{Weight}_{i,t} = \text{Weight}_{\text{total},t} \cdot p_{i,t},
$$

其中 $p_{i,t}$ 表示国家 $i$ 在三国总进口量中的份额。  
对应的贸易额为

$$
\text{Value}_{i,t} = \text{Price}_{i,t} \cdot \text{Weight}_{i,t}.
$$

---

### 2. 时间序列部分：自回归

总量和价格都采用自回归（AR）模型刻画 2015–2025 年的时间演化：

- $\ln \text{Weight}_{\text{total},t}$
- $\ln \text{Price}_{i,t}$

分别对各自的滞后项（以及可能的共同宏观变量）回归，  
用于描述整体进口规模与价格水平随时间的平滑变化。

---

### 3. 份额部分 $p_i$：以美国为基准的重力型对数线性回归

选择美国为基准国，用重力模型思想的对数线性回归刻画国家份额 $p_{i,t}$。

对 $i \in \{\text{阿根廷}, \text{巴西}\}$，有

$$
\ln \frac{p_{i,t}}{p_{\mathrm{US},t}}
= \alpha_i + \beta_1 \ln\big(\text{美国对中国的总体关税}_t\big) + \beta_2 \ln\Big(\text{三国产量中属于 i 的比例}_t\Big) + \beta_3 \ln\big(\text{中国对美国大豆加征的关税}_t\big) + \beta_4 \varepsilon_{i,t}.
$$

其中：

- $\alpha_i$ 为国家 $i$ 的偏置项（固定效应）；
- $\ln(\text{美国对中国的总体关税}_t)$ 刻画美国对中国的总体关税水平；
- 产量项为国家 $i$ 在三国大豆最终产量中的占比的对数；
- $\ln(\text{中国对美国大豆加征的关税}_t)$ 为中国对美国大豆的报复性关税；
- $\varepsilon_{i,t}$ 为弹性项（见第 5 点）。

可以理解为：我们把传统重力模型中“$\ln X$ 回归”的右侧变量，转写到了“$\ln (p_i / p_{\mathrm{US}})$” 的回归中。

---

### 4. 偏置项与识别问题

原始设定中，对美国、阿根廷、巴西各有一个偏置项。  
但在写成

$$
\ln \frac{p_{i,t}}{p_{\mathrm{US},t}}
$$

的形式后，只能识别相对偏差，因此可以将美国的偏置设为 0，  
仅估计阿根廷和巴西的两个截距 $\alpha_{\mathrm{AR}}$ 和 $\alpha_{\mathrm{BR}}$。  
这与多项 logit 与重力方程的标准做法一致。

---

### 5. 弹性项 $\dfrac{\partial \text{Weight}_i}{\partial \text{Price}_i}$ 的含义

$\dfrac{\partial \text{Weight}_i}{\partial \text{Price}_i}$代表进口需求弹性：

$$
\varepsilon_{i,t} = \dfrac{\partial \text{Weight}_i}{\partial \text{Price}_i},
$$

用于衡量“中国自国家 $i$ 的进口量”对“该国出口价格”变化的敏感程度。  
在国际贸易与重力模型中，这类贸易弹性是分析关税或价格冲击如何传导到双边贸易流量变化的关键参数，也可以理解为进口方与出口方之间“议价能力 / 平衡”的量化指标。

# 第二问第一阶段建模总结 / Stage I Model Summary for Question 2

---

## 一、中文版（用于正文）

### 1. 研究范围与总体思路

* 只考虑 **美国（US）–日本（JP）–墨西哥（MX）** 三者之间的汽车贸易与生产互动。
* 目标：

  1. 在给定/预测的关税路径下，预测日本与墨西哥对美汽车进口额及份额；
  2. 从进口额反推三国汽车产量；
  3. 在此基础上，为日本构造一个“**有限调整幅度下的供应链优化问题**”，体现日本的对策空间；
  4. 利用美国“**46% 的汽车为进口**”这一事实来标定美国本土产量与进口之间的关系，从而推导美国本土产量。

---

### 2. 国别层：份额模型与进口额预测

**(1) 变量与数据**

* 时间：$t = 2020,\dots,2024$，2025 为预测年。
* 国家：$i \in {\text{US}, \text{JP}, \text{MX}}$。
* $M_{i,t}$：美国自国家 $i$ 进口的汽车贸易额（Customs Value），数据来自 USITC DataWeb（HS 87 或 8703）。
* 总进口额：
  $$M_{\text{tot},t} = \sum_{i\in{\text{JP},\text{MX}}} M_{i,t} \quad (\text{只统计进口来源国}).$$
* 份额：
  $$p_{i,t} = \frac{M_{i,t}}{M_{\text{tot},t}}, \quad i\in{\text{JP},\text{MX}}.$$
* 有效关税率：
  $$\text{tariff}*{i,t} = \frac{\text{Duties}*{i,t}}{\text{Customs Value}_{i,t}},$$
  同样由 DataWeb 导出。

**(2) 模型形式**

我们对每个出口国 $i\in{\text{JP},\text{MX}}$ 的对美份额对数建模：

$$
\boxed{
y_{i,t} \equiv \ln p_{i,t}
==========================

\alpha_i

* \rho, y_{i,t-1}
* \beta, \ln(1+\text{tariff}_{i,t})
* \varepsilon_{i,t}
  }
  $$

- $\alpha_i$：国家固定效应（距离、品牌偏好、历史关系等隐含其中）。
- $\rho$：份额惯性（上一年份额的影响）。
- $\beta<0$：关税项的平均弹性——关税提高，份额下降。
- $\varepsilon_{i,t}$：噪声项。

在样本极少的情况下：

* 可以通过 OLS 粗略估计 $(\alpha_i,\rho,\beta)$；
* 也可以用文献与第一问中的弹性结果，对 $\beta$ 做标定并在一定区间内做敏感性分析。

**(3) 预测 2025 年份额与进口额**

1. 在题目给定的“新关税安排”下，设定 2025 年的 $\text{tariff}_{i,2025}$。
2. 利用上式得到：
   $$
   \hat y_{i,2025}
   = \hat\alpha_i

   * \hat\rho, y_{i,2024}
   * \hat\beta, \ln(1+\text{tariff}_{i,2025}).
     $$
3. 恢复份额：
   $$
   \hat p_{i,2025} = \exp(\hat y_{i,2025}), \quad
   \text{再归一化使 } \sum_{i\in{\text{JP},\text{MX}}} \hat p_{i,2025}=1.
   $$
4. 用一个简单自回归（AR(1) 或 ARIMA）预测总进口额：
   $$
   M_{\text{tot},t} = \phi_0 + \phi_1 M_{\text{tot},t-1} + u_t,
   $$
   得到 $\hat M_{\text{tot},2025}$。
5. 于是：
   $$
   \boxed{M_{i,2025} = \hat p_{i,2025} \cdot \hat M_{\text{tot},2025}},
   \quad i\in{\text{JP},\text{MX}}.
   $$

---

### 3. 反推三国产量：$Q_{i,t}$ 的定义与标定

为了把进口额与本国生产联系起来，我们引入“**出口到美国在本国生产中的占比**”$s_{i,t}$：

* $s_{i,t}$：美国市场在国家 $i$ 汽车总产量中的份额，例如
  $s_{\text{JP},t} \approx \frac{\text{对美出口产量}}{\text{日本总产量}}$。

我们利用之前拟合的供应链结构表（即“基期中日本出口/北美本地生产的占比拟合表”）对 $s_{i,t}$ 做标定或构造合理数值区间。

于是对 $i \in {\text{JP},\text{MX}}$，有：

$$
\boxed{
Q_{i,t} = \frac{M_{i,t}}{s_{i,t}}
}
$$

* 其中 $Q_{i,t}$ 可以理解为“为全球市场生产、其中一部分流向美国”的总产量 proxy。
* $s_{i,t}$ 越大，意味着该国对美国市场依赖度越高。

**注意：**

* 对于美国本身，我们不直接用 $Q_{i,t} = M_{i,t}/s_{i,t}$ 的形式，而是单独处理（见第 5 点）。

---

### 4. 日本–墨西哥–美国路径上的优化问题（结构搭建）

在得到日本和墨西哥对美出口规模 $M_{\text{JP},2025}$、$M_{\text{MX},2025}$ 后，我们进一步只关注 **日本→美国** 这部分贸易，构造日本在三条路径上的决策：

* $x_1$：日本本土生产 → 直接出口到美国（JP→US）；
* $x_2$：在墨西哥生产（或经墨西哥中转） → 美国（JP→MX→US）；
* $x_3$：日本车企在美国本土工厂生产 → 美国销售（JP–FDI–US）。

令 $Q_{\text{JP},2025}^{\text{US}}$ 为第 2.3 节得到的日本对美总出口量（或将 $M_{\text{JP},2025}$ 按平均车价换算为产量），满足：

$$
x_1 + x_2 + x_3 = Q_{\text{JP},2025}^{\text{US}}.
$$

我们构造三条路径的**相对单位成本** $c_1,c_2,c_3$（“生产 + 运输 + 关税”的合成指数），例如：

* $c_1$：日本本土生产 + 远洋运输 + 新关税（最高）；
* $c_2$：墨西哥生产（工资低） + 短运输 + USMCA 下低关税（最低）；
* $c_3$：美国本地生产（工资高） + 低运输 + 无关税（介于两者之间）。

单位成本的具体数值可取：

* $c_1 = 115,\ c_2 = 77,\ c_3 = 95$
  （前面我们已构造并给出详细经济解释：日本直出最贵，墨西哥路线最便宜，美国本地居中）。

同时，为避免“供应链一夜大搬家”，加入**调整幅度约束**：

$$
|x_r - x_{r,2024}| \le 0.05,x_{r,2024}, \quad r=1,2,3,
$$

即每条路径在一年内调整不超过原有规模的 5%。

于是日本的对策优化问题为：

$$
\begin{aligned}
\min_{x_1,x_2,x_3} \quad & c_1 x_1 + c_2 x_2 + c_3 x_3 \
\text{s.t.} \quad
& x_1 + x_2 + x_3 = Q_{\text{JP},2025}^{\text{US}}, \
& |x_r - x_{r,2024}| \le 0.05 x_{r,2024}, \quad r=1,2,3, \
& x_r \ge 0.
\end{aligned}
$$

> 这一阶段我们重点是**搭建优化结构并标定 $c_1,c_2,c_3$ 与约束形式**，真正求解可以放在第二阶段（情景分析）完成。

---

### 5. 美国本土产量 $Q_{\text{US},t}$ 的标定与推导

根据最近关于美国汽车进口的报道：

> “In 2024, 46% of cars sold in the U.S. were imported.”

这意味着：

* 2024 年美国汽车销量 $D_{2024}$ 中：

  * 进口车占比约 $0.46$，
  * 本土生产车占比约 $0.54$。

记 2024 年美国汽车**进口总量**为 $M_{\text{tot},2024}$，则有：

$$
\frac{M_{\text{tot},2024}}{D_{2024}} \approx 0.46,
\quad
\frac{Q_{\text{US},2024}}{D_{2024}} \approx 0.54.
$$

从而可解得：

$$
Q_{\text{US},2024}
\approx
\frac{0.54}{0.46} M_{\text{tot},2024}.
$$

在此基础上，我们可以：

1. 用一个简单的时间序列模型（如 AR(1) 或固定增速）预测 2025 年美国总销量 $D_{2025}$；
2. 在不同情景下，利用第一层得到的 2025 年进口总量 $M_{\text{tot},2025}$；
3. 由恒等式：
   $$
   Q_{\text{US},2025} = D_{2025} - M_{\text{tot},2025}
   $$
   推导美国本土产量；
4. 再用 BLS/FRED 的就业–产量关系（如 $L_{\text{US},t} = \beta_{\text{emp}} Q_{\text{US},t}$）将产量变化传导为就业变化（这一部分可在第二阶段详细展开）。

---

### 6. 第一阶段小结（中文）

* **国家范围**：暂时只聚焦美国、日本、墨西哥三国，简化全球环境；
* **国别份额模型**：用一个小样本友好的自回归–关税弹性模型预测日本、墨西哥在美国汽车进口中的份额及规模；
* **生产量反推**：通过“出口到美国占本国产量的份额” $s_{i,t}$ 将进口额 $M_{i,t}$ 反推各国产量 $Q_{i,t}$；
* **日本对策优化**：构造三条路径（直出日本、墨西哥、中美本地生产）的成本结构和线性规划问题，并加入“年度调整不超过 5%”的约束，体现现实中的调整摩擦；
* **美国本土产量标定**：利用 2024 年“46% 为进口车”的事实标定 $Q_{\text{US},2024}$，并在不同情景下通过“总销量–总进口”关系推导 2025 年的美国本土产量，为后续就业分析打下基础。

这一阶段的任务是**把第 2 问的结构模型框架搭好、参数和约束形式说清楚**，第二阶段再在此基础上进行具体情景（如高关税、谈判降税、日本调整供应链）的求解与比较。

---

## II. English Version (for report / appendix)

### 1. Scope and overall idea

* We focus on interactions among **the U.S. (US), Japan (JP), and Mexico (MX)** in the automotive sector.
* Stage I of Question 2 does three things:

  1. Model how **tariffs affect the U.S. import shares and volumes** from Japan and Mexico;
  2. Back out implied **production volumes** in each country from the import data;
  3. Set up an **optimization problem for Japan’s supply-chain response** under limited adjustment;
  4. Calibrate **U.S. domestic production** using the fact that **46% of cars sold in the U.S. in 2024 were imported**.

---

### 2. Country-level share model and import forecasts

**(1) Variables and data**

* Time: $t = 2020,\dots,2024$, with 2025 as the forecast year.
* Countries: $i \in {\text{US}, \text{JP}, \text{MX}}$.
* $M_{i,t}$: U.S. imports of vehicles (Customs Value) from country $i$, obtained from USITC DataWeb (HS 87 or 8703).
* Total imports:
  $$
  M_{\text{tot},t} = \sum_{i\in{\text{JP},\text{MX}}} M_{i,t}.
  $$
* Import shares:
  $$
  p_{i,t} = \frac{M_{i,t}}{M_{\text{tot},t}},\quad i\in{\text{JP},\text{MX}}.
  $$
* Effective tariff rates:
  $$
  \text{tariff}*{i,t} = \frac{\text{Duties}*{i,t}}{\text{Customs Value}_{i,t}},
  $$
  constructed from the duties and customs value in DataWeb.

**(2) Model specification**

For each exporter $i\in{\text{JP},\text{MX}}$ we model the log share as:

```math
y_{i,t} \equiv \ln p_{i,t}
=
\alpha_i
+ \rho\, y_{i,t-1}
+ \beta\, \ln(1+\text{tariff}_{i,t})
+ \varepsilon_{i,t}.
```

* $\alpha_i$: country fixed effects (distance, brand preference, FTA, etc.).
* $\rho$: persistence in import shares.
* $\beta<0$: average tariff elasticity (higher tariffs → lower import share).
* $\varepsilon_{i,t}$: error term.

Given the extremely small sample, we can either:

* run a simple OLS to get rough estimates; or
* impose a **calibrated range** for $\beta$ based on literature and Question 1, and use the model mainly for scenario analysis.

**(3) Forecasting 2025 shares and imports**

1. Under the new tariff schedule, specify $\text{tariff}_{i,2025}$.
2. Compute
   $$
   \hat y_{i,2025}
   = \hat\alpha_i + \hat\rho, y_{i,2024}

   * \hat\beta \ln(1+\text{tariff}_{i,2025}).
     $$
3. Recover predicted shares and renormalize:
   $$
   \hat p_{i,2025} = \exp(\hat y_{i,2025}),\quad
   \sum_{i\in{\text{JP},\text{MX}}} \hat p_{i,2025}=1.
   $$
4. Forecast total imports using an AR model:
   $$
   M_{\text{tot},t} = \phi_0 + \phi_1 M_{\text{tot},t-1} + u_t,
   $$
   obtaining $\hat M_{\text{tot},2025}$.
5. Then
   $$
   M_{i,2025} = \hat p_{i,2025}, \hat M_{\text{tot},2025},
   \quad i\in{\text{JP},\text{MX}}.
   $$

---

### 3. Backing out production volumes $Q_{i,t}$

To link imports to production, we define for $i\in{\text{JP},\text{MX}}$:

* $s_{i,t}$: share of country $i$’s vehicle production that is sold in the U.S. market, i.e.,
  $s_{i,t} \approx \text{(shipments to U.S.)} / \text{(total production of }i\text{)}$.

Using our **calibrated supply-chain table** (from earlier steps) we assign plausible values or ranges for $s_{i,t}$.

Then:

```math
Q_{i,t} = \frac{M_{i,t}}{s_{i,t}},
\quad i\in\{\text{JP},\text{MX}\}.
```

$Q_{i,t}$ is interpreted as an implied production level consistent with the observed U.S.-bound flows and the assumed export share $s_{i,t}$.

> The U.S. is treated differently (see Section 5).

---

### 4. Japanese supply-chain optimization across three routes

Given the total volume of Japanese vehicles sold in the U.S., $Q_{\text{JP},2025}^{\text{US}}$, we consider three supply routes:

* $x_1$: produced in Japan and exported directly to the U.S. (JP→US);
* $x_2$: produced in or routed via Mexico, then exported to the U.S. (JP→MX→US);
* $x_3$: produced in Japanese-owned plants in the U.S. (JP–FDI–US).

They satisfy:

```math
x_1 + x_2 + x_3 = Q_{\text{JP},2025}^{\text{US}}.
```

We construct **relative unit costs** $(c_1,c_2,c_3)$ combining production, transportation, and tariff components:

* $c_1$: highest (Japan → U.S., long-distance shipping + new higher tariff);
* $c_2$: lowest (Mexico route, lower wages + shorter shipping + near-zero USMCA tariff);
* $c_3$: intermediate (U.S. local assembly, high wages but no tariff and low shipping).

A plausible numerical example (in abstract cost units) is:

```math
c_1 = 115,\quad c_2 = 77,\quad c_3 = 95.
```

To avoid unrealistic one-year shifts, we impose **adjustment constraints**:

```math
|x_r - x_{r,2024}| \le 0.05\, x_{r,2024}, \quad r=1,2,3,
```

i.e. each route can only adjust by at most 5% of its baseline level.

**Optimization problem (cost minimization):**

```math
\begin{aligned}
\min_{x_1,x_2,x_3} \quad & 115 x_1 + 77 x_2 + 95 x_3 \\
\text{s.t.} \quad
& x_1 + x_2 + x_3 = Q_{\text{JP},2025}^{\text{US}}, \\
& |x_r - x_{r,2024}| \le 0.05 x_{r,2024}, \quad r=1,2,3, \\
& x_r \ge 0.
\end{aligned}
```

Stage I focuses on **specifying the structure and parameters** of this optimization; solving it under different policy scenarios (e.g. with/without adjustment) belongs to Stage II.

---

### 5. U.S. domestic production $Q_{\text{US},t}$

A recent report on the U.S. auto market states:

> “In 2024, 46% of cars sold in the U.S. were imported.”

Thus:

* In 2024, if total sales are $D_{2024}$,
  $$
  \frac{M_{\text{tot},2024}}{D_{2024}} \approx 0.46,\quad
  \frac{Q_{\text{US},2024}}{D_{2024}} \approx 0.54.
  $$

This implies:

```math
Q_{\text{US},2024}
\approx \frac{0.54}{0.46}\, M_{\text{tot},2024}.
```

For 2025 we proceed as follows:

1. Forecast total U.S. vehicle sales $D_{2025}$ (e.g. AR(1) or a fixed growth rate);
2. Use the country-level model to obtain $M_{\text{tot},2025}$ under each tariff scenario;
3. Compute domestic production:

```math
Q_{\text{US},2025} = D_{2025} - M_{\text{tot},2025}.
```

In later stages, $Q_{\text{US},t}$ can be linked to U.S. automotive **employment** via a simple linear relation calibrated from BLS/FRED data, e.g.

```math
L_{\text{US},t} = \beta_{\text{emp}}\, Q_{\text{US},t},
```

with $\beta_{\text{emp}}$ calibrated using 2020–2024 employment and assembly data.

---

### 6. Stage I summary (English)

* We **restrict the country set** to the U.S., Japan, and Mexico to keep the structure tractable.
* We specify a **small-sample-friendly log-share model** linking tariffs to U.S. import shares and volumes for Japan and Mexico.
* We **back out production volumes** $Q_{i,t}$ for Japan and Mexico using assumed export shares $s_{i,t}$ calibrated from our supply-chain table.
* We formulate a **linear cost-minimization problem** for Japan’s supply-chain response across three routes (JP→US, JP→MX→US, JP–FDI–US), with a realistic **5% annual adjustment cap** on each route.
* We **calibrate U.S. domestic production** using the empirical fact that 46% of cars sold in the U.S. in 2024 were imported, and we use the identity $Q_{\text{US},t} = D_t - M_{\text{tot},t}$ to obtain 2025 domestic production under each scenario.

Stage I thus delivers a **coherent structural framework** for Question 2; Stage II will plug in specific tariff scenarios, solve the optimization problem, and evaluate the implications for Japanese trade patterns and U.S. automotive production and employment.


