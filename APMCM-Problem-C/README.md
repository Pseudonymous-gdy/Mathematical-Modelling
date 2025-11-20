## Model Summary (English)

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
