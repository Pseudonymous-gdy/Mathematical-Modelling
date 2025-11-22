# 关税拉弗曲线分析

下面这一版就是**重写后的第四问建模总结**，只围绕：

*   关税（平均税率）
*   关税收入（revenue）
*   单位价格（unit price）

来做一个 **ARIMAX 形式的 ARIMA**，公式尽量压缩成“一行”。

* * *

1\. 从数据到变量：我们怎么把 Excel 变成时序序列？
------------------------------

我们现在有（按月、2008–2025）的 DataWeb 数据（类似你给我的那几份）：

*   `General Customs Value`：每月总进口额  
     $\Rightarrow$  记为  $\text{CustomsValue}_t$ 
*   `General Import Charges`：每月总的 Import Charges  
     $\Rightarrow$  记为  $\text{ImportCharges}_t$ 
*   `General 1st Unit of Qty`：每月总数量  
     $\Rightarrow$  记为  $\text{Qty}_t$ 
*   （可选）`General CIF Imports Val`：CIF 价值  
     $\Rightarrow$  如果用它来算价格也可以

在第四问中，我们只关心 3 个**核心时间序列**：

### (1) 关税收入（revenue）

直接用 Import Charges 的“Value for: number”那一行：

$$
R_t = \text{ImportCharges}_t
$$

建模时用对数形式：

$$
y_t = \ln R_t
$$

### (2) 平均关税税率（tariff rate）

用“关税收入 ÷ 进口额”的比值做月度平均税率：

$$
\tau_t = \frac{\text{ImportCharges}_t}{\text{CustomsValue}_t}
$$

再做一个平滑的特征（避免 0 问题）：

$$
x_t^{(\tau)} = \ln(1+\tau_t)
$$

### (3) 单位价格（unit price）

用“金额 ÷ 数量”得到平均单位价格的月度序列：

$$
P_t = \frac{\text{CustomsValue}_t}{\text{Qty}_t} \quad\text{（或者用 CIF Value ÷ Qty，一致即可）}
$$

同样取对数，后面方便线性建模：

$$
x_t^{(p)} = \ln P_t
$$

### 小结

*    $y_t$ ：要预测的“log 关税收入”；
*    $x_t^{(\tau)}$ ：关于关税率的特征；
*    $x_t^{(p)}$ ：关于单位价格的特征；

所有这些都是 **按月、2008–2025 的时间序列**。

* * *

2\. ARIMA（ARIMAX）模型：最简洁的一行公式
----------------------------

我们对“关税收入的对数”做差分（如果需要）：

$$
w_t = \nabla^d y_t
$$

其中  $d=0$  或  $1$ ，由平稳性检验决定。

**一行写下 ARIMAX(p,d,q)（ARIMA 带外生变量）的形式：**

$$
\boxed{ \quad \nabla^d y_t = c + \sum_{i=1}^p \phi_i\, \nabla^d y_{t-i} + \beta_1\, x_{t-1}^{(\tau)} + \beta_2\, x_{t-1}^{(p)} + \sum_{j=1}^q \theta_j\, \varepsilon_{t-j} + \varepsilon_t \quad}
$$

解释尽量压缩成一口气：

*   左边  $\nabla^d y_t$ ：关税收入（对数）的“差分后序列”（如果  $d=1$ ，就是增长率）；
*   右边：
    *    $\sum \phi_i \nabla^d y_{t-i}$ ：自身的时间惯性（过去几个月收入的走势）；
    *    $\beta_1 x_{t-1}^{(\tau)}$ ：**上一期平均关税率**对本期收入变化的影响；
    *    $\beta_2 x_{t-1}^{(p)}$ ：**上一期单位价格**对本期收入变化的影响（价格变动会改变名义税基）；
    *    $\sum \theta_j \varepsilon_{t-j}$ ：噪声的自相关（MA 部分）；
    *    $\varepsilon_t$ ：白噪声。

**p, d, q 怎么办？**

*   我们不在公式里硬性限制  $p,d,q$  的具体数值；
*   实际上用 2008–2025 的长序列，给定一个合理的候选集合（如  $p,q$  在 0–4， $d\in\{0,1\}$ ），通过 AIC/BIC + 残差检验来自动选出最合适的  $(p,d,q)$  组合即可。
*   在论文里可以一句话写：  
    “We estimate an ARIMAX(p,d,q) model for log tariff revenue with tariff rate and unit price as lagged exogenous regressors, and select p,d,q by AIC/BIC and residual diagnostics.”

* * *

3\. 模型的直觉 & 优势（含 Laffer 借鉴）
---------------------------

### 3.1 直觉：这其实是“简化版 Laffer 曲线”的时间序列实现

*   真实世界里，Laffer 曲线告诉我们：

$$
R_t(\tau_t) = \tau_t \cdot \text{TaxBase}_t(\tau_t)
$$

    税率太低：乘数小 → 收入低；  
    税率太高：税基缩得太厉害 → 收入也下降，中间有个“峰值”。
*   我们现在做的事是：**用一个 ARIMAX** 去直接学 **“关税率 + 单位价格 → 关税收入的变化”**：
    *    $\beta_1$  捕捉“关税率变化如何影响收入增长”；
    *    $\beta_2$  捕捉“价格变化（通胀、进口结构变化）如何影响名义税基”。

> 虽然我们没有显式写出一个解析的 Laffer 曲线  $R(\tau)$ ，  
> 但通过长期月度数据估计  $\beta_1,\beta_2$ ，再在不同关税路径下重新预测  $R_t$ ，  
> 本质上就是在做一个\*\*“经验 Laffer 曲线”\*\*：  
> 看看当你把平均关税从历史水平推到更高时，关税收入的整条路径会怎样变化。

### 3.2 模型的特有优势

1.  **数据成本低、变量简单：**
    *   只用 DataWeb 的 3 类列：总进口额、Import Charges、数量；
    *   通过简单的除法和取对数就得到三个关键时序  $y_t, x_t^{(\tau)}, x_t^{(p)}$ 。
2.  **结构清晰、解释直接：**
    *    $y_t$  是“关税收入”（题目第四问最关心的量）；
    *    $x_{t-1}^{(\tau)}$  是“上一期关税率”（政策工具）；
    *    $x_{t-1}^{(p)}$  是“上一期单位价格”（名义税基的价格因素）；
    *   一条式子就能说清楚“**过去的收入走势 + 过去的关税 + 过去的价格 → 当前的收入**”。
3.  **可以自然做政策情景：**
    *   因为关税和价格以特征形式进入模型，我们只要替换不同的“未来关税路径”，就能得到不同的  $R_t$  预测路径；
    *   这正是第四问想要的：**在高关税政策 vs 基准情况下，美国未来若干年的关税收入会怎样变**。
4.  **与 Laffer 文献对得上口径：**
    *   既体现“关税率通过税基影响收入”的思路，又不用搭多国一般均衡；
    *   适合 MCM/ICM 的时间和数学水平，同时在文献综述里可以挂上 tariff Laffer curve 的名字。

* * *

4\. 用这个模型具体怎么回答第四问？
-------------------

第四问核心：**在“互惠高关税”政策下，美国在第二任期内的关税收入会怎样变化？和不加税的基准相比，净变化是多少？**

用上面的模型，可以分三步走：

### Step 1：用 2008–2024 数据估计模型

*   用 2008–2024 的月度数据，构造：
    *    $y_t = \ln R_t$ 
    *    $x_t^{(\tau)} = \ln(1+\tau_t)$ 
    *    $x_t^{(p)} = \ln P_t$ 
*   做单位根与平稳性检验，确定  $d = 0$  还是  $1$ ；
*   在若干  $(p,d,q)$  组合中，用 AIC/BIC + 残差检验选出一个最终的 ARIMAX 模型；
*   得到估计系数  $\hat{c}, \hat{\phi}_i, \hat{\theta}_j, \hat{\beta}_1, \hat{\beta}_2$ 。

### Step 2：设定“基准”和“高关税”两条未来路径

1.  **基准情景（Baseline）**
    *   对历史  $\tau_t$ 、 $P_t$  做各自的简单 ARIMA 或趋势预测，得到 2025–2028 的：

$$
\tau_t^{\text{base}},\quad P_t^{\text{base}},\quad x_t^{(\tau,\text{base})},\ x_t^{(p,\text{base})}
$$

用这些特征喂进 ARIMAX，递推得到  $y_t^{\text{base}}$  和  $R_t^{\text{base}} = \exp(y_t^{\text{base}})$ 。
1.  **高关税情景（Policy）**
    *   根据题意，把 2025 年起的平均税率从当前水平抬升到 20.11%（或题目给定数），构造：

$$
\tau_t^{\text{policy}}
$$

对单位价格  $P_t^{\text{policy}}$  可以：
    *   假设维持基准情景的路径（“高关税主要作用于数量”）；或者
    *   加一点简单的价格传导假设（例如价格略有上升）；  
        二者都可以在文中说明。
计算：

$$
x_t^{(\tau,\text{policy})} = \ln(1+\tau_t^{\text{policy}}),\quad x_t^{(p,\text{policy})} = \ln P_t^{\text{policy}}
$$

用同一个 ARIMAX 模型，递推得到  $y_t^{\text{policy}}$  和  $R_t^{\text{policy}} = \exp(y_t^{\text{policy}})$ 。

### Step 3：比较两条收入路径，给出第四问的答案

*   月度差值：

$$
\Delta R_t = R_t^{\text{policy}} - R_t^{\text{base}}
$$

*   对“第二任期”（例如 2025–2028）求和：
  
$$
\Delta R = \sum_{t=2025\text{-}01}^{2028\text{-}12} \Delta R_t
$$

*   画图 + 文字解释：
    *   如果短期  $\Delta R_t > 0$  很多，说明刚加税时收入大涨；
    *   如果中后期  $\Delta R_t$  逐渐缩小甚至变负，说明“关税太高 → 税基萎缩 → 收入开始掉头”，就是 Laffer 曲线的影子；
    *   最终的  $\Delta R$  就是题目问的“第二任期内，在高关税政策下，美国关税收入相对基准情景的净变化”。
