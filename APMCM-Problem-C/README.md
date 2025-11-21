# 第二问第一阶段建模总结 / Stage I Model Summary for Question 2
显式为

$$
y_{i,t}=\ln p_{i,t}:=\alpha_i+\rho y_{i,t-1}+\beta\ln(1+\text{tariff}_{i,t})
$$

计算 $M_{i,2025}=p_{i,2025}M_{2025}^{tot}$ ，其中 $M_{2025}^{tot}$ 可由自回归得到，意思为各国出口车的贸易总量。

计算 $Q_{i,t}=\frac{M_{i,t}}{s_{i,t}}$ ，其中 $s_{i,t}$ 是国家在本国生产量除以国家实际生产量。

实现优化问题：
<img width="872" height="280" alt="image" src="https://github.com/user-attachments/assets/9d18d407-78f2-478c-a3a0-569186a8f13f" />
