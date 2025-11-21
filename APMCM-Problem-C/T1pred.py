import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import math
import matplotlib.pyplot as plt

# =========================================
# 1. 读入数据
# =========================================

trade_file = "TradeData.xlsx"        # 美/巴/阿对华出口数据（国家面板）
annual_file = "TradeDataAnnual.xlsx" # 年度总量 + 关税 + 产量（包括 2025）

df_trade = pd.read_excel(trade_file)
df_annual = pd.read_excel(annual_file)

# 只取需要的列并改名
df_trade = df_trade[['period', 'reporterDesc', 'netWgt', 'primaryValue']].copy()
df_trade = df_trade.rename(columns={
    'period': 'year',
    'reporterDesc': 'country',
    'netWgt': 'weight',
    'primaryValue': 'value'
})
df_annual = df_annual.rename(columns={'period': 'year'})

# 按年份合并年度信息（关税、产量、总进口）到国家数据
df = df_trade.merge(df_annual, on='year', how='left')

# =========================================
# 2. 价格与含关税价格（只对美国乘中国对美大豆关税）
# =========================================

df['price'] = df['value'] / df['weight']

# eff_price = price * (1 + CN→US 税率/100)，非美国设为 0 冲击（价不变）
df['eff_price'] = df['price'] * (
    1 + np.where(df['country'] == 'USA', df['tariff cn-us'] / 100.0, 0.0)
)

# 避免 log 出错：非正数一律视为缺失
for col in ['price', 'eff_price', 'weight', 'total import wgt']:
    df.loc[df[col] <= 0, col] = np.nan

# =========================================
# 3. 三国产量份额（年度变量）
# =========================================

df['prod_total'] = (
    df['us production'] +
    df['brazil production'] +
    df['argentina production']
)

def prod_for_row(row):
    if row['country'] == 'USA':
        return row['us production']
    elif row['country'] == 'Brazil':
        return row['brazil production']
    elif row['country'] == 'Argentina':
        return row['argentina production']
    else:
        return np.nan

df['prod_i'] = df.apply(prod_for_row, axis=1)
df['prod_share'] = df['prod_i'] / df['prod_total']

# =========================================
# 4. 各国份额 p_i（用重量份额）
# =========================================

df['p_i'] = df['weight'] / df['total import wgt']

# =========================================
# 5. 含关税价格弹性 elasticity（方案 2B）
#    eta = d log W / d log P_eff
#    elasticity_i,t = (eta_{t-1} + eta_{t-2}) / 2
#    极小价格变化用 1e-6 替代避免爆炸
# =========================================

df = df.sort_values(['country', 'year'])

df['logW'] = np.log(df['weight'])
df['logPe'] = np.log(df['eff_price'])

df['dlogW'] = df.groupby('country')['logW'].diff()
df['dlogP'] = df.groupby('country')['logPe'].diff()

eps_den = 1e-6
mask_small = df['dlogP'].abs() < eps_den
df.loc[mask_small & df['dlogP'].notna(), 'dlogP'] = df.loc[
    mask_small & df['dlogP'].notna(), 'dlogP'
].apply(lambda x: eps_den if x >= 0 else -eps_den)

df['eta'] = df['dlogW'] / df['dlogP']
df['elasticity'] = df.groupby('country')['eta'].transform(
    lambda x: (x.shift(1) + x.shift(2)) / 2.0
)

# =========================================
# 6. ln(p_i / p_US) 作为回归因变量
# =========================================

p_us = df[df['country'] == 'USA'][['year', 'p_i']].rename(columns={'p_i': 'p_us'})
df = df.merge(p_us, on='year', how='left')

df['ln_share_ratio'] = np.log(df['p_i'] / df['p_us'])

# =========================================
# 7. 份额回归（Argentina & Brazil）
#    ln(p_i / p_US) ~ trend + ln(prod_share)
#                     + ln(1+tariff_us-cn) + ln(1+tariff_cn-us)
#                     + elasticity + Brazil dummy
# =========================================

share_df = df[df['country'].isin(['Brazil', 'Argentina'])].copy()

share_df['trend'] = share_df['year']
share_df['ln_prod_share'] = np.log(share_df['prod_share'])

# 使用 log(1 + 税率/100)，0 税率 -> ln(1) = 0 合法
share_df['ln_tariff_us_cn'] = np.log1p(share_df['tariff us-cn'] / 100.0)
share_df['ln_tariff_cn_us'] = np.log1p(share_df['tariff cn-us'] / 100.0)

share_df = share_df.replace([np.inf, -np.inf], np.nan)

needed_cols = [
    'ln_share_ratio',
    'ln_prod_share',
    'ln_tariff_us_cn',
    'ln_tariff_cn_us',
    'elasticity',
    'trend'
]
share_df = share_df.dropna(subset=needed_cols)

if share_df.shape[0] < 3:
    print("份额回归：有效观测太少，无法估计。")
    share_model = None
else:
    X = share_df[['trend',
                  'ln_prod_share',
                  'ln_tariff_us_cn',
                  'ln_tariff_cn_us',
                  'elasticity']].copy()
    # 国家虚拟变量（基准 Argentina，Brazil = 1）
    dummies = pd.get_dummies(share_df['country'], drop_first=True)
    X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)
    y = share_df['ln_share_ratio']

    X = X.astype(float)
    y = y.astype(float)

    share_model = sm.OLS(y, X).fit(cov_type='HC1')

    print("\n====== 份额回归（ln(p_i / p_US)） ======\n")
    print(share_model.summary())

# =========================================
# 8. 用 2025 年年度变量预测份额（softmax）
# =========================================

pred_2025 = None
p_hat_25 = None
W_total_25 = None

if share_model is not None:
    annual_2025 = df_annual[df_annual['year'] == 2025]
    if annual_2025.empty:
        print("\n年度表中没有 2025 行，无法预测 2025。")
    else:
        a25 = annual_2025.iloc[0]

        # 2025 年三国产量份额
        prod_total_25 = (
            a25['us production'] +
            a25['brazil production'] +
            a25['argentina production']
        )

        rows_25 = []
        for c in ['USA', 'Brazil', 'Argentina']:
            if c == 'USA':
                prod_i_25 = a25['us production']
            elif c == 'Brazil':
                prod_i_25 = a25['brazil production']
            else:
                prod_i_25 = a25['argentina production']

            prod_share_25 = prod_i_25 / prod_total_25 if prod_total_25 > 0 else np.nan

            rows_25.append({
                'year': 2025,
                'country': c,
                'prod_share': prod_share_25,
                'tariff us-cn': a25['tariff us-cn'],
                'tariff cn-us': a25['tariff cn-us']
            })

        df_25 = pd.DataFrame(rows_25)

        # elasticity：用各国最后一个非空 elasticity 作为 2025 的近似
        eps_last = {}
        for c in ['USA', 'Brazil', 'Argentina']:
            tmp = df[(df['country'] == c) & df['elasticity'].notna()].sort_values('year')
            eps_last[c] = tmp['elasticity'].iloc[-1] if not tmp.empty else np.nan

        df_25['elasticity'] = df_25['country'].map(eps_last)

        # 只对 Brazil & Argentina 构造回归自变量
        df_25_sub = df_25[df_25['country'].isin(['Brazil', 'Argentina'])].copy()
        df_25_sub['trend'] = df_25_sub['year']
        df_25_sub['ln_prod_share'] = np.log(df_25_sub['prod_share'])
        df_25_sub['ln_tariff_us_cn'] = np.log1p(df_25_sub['tariff us-cn'] / 100.0)
        df_25_sub['ln_tariff_cn_us'] = np.log1p(df_25_sub['tariff cn-us'] / 100.0)

        df_25_sub = df_25_sub.replace([np.inf, -np.inf], np.nan)
        df_25_sub = df_25_sub.dropna(subset=[
            'ln_prod_share',
            'ln_tariff_us_cn',
            'ln_tariff_cn_us',
            'elasticity',
            'trend'
        ])

        if df_25_sub.shape[0] >= 1:
            X_25 = df_25_sub[['trend',
                              'ln_prod_share',
                              'ln_tariff_us_cn',
                              'ln_tariff_cn_us',
                              'elasticity']].copy()
            dummies_25 = pd.get_dummies(df_25_sub['country'], drop_first=True)
            X_25 = pd.concat([X_25, dummies_25], axis=1)
            X_25 = sm.add_constant(X_25, has_constant='add')

            # 对齐训练时的列名顺序
            train_cols = list(share_model.params.index)
            for col in train_cols:
                if col not in X_25.columns:
                    X_25[col] = 0.0
            X_25 = X_25[train_cols]
            X_25 = X_25.astype(float)

            u_hat_25 = share_model.predict(X_25)

            # 构造 2025 效用：USA 作为基准 = 0
            util_25 = {'USA': 0.0}
            for (_, row), u in zip(df_25_sub.iterrows(), u_hat_25):
                util_25[row['country']] = float(u)

            exps = {c: math.exp(u) for c, u in util_25.items()}
            denom = sum(exps.values())
            p_hat_25 = {c: exps[c] / denom for c in ['USA', 'Brazil', 'Argentina']}

            print("\n====== 2025 预测份额 p_hat_2025 ======\n")
            for c in ['USA', 'Brazil', 'Argentina']:
                print(f"{c}: {p_hat_25[c]:.6f}")
        else:
            print("\n2025 年 Brazil/Argentina 自变量不完整，无法预测份额。")

# =========================================
# 9. 对 ln(W_total) 做 AR(2)，预测 2025 总进口量
# =========================================

wt_nonan = df_annual[df_annual['total import wgt'].notna()].sort_values('year')
if wt_nonan.shape[0] < 4:
    print("\nW_total 有效样本太少，无法估计 AR(2)。")
else:
    years_obs = wt_nonan['year'].values
    logW = np.log(wt_nonan['total import wgt'].values)

    ar2_model = AutoReg(logW, lags=2, old_names=False).fit()

    print("\n====== ln(W_total) 的 AR(2) 结果 ======\n")
    print(ar2_model.summary())

    logW_25 = ar2_model.forecast(steps=1)[0]
    W_total_25 = float(np.exp(logW_25))

    print(f"\n预测 2025 年总进口量 W_total(2025) ≈ {W_total_25:.2f}")

# =========================================
# 10. 合成 2025 各国 W_hat_2025 并画图
# =========================================

if (p_hat_25 is not None) and (W_total_25 is not None):
    pred_2025 = pd.DataFrame({
        'year': [2025, 2025, 2025],
        'country': ['USA', 'Brazil', 'Argentina'],
        'p_hat_2025': [
            p_hat_25['USA'],
            p_hat_25['Brazil'],
            p_hat_25['Argentina']
        ]
    })
    pred_2025['W_total_2025_pred'] = W_total_25
    pred_2025['W_hat_2025'] = pred_2025['p_hat_2025'] * W_total_25

    print("\n====== 2025 各国预测进口量 W_hat_2025 ======\n")
    print(pred_2025)

    # 图1：2025 份额柱状图
    plt.figure(figsize=(6, 4))
    plt.bar(pred_2025['country'], pred_2025['p_hat_2025'])
    plt.ylabel('Predicted share p_hat (2025)')
    plt.title('Predicted Import Share in 2025')
    plt.tight_layout()
    plt.savefig("fig_share_2025.png", dpi=300)
    plt.close()

    # 图2：2025 进口量柱状图
    plt.figure(figsize=(6, 4))
    plt.bar(pred_2025['country'], pred_2025['W_hat_2025'])
    plt.ylabel('Predicted import volume W_hat (2025)')
    plt.title('Predicted Import Volume in 2025')
    plt.tight_layout()
    plt.savefig("fig_volume_2025.png", dpi=300)
    plt.close()

    # 图3：历史 + 预测总进口趋势
    plt.figure(figsize=(7, 4))
    plt.plot(wt_nonan['year'], wt_nonan['total import wgt'],
             marker='o', label='Observed W_total')
    plt.scatter([2025], [W_total_25], color='red',
                label='Predicted W_total(2025)')
    plt.xlabel('Year')
    plt.ylabel('Total import weight')
    plt.title('Total Import Weight: Observed and AR(2) Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_total_import_AR2.png", dpi=300)
    plt.close()

    print("\n图已保存：fig_share_2025.png, fig_volume_2025.png, fig_total_import_AR2.png")
else:
    print("\n由于份额或总量预测缺失，未生成 2025 预测图。")

# ================================================
# 份额折线图：历史 p_i + 预测 p_hat_2025
# ================================================

# 1. 提取历史份额（按重量）
hist = df[['year','country','p_i']].dropna()
hist = hist[hist['year'] <= 2024]

# pivot：year × country
share_hist = hist.pivot(index='year', columns='country', values='p_i')

# 2. 加入预测 2025 份额（来自 p_hat_25）
if p_hat_25 is not None:
    for c in ['USA','Brazil','Argentina']:
        share_hist.loc[2025, c] = p_hat_25[c]

# 3. 绘图
plt.figure(figsize=(8,5))
for c in ['USA','Brazil','Argentina']:
    if c in share_hist.columns:
        plt.plot(share_hist.index, share_hist[c], marker='o', label=c)

plt.xlabel('Year')
plt.ylabel('Import share (by weight)')
plt.title('China Soybean Import Share (2015–2025, predicted)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("fig_share_2015_2025.png", dpi=300)
plt.close()

print("\n图已保存：fig_share_2015_2025.png")
