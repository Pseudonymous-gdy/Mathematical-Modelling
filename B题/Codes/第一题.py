
import math
import pandas as pd

# -----------------------------
# 1. 参数区
# -----------------------------
TOTAL_RB    = 50           # 基站最大可用资源块
RB_BW       = 360e3        # Hz，单个RB带宽
P_TX_DBM    = 30           # 发射功率（dBm）, 可根据需要调整
NOISE_FIG   = 7            # dB

# URLLC
URLLC_USERS = ['U1','U2']
ALPHA       = 0.95
P_URLLC     = 5
D_SLA_URLLC = 5            # ms

# eMBB
EMBB_USERS  = ['e1','e2','e3','e4']
R_SLA_EMBB  = 50           # Mbps
T_SLA_EMBB  = 100          # ms
P_EMBB      = 3

# mMTC
MMTC_USERS  = [f"m{i}" for i in range(1,11)]
T_SLA_MMT   = 500          # ms
P_MMT       = 1

# -----------------------------
# 2. 读取数据
# -----------------------------
xls      = pd.ExcelFile(r"D:\2025年第六届华数杯数学建模竞赛赛题\B题\附件\附件1\channel_data.xlsx")
L0       = pd.read_excel(xls, '大规模衰减').iloc[0]
h2_0     = pd.read_excel(xls, '小规模瑞丽衰减').iloc[0]
flow0    = pd.read_excel(xls, '任务流').iloc[0]

# -----------------------------
# 3. 预计算：噪声 & 每RB速率
# -----------------------------
noise_dbm = -174 + 10*math.log10(RB_BW) + NOISE_FIG
noise_mw  = 10**(noise_dbm/10)
P_tx_mw   = 10**(P_TX_DBM/10)

rate_per_rb = {}
for u in URLLC_USERS + EMBB_USERS + MMTC_USERS:
    L_db  = L0[u]
    h2    = h2_0[u]
    sinr = (P_tx_mw * 10**(-L_db/10) * h2) / noise_mw
    rate_per_rb[u] = RB_BW * math.log2(1 + sinr) / 1e6

# -----------------------------
# 4. 服务质量函数
# -----------------------------
def q_urlcc(delay_ms):
    return ALPHA ** (delay_ms) if delay_ms <= D_SLA_URLLC else -P_URLLC

def q_embb(Rtot, delay_ms):
    if delay_ms > T_SLA_EMBB:
        return -P_EMBB
    return 1.0 if Rtot >= R_SLA_EMBB else (Rtot / R_SLA_EMBB)

def q_mmtc(success_count, total_count, worst_delay):
    return -P_MMT if worst_delay > T_SLA_MMT else (success_count / total_count)

best = {'score': -1e9}

for n_url in range(len(URLLC_USERS) + 1):
    rb_url = 10 * n_url
    for n_emb in range(len(EMBB_USERS) + 1):
        rb_emb = 5 * n_emb

        # 必须保证还有至少 10 RB 留给 mMTC
        if TOTAL_RB - rb_url - rb_emb < 10:
            continue

        # 固定 mMTC 资源块数为 10（每任务 2 RB -> 首时隙服务 5 个）
        n_mmt = min(10 // 2, len(MMTC_USERS))  # =5，但写成表达式更稳
        rb_mmt = 2 * n_mmt  # =10

        # URLLC 得分（无排队或假设资源足够）
        score_url = 0.0
        for u in URLLC_USERS[:n_url]:
            size = flow0[u]
            Rtot = rate_per_rb[u] * 10
            t_tx = size / Rtot * 1e3
            score_url += q_urlcc(t_tx)
        score_url += (len(URLLC_USERS) - n_url) * (-P_URLLC)

        # eMBB 得分
        score_emb = 0.0
        for u in EMBB_USERS[:n_emb]:
            size = flow0[u]
            Rtot = rate_per_rb[u] * 5
            t_tx = size / Rtot * 1e3
            score_emb += q_embb(Rtot, t_tx)
        score_emb += (len(EMBB_USERS) - n_emb) * (-P_EMBB)

        # mMTC 得分（支持排队至下一时隙，排队延迟算1ms）
        total_tasks = len(MMTC_USERS)
        delays = []
        # 第一时隙服务前 n_mmt 任务（=5 个）
        for u in MMTC_USERS[:n_mmt]:
            size = flow0[u]
            Rtot = rate_per_rb[u] * 2
            t_tx = size / Rtot * 1e3
            delays.append(t_tx)
        # 剩余任务排队，下一时隙开始，排队延迟1ms
        for u in MMTC_USERS[n_mmt:]:
            size = flow0[u]
            Rtot = rate_per_rb[u] * 2
            t_tx = size / Rtot * 1e3
            delays.append(t_tx + 1.0)
        success = sum(1 for d in delays if d <= T_SLA_MMT)
        worst  = max(delays) if delays else 0.0
        score_mmt = q_mmtc(success, total_tasks, worst)

        total_score = score_url + score_emb + score_mmt
        if total_score > best['score']:
            best = {
                'score': total_score,
                'n_url': n_url, 'rb_url': rb_url,
                'n_emb': n_emb, 'rb_emb': rb_emb,
                'n_mmt': n_mmt, 'rb_mmt': rb_mmt,
                'score_url': score_url,
                'score_emb': score_emb,
                'score_mmt': score_mmt
            }
mmt_display = best['score_mmt'] * 10.0
total_display = best['score_url'] + best['score_emb'] + mmt_display
print("===== 最优资源分配方案（含排队延迟） =====")
print(f" URL﻿LLC: 服务 {best['n_url']} 个任务，用 {best['rb_url']} RB")
print(f" eMBB : 服务 {best['n_emb']} 个任务，用 {best['rb_emb']} RB")
print(f" mMTC : 服务 {len(MMTC_USERS)} 个任务，用 {best['rb_mmt']} RB")
print(f" 空闲  : {TOTAL_RB - best['rb_url'] - best['rb_emb'] - best['rb_mmt']} RB\n")

print(f"分片得分 (URLLC, eMBB, mMTC) = "
      f"{best['score_url']:.3f}, {best['score_emb']:.3f}, {mmt_display:.3f}")
print(f"→ 总服务质量评分 = {total_display:.3f}")
