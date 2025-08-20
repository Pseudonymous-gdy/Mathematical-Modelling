import pandas as pd
import numpy as np
import math
from heapq import nlargest


# 预留：基站2/3（如需）
# BS2 = BaseStation(bs_id=2, name="BS2", tx_power_dbm=..., rb_total=RB_LIMIT)
# BS3 = BaseStation(bs_id=3, name="BS3", tx_power_dbm=..., rb_total=RB_LIMIT)

# === Parameters ===
RB_LIMIT = 50
GROUP_SIZE = {'URLLC': 10, 'eMBB': 5, 'mMTC': 2}
BEAM_WIDTH = 10

# Physical parameters
B = 360e3       # RB bandwidth in Hz
DELTA_T = 1e-3  # Time slot duration in seconds

# === Noise / Receiver constants (from appendix) ===
NOISE_DENSITY_DBM_PER_HZ = -174.0
RECEIVER_NOISE_FIGURE_DB = 7.0  # NF

# === Data Loading ===
ray_df = pd.read_excel(r"D:\2025年第六届华数杯数学建模竞赛赛题\B题\附件\附件2\channel_data.xlsx",sheet_name='小规模瑞丽衰减')
channel_df = pd.read_excel(r"D:\2025年第六届华数杯数学建模竞赛赛题\B题\附件\附件2\channel_data.xlsx", sheet_name='大规模衰减')
tasks_df = pd.read_excel(r"D:\2025年第六届华数杯数学建模竞赛赛题\B题\附件\附件2\channel_data.xlsx", sheet_name='用户任务流')

ray_df['ms'] = (ray_df['Time'] * 1000).round().astype(int)
channel_df['ms'] = (channel_df['Time'] * 1000).round().astype(int)
tasks_df['ms']   = (tasks_df['Time']   * 1000).round().astype(int)

class Task:
    def __init__(self, user, slice_type, arrival, size):
        self.user = user
        self.slice = slice_type
        self.arrival = arrival  # in ms
        self.remaining = size   # in Mbit
        self.start_time = None
        self.finish_time = None
        self.transmission_rate = 0

def expand_state(state):
    u, e, m = state
    successors = []
    for du, de, dm in [(1,0,0), (0,1,0), (0,0,1)]:
        nu, ne, nm = u+du, e+de, m+dm
        if GROUP_SIZE['URLLC']*nu + GROUP_SIZE['eMBB']*ne + GROUP_SIZE['mMTC']*nm <= RB_LIMIT:
            successors.append((nu, ne, nm))
    return successors

class BaseStation:
    """
    基站模型：
      - tx_power_dbm: 总发射功率（dBm），默认均匀分到 rb_total 个 RB
      - rb_total: 基站可用 RB 总数（默认为 RB_LIMIT）
    """
    def __init__(self, bs_id: int, name: str, tx_power_dbm: float, rb_total: int = RB_LIMIT):
        self.id = bs_id
        self.name = name
        self.tx_power_dbm = float(tx_power_dbm)
        self.rb_total = int(rb_total)

    def power_per_rb_dbm(self) -> float:
        """每个 RB 的等效发射功率（dBm），假设均匀分配到 rb_total 个 RB。"""
        return self.tx_power_dbm - 10 * math.log10(self.rb_total)

    def rx_power_per_rb_dbm(self, pl_db: float, ray_db: float = 0.0) -> float:
        """
        接收端单个 RB 上的接收功率（dBm）
        pl_db: 路径损耗(dB)；ray_db: 小尺度瑞利衰落(dB)
        """
        return self.power_per_rb_dbm() - pl_db + ray_db

    @staticmethod
    def noise_power_dbm(n_rb: int = 1, rb_bandwidth_hz: float = B,
                        noise_figure_db: float = RECEIVER_NOISE_FIGURE_DB) -> float:
        """
        热噪声功率（dBm）。注意 n_rb*B 是总带宽。
        """
        return NOISE_DENSITY_DBM_PER_HZ + 10 * math.log10(n_rb * rb_bandwidth_hz) + noise_figure_db

# 全局常量（示例，按需调整）
ALPHA = 0.95
SLA_DELAY = {'URLLC': 5, 'eMBB': 100, 'mMTC': 500}       # ms
SLA_RATE   = {'URLLC': None, 'eMBB': 50.0, 'mMTC': None}  # Mbps
PENALTIES  = {'URLLC': 5.0, 'eMBB': 3.0, 'mMTC': 1.0}     # 附件表1 中的惩罚系数

def compute_qos(task):
    """
    依据附件（3.用户服务质量定义）中的公式计算单个任务的 QoS：
    - URLLC: α^D                         , D ≤ D_SLA;  -P_URLLC , D > D_SLA
    - eMBB: (r / r_SLA) 或 1           , D ≤ D_SLA;  -P_eMBB , D > D_SLA
    - mMTC: 1 或 （按周期汇总后再折扣） , D ≤ D_SLA;  -P_mMTC , D > D_SLA
    """
    # 1. 计算总延迟 D = 排队延迟 + 传输延迟（ms）
    D = task.finish_time - task.arrival

    # 2. 超时处罚（任何切片超出 SLA 都直接罚分）
    if D >= SLA_DELAY[task.slice]:
        return -PENALTIES[task.slice]

    # 3. 各切片在未超时情况下的正向 QoS
    if task.slice == 'URLLC':
        # Q = α^D, α=0.95，延迟越小 QoS 越接近 1 :contentReference[oaicite:0]{index=0}
        return ALPHA ** D

    elif task.slice == 'eMBB':
        # Q = r/r_SLA (当 r<r_SLA) 或 Q=1 (当 r≥r_SLA) :contentReference[oaicite:1]{index=1}
        r = task.transmission_rate  # 单位：Mbps，需在调度时记录
        if r < SLA_RATE['eMBB']:
            return r / SLA_RATE['eMBB']
        else:
            return 1.0

    elif task.slice ==  'mMTC':
        # mMTC 在任务级别只做接入成功（=1）或周期末统一折扣：
        # 这里返回 1，后续在周期汇总时再除以总任务数 :contentReference[oaicite:2]{index=2}
        return 1.0

def simulate_one_round(partition, inherited_tasks, channel_period, tasks_period, round_start_ms,
                       ray_period=None, serving_bs: BaseStation | None = None,
                       interferers: list | None = None):
    if serving_bs is None:
        serving_bs = BS1

    """
    Simulate one 100 ms round of scheduling.

    Args:
        partition: tuple (u_groups, e_groups, m_groups) pre-partitioned RB groups.
        inherited_tasks: list of unfinished mMTC Task objects from previous round.
        channel_period: DataFrame with 100 rows of channel quality for this round.
        tasks_period: DataFrame with new arrivals for this round.
        round_start_ms: int/float, global start time of this round in ms.

    Returns:
        total_qos: float, cumulative QoS for this round.
        leftovers: list of Task objects (unfinished mMTC) to carry to next round.
        slice_qos: dict, per-slice QoS breakdown like {'URLLC': x, 'eMBB': y, 'mMTC': z}
    """
    # 1. Initialize FIFO queues for each slice
    queues = {'URLLC': [], 'eMBB': [], 'mMTC': []}

    # 额外新增：分片 QoS 统计
    slice_qos = {'URLLC': 0.0, 'eMBB': 0.0, 'mMTC': 0.0}

    # 2. Inherit unfinished mMTC tasks without modifying their original arrival
    for task in inherited_tasks:
        queues[task.slice].append(task)

    # 3. Build new arrivals list (global times)
    arrival_list = []
    for _, row in tasks_period.iterrows():
        t_global = int(round(row['Time'] * 1000))
        # URLLC users U1, U2
        for user in ['U1', 'U2']:
            size = row[user]
            if size > 0:
                arrival_list.append(Task(user, 'URLLC', t_global, size))
        # eMBB users e1–e4
        for user in ['e1', 'e2', 'e3', 'e4']:
            size = row[user]
            if size > 0:
                arrival_list.append(Task(user, 'eMBB', t_global, size))
        # mMTC users m1–m10
        for i in range(1, 11):
            user = f'm{i}'
            size = row[user]
            if size > 0:
                arrival_list.append(Task(user, 'mMTC', t_global, size))
    arrival_list.sort(key=lambda x: x.arrival)

    total_qos = 0.0
    leftovers = []

    # 4. Simulate each of 100 slots (1 ms each)
    for local_t in range(100):
        if local_t == 0:
            first_ms_arrivals = sum(1 for t in arrival_list if t.arrival == round_start_ms)
            # print(f"[DEBUG] round@{round_start_ms}ms: arrival_list={len(arrival_list)}, first_ms={first_ms_arrivals}")
        t_global = int(round_start_ms) + local_t

        # 4.1 Enqueue new arrivals at this global time
        while arrival_list and arrival_list[0].arrival == t_global:
            task = arrival_list.pop(0)
            queues[task.slice].append(task)

        # 4.2 Schedule each slice type
        for slice_type, groups in zip(['URLLC', 'eMBB', 'mMTC'], partition):
            if queues[slice_type]:
                kept = []
                for task in queues[slice_type]:
                    # 用 > 保证触发罚分；若你坚持 >=，请把 finish_time 设为 arrival + SLA + 1
                    if (t_global - task.arrival) >= SLA_DELAY[slice_type]:
                        task.finish_time = t_global
                        q = compute_qos(task)
                        total_qos += q
                        slice_qos[task.slice] += q
                        # 不放回队列
                    else:
                        kept.append(task)
                queues[slice_type] = kept
            # ----------------------------------

            free_rb = GROUP_SIZE[slice_type] * groups
            while free_rb >= GROUP_SIZE[slice_type] and queues[slice_type]:
                task = queues[slice_type].pop(0)
                if task.start_time is None:
                    task.start_time = t_global

                # —— 每毫秒/用户：计算“每RB”的 SINR ——
                pl_db  = channel_period.at[local_t, task.user]
                ray_db = ray_period.at[local_t, task.user] if ray_period is not None else 0.0

                sinr_lin, sinr_db = compute_sinr(
                    serving_bs=serving_bs,
                    pl_db_serv=pl_db,
                    ray_db_serv=ray_db,
                    interfering_links=interferers,  # 暂不使用干扰则传 None
                    n_rb_noise=1,                   # 噪声按单RB带宽计算，避免重复
                    rb_bandwidth_hz=B
                )

                # —— 保持原单位：c_rb, rate 仍为 Mbit/s ——
                n_rb = GROUP_SIZE[slice_type]
                free_rb -= n_rb
                rate_bps, c_rb_bps = compute_rate_from_sinr(
                    sinr_linear=sinr_lin,
                    n_rb=n_rb,
                    rb_bandwidth_hz=B
                )
                c_rb = c_rb_bps / 1e6            # 单 RB 容量 Mbit/s
                rate = rate_bps / 1e6            # 该用户当毫秒速率 Mbit/s
                task.transmission_rate = rate
                transmitted = rate * DELTA_T     # Mbit
                task.remaining -= transmitted

                if task.remaining <= 0:
                    task.finish_time = t_global + max(0.001,task.remaining/rate)
                    q = compute_qos(task)
                    total_qos += q
                    slice_qos[task.slice] += q
                else:
                    queues[slice_type].insert(0, task)

                    

    round_end_ms = round_start_ms + 100  # 本轮结束的全局时间（ms）

    for slice_type in ['URLLC', 'eMBB', 'mMTC']:
        while queues[slice_type]:
            task = queues[slice_type].pop(0)

            # 若任务尚未完成：根据是否超时决定“结算或带入下一轮”
            if (round_end_ms - task.arrival) >= SLA_DELAY[slice_type]:
                # 在“到达 + SLA”这一刻结算（而不是强行记到本轮末）
                task.finish_time = task.arrival + SLA_DELAY[slice_type]
                q = compute_qos(task)
                total_qos += q
                slice_qos[task.slice] += q
                # 超时任务在此被结算并移除；不再带入下一轮
            else:
                # 未超时 → 不结算，带入下一轮继续
                leftovers.append(task)
    return total_qos, leftovers, slice_qos

def beam_search_for_round(inherited_tasks, channel_period, tasks_period, round_start_ms):
    beam = [(1,1,1)]
    leftovers_map = {}
    while True:
        candidates = set()
        for state in beam:
            for ns in expand_state(state):
                candidates.add(ns)
        if not candidates:
            break
        scored = []
        for s in candidates:
            qos, left, _ = simulate_one_round(s, inherited_tasks, channel_period, tasks_period, round_start_ms)
            scored.append((qos, s, left))
        top = nlargest(BEAM_WIDTH, scored, key=lambda x: x[0])
        beam = [s for (_,s,_) in top]
        leftovers_map = {s:left for (_,s,left) in top}
    # final pick
    final = max([(simulate_one_round(s, inherited_tasks, channel_period, tasks_period, round_start_ms)[0], s)
                  for s in beam], key=lambda x: x[0])
    best_state = final[1]
    best_qos, best_left, best_slice_qos = simulate_one_round(best_state, inherited_tasks, channel_period, tasks_period, round_start_ms)
    return best_state, best_qos, best_left, best_slice_qos

def schedule_all_rounds(num_rounds):
    results = []
    inherited = []
    for r in range(num_rounds):
        round_start_ms = r * 100
        round_end_ms   = round_start_ms + 100

        ch_period = channel_df[(channel_df['ms'] >= round_start_ms) &
                               (channel_df['ms'] <  round_end_ms)].drop(columns=['ms']).reset_index(drop=True)
        tasks_period = tasks_df[(tasks_df['ms'] >= round_start_ms) &
                                (tasks_df['ms'] <  round_end_ms)].drop(columns=['ms']).reset_index(drop=True)

        # 现在一定是 100 行
        assert len(ch_period) == 100, f"round {r}: ch_period rows={len(ch_period)}"
        # tasks_period 行数可 99/100/101 都行

        part, qos, left, slice_qos = beam_search_for_round(inherited, ch_period, tasks_period, round_start_ms)
        results.append((r+1, part, qos, slice_qos['URLLC'], slice_qos['eMBB'], slice_qos['mMTC']))
        inherited = left
    return results


def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)

def compute_sinr(serving_bs: BaseStation,
                 pl_db_serv: float, ray_db_serv: float,
                 interfering_links: list | None = None,
                 n_rb_noise: int = 1,
                 rb_bandwidth_hz: float = B,
                 noise_figure_db: float = RECEIVER_NOISE_FIGURE_DB):
    """
    计算“每RB”的 SINR。
    - n_rb_noise: 用于噪声计算的带宽= n_rb_noise * B。为了得到“每 RB”的 SINR，这里应传 1。
    - interfering_links: [(bs2, pl2_db, ray2_db), (bs3, pl3_db, ray3_db), ...]
    返回 (sinr_linear, sinr_db)
    """

    if serving_bs is None:
        serving_bs = BS1
    
    # 服役链路接收功率（单RB）
    prx_serv_dbm = serving_bs.rx_power_per_rb_dbm(pl_db_serv, ray_db_serv)
    prx_serv_mw  = dbm_to_mw(prx_serv_dbm)

    # 干扰功率（如无就为 0）
    interf_mw = 0.0
    if interfering_links:
        for bs_i, pl_i_db, ray_i_db in interfering_links:
            prx_i_dbm = bs_i.rx_power_per_rb_dbm(pl_i_db, ray_i_db)
            interf_mw += dbm_to_mw(prx_i_dbm)

    # 噪声功率（dBm → mW）
    noise_dbm = BaseStation.noise_power_dbm(n_rb=n_rb_noise, rb_bandwidth_hz=rb_bandwidth_hz,
                                            noise_figure_db=noise_figure_db)
    noise_mw = dbm_to_mw(noise_dbm)

    sinr_linear = prx_serv_mw / (interf_mw + noise_mw)
    sinr_db = 10 * math.log10(max(sinr_linear, 1e-12))
    return sinr_linear, sinr_db


def compute_rate_from_sinr(
    sinr_linear: float,
    n_rb: int,
    rb_bandwidth_hz: float = B,
    time_fraction: float = 1.0,      # 本毫秒内分给该用户的时间占比(0~1)
    rb_fraction: float = 1.0,        # 分到的RB中实际可用占比(0~1)
    n_layers: int = 1,               # MIMO层数
    overhead_fraction: float = 1.0,  # 协议/导频开销后的有效比例
    se_cap_bps_per_hz: float | None = None  # MCS上限(比特/秒/赫兹)
) -> tuple[float, float]:
    """
    返回: (rate_bps, c_rb_bps) —— 总速率(比特/秒) 与 单RB容量(比特/秒)
    说明: 先按香农算每RB容量；如给定 MCS 上限则封顶；再乘以 RB 数与各类占比/层数。
    """
    c_rb_bps = rb_bandwidth_hz * math.log2(1.0 + max(sinr_linear, 1e-12))
    if se_cap_bps_per_hz is not None:
        c_rb_bps = min(c_rb_bps, se_cap_bps_per_hz * rb_bandwidth_hz)
    rate_bps = c_rb_bps * n_rb * rb_fraction * time_fraction * max(1, int(n_layers)) * overhead_fraction
    return rate_bps, c_rb_bps

BS1 = BaseStation(bs_id=1, name="BS1", tx_power_dbm=30.0, rb_total=RB_LIMIT)

# Example usage:
if __name__ == "__main__":
    rounds = schedule_all_rounds(num_rounds=10)
    for rnd, part, score, q_u, q_e, q_m in rounds:
        print(f"Round {rnd}: partition {part}, QoS_total {score:.2f}, "
              f"QoS_URLLC {q_u:.2f}, QoS_eMBB {q_e:.2f}, QoS_mMTC {q_m:.2f}")
