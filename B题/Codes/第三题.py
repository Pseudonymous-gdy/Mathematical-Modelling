from pathlib import Path
import math
import pandas as pd
from heapq import nlargest

# ---------------------- Constants & Lists ----------------------
RB_LIMIT = 50
GROUP_SIZE = {'URLLC': 10, 'eMBB': 5, 'mMTC': 2}
BEAM_WIDTH = 10

B = 360e3       # RB bandwidth in Hz
DELTA_T = 1e-3  # 1 ms

NOISE_DENSITY_DBM_PER_HZ = -174.0
RECEIVER_NOISE_FIGURE_DB = 7.0

ALPHA = 0.95
SLA_DELAY = {'URLLC': 5, 'eMBB': 100, 'mMTC': 500}       # ms
SLA_RATE   = {'URLLC': None, 'eMBB': 50.0, 'mMTC': None}  # Mbps
PENALTIES  = {'URLLC': 5.0, 'eMBB': 3.0, 'mMTC': 1.0}

URLLC_USERS = [f"U{i}" for i in range(1, 7)]
EMBB_USERS  = [f"e{i}" for i in range(1, 13)]
MMTC_USERS  = [f"m{i}" for i in range(1, 31)]
ALL_USERS = URLLC_USERS + EMBB_USERS + MMTC_USERS

BS_COORDS = {
    1: (0.0, 500.0),
    2: (-433.0127, -250.0),
    3: (433.0127, -250.0),
    # 4: (0.0, 0.0),  # 新基站bs4(mbs1)
}

# ---------------------- Utilities ----------------------
def dbm_to_mw(dbm: float) -> float:
    return 10 ** (dbm / 10.0)

def noise_power_mw(n_rb: int = 1, rb_bandwidth_hz: float = B,
                   noise_figure_db: float = RECEIVER_NOISE_FIGURE_DB) -> float:
    noise_dbm = NOISE_DENSITY_DBM_PER_HZ + 10 * math.log10(n_rb * rb_bandwidth_hz) + noise_figure_db
    return dbm_to_mw(noise_dbm)

NOISE_P_MW_1RB = noise_power_mw(1, B, RECEIVER_NOISE_FIGURE_DB)

def expand_state(state, rb_limit=RB_LIMIT):
    # 自动兼容单基站和多基站输入
    if isinstance(state[0], int):
        # 单基站情况，原逻辑
        u, e, m = state
        succ = []
        for i, g in enumerate([u, e, m]):
            ng = [u, e, m]
            ng[i] = g + 1
            if (ng[0] <= len(URLLC_USERS) and
                ng[1] <= len(EMBB_USERS) and
                ng[2] <= len(MMTC_USERS) and
                10 * ng[0] + 5 * ng[1] + 2 * ng[2] <= rb_limit):
                succ.append(tuple(ng))
        return succ
    else:
        # 多基站情况
        succ = []
        for bs_idx in range(len(state)):
            for slice_idx in range(3):
                new_state = [list(s) for s in state]
                new_state[bs_idx][slice_idx] += 1
                # 统计所有基站同类分组总和
                total_url = sum(s[0] for s in new_state)
                total_embb = sum(s[1] for s in new_state)
                total_mmtc = sum(s[2] for s in new_state)
                # 判断是否超限
                if (total_url <= len(URLLC_USERS) and
                    total_embb <= len(EMBB_USERS) and
                    total_mmtc <= len(MMTC_USERS)):
                    # 分别判断每个基站的RB限制
                    rb_limits = [RB_LIMIT] * len(new_state)
                    if all(10*s[0] + 5*s[1] + 2*s[2] <= rb_limits[i] for i, s in enumerate(new_state)):
                        succ.append(tuple(tuple(s) for s in new_state))
        return succ

class Task:
    def __init__(self, user, slice_type, arrival, size):
        self.user = user
        self.slice = slice_type
        self.arrival = arrival  # in ms
        self.remaining = size   # in Mbit
        self.start_time = None
        self.finish_time = None
        self.transmission_rate = 0.0  # Mbps

def compute_qos(task: 'Task') -> float:
    D = task.finish_time - task.arrival  # ms
    if D >= SLA_DELAY[task.slice]:
        return -PENALTIES[task.slice]
    if task.slice == 'URLLC':
        return (ALPHA ** D)
    elif task.slice == 'eMBB':
        r = task.transmission_rate  # Mbps
        return min(1.0, r / SLA_RATE['eMBB'])
    elif task.slice == 'mMTC':
        return 1.0
    return 0.0

class BaseStation:
    def __init__(self, bs_id: int, name: str, tx_power_dbm: float = 30.0, rb_total: int = RB_LIMIT):
        self.bs_id = bs_id
        self.name = name
        self.tx_power_dbm = float(tx_power_dbm)
        self.rb_total = int(rb_total)

    def power_per_rb_dbm(self) -> float:
        # IMPORTANT: per your rule, do NOT divide across RBs; one BS uses a single p for all RBs.
        return self.tx_power_dbm

# ---------------------- Signal Model ----------------------
def compute_sinr_linear(
    bs_serv: BaseStation,
    pl_db_serv: float,
    ray_db_serv: float,
    interferers: list,  # list of tuples (bs_obj, pl_db, ray_db)
) -> float:
    prx_serv_dbm = bs_serv.power_per_rb_dbm() - pl_db_serv + ray_db_serv
    S = dbm_to_mw(prx_serv_dbm)

    I = 0.0
    for (bs_i, pl_i_db, ray_i_db) in interferers:
        prx_i_dbm = bs_i.power_per_rb_dbm() - pl_i_db + ray_i_db
        I += dbm_to_mw(prx_i_dbm)

    sinr = S / (I + NOISE_P_MW_1RB)
    if sinr <= 1e-14:
        sinr = 1e-14
    return sinr

def compute_rate_bps_from_sinr(sinr_linear: float, n_rb: int) -> tuple[float, float]:
    c_rb_bps = B * math.log2(1.0 + sinr_linear)
    rate_bps = c_rb_bps * n_rb
    return rate_bps, c_rb_bps

# ---------------------- Data Loading ----------------------
def load_all_data(base_dir: Path):
    # Each BS file has sheets: '大规模衰减', '小规模瑞丽衰减', with columns: Time + users
    bs_data = {}
    for bs_id in [1, 2, 3]:
        xls = pd.ExcelFile(base_dir / fr"D:\2025年第六届华数杯数学建模竞赛赛题\B题\附件\附件3\BS{bs_id}.xlsx")
        large = pd.read_excel(xls, sheet_name='大规模衰减').copy()
        small = pd.read_excel(xls, sheet_name='小规模瑞丽衰减').copy()
        large['ms'] = (large['Time'] * 1000).round().astype(int)
        small['ms'] = (small['Time'] * 1000).round().astype(int)
        large = large.drop(columns=['Time'])
        small = small.drop(columns=['Time'])
        bs_data[bs_id] = {'large': large, 'small': small}

    # Taskflow: 用户任务流（每 ms 各用户任务大小, Mbit） + 用户位置（每 ms 坐标）
    tf_xls = pd.ExcelFile(base_dir / r"D:\2025年第六届华数杯数学建模竞赛赛题\B题\附件\附件3\taskflow.xlsx")
    tasks = pd.read_excel(tf_xls, sheet_name='用户任务流').copy()
    pos   = pd.read_excel(tf_xls, sheet_name='用户位置').copy()
    tasks['ms'] = (tasks['Time'] * 1000).round().astype(int)
    pos['ms']   = (pos['Time']   * 1000).round().astype(int)
    tasks = tasks.drop(columns=['Time'])
    pos   = pos.drop(columns=['Time'])
    return bs_data, tasks, pos

# ---------------------- Binding by median distance of this period ----------------------
def bind_users_for_round(t_round: int, pos_df: pd.DataFrame) -> dict:
    start_ms = t_round * 100
    end_ms   = start_ms + 100
    pos_win = pos_df[(pos_df['ms'] >= start_ms) & (pos_df['ms'] < end_ms)].reset_index(drop=True)

    BS_COORDS = {
        1: (0.0, 500.0),
        2: (-433.0127, -250.0),
        3: (433.0127, -250.0),
    }

    binding = {}
    for u in ALL_USERS:
        x_col, y_col = f"{u}_X", f"{u}_Y"
        if x_col not in pos_win.columns or y_col not in pos_win.columns:
            binding[u] = 1
            continue
        
        distances = []
        for bs_id, coords in BS_COORDS.items():
            dx = pos_win[x_col] - coords[0]
            dy = pos_win[y_col] - coords[1]
            dist = (dx * dx + dy * dy).pow(0.5).median()
            distances.append((dist, bs_id))

        best_bs_id = min(distances, key=lambda x: (x[0], x[1]))[1]
        binding[u] = best_bs_id
    return binding

# ---------------------- Period slices search (single BS, beam search) ----------------------
def beam_search_for_round_single_bs(
    inherited_tasks: list,
    channel_large_period: pd.DataFrame,
    channel_small_period: pd.DataFrame,
    tasks_period: pd.DataFrame,
    round_start_ms: int,
    serving_bs: BaseStation,
    other_bs_periods: list,  # list of dicts: {'bs': BaseStation, 'large': df, 'small': df}
):
    def simulate_one_round_single(
        partition, inherited_tasks_local, ch_large, ch_small, tasks_per, round_start_ms_local,
        bs_serv: BaseStation, others_periods: list
    ):
        queues = {'URLLC': [], 'eMBB': [], 'mMTC': []}
        slice_qos = {'URLLC': 0.0, 'eMBB': 0.0, 'mMTC': 0.0}

        # inherit unfinished mMTC
        for t in inherited_tasks_local:
            queues[t.slice].append(t)

        # build arrivals list
        arrival_list = []
        for _, row in tasks_per.iterrows():
            t_global = int(row['ms'])
            for u in URLLC_USERS:
                if u in tasks_per.columns:
                    size = row[u]
                    if size > 0:
                        arrival_list.append(Task(u, 'URLLC', t_global, size))
            for u in EMBB_USERS:
                if u in tasks_per.columns:
                    size = row[u]
                    if size > 0:
                        arrival_list.append(Task(u, 'eMBB', t_global, size))
            for u in MMTC_USERS:
                if u in tasks_per.columns:
                    size = row[u]
                    if size > 0:
                        arrival_list.append(Task(u, 'mMTC', t_global, size))

        arrival_list.sort(key=lambda x: x.arrival)

        total_qos = 0.0
        leftovers = []

        # run 100 ms
        for local_t in range(100):
            t_global = int(round_start_ms_local) + local_t

            while arrival_list and arrival_list[0].arrival == t_global:
                queues[arrival_list[0].slice].append(arrival_list.pop(0))

            for slice_type, groups in zip(['URLLC', 'eMBB', 'mMTC'], partition):
                if queues[slice_type]:
                    kept = []
                    for task in queues[slice_type]:
                        if (t_global - task.arrival) >= SLA_DELAY[slice_type]:
                            task.finish_time = t_global
                            q = compute_qos(task)
                            total_qos += q
                            slice_qos[task.slice] += q
                        else:
                            kept.append(task)
                    queues[slice_type] = kept

                free_rb = GROUP_SIZE[slice_type] * groups
                while free_rb >= GROUP_SIZE[slice_type] and queues[slice_type]:
                    task = queues[slice_type].pop(0)
                    if task.start_time is None:
                        task.start_time = t_global

                    pl_db  = ch_large.at[local_t, task.user]
                    ray_db = ch_small.at[local_t, task.user]

                    interfs = []
                    for od in others_periods:
                        pl_i  = od['large'].at[local_t, task.user]
                        ray_i = od['small'].at[local_t, task.user]
                        interfs.append((od['bs'], pl_i, ray_i))

                    sinr_lin = compute_sinr_linear(bs_serv, pl_db, ray_db, interfs)
                    n_rb = GROUP_SIZE[slice_type]
                    free_rb -= n_rb
                    rate_bps, c_rb_bps = compute_rate_bps_from_sinr(sinr_lin, n_rb)
                    rate_mbps = rate_bps / 1e6
                    task.transmission_rate = rate_mbps
                    transmitted = rate_mbps * DELTA_T  # Mbit
                    task.remaining -= transmitted

                    if task.remaining <= 0:
                        task.finish_time = t_global + max(0.001, task.remaining / max(rate_mbps, 1e-9))
                        q = compute_qos(task)
                        total_qos += q
                        slice_qos[task.slice] += q
                    else:
                        queues[slice_type].insert(0, task)

        round_end_ms = round_start_ms_local + 100
        for slice_type in ['URLLC', 'eMBB', 'mMTC']:
            while queues[slice_type]:
                t = queues[slice_type].pop(0)
                if (round_end_ms - t.arrival) >= SLA_DELAY[slice_type]:
                    t.finish_time = round_end_ms
                    q = compute_qos(t)
                    total_qos += q
                    slice_qos[t.slice] += q
                else:
                    if slice_type == 'mMTC':
                        leftovers.append(t)

        return total_qos, leftovers, slice_qos

    # Beam search over partitions
    beam = [(1, 1, 1)]
    leftovers_map = {}
    while True:
        cands = set()
        for st in beam:
            for ns in expand_state(st, serving_bs.rb_total):
                cands.add(ns)
        if not cands:
            break
        scored = []
        for s in cands:
            qos, left, _ = simulate_one_round_single(
                s, inherited_tasks, channel_large_period, channel_small_period, tasks_period,
                round_start_ms, serving_bs, other_bs_periods
            )
            scored.append((qos, s, left))
        top = nlargest(BEAM_WIDTH, scored, key=lambda x: x[0])
        beam = [s for (_, s, _) in top]
        leftovers_map = {s: left for (_, s, left) in top}

    final = max(
        [(simulate_one_round_single(s, inherited_tasks, channel_large_period, channel_small_period, tasks_period,
                                    round_start_ms, serving_bs, other_bs_periods)[0], s) for s in beam],
        key=lambda x: x[0]
    )
    best_state = final[1]
    best_qos, best_left, best_slice_qos = simulate_one_round_single(
        best_state, inherited_tasks, channel_large_period, channel_small_period, tasks_period,
        round_start_ms, serving_bs, other_bs_periods
    )
    return best_state, best_qos, best_left, best_slice_qos

# ---------------------- Orchestration for 4 BS ----------------------
def schedule_all_rounds_multi_bs(num_rounds=10, p_init=(30, 30, 30)):
    base_dir = Path(__file__).resolve().parent
    bs_data, tasks_df, pos_df = load_all_data(base_dir)

    BS = {
        1: BaseStation(1, "BS1", tx_power_dbm=float(p_init[0]), rb_total=RB_LIMIT),
        2: BaseStation(2, "BS2", tx_power_dbm=float(p_init[1]), rb_total=RB_LIMIT),
        3: BaseStation(3, "BS3", tx_power_dbm=float(p_init[2]), rb_total=RB_LIMIT),

    }
 
    inherited = {1: [], 2: [], 3: []}
    p_prev = list(p_init)

    per_round_results = []

    for r in range(num_rounds):
        round_start_ms = r * 100
        round_end_ms   = round_start_ms + 100

        binding = bind_users_for_round(r, pos_df)

        tasks_win = tasks_df[(tasks_df['ms'] >= round_start_ms) & (tasks_df['ms'] < round_end_ms)].reset_index(drop=True)

        tasks_bs = {b: tasks_win[['ms']].copy() for b in [1,2,3]}
        for u in ALL_USERS:
            assigned_bs = binding[u]
            if u in tasks_win.columns:
                tasks_bs[assigned_bs][u] = tasks_win[u]

        ch_large = {}
        ch_small = {}
        for b in [1, 2, 3]:
            large = bs_data[b]['large']
            small = bs_data[b]['small']
            win_large = large[(large['ms'] >= round_start_ms) & (large['ms'] < round_end_ms)].drop(columns=['ms']).reset_index(drop=True)
            win_small = small[(small['ms'] >= round_start_ms) & (small['ms'] < round_end_ms)].drop(columns=['ms']).reset_index(drop=True)
            assert len(win_large) == 100 and len(win_small) == 100, f"round {r}: BS{b} channel rows != 100"
            ch_large[b] = win_large
            ch_small[b] = win_small

        # Step 1: per-BS slicing search with previous powers
        for b in [1, 2, 3]:
            BS[b].tx_power_dbm = float(p_prev[b-1])

        partitions = {}
        slice_qos_parts = {}
        for b in [1, 2, 3]:
            other_bses = [bb for bb in [1,2,3] if bb != b]
            other_periods = [{'bs': BS[bb], 'large': ch_large[bb], 'small': ch_small[bb]} for bb in other_bses]

            cols = ['ms'] + [c for c in tasks_bs[b].columns if c != 'ms' and c in ch_large[b].columns]
            tasks_b = tasks_bs[b][cols].copy()

            best_part, best_qos, best_left, best_slice_qos = beam_search_for_round_single_bs(
                inherited[b], ch_large[b], ch_small[b], tasks_b, round_start_ms,
                BS[b], other_periods
            )
            partitions[b] = best_part
            slice_qos_parts[b] = best_slice_qos
            inherited[b] = best_left  # provisional

        # Step 2: joint power enumeration ±1 dB from previous round's best
        def neighborhood(p, p_min, p_max):
            base = int(round(p))
            vals = list(range(base - 3, base + 4))
            return [max(p_min, min(p_max, v)) for v in vals]

        cand_p = [
            neighborhood(p_prev[0], 10, 30),
            neighborhood(p_prev[1], 10, 30),
            neighborhood(p_prev[2], 10, 30),
        ]

        best_tuple = None
        best_qos_total = -1e18

        for p1 in cand_p[0]:
            for p2 in cand_p[1]:
                for p3 in cand_p[2]:
                    # The values from cand_p are already clamped, so direct assignment is fine.
                    BS[1].tx_power_dbm, BS[2].tx_power_dbm, BS[3].tx_power_dbm = float(p1), float(p2), float(p3)

                    total_qos_sum = 0.0
                    for b in [1, 2, 3]:
                        other_bses = [bb for bb in [1,2,3] if bb != b]
                        other_periods = [{'bs': BS[bb], 'large': ch_large[bb], 'small': ch_small[bb]} for bb in other_bses]

                        # Move sim_fixed definition outside the for loop to correct indentation
                    def sim_fixed(partition, inh, chL, chS, tasksB, start_ms, bsS, othersP):
                        queues = {'URLLC': [], 'eMBB': [], 'mMTC': []}
                        slice_qos = {'URLLC': 0.0, 'eMBB': 0.0, 'mMTC': 0.0}
                        for t in inh:
                            queues[t.slice].append(t)
                        arrival_list = []
                        for _, row in tasksB.iterrows():
                            t_glob = int(row['ms'])
                            for uu in URLLC_USERS:
                                if uu in tasksB.columns:
                                    sz = row[uu]
                                    if sz > 0: arrival_list.append(Task(uu, 'URLLC', t_glob, sz))
                            for uu in EMBB_USERS:
                                if uu in tasksB.columns:
                                    sz = row[uu]
                                    if sz > 0: arrival_list.append(Task(uu, 'eMBB', t_glob, sz))
                            for uu in MMTC_USERS:
                                if uu in tasksB.columns:
                                    sz = row[uu]
                                    if sz > 0: arrival_list.append(Task(uu, 'mMTC', t_glob, sz))
                        arrival_list.sort(key=lambda x: x.arrival)
                        total_q = 0.0
                        for local_t in range(100):
                            t_glob = int(start_ms) + local_t
                            while arrival_list and arrival_list[0].arrival == t_glob:
                                queues[arrival_list[0].slice].append(arrival_list.pop(0))
                            for s_type, groups in zip(['URLLC', 'eMBB', 'mMTC'], partition):
                                if queues[s_type]:
                                    kept = []
                                    for tk in queues[s_type]:
                                        if (t_glob - tk.arrival) >= SLA_DELAY[s_type]:
                                            tk.finish_time = t_glob
                                            total_q += compute_qos(tk)
                                        else:
                                            kept.append(tk)
                                    queues[s_type] = kept
                                free_rb = GROUP_SIZE[s_type] * groups
                                while free_rb >= GROUP_SIZE[s_type] and queues[s_type]:
                                    tk = queues[s_type].pop(0)
                                    if tk.start_time is None: tk.start_time = t_glob
                                    pl_db  = chL.at[local_t, tk.user]
                                    ray_db = chS.at[local_t, tk.user]
                                    interfs = []
                                    for od in othersP:
                                        pl_i  = od['large'].at[local_t, tk.user]
                                        ray_i = od['small'].at[local_t, tk.user]
                                        interfs.append((od['bs'], pl_i, ray_i))
                                    sinr_lin = compute_sinr_linear(bsS, pl_db, ray_db, interfs)
                                    n_rb = GROUP_SIZE[s_type]
                                    free_rb -= n_rb
                                    rate_bps, _ = compute_rate_bps_from_sinr(sinr_lin, n_rb)
                                    rate_mbps = rate_bps / 1e6
                                    tk.transmission_rate = rate_mbps
                                    tk.remaining -= rate_mbps * DELTA_T
                                    if tk.remaining <= 0:
                                        tk.finish_time = t_glob + max(0.001, tk.remaining / max(rate_mbps, 1e-9))
                                        total_q += compute_qos(tk)
                                    else:
                                        queues[s_type].insert(0, tk)
                        round_end = start_ms + 100
                        for s_type in ['URLLC', 'eMBB', 'mMTC']:
                            while queues[s_type]:
                                tk = queues[s_type].pop(0)
                                if (round_end - tk.arrival) >= SLA_DELAY[s_type]:
                                    tk.finish_time = round_end
                                    total_q += compute_qos(tk)
                        return total_q

                    for b in [1, 2, 3]:
                        other_bses = [bb for bb in [1,2,3] if bb != b]
                        other_periods = [{'bs': BS[bb], 'large': ch_large[bb], 'small': ch_small[bb]} for bb in other_bses]
                        total_qos_sum += sim_fixed(partitions[b], inherited[b], ch_large[b], ch_small[b],
                                                   tasks_bs[b], round_start_ms, BS[b], other_periods)

                        cand_tuple = (p1, p2, p3)
                        if (total_qos_sum > best_qos_total) or (abs(total_qos_sum - best_qos_total) < 1e-12 and (best_tuple is None or cand_tuple < best_tuple)):
                            best_qos_total = total_qos_sum
                            best_tuple = cand_tuple

        p_curr = list(best_tuple)
        BS[1].tx_power_dbm, BS[2].tx_power_dbm, BS[3].tx_power_dbm = map(float, p_curr)

        # Final simulation to produce leftovers & slice QoS with chosen powers
        total_qos_final = 0.0 
        inherited_final = {1: [], 2: [], 3: []}
        slice_qos_final = {'URLLC': 0.0, 'eMBB': 0.0, 'mMTC': 0.0}
        for b in [1, 2, 3]:
            other_bses = [bb for bb in [1,2,3] if bb != b]
            other_periods = [{'bs': BS[bb], 'large': ch_large[bb], 'small': ch_small[bb]} for bb in other_bses]

            def simulate_final(partition, inh, chL, chS, tasksB, start_ms, bsS, othersP):
                queues = {'URLLC': [], 'eMBB': [], 'mMTC': []}
                slice_q = {'URLLC': 0.0, 'eMBB': 0.0, 'mMTC': 0.0}
                for t in inh:
                    queues[t.slice].append(t)
                arrival_list = []
                for _, row in tasksB.iterrows():
                    t_glob = int(row['ms'])
                    for uu in URLLC_USERS:
                        if uu in tasksB.columns:
                            sz = row[uu]
                            if sz > 0: arrival_list.append(Task(uu, 'URLLC', t_glob, sz))
                    for uu in EMBB_USERS:
                        if uu in tasksB.columns:
                            sz = row[uu]
                            if sz > 0: arrival_list.append(Task(uu, 'eMBB', t_glob, sz))
                    for uu in MMTC_USERS:
                        if uu in tasksB.columns:
                            sz = row[uu]
                            if sz > 0: arrival_list.append(Task(uu, 'mMTC', t_glob, sz))
                arrival_list.sort(key=lambda x: x.arrival)
                total_q = 0.0
                leftovers_out = []
                for local_t in range(100):
                    t_glob = int(start_ms) + local_t
                    while arrival_list and arrival_list[0].arrival == t_glob:
                        queues[arrival_list[0].slice].append(arrival_list.pop(0))
                    for s_type, groups in zip(['URLLC', 'eMBB', 'mMTC'], partition):
                        if queues[s_type]:
                            kept = []
                            for tk in queues[s_type]:
                                if (t_glob - tk.arrival) >= SLA_DELAY[s_type]:
                                    tk.finish_time = t_glob
                                    total_q += compute_qos(tk)
                                    slice_q[tk.slice] += compute_qos(tk)
                                else:
                                    kept.append(tk)
                            queues[s_type] = kept
                        free_rb = GROUP_SIZE[s_type] * groups
                        while free_rb >= GROUP_SIZE[s_type] and queues[s_type]:
                            tk = queues[s_type].pop(0)
                            if tk.start_time is None: tk.start_time = t_glob
                            pl_db  = chL.at[local_t, tk.user]
                            ray_db = chS.at[local_t, tk.user]
                            interfs = []
                            for od in othersP:
                                pl_i  = od['large'].at[local_t, tk.user]
                                ray_i = od['small'].at[local_t, tk.user]
                                interfs.append((od['bs'], pl_i, ray_i))
                            sinr_lin = compute_sinr_linear(bsS, pl_db, ray_db, interfs)
                            n_rb = GROUP_SIZE[s_type]
                            free_rb -= n_rb
                            rate_bps, _ = compute_rate_bps_from_sinr(sinr_lin, n_rb)
                            rate_mbps = rate_bps / 1e6
                            tk.transmission_rate = rate_mbps
                            tk.remaining -= rate_mbps * DELTA_T
                            if tk.remaining <= 0:
                                tk.finish_time = t_glob + max(0.001, tk.remaining / max(rate_mbps, 1e-9))
                                q = compute_qos(tk)
                                total_q += q
                                slice_q[tk.slice] += q
                            else:
                                queues[s_type].insert(0, tk)
                round_end = start_ms + 100
                for s_type in ['URLLC', 'eMBB', 'mMTC']:
                    while queues[s_type]:
                        tk = queues[s_type].pop(0)
                        if (round_end - tk.arrival) >= SLA_DELAY[s_type]:
                            tk.finish_time = round_end
                            q = compute_qos(tk)
                            total_q += q
                            slice_q[tk.slice] += q
                        else:
                            if s_type == 'mMTC':
                                leftovers_out.append(tk)
                return total_q, leftovers_out, slice_q

            qos_b, left_b, slice_q_b = simulate_final(
                partitions[b], inherited[b], ch_large[b], ch_small[b],
                tasks_bs[b], round_start_ms, BS[b], other_periods
            )
            total_qos_final += qos_b
            inherited_final[b] = left_b
            for k in slice_qos_final:
                slice_qos_final[k] += slice_q_b.get(k, 0.0)

        per_round_results.append({
            'round': r+1,
            'partition_BS1': partitions[1],
            'partition_BS2': partitions[2],
            'partition_BS3': partitions[3],
            'p_BS1': p_curr[0],
            'p_BS2': p_curr[1],
            'p_BS3': p_curr[2],
            'QoS_total': total_qos_final,
            'QoS_URLLC': slice_qos_final['URLLC'],
            'QoS_eMBB' : slice_qos_final['eMBB'],
            'QoS_mMTC' : slice_qos_final['mMTC'],
        })

        inherited = inherited_final
        p_prev = p_curr                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

    return per_round_results

if __name__ == "__main__":
    # Example run: 10 round
    results = schedule_all_rounds_multi_bs(num_rounds=10, p_init=(30,30,30))
    for r in results:        print(f"Round {r['round']}: "
              f"BS1 part={r['partition_BS1']}, BS2 part={r['partition_BS2']}, BS3 part={r['partition_BS3']}; "
              f"p=({r['p_BS1']},{r['p_BS2']},{r['p_BS3']}), "
              f"QoS_total={r['QoS_total']:.3f}, "
              f"URLLC={r['QoS_URLLC']:.3f}, eMBB={r['QoS_eMBB']:.3f}, mMTC={r['QoS_mMTC']:.3f}")