# -*- coding: utf-8 -*-
"""
高频订单因子计算（均值版）
"""
import os
import time as tm
import itertools
import warnings
from datetime import time, datetime
from typing import Optional
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# -------------------- 1. 参数 --------------------
DEBUG = False
AUCTION_AM = (time(9, 15), time(9, 25))
FINAL_CALL = (time(9, 25), time(9, 30))
CONT_AM = (time(9, 30), time(11, 30))
NOON_BREAK = (time(11, 30), time(13, 0))
CONT_PM = (time(13, 0), time(14, 57))
MORNING_END_INCLUSIVE = time(11, 30)     # 包含上午结束时间 11:30:00
AFTERNOON_START_INCLUSIVE = time(13, 0)

# -------------------- 2. 工具 --------------------
def calc_effective_duration_vec(start: pd.Series, end: pd.Series) -> np.ndarray:
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    day = start.dt.normalize()
    am_end, pm_start = day + pd.Timedelta("11:30:00"), day + pd.Timedelta("13:00:00")
    no_span = (end <= am_end) | (start >= pm_start)
    dur_no_span = (end - start).dt.total_seconds()
    dur_morning = (am_end - start).dt.total_seconds().clip(lower=0)
    dur_afternoon = (end - pm_start).dt.total_seconds().clip(lower=0)
    dur_span = dur_morning + dur_afternoon
    return np.where(no_span, dur_no_span, dur_span)


def calculate_threshold_mean_std(s: pd.Series) -> float:
    """计算阈值：均值+标准差"""
    if len(s) < 2:
        return s.max() if len(s) else 0.0
    mean, std = s.mean(), s.std()
    return mean if std == 0 else mean + std


# -------------------- 3. 单日单股 --------------------
def one_day_stock_factor(secucode: str, date: str, df: pd.DataFrame) -> Optional[dict]:
    """处理单日单只股票的高频订单因子"""
    total_volume = df["Volume"].sum()
    if total_volume == 0:
        return None

    # 3.1 买单聚合
    buy = (df.groupby("BuyOrderID")
           .agg(buy_volume=("Volume", "sum"),
                buy_first_time=("TradeTime", "min"),
                buy_last_time=("TradeTime", "max"))
           .reset_index())
    buy["buy_duration"] = calc_effective_duration_vec(buy["buy_first_time"], buy["buy_last_time"])

    # 3.2 卖单聚合
    sell = (df.groupby("SaleOrderID")
            .agg(sell_volume=("Volume", "sum"),
                 sell_first_time=("TradeTime", "min"),
                 sell_last_time=("TradeTime", "max"))
            .reset_index())
    sell["sell_duration"] = calc_effective_duration_vec(sell["sell_first_time"], sell["sell_last_time"])

    # 3.3 阈值计算
    buy_big_thr = calculate_threshold_mean_std(buy["buy_volume"])
    sell_big_thr = calculate_threshold_mean_std(sell["sell_volume"])
    buy_long_thr = calculate_threshold_mean_std(buy["buy_duration"])
    sell_long_thr = calculate_threshold_mean_std(sell["sell_duration"])

    df = (df.merge(buy[["BuyOrderID", "buy_volume", "buy_duration"]], how="left", on="BuyOrderID")
          .merge(sell[["SaleOrderID", "sell_volume", "sell_duration"]], how="left", on="SaleOrderID"))
    df[["buy_volume", "sell_volume", "buy_duration", "sell_duration"]] = \
        df[["buy_volume", "sell_volume", "buy_duration", "sell_duration"]].fillna(0)

    # 3.4 标记
    df["is_big_buy"] = df["buy_volume"] > buy_big_thr
    df["is_big_sell"] = df["sell_volume"] > sell_big_thr
    df["is_long_buy"] = df["buy_duration"] > buy_long_thr
    df["is_long_sell"] = df["sell_duration"] > sell_long_thr

    # 3.5 子因子计算
    def vol_ratio(mask):
        return df.loc[mask, "Volume"].sum() / total_volume if mask.any() else 0.0

    bb_ns = vol_ratio(df["is_big_buy"] & ~df["is_big_sell"])
    nb_bs = vol_ratio(~df["is_big_buy"] & df["is_big_sell"])
    bb_bs = vol_ratio(df["is_big_buy"] & df["is_big_sell"])
    lb_nls = vol_ratio(df["is_long_buy"] & ~df["is_long_sell"])
    nb_ls = vol_ratio(~df["is_long_buy"] & df["is_long_sell"])
    lb_ls = vol_ratio(df["is_long_buy"] & df["is_long_sell"])

    # 3.6 核心因子
    vol_big_orig = bb_ns + nb_bs + 2 * bb_bs
    vol_big = -bb_ns - nb_bs + bb_bs
    vol_long = lb_nls + nb_ls + 2 * lb_ls
    vol_long_big = vol_big + vol_long

    # 3.7 16类订单
    order_type = {}
    for bb, bs, lb, ls in itertools.product([0, 1], repeat=4):
        mask = (df["is_big_buy"].eq(bool(bb)) &
                df["is_big_sell"].eq(bool(bs)) &
                df["is_long_buy"].eq(bool(lb)) &
                df["is_long_sell"].eq(bool(ls)))
        order_type[f"BB{bb}_BS{bs}_LB{lb}_LS{ls}"] = vol_ratio(mask)

    select = np.mean([order_type["BB1_BS1_LB1_LS1"],
                      order_type["BB1_BS1_LB0_LS1"],
                      order_type["BB1_BS1_LB1_LS0"],
                      order_type["BB0_BS1_LB0_LS1"],
                      -order_type["BB1_BS0_LB0_LS0"]])

    # 3.8 返回结果
    return dict(
        secucode=secucode,
        date=date,
        total_volume=total_volume,
        total_trades=len(df),
        buy_orders=len(buy),
        sell_orders=len(sell),
        buy_big_threshold=buy_big_thr,
        sell_big_threshold=sell_big_thr,
        buy_long_threshold=buy_long_thr,
        sell_long_threshold=sell_long_thr,
        big_buy_non_big_sell=bb_ns,
        non_big_buy_big_sell=nb_bs,
        big_buy_big_sell=bb_bs,
        long_buy_non_long_sell=lb_nls,
        non_long_buy_long_sell=nb_ls,
        long_buy_long_sell=lb_ls,
        VolumeBigOrigin=vol_big_orig,
        VolumeBig=vol_big,
        VolumeLong=vol_long,
        VolumeLongBig=vol_long_big,
        VolumeLongBigSelect=select,
        **order_type)


# -------------------- 4. 主控--------------------
def calculate_all_hfa_factors(data_path: str, output_dir: str = None):
    """主计算函数 """
    timings = {}

    # 4.1 列裁剪 + 内存优化
    t0_start = tm.time()
    columns_needed = ["secucode", "date", "Time", "Volume", "BuyOrderID", "SaleOrderID"]
    table = pq.read_table(data_path, columns=columns_needed, memory_map=True)
    df = table.to_pandas()
    timings['数据加载'] = tm.time() - t0_start
    print(f"   数据加载耗时: {timings['数据加载']:.2f}s ，形状: {df.shape}")

    # 4.2 优化时间处理
    print(f"\n   🚀 优化时间处理（避免字符串转换）")
    t1_start = tm.time()

    # 检查数据类型
    print(f"      date列类型: {df['date'].dtype}, Time列类型: {df['Time'].dtype}")

    # 直接数值运算
    try:
        df['TradeTime'] = df['date'] + (df['Time'] - df['Time'].dt.normalize())
        print(f"      使用直接数值运算合并时间")
    except Exception as e:
        #备用方案（如果需要处理格式问题）
        print(f"      方法1失败，使用备用方案: {e}")
        df['TradeTime'] = pd.to_datetime(
            df['date'].dt.date.astype(str) + ' ' + df['Time'].dt.time.astype(str),
            format='%Y-%m-%d %H:%M:%S.%f',
            errors='coerce'
        )

    if df['TradeTime'].isna().any():
        print(f"      ⚠️  警告: {df['TradeTime'].isna().sum()} 条记录时间转换失败")
        df = df.dropna(subset=['TradeTime']).copy()

    # 提取时间部分并过滤
    df['time_only'] = df['TradeTime'].dt.time
    mask = (((df["time_only"] >= CONT_AM[0]) & (df["time_only"] <= MORNING_END_INCLUSIVE)) |
            ((df["time_only"] >= AFTERNOON_START_INCLUSIVE ) & (df["time_only"] < CONT_PM[1])))
    df = df[mask].copy()

    timings['时间处理过滤'] = tm.time() - t1_start
    print(f"   时间处理+过滤耗时: {timings['时间处理过滤']:.2f}s ，连续竞价记录: {len(df):,}")

    # 4.3 优化分组准备
    print(f"\n   🚀 优化分组准备（避免字符串转换）")
    t2_start = tm.time()

    df = df.sort_values(["secucode", "date"]).reset_index(drop=True)
    groups = []
    group_count = 0

    for (stk, date_val), sub in df.groupby(["secucode", "date"], sort=False):
        date_str = date_val.strftime('%Y%m%d')
        groups.append((stk, date_str, sub[["TradeTime", "Volume", "BuyOrderID", "SaleOrderID"]].copy()))
        group_count += 1

    timings['分组准备'] = tm.time() - t2_start
    print(f"   分组准备耗时: {timings['分组准备']:.2f}s ，任务数: {len(groups)}")

    # 4.4 并行计算
    t3_start = tm.time()
    n_tasks = len(groups)
    n_jobs = min(os.cpu_count(), n_tasks, 28)
    print(f"\n   ⚡ 使用 {n_jobs} 个进程进行并行计算")

    with parallel_backend('loky', n_jobs=n_jobs):
        results = Parallel(verbose=10, batch_size='auto', max_nbytes='1M')(
            delayed(one_day_stock_factor)(stk, d, sub)
            for stk, d, sub in groups
        )

    results = [r for r in results if r is not None]
    timings['并行计算'] = tm.time() - t3_start
    print(f"   并行计算耗时: {timings['并行计算']:.2f}s ，有效结果: {len(results)}")

    # 4.5 结果保存
    t4_start = tm.time()
    if results:
        factors_df = pd.DataFrame(results)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "高频订单因子_优化版.feather")
            factors_df.reset_index(drop=True).to_feather(out_path)

            csv_path = os.path.join(output_dir, "高频订单因子_优化版.csv")
            factors_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            timings['结果保存'] = tm.time() - t4_start

            print(f"\n💾 结果已保存 -> {out_path}")
            print(f"   同时保存为CSV -> {csv_path}")
            print(f"   结果保存耗时: {timings['结果保存']:.2f}s")
            print(f"   结果形状: {factors_df.shape}")

        return factors_df, timings
    else:
        print("⚠️  没有有效结果")
        return pd.DataFrame(), timings


# -------------------- 5. 入口 --------------------
if __name__ == "__main__":
    prog_start = tm.time()
    print(f"程序开始: {datetime.now():%F %T}")

    DATA = r"D:/pycharm/pythonProject/dataExample_5k.parquet"
    OUT_DIR = r"D:/pycharm/pythonProject"

    # 运行主程序
    df_fac, timings = calculate_all_hfa_factors(DATA, OUT_DIR)

    total_time = tm.time() - prog_start

    print(f"\n" + "=" * 80)
    print("📈 性能分析报告")
    print("=" * 80)

    # 汇总所有耗时
    print(f"\n总运行时间: {total_time:.3f}秒")
    print(f"程序开始: {datetime.fromtimestamp(prog_start):%F %T}")
    print(f"程序结束: {datetime.now():%F %T}")

    print(f"\n各阶段耗时详情:")
    print("-" * 60)

    # 按阶段分类显示
    stage_times = {
        '数据加载': timings.get('数据加载', 0),
        '时间处理过滤': timings.get('时间处理过滤', 0),
        '分组准备': timings.get('分组准备', 0),
        '并行计算': timings.get('并行计算', 0),
        '结果保存': timings.get('结果保存', 0)
    }

    for stage_name, stage_time in stage_times.items():
        if stage_time > 0:
            percentage = (stage_time / total_time) * 100
            print(f"{stage_name}: {stage_time:.3f}s ({percentage:.1f}%)")

    # 显示部分结果
    if not df_fac.empty:
        print(f"\n📋 计算结果预览 (前5行):")
        print(df_fac[['secucode', 'date', 'total_volume', 'total_trades', 'VolumeBig', 'VolumeLong',
                      'VolumeLongBig']].head())