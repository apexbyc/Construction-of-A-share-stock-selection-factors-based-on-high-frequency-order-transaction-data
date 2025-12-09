# -*- coding: utf-8 -*-
"""
高频订单因子计算 - 单日版
核心功能：处理单日股票高频交易数据，计算订单行为特征因子
优化特点：
1. 不使用date列（因为所有数据都是同一天）
2. 输出CSV和Parquet格式
"""
import os
import time as tm
import warnings
from datetime import time, datetime
from typing import Optional
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# -------------------- 1. 全局参数定义 --------------------
CONT_AM = (time(9, 30), time(11, 30))  # 上午连续竞价时间段
CONT_PM = (time(13, 0), time(14, 57))  # 下午连续竞价时间段


# -------------------- 2. 核心计算函数 --------------------
def compute_factors_ultimate_single_day(secucode: str, target_date: str, df: pd.DataFrame) -> Optional[dict]:
    """
    单日优化版因子计算函数

    功能：计算股票的高频订单因子
    输入：
        secucode: 股票代码
        target_date: 交易日期（字符串格式）
        df: 单只股票的单日交易数据
    输出：
        dict: 包含所有计算因子的字典，如果数据无效返回None
    """
    # 数据有效性检查：空数据或成交量为0的数据直接跳过
    if df.empty or df["Volume"].sum() == 0:
        return None

    # 总成交量：用于后续比例计算
    total_volume = float(df["Volume"].sum())

    # -------------------- 步骤1：订单聚合 --------------------
    # 按买方订单ID分组，计算每个买单的特征
    # observed=True：优化groupby性能，减少内存使用
    buy = df.groupby("BuyOrderID", observed=True).agg(
        buy_volume=("Volume", "sum"),  # 买单总成交量
        buy_first_time=("TradeTime", "min"),  # 买单首次成交时间
        buy_last_time=("TradeTime", "max")  # 买单末次成交时间
    ).reset_index()

    # 按卖方订单ID分组，计算每个卖单的特征
    sell = df.groupby("SaleOrderID", observed=True).agg(
        sell_volume=("Volume", "sum"),  # 卖单总成交量
        sell_first_time=("TradeTime", "min"),  # 卖单首次成交时间
        sell_last_time=("TradeTime", "max")  # 卖单末次成交时间
    ).reset_index()

    # -------------------- 步骤2：计算订单持续时间 --------------------
    def calc_duration_fast(start, end):
        """
        快速计算订单持续时间，考虑午休时间扣除

        逻辑：
        1. 计算原始持续时间（秒）
        2. 如果订单跨越午休（开始<11:30且结束>13:00），扣除5400秒（1.5小时）
        3. 返回调整后的持续时间
        """
        duration = (end - start).dt.total_seconds()
        spans_noon = (start.dt.hour < 12) & (end.dt.hour >= 13)
        return duration - spans_noon.astype(int) * 5400

    # 计算买单和卖单的持续时间
    buy["buy_duration"] = calc_duration_fast(buy["buy_first_time"], buy["buy_last_time"])
    sell["sell_duration"] = calc_duration_fast(sell["sell_first_time"], sell["sell_last_time"])

    # -------------------- 步骤3：计算阈值（90%分位数） --------------------
    def threshold_fast(series):
        """
        快速计算阈值（90%分位数）

        逻辑：
        1. 提取序列值
        2. 如果数据少于2个，返回唯一值或0
        3. 使用numpy的percentile计算90%分位数，比pandas更快
        """
        vals = series.values
        if len(vals) < 2:
            return float(vals[0]) if len(vals) == 1 else 0.0
        return float(np.percentile(vals[~np.isnan(vals)], 90))

    # 计算4个阈值：大买单、大卖单、长买单、长卖单的阈值
    buy_big_thr = threshold_fast(buy["buy_volume"])  # 大买单成交量阈值
    sell_big_thr = threshold_fast(sell["sell_volume"])  # 大卖单成交量阈值
    buy_long_thr = threshold_fast(buy["buy_duration"])  # 长买单持续时间阈值
    sell_long_thr = threshold_fast(sell["sell_duration"])  # 长卖单持续时间阈值

    # -------------------- 步骤4：将订单特征映射回原始数据 --------------------
    # 创建字典映射：订单ID -> 订单特征（比merge更高效）
    buy_vol_map = dict(zip(buy["BuyOrderID"], buy["buy_volume"]))
    buy_dur_map = dict(zip(buy["BuyOrderID"], buy["buy_duration"]))
    sell_vol_map = dict(zip(sell["SaleOrderID"], sell["sell_volume"]))
    sell_dur_map = dict(zip(sell["SaleOrderID"], sell["sell_duration"]))

    # 使用map函数快速映射（向量化操作）
    df["buy_volume"] = df["BuyOrderID"].map(buy_vol_map).fillna(0)
    df["sell_volume"] = df["SaleOrderID"].map(sell_vol_map).fillna(0)
    df["buy_duration"] = df["BuyOrderID"].map(buy_dur_map).fillna(0)
    df["sell_duration"] = df["SaleOrderID"].map(sell_dur_map).fillna(0)

    # -------------------- 步骤5：标记大单和长单 --------------------
    # 布尔标记：是否为大于阈值的订单
    df["is_big_buy"] = df["buy_volume"] > buy_big_thr  # 大买单标记
    df["is_big_sell"] = df["sell_volume"] > sell_big_thr  # 大卖单标记
    df["is_long_buy"] = df["buy_duration"] > buy_long_thr  # 长买单标记
    df["is_long_sell"] = df["sell_duration"] > sell_long_thr  # 长卖单标记

    # -------------------- 步骤6：计算16类订单比例因子 --------------------
    # 使用二进制编码将4个布尔标记转换为0-15的整数（16种组合）
    # 编码规则：is_big_buy(8) + is_big_sell(4) + is_long_buy(2) + is_long_sell(1)
    code = (df["is_big_buy"].astype(int) * 8 +
            df["is_big_sell"].astype(int) * 4 +
            df["is_long_buy"].astype(int) * 2 +
            df["is_long_sell"].astype(int))

    # 按编码分组，计算每类订单的成交量占总成交量的比例
    grouped = df.groupby(code)["Volume"].sum() / total_volume

    # 构建16个订单类型因子的字典
    # 命名格式：BB{大买单标记}_BS{大卖单标记}_LB{长买单标记}_LS{长卖单标记}
    order_type = {}
    for i in range(16):
        # 从编码中提取4个标记
        bb = (i & 8) // 8  # 提取大买单标记（第4位）
        bs = (i & 4) // 4  # 提取大卖单标记（第3位）
        lb = (i & 2) // 2  # 提取长买单标记（第2位）
        ls = i & 1  # 提取长卖单标记（第1位）

        # 获取该类型的比例，如果不存在则为0.0
        order_type[f"BB{bb}_BS{bs}_LB{lb}_LS{ls}"] = float(grouped.get(i, 0.0))

    # -------------------- 步骤7：计算6个子因子 --------------------
    # 从16类订单中组合出6个有意义的子因子

    # bb_ns: 大买单非大卖单（编码8-11：大买单=1，大卖单=0）
    bb_ns_codes = [8, 9, 10, 11]
    bb_ns = sum(grouped.get(c, 0.0) for c in bb_ns_codes)

    # nb_bs: 非大买单大卖单（编码4-7：大买单=0，大卖单=1）
    nb_bs_codes = [4, 5, 6, 7]
    nb_bs = sum(grouped.get(c, 0.0) for c in nb_bs_codes)

    # bb_bs: 大买单大卖单（编码12-15：大买单=1，大卖单=1）
    bb_bs_codes = [12, 13, 14, 15]
    bb_bs = sum(grouped.get(c, 0.0) for c in bb_bs_codes)

    # lb_nls: 长买单非长卖单（长买单=1，长卖单=0）
    lb_nls_codes = [2, 3, 6, 7, 10, 11, 14, 15]
    lb_nls = sum(grouped.get(c, 0.0) for c in lb_nls_codes)

    # nb_ls: 非长买单长卖单（长买单=0，长卖单=1）
    nb_ls_codes = [1, 5, 9, 13]
    nb_ls = sum(grouped.get(c, 0.0) for c in nb_ls_codes)

    # lb_ls: 长买单长卖单（长买单=1，长卖单=1）
    lb_ls_codes = [3, 7, 11, 15]
    lb_ls = sum(grouped.get(c, 0.0) for c in lb_ls_codes)

    # -------------------- 步骤8：计算4个核心因子 --------------------
    # VolumeBigOrigin: 大单原始比例（加权计算）
    vol_big_orig = bb_ns + nb_bs + 2 * bb_bs

    # VolumeBig: 大单净流向因子（多头为正，空头为负）
    vol_big = -bb_ns - nb_bs + bb_bs

    # VolumeLong: 长单净流向因子
    vol_long = lb_nls + nb_ls + 2 * lb_ls

    # VolumeLongBig: 长单大单综合因子
    vol_long_big = vol_big + vol_long

    # -------------------- 步骤9：构建返回结果 --------------------
    return {
        # 基本信息
        "secucode": secucode,  # 股票代码
        "date": target_date,  # 交易日期
        "total_volume": total_volume,  # 总成交量
        "total_trades": len(df),  # 总交易笔数
        "buy_orders": len(buy),  # 买单数量
        "sell_orders": len(sell),  # 卖单数量

        # 阈值信息
        "buy_big_threshold": buy_big_thr,  # 大买单阈值
        "sell_big_threshold": sell_big_thr,  # 大卖单阈值
        "buy_long_threshold": buy_long_thr,  # 长买单阈值
        "sell_long_threshold": sell_long_thr,  # 长卖单阈值

        # 6个子因子
        "big_buy_non_big_sell": bb_ns,  # 大买单非大卖单比例
        "non_big_buy_big_sell": nb_bs,  # 非大买单大卖单比例
        "big_buy_big_sell": bb_bs,  # 大买单大卖单比例
        "long_buy_non_long_sell": lb_nls,  # 长买单非长卖单比例
        "non_long_buy_long_sell": nb_ls,  # 非长买单长卖单比例
        "long_buy_long_sell": lb_ls,  # 长买单长卖单比例

        # 4个核心因子
        "VolumeBigOrigin": vol_big_orig,  # 大单原始比例
        "VolumeBig": vol_big,  # 大单净流向因子
        "VolumeLong": vol_long,  # 长单净流向因子
        "VolumeLongBig": vol_long_big,  # 长单大单综合因子

        # 16个订单类型因子（展开到字典中）
        **order_type
    }


# -------------------- 3. 主函数 --------------------
def calculate_factors_single_day_complete(data_path: str, target_date: str, output_dir: str = None):
    """
    主函数：单日数据完整因子计算流程

    功能：组织完整的因子计算流程，包括数据加载、预处理、计算和保存
    输入：
        data_path: 数据文件路径（Parquet格式）
        target_date: 目标日期（字符串格式，如"2024-01-15"）
        output_dir: 输出目录路径
    输出：
        tuple: (结果DataFrame, 各阶段耗时字典)
    """
    print("=" * 80)
    print("📊 高频订单因子计算 - 单日完整版")
    print(f"📅 目标日期: {target_date}")
    print("⚡ 特点：不使用date列，用16个因子替换select因子")
    print("=" * 80)

    timings = {}  # 记录各阶段耗时

    # -------------------- 阶段1：数据加载 --------------------
    t0 = tm.time()
    print("\n   1. 数据加载...")

    # 仅读取必要的列（不包含date列）
    columns_needed = ["secucode", "Time", "Volume", "BuyOrderID", "SaleOrderID"]
    df = pq.read_table(data_path, columns=columns_needed).to_pandas()

    timings['数据加载'] = tm.time() - t0
    print(f"      ✓ 耗时: {timings['数据加载']:.1f}s, 记录数: {len(df):,}")

    # -------------------- 阶段2：时间处理与过滤 --------------------
    t1 = tm.time()
    print("   2. 时间处理与过滤...")

    # 转换Time列为datetime格式（原始数据可能是字符串）
    df['Time'] = pd.to_datetime(df['Time'])

    # 创建基础日期时间戳（使用传入的target_date）
    base_date = pd.to_datetime(target_date)

    # 合并时间：将基础日期与时间部分组合成完整的时间戳
    df['TradeTime'] = base_date + (df['Time'] - df['Time'].dt.normalize())

    # 获取时间部分（用于过滤）
    time_only = df['TradeTime'].dt.time

    # 标记需要调整的时间点
    mask_pre = time_only < time(9, 30)  # 早于9:30
    mask_noon = (time_only > time(11, 30)) & (time_only < time(13, 0))  # 午休时间
    mask_close = time_only >= time(14, 57)  # 收盘后

    # 调整非连续竞价时间到最近的连续竞价时间
    df.loc[mask_pre, 'TradeTime'] = base_date + pd.Timedelta(hours=9, minutes=30)
    df.loc[mask_noon, 'TradeTime'] = base_date + pd.Timedelta(hours=13, minutes=0)
    df.loc[mask_close, 'TradeTime'] = base_date + pd.Timedelta(hours=14, minutes=57)

    # 重新获取调整后的时间
    time_only = df['TradeTime'].dt.time

    # 过滤：只保留连续竞价时间段的数据
    mask = ((time_only >= time(9, 30)) & (time_only <= time(11, 30))) | \
           ((time_only >= time(13, 0)) & (time_only < time(14, 57)))
    df = df[mask].copy()  # 使用copy避免SettingWithCopyWarning

    timings['时间处理过滤'] = tm.time() - t1
    print(f"      ✓ 耗时: {timings['时间处理过滤']:.1f}s, 过滤后: {len(df):,}")

    # -------------------- 阶段3：分组准备 --------------------
    t2 = tm.time()
    print("   3. 分组准备...")

    groups = []
    # 按股票代码分组（因为是单日数据，不需要再按date分组）
    for secucode, sub_df in df.groupby("secucode"):
        # 只保留计算所需的列，减少内存占用
        groups.append((secucode, target_date, sub_df[["TradeTime", "Volume", "BuyOrderID", "SaleOrderID"]]))

    timings['分组准备'] = tm.time() - t2
    print(f"      ✓ 耗时: {timings['分组准备']:.1f}s, 分组数: {len(groups):,}")

    # -------------------- 阶段4：因子计算 --------------------
    t3 = tm.time()
    print("   4. 因子计算...")

    results = []  # 存储所有股票的计算结果
    total = len(groups)
    start_time = tm.time()  # 用于进度显示

    # 遍历每只股票进行计算
    for i, (stk, date_str, sub_df) in enumerate(groups):
        # 调用核心计算函数
        result = compute_factors_ultimate_single_day(stk, date_str, sub_df)
        if result:
            results.append(result)

        # 进度显示（每100只股票或每5秒显示一次）
        if (i + 1) % 100 == 0 or tm.time() - start_time >= 5:
            elapsed = tm.time() - t3
            progress = (i + 1) / total * 100
            # 计算剩余时间
            remaining = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0

            print(f"        进度: {i + 1}/{total} ({progress:.1f}%) - "
                  f"已用: {elapsed:.1f}s - 剩余: {remaining:.1f}s")
            start_time = tm.time()

    timings['因子计算'] = tm.time() - t3
    print(f"      ✓ 耗时: {timings['因子计算']:.1f}s, 有效结果: {len(results):,}")

    # -------------------- 阶段5：保存结果 --------------------
    t4 = tm.time()
    if results:  # 如果有有效结果
        # 将结果列表转换为DataFrame
        factors_df = pd.DataFrame(results)

        if output_dir:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)

            # 生成文件名（包含日期）
            date_str_for_filename = target_date.replace('-', '')[:8]  # 格式化为YYYYMMDD
            base_filename = f"高频订单因子_单日完整_{date_str_for_filename}"

            # 保存为CSV格式（便于人工查看）
            csv_path = os.path.join(output_dir, f"{base_filename}.csv")
            factors_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            # 保存为Parquet格式（高性能二进制格式，便于后续分析）
            parquet_path = os.path.join(output_dir, f"{base_filename}.parquet")
            factors_df.to_parquet(parquet_path, index=False)

            timings['结果保存'] = tm.time() - t4

            # 输出保存信息
            print(f"\n💾 结果已保存:")
            print(f"   CSV: {csv_path}")
            print(f"   Parquet: {parquet_path}")
            print(f"   结果形状: {factors_df.shape}")
            print(f"   列数: {len(factors_df.columns)}")

            # 显示详细的列统计信息
            print(f"\n📊 输出列统计:")
            print(f"   基本信息列: 7个")
            print(f"   阈值列: 4个")
            print(f"   子因子列: 6个")
            print(f"   核心因子列: 4个")
            print(f"   订单类型因子: 16个")
            print(f"   总列数: {7 + 4 + 6 + 4 + 16}个")

        return factors_df, timings

    # 如果没有结果，返回空的DataFrame
    return pd.DataFrame(), timings


# -------------------- 4. 程序入口 --------------------
if __name__ == "__main__":
    # 记录程序开始时间
    prog_start = tm.time()
    print(f"程序开始: {datetime.now():%F %T}")

    # ==================== 配置参数 ====================
    # 数据文件路径
    DATA = r"D:/pycharm/pythonProject/dataExample.parquet"

    # 输出目录
    OUT_DIR = r"D:/pycharm/pythonProject"

    # 目标日期（根据实际数据设置）
    TARGET_DATE = "2024-01-15"
    try:
        # 执行因子计算
        df_fac, timings = calculate_factors_single_day_complete(
            data_path=DATA,
            target_date=TARGET_DATE,
            output_dir=OUT_DIR
        )

        # 计算总运行时间
        total_time = tm.time() - prog_start

        # 输出运行时间统计
        print(f"\n总运行时间: {total_time:.3f}秒")
        print(f"程序开始: {datetime.fromtimestamp(prog_start):%F %T}")
        print(f"程序结束: {datetime.now():%F %T}")

        # 输出各阶段耗时详情
        print(f"\n各阶段耗时详情:")
        print("-" * 60)
        for name, time_val in timings.items():
            if time_val > 0:
                percentage = (time_val / total_time) * 100
                print(f"{name}: {time_val:.3f}s ({percentage:.1f}%)")

        # 如果计算结果不为空，显示预览和统计信息
        if not df_fac.empty:
            print(f"\n📋 计算结果预览 (前3行):")
            # 选择关键列进行预览
            key_columns = ['secucode', 'date', 'total_volume', 'total_trades',
                           'VolumeBig', 'VolumeLong', 'VolumeLongBig']
            # 找出所有16个订单类型因子
            factor_columns = [col for col in df_fac.columns if col.startswith('BB')]
            # 合并预览列：关键列 + 前3个订单类型因子
            preview_cols = key_columns + factor_columns[:3]

            # 显示前3行数据
            print(df_fac[preview_cols].head(3))

            # 输出统计信息
            print(f"\n📊 因子统计:")
            print(f"   股票数量: {len(df_fac)}")
            print(f"   总因子数: {len(df_fac.columns)}")
            print(f"   16个订单类型因子: {len(factor_columns)}个")

            # 验证：检查是否包含所有应有的因子
            expected_columns = [
                'secucode', 'date', 'total_volume', 'total_trades', 'buy_orders', 'sell_orders',
                'buy_big_threshold', 'sell_big_threshold', 'buy_long_threshold', 'sell_long_threshold',
                'big_buy_non_big_sell', 'non_big_buy_big_sell', 'big_buy_big_sell',
                'long_buy_non_long_sell', 'non_long_buy_long_sell', 'long_buy_long_sell',
                'VolumeBigOrigin', 'VolumeBig', 'VolumeLong', 'VolumeLongBig'
            ]

            # 检查是否有缺失的因子
            missing = [col for col in expected_columns if col not in df_fac.columns]
            if missing:
                print(f"   ⚠️  缺少的原有因子: {missing}")
            else:
                print("   ✅ 所有原有因子都在输出中")

    # 异常处理
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
        import traceback

        traceback.print_exc()