"""
optimizer_lp_dual.py

這個模組提供企業 PPA 投資組合的線性規劃（LP）求解器。

========================
【這個模組要吃什麼 input file？】
========================
1. plant_load.csv
   - 每列代表一個小時
   - timestamp 為時間欄位
   - 其他欄位為各廠的負載（kWh）

2. renewable_profiles.csv
   - 每列代表一個小時
   - timestamp 為時間欄位
   - 其他欄位為各案場的發電 profile
   - 值的意義：每 1 kW 簽約容量，在該小時可提供多少 kWh

3. candidate_config.csv
   - 每列代表一個案場
   - 必要欄位：
     asset_name, asset_type, min_kw, max_kw, ppa_price_per_kwh
   - step_kw 可保留，但這個 LP 版本不使用

========================
【模型最終結果是什麼？】
========================
1. 每個案場的最佳固定簽約容量（contract_capacity_kw）
2. CFE ratio（全年 clean served / total load）
3. CFE hourly compliance（有多少比例的小時達到 100% clean）
4. RE ratio（全年綠電發電量覆蓋負載的比例）
5. Total cost / Avg cost per kWh
6. 每小時的 dispatch 結果：
   - renewable supply total
   - storage charge / discharge
   - grid purchase
   - curtailment
   - storage SOC

========================
【目標函數是什麼？】
========================
支援兩種模式：

A. weighted（加權模式）
   max  w_cfe * CFE + w_re * RE - w_cost * normalized_cost

B. target_min_cost（門檻模式）
   min total_cost
   subject to:
       CFE >= cfe_target
       RE  >= re_target

========================
【重要建模假設】
========================
1. 各案場簽約容量是「連續變數」
   - 例如可以是 1234.56 kW
   - 不做 500 kW 一格
   - 不做 0/1 是否簽約

2. 儲能只允許由再生能源充電
   - 不允許 grid 充電

3. 本版本是 LP，不是 MILP
   - 優點：8760 小時、多案場仍相對可解
   - 缺點：容量不是離散的，不含 binary 合約選址

4. LP 只用來求最佳容量
   - 最終 dispatch 不直接採用 LP 內部變數
   - 而是改用實務調度邏輯模擬：
     renewable -> load -> storage -> curtailment / discharge -> grid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pulp


# =========================================================
# 資料結構
# =========================================================
@dataclass
class PlantLoad:
    name: str
    series: pd.Series


@dataclass
class RenewableAsset:
    name: str
    asset_type: str
    generation_profile_per_kw: pd.Series
    ppa_price_per_kwh: float
    min_capacity_kw: float = 0.0
    max_capacity_kw: float = 0.0


@dataclass
class StorageAsset:
    name: str
    energy_capacity_kwh: float
    power_capacity_kw: float
    round_trip_efficiency: float = 0.90
    initial_soc_ratio: float = 0.50
    min_soc_ratio: float = 0.10
    max_soc_ratio: float = 0.90
    annual_cost: float = 0.0


# =========================================================
# 輸入資料讀取與檢查
# =========================================================
def validate_time_indexes(
    plant_loads: List[PlantLoad],
    renewable_assets: List[RenewableAsset],
) -> None:
    if not plant_loads:
        raise ValueError("plant_loads 不可為空。")
    if not renewable_assets:
        raise ValueError("renewable_assets 不可為空。")

    base_index = plant_loads[0].series.index

    for plant in plant_loads:
        if not plant.series.index.equals(base_index):
            raise ValueError(f"廠區 {plant.name} 的時間序列與其他廠區不一致。")

    for asset in renewable_assets:
        if not asset.generation_profile_per_kw.index.equals(base_index):
            raise ValueError(f"再生能源 {asset.name} 的時間序列與廠區負載不一致。")


def load_plant_loads(df_load: pd.DataFrame) -> List[PlantLoad]:
    if "timestamp" not in df_load.columns:
        raise ValueError("負載檔必須包含 timestamp 欄位。")

    df = df_load.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    plant_loads: List[PlantLoad] = []
    for col in df.columns:
        if col == "timestamp":
            continue
        plant_loads.append(
            PlantLoad(
                name=col,
                series=pd.Series(df[col].astype(float).values, index=df["timestamp"], name=col),
            )
        )

    if not plant_loads:
        raise ValueError("負載檔至少要有一個廠區欄位。")

    return plant_loads


def load_renewables(df_re: pd.DataFrame, config_df: pd.DataFrame) -> List[RenewableAsset]:
    if "timestamp" not in df_re.columns:
        raise ValueError("再生能源檔必須包含 timestamp 欄位。")

    required_cols = {
        "asset_name",
        "asset_type",
        "min_kw",
        "max_kw",
        "ppa_price_per_kwh",
    }
    if not required_cols.issubset(set(config_df.columns)):
        raise ValueError(f"candidate_config.csv 缺少必要欄位：{required_cols}")

    df = df_re.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    cfg = config_df.copy()
    cfg["asset_name"] = cfg["asset_name"].astype(str)

    assets: List[RenewableAsset] = []
    for _, row in cfg.iterrows():
        asset_name = row["asset_name"]
        if asset_name not in df.columns:
            raise ValueError(f"renewable_profiles.csv 缺少資源欄位：{asset_name}")

        series = pd.Series(
            df[asset_name].astype(float).values,
            index=df["timestamp"],
            name=asset_name,
        )

        assets.append(
            RenewableAsset(
                name=asset_name,
                asset_type=str(row["asset_type"]),
                generation_profile_per_kw=series,
                ppa_price_per_kwh=float(row["ppa_price_per_kwh"]),
                min_capacity_kw=float(row["min_kw"]),
                max_capacity_kw=float(row["max_kw"]),
            )
        )

    return assets


# =========================================================
# 實務 dispatch 模擬器
# =========================================================
def simulate_practical_dispatch(
    index_: pd.Index,
    total_load: pd.Series,
    renewable_assets: List[RenewableAsset],
    cap_solution: Dict[str, float],
    storage: Optional[StorageAsset] = None,
) -> Dict:
    """
    用「實務調度順序」重建 dispatch：

    1. renewable 先供 load
    2. surplus 再充電
    3. 剩餘才 curtail
    4. 若 load 還不足，再由 storage 放電
    5. 最後才由 grid 補足

    注意：
    - 這裡是 rule-based dispatch
    - 不直接使用 LP 內部的 ren_to_load / discharge / curtail 等變數
    """

    renewable_supply_by_asset = {}
    renewable_supply_total = pd.Series(0.0, index=index_, name="renewable_supply_total")

    for a in renewable_assets:
        series = a.generation_profile_per_kw * float(cap_solution[a.name])
        renewable_supply_by_asset[a.name] = series.rename(f"supply_{a.name}")
        renewable_supply_total = renewable_supply_total.add(series, fill_value=0.0)

    # 無儲能情況
    if storage is None or storage.energy_capacity_kwh <= 0 or storage.power_capacity_kw <= 0:
        ren_to_load_vals = []
        charge_vals = []
        discharge_vals = []
        soc_vals = []
        grid_vals = []
        curtail_vals = []

        for t in index_:
            load_t = float(total_load.loc[t])
            ren_t = float(renewable_supply_total.loc[t])

            ren_used = min(load_t, ren_t)
            surplus = max(ren_t - ren_used, 0.0)
            grid_t = max(load_t - ren_used, 0.0)

            ren_to_load_vals.append(ren_used)
            charge_vals.append(0.0)
            discharge_vals.append(0.0)
            soc_vals.append(np.nan)
            grid_vals.append(grid_t)
            curtail_vals.append(surplus)

        ren_to_load = pd.Series(ren_to_load_vals, index=index_, name="ren_to_load")
        storage_charge = pd.Series(charge_vals, index=index_, name="storage_charge")
        storage_discharge = pd.Series(discharge_vals, index=index_, name="storage_discharge")
        storage_soc = pd.Series(soc_vals, index=index_, name="storage_soc")
        grid_purchase = pd.Series(grid_vals, index=index_, name="grid_purchase")
        curtailment = pd.Series(curtail_vals, index=index_, name="curtailment")
        clean_served = (ren_to_load + storage_discharge).rename("clean_served")

    else:
        eta_c = np.sqrt(float(storage.round_trip_efficiency))
        eta_d = np.sqrt(float(storage.round_trip_efficiency))

        soc_min = float(storage.energy_capacity_kwh) * float(storage.min_soc_ratio)
        soc_max = float(storage.energy_capacity_kwh) * float(storage.max_soc_ratio)
        soc = float(storage.energy_capacity_kwh) * float(storage.initial_soc_ratio)

        ren_to_load_vals = []
        charge_vals = []
        discharge_vals = []
        soc_vals = []
        grid_vals = []
        curtail_vals = []

        for t in index_:
            load_t = float(total_load.loc[t])
            ren_t = float(renewable_supply_total.loc[t])

            # 1) renewable 先供 load
            ren_used = min(load_t, ren_t)

            # 2) surplus 再充電
            surplus = max(ren_t - ren_used, 0.0)
            charge_limit_by_soc = max((soc_max - soc) / eta_c, 0.0) if eta_c > 0 else 0.0
            charge_t = min(float(storage.power_capacity_kw), surplus, charge_limit_by_soc)

            soc = soc + charge_t * eta_c

            # 3) 若 load 還不足，再由 storage 放電
            load_after_ren = load_t - ren_used
            discharge_limit_by_soc = max((soc - soc_min) * eta_d, 0.0) if eta_d > 0 else 0.0
            discharge_t = min(float(storage.power_capacity_kw), load_after_ren, discharge_limit_by_soc)

            soc = soc - discharge_t / eta_d if eta_d > 0 else soc

            # 4) 最後才 grid
            grid_t = max(load_after_ren - discharge_t, 0.0)

            # 5) 剩餘才 curtail
            curtail_t = max(surplus - charge_t, 0.0)

            ren_to_load_vals.append(ren_used)
            charge_vals.append(charge_t)
            discharge_vals.append(discharge_t)
            soc_vals.append(soc)
            grid_vals.append(grid_t)
            curtail_vals.append(curtail_t)

        ren_to_load = pd.Series(ren_to_load_vals, index=index_, name="ren_to_load")
        storage_charge = pd.Series(charge_vals, index=index_, name="storage_charge")
        storage_discharge = pd.Series(discharge_vals, index=index_, name="storage_discharge")
        storage_soc = pd.Series(soc_vals, index=index_, name="storage_soc")
        grid_purchase = pd.Series(grid_vals, index=index_, name="grid_purchase")
        curtailment = pd.Series(curtail_vals, index=index_, name="curtailment")
        clean_served = (ren_to_load + storage_discharge).rename("clean_served")

    hourly_clean_ratio = (
        clean_served / total_load.replace(0, pd.NA)
    ).fillna(0).clip(lower=0, upper=1).rename("hourly_clean_ratio")

    timeseries_df = pd.DataFrame(
        {
            "total_load": total_load,
            "renewable_supply_total": renewable_supply_total,
            "ren_to_load": ren_to_load,
            "storage_charge": storage_charge,
            "storage_discharge": storage_discharge,
            "grid_purchase": grid_purchase,
            "curtailment": curtailment,
            "storage_soc": storage_soc,
            "clean_served": clean_served,
            "hourly_clean_ratio": hourly_clean_ratio,
        }
    )

    for name, series in renewable_supply_by_asset.items():
        timeseries_df[series.name] = series

    return {
        "timeseries": timeseries_df,
        "renewable_supply_total": renewable_supply_total,
        "renewable_supply_by_asset": renewable_supply_by_asset,
        "ren_to_load": ren_to_load,
        "storage_charge": storage_charge,
        "storage_discharge": storage_discharge,
        "storage_soc": storage_soc,
        "grid_purchase": grid_purchase,
        "curtailment": curtailment,
        "clean_served": clean_served,
        "hourly_clean_ratio": hourly_clean_ratio,
    }


# =========================================================
# 核心求解器
# =========================================================
def solve_portfolio_lp(
    plant_loads: List[PlantLoad],
    renewable_assets: List[RenewableAsset],
    storage: Optional[StorageAsset] = None,
    grid_price_per_kwh: float = 4.0,
    mode: str = "weighted",
    objective_weights: Optional[Dict[str, float]] = None,
    cfe_target: Optional[float] = None,
    re_target: Optional[float] = None,
    solver_name: str = "CBC",
    time_limit_seconds: Optional[int] = None,
) -> Dict:
    validate_time_indexes(plant_loads, renewable_assets)

    mode = mode.lower().strip()
    if mode not in {"weighted", "target_min_cost"}:
        raise ValueError("mode 只支援 'weighted' 或 'target_min_cost'。")

    if objective_weights is None:
        objective_weights = {"cfe": 0.45, "re": 0.25, "cost": 0.30}

    if mode == "weighted":
        if abs(sum(objective_weights.values()) - 1.0) > 1e-6:
            raise ValueError("weighted 模式下，objective_weights 加總必須為 1。")

    if mode == "target_min_cost":
        if cfe_target is None or re_target is None:
            raise ValueError("target_min_cost 模式下，必須提供 cfe_target 與 re_target。")
        if not (0 <= cfe_target <= 1 and 0 <= re_target <= 1):
            raise ValueError("cfe_target 與 re_target 必須介於 0~1。")

    index_ = plant_loads[0].series.index
    time_labels = list(range(len(index_)))

    # -----------------------------------------------------
    # Step 1. 匯總總負載
    # -----------------------------------------------------
    total_load = pd.Series(0.0, index=index_, name="total_load")
    for plant in plant_loads:
        total_load = total_load.add(plant.series, fill_value=0.0)

    total_load_sum = float(total_load.sum())
    if total_load_sum <= 0:
        raise ValueError("總用電量必須大於 0。")

    load_by_t = {t: float(total_load.iloc[t]) for t in time_labels}

    # -----------------------------------------------------
    # Step 2. 建立 LP 模型
    # -----------------------------------------------------
    sense = pulp.LpMaximize if mode == "weighted" else pulp.LpMinimize
    prob = pulp.LpProblem("Enterprise_PPA_Portfolio_Optimization", sense)

    # -----------------------------------------------------
    # Step 3. 容量決策變數
    # -----------------------------------------------------
    cap_vars = {
        a.name: pulp.LpVariable(
            f"cap_{a.name}",
            lowBound=float(a.min_capacity_kw),
            upBound=float(a.max_capacity_kw),
            cat="Continuous",
        )
        for a in renewable_assets
    }

    # -----------------------------------------------------
    # Step 4. 簡化用的 LP dispatch 變數
    # -----------------------------------------------------
    # 注意：
    # 這些變數只用來讓 LP 有能力評估容量的優劣，
    # 最後輸出的 dispatch 不直接採用它們。
    ren_to_load = {t: pulp.LpVariable(f"ren_to_load_{t}", lowBound=0) for t in time_labels}
    grid_to_load = {t: pulp.LpVariable(f"grid_to_load_{t}", lowBound=0) for t in time_labels}
    curtail = {t: pulp.LpVariable(f"curtail_{t}", lowBound=0) for t in time_labels}

    if storage is not None and storage.energy_capacity_kwh > 0 and storage.power_capacity_kw > 0:
        ren_to_storage = {
            t: pulp.LpVariable(
                f"ren_to_storage_{t}",
                lowBound=0,
                upBound=float(storage.power_capacity_kw),
            )
            for t in time_labels
        }
        discharge = {
            t: pulp.LpVariable(
                f"discharge_{t}",
                lowBound=0,
                upBound=float(storage.power_capacity_kw),
            )
            for t in time_labels
        }

        soc_min = float(storage.energy_capacity_kwh) * float(storage.min_soc_ratio)
        soc_max = float(storage.energy_capacity_kwh) * float(storage.max_soc_ratio)
        soc_init = float(storage.energy_capacity_kwh) * float(storage.initial_soc_ratio)

        soc = {
            t: pulp.LpVariable(
                f"soc_{t}",
                lowBound=soc_min,
                upBound=soc_max,
            )
            for t in time_labels
        }

        eta_c = np.sqrt(float(storage.round_trip_efficiency))
        eta_d = np.sqrt(float(storage.round_trip_efficiency))
    else:
        ren_to_storage = {t: 0 for t in time_labels}
        discharge = {t: 0 for t in time_labels}
        soc = {}
        eta_c = None
        eta_d = None
        storage = None

    # -----------------------------------------------------
    # Step 5. 每小時總綠電供應表達式
    # -----------------------------------------------------
    renewable_supply_expr = {}
    for t in time_labels:
        renewable_supply_expr[t] = pulp.lpSum(
            float(a.generation_profile_per_kw.iloc[t]) * cap_vars[a.name]
            for a in renewable_assets
        )

    # -----------------------------------------------------
    # Step 6. LP 供需平衡約束
    # -----------------------------------------------------
    for t in time_labels:
        prob += (
            renewable_supply_expr[t]
            == ren_to_load[t] + ren_to_storage[t] + curtail[t],
            f"renewable_balance_{t}",
        )

        prob += (
            load_by_t[t] == ren_to_load[t] + discharge[t] + grid_to_load[t],
            f"load_balance_{t}",
        )

    # -----------------------------------------------------
    # Step 7. LP 儲能 SOC 約束
    # -----------------------------------------------------
    if storage is not None:
        for t in time_labels:
            if t == 0:
                prob += (
                    soc[t] == soc_init + eta_c * ren_to_storage[t] - discharge[t] / eta_d,
                    f"soc_init_{t}",
                )
            else:
                prob += (
                    soc[t] == soc[t - 1] + eta_c * ren_to_storage[t] - discharge[t] / eta_d,
                    f"soc_dyn_{t}",
                )

        # 期末 SOC 回到初始值
        prob += (soc[time_labels[-1]] == soc_init, "terminal_soc")

    # -----------------------------------------------------
    # Step 8. LP 用的 CFE / RE 指標表達式
    # -----------------------------------------------------
    clean_served_expr = pulp.lpSum(ren_to_load[t] + discharge[t] for t in time_labels)
    total_renewable_generation_expr = pulp.lpSum(renewable_supply_expr[t] for t in time_labels)

    re_credit = pulp.LpVariable(
        "re_credit",
        lowBound=0,
        upBound=total_load_sum,
        cat="Continuous",
    )
    prob += (re_credit <= total_renewable_generation_expr, "re_credit_cap_supply")
    prob += (re_credit <= total_load_sum, "re_credit_cap_load")

    # -----------------------------------------------------
    # Step 9. 成本表達式
    # -----------------------------------------------------
    total_ppa_cost_expr = pulp.lpSum(
        float(a.ppa_price_per_kwh) * pulp.lpSum(
            float(a.generation_profile_per_kw.iloc[t]) * cap_vars[a.name]
            for t in time_labels
        )
        for a in renewable_assets
    )
    total_grid_cost_expr = float(grid_price_per_kwh) * pulp.lpSum(grid_to_load[t] for t in time_labels)
    total_storage_cost = float(storage.annual_cost) if storage is not None else 0.0
    total_cost_expr = total_ppa_cost_expr + total_grid_cost_expr + total_storage_cost

    baseline_all_grid_cost = total_load_sum * float(grid_price_per_kwh)
    if baseline_all_grid_cost <= 0:
        baseline_all_grid_cost = 1.0

    # -----------------------------------------------------
    # Step 10. 目標函數
    # -----------------------------------------------------
    if mode == "weighted":
        objective = (
            float(objective_weights["cfe"]) * (clean_served_expr / total_load_sum)
            + float(objective_weights["re"]) * (re_credit / total_load_sum)
            - float(objective_weights["cost"]) * (total_cost_expr / baseline_all_grid_cost)
        )
        prob += objective
    else:
        prob += clean_served_expr >= float(cfe_target) * total_load_sum, "cfe_target_constraint"
        prob += re_credit >= float(re_target) * total_load_sum, "re_target_constraint"
        prob += total_cost_expr

    # -----------------------------------------------------
    # Step 11. Solver
    # -----------------------------------------------------
    solver_name = solver_name.upper()
    if solver_name == "CBC":
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    elif solver_name == "HIGHS":
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit_seconds)
    else:
        raise ValueError("solver_name 目前只支援 'CBC' 或 'HiGHS'。")

    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status in {"Infeasible", "Unbounded"}:
        raise RuntimeError(f"模型無可行解，求解狀態：{status}")

    # -----------------------------------------------------
    # Step 12. 最佳容量解
    # -----------------------------------------------------
    cap_solution = {a.name: float(cap_vars[a.name].value() or 0.0) for a in renewable_assets}

    # =====================================================
    # Step 13. 用實務調度邏輯重建 dispatch
    # =====================================================
    dispatch = simulate_practical_dispatch(
        index_=index_,
        total_load=total_load,
        renewable_assets=renewable_assets,
        cap_solution=cap_solution,
        storage=storage,
    )

    renewable_supply_total = dispatch["renewable_supply_total"]
    renewable_supply_by_asset = dispatch["renewable_supply_by_asset"]
    ren_to_load_series = dispatch["ren_to_load"]
    ren_to_storage_series = dispatch["storage_charge"]
    discharge_series = dispatch["storage_discharge"]
    grid_series = dispatch["grid_purchase"]
    curtail_series = dispatch["curtailment"]
    soc_series = dispatch["storage_soc"]
    clean_served_series = dispatch["clean_served"]
    hourly_clean_ratio = dispatch["hourly_clean_ratio"]
    timeseries_df = dispatch["timeseries"]

    # -----------------------------------------------------
    # Step 14. KPI
    # -----------------------------------------------------
    total_ppa_cost = sum(
        float(renewable_supply_by_asset[a.name].sum()) * float(a.ppa_price_per_kwh)
        for a in renewable_assets
    )
    total_grid_cost = float(grid_series.sum()) * float(grid_price_per_kwh)
    total_cost = total_ppa_cost + total_grid_cost + total_storage_cost
    avg_cost_per_kwh = total_cost / total_load_sum

    cfe_ratio = float(clean_served_series.sum() / total_load_sum)
    cfe_hourly_compliance = float((hourly_clean_ratio >= 0.999999).mean())
    re_ratio = float(min(renewable_supply_total.sum(), total_load_sum) / total_load_sum)
    grid_ratio = float(grid_series.sum() / total_load_sum)
    curtailment_ratio = (
        float(curtail_series.sum() / renewable_supply_total.sum())
        if renewable_supply_total.sum() > 0
        else 0.0
    )

    # -----------------------------------------------------
    # Step 15. 輸出表格
    # -----------------------------------------------------
    capacities_df = pd.DataFrame(
        {
            "asset_name": [a.name for a in renewable_assets],
            "asset_type": [a.asset_type for a in renewable_assets],
            "ppa_price_per_kwh": [a.ppa_price_per_kwh for a in renewable_assets],
            "contract_capacity_kw": [cap_solution[a.name] for a in renewable_assets],
        }
    )

    ppa_cost_by_asset_df = pd.DataFrame(
        {
            "asset_name": [a.name for a in renewable_assets],
            "ppa_generation_kwh": [renewable_supply_by_asset[a.name].sum() for a in renewable_assets],
            "ppa_price_per_kwh": [a.ppa_price_per_kwh for a in renewable_assets],
        }
    )
    ppa_cost_by_asset_df["ppa_cost"] = (
        ppa_cost_by_asset_df["ppa_generation_kwh"] * ppa_cost_by_asset_df["ppa_price_per_kwh"]
    )

    return {
        "mode": mode,
        "status": status,
        "objective_value": float(pulp.value(prob.objective) or 0.0),
        "cfe_ratio": cfe_ratio,
        "cfe_hourly_compliance": cfe_hourly_compliance,
        "re_ratio": re_ratio,
        "grid_ratio": grid_ratio,
        "curtailment_ratio": curtailment_ratio,
        "total_load_kwh": float(total_load_sum),
        "total_renewable_generation_kwh": float(renewable_supply_total.sum()),
        "total_clean_served_kwh": float(clean_served_series.sum()),
        "total_grid_purchase_kwh": float(grid_series.sum()),
        "total_curtailment_kwh": float(curtail_series.sum()),
        "total_ppa_cost": float(total_ppa_cost),
        "total_grid_cost": float(total_grid_cost),
        "total_storage_cost": float(total_storage_cost),
        "total_cost": float(total_cost),
        "avg_cost_per_kwh": float(avg_cost_per_kwh),
        "baseline_all_grid_cost": float(baseline_all_grid_cost),
        "timeseries": timeseries_df,
        "capacities": capacities_df,
        "ppa_cost_by_asset": ppa_cost_by_asset_df,
    }