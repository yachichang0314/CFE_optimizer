"""
appV4_lp_full.py

完整版本：
1. 使用 LP 求解，不用 brute force
2. Dispatch 圖可指定日期
3. Heatmap 可指定月份
4. 切換日期 / 月份時，不會要求重新求解（使用 session_state 快取）
"""

from __future__ import annotations

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from optimizer import (
    StorageAsset,
    load_plant_loads,
    load_renewables,
    solve_portfolio_lp,
)

st.set_page_config(
    page_title="Enterprise PPA Optimizer (LP Dual Mode)",
    page_icon="⚡",
    layout="wide",
)

# -------------------------------------------------
# Session state：避免切換日期 / 月份後重新求解
# -------------------------------------------------
if "solve_result" not in st.session_state:
    st.session_state["solve_result"] = None

# -------------------------------------------------
# 樣式
# -------------------------------------------------
st.markdown(
    """
    <style>
    .main {background: linear-gradient(180deg, #f5f7fb 0%, #eef3f8 100%);}
    .block-container {max-width: 1450px; padding-top: 2rem; padding-bottom: 2rem;}
    .hero {
        background: linear-gradient(135deg, #0f2f4f 0%, #1f5c8c 55%, #4c8db5 100%);
        color: white; border-radius: 22px; padding: 28px 34px;
        box-shadow: 0 10px 28px rgba(16, 44, 74, 0.20); margin-bottom: 1.2rem;
    }
    .hero-title {font-size: 2rem; font-weight: 800; margin-bottom: 0.35rem;}
    .hero-subtitle {font-size: 1rem; opacity: 0.95; line-height: 1.6;}
    .panel-card, .kpi-card {
        background: white; border-radius: 18px; padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        border: 1px solid rgba(20, 30, 50, 0.05); margin-bottom: 1rem;
    }
    .kpi-title {color: #5e6c7b; font-size: 0.92rem; font-weight: 700; margin-bottom: 0.35rem;}
    .kpi-value {color: #14314b; font-size: 1.65rem; font-weight: 800; margin-bottom: 0.25rem;}
    .kpi-note {color: #6f7c8b; font-size: 0.84rem; line-height: 1.45;}
    .pill {
        display: inline-block; padding: 6px 12px; border-radius: 999px; background: #e8f2fb;
        color: #1e5d93; font-size: 0.82rem; font-weight: 700; margin-right: 8px;
    }
    .small-text {color: #5d6a79; font-size: 0.93rem; line-height: 1.65;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# 工具函式
# -------------------------------------------------
def uploaded_file_to_df(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    df = pd.read_csv(io.BytesIO(file_bytes))

    if len(df.columns) == 1 and "," in str(df.columns[0]):
        text = file_bytes.decode("utf-8-sig")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        split_rows = [line.split(",") for line in lines]
        df = pd.DataFrame(split_rows[1:], columns=split_rows[0])

    elif len(df.columns) == 1 and "，" in str(df.columns[0]):
        text = file_bytes.decode("utf-8-sig").replace("，", ",")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        split_rows = [line.split(",") for line in lines]
        df = pd.DataFrame(split_rows[1:], columns=split_rows[0])

    return df


def make_kpi_card(title: str, value: str, note: str):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_currency(v: float) -> str:
    return f"{v:,.0f}"


# -------------------------------------------------
# 頁首
# -------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">⚡ Enterprise PPA Optimizer (LP Dual Mode)</div>
        <div class="hero-subtitle">
            協助企業滿足24/7、RE100等綠能倡議的最適綠電投資組合。
            兩種模式：加權模式(24/7、RE100、成本)與門檻模式。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.title("控制面板")

    load_file = st.file_uploader("1) plant_load.csv", type=["csv"])
    renewable_file = st.file_uploader("2) renewable_profiles.csv", type=["csv"])
    candidate_file = st.file_uploader("3) candidate_config.csv", type=["csv"])

    st.markdown("---")
    mode = st.radio(
        "求解模式",
        ["weighted", "target_min_cost"],
        format_func=lambda x: "加權模式" if x == "weighted" else "門檻模式（最低成本）",
    )

    if mode == "weighted":
        st.markdown("### 加權模式參數")
        w_cfe = st.slider("CFE 權重", 0.0, 1.0, 0.45, 0.05)
        w_re = st.slider("RE 權重", 0.0, 1.0, 0.25, 0.05)
        w_cost = st.slider("成本權重", 0.0, 1.0, 0.30, 0.05)
        weight_sum = w_cfe + w_re + w_cost
        st.caption(f"目前權重加總：{weight_sum:.2f}（必須等於 1.00）")
        cfe_target = None
        re_target = None
    else:
        st.markdown("### 門檻模式參數")
        cfe_target = st.slider("CFE 目標值", 0.0, 1.0, 0.80, 0.01)
        re_target = st.slider("RE 目標值", 0.0, 1.0, 0.95, 0.01)
        w_cfe = w_re = w_cost = None
        weight_sum = None

    st.markdown("---")
    grid_price_per_kwh = st.number_input("電網購電價格 (元/kWh)", min_value=0.0, value=4.0, step=0.1)

    st.markdown("---")
    st.markdown("### 儲能設定")
    use_storage = st.checkbox("啟用儲能", value=True)
    storage_energy_kwh = st.number_input("儲能容量 (kWh)", min_value=0.0, value=4000.0, step=100.0)
    storage_power_kw = st.number_input("儲能功率 (kW)", min_value=0.0, value=1100.0, step=100.0)
    storage_eff = st.slider("Round-trip efficiency", 0.50, 1.00, 0.90, 0.01)
    storage_init_soc = st.slider("初始 SOC", 0.00, 1.00, 0.50, 0.05)
    storage_min_soc = st.slider("最小 SOC", 0.00, 1.00, 0.10, 0.05)
    storage_max_soc = st.slider("最大 SOC", 0.00, 1.00, 0.90, 0.05)
    storage_annual_cost = st.number_input("儲能年成本 (元/年)", min_value=0.0, value=500000.0, step=100000.0)

    st.markdown("---")
    solver_name = st.selectbox("Solver", ["CBC", "HiGHS"], index=0)
    time_limit_seconds = st.number_input("求解上限秒數（0=不限制）", min_value=0, value=60, step=10)

    run_btn = st.button("開始求解", type="primary", use_container_width=True)
    clear_btn = st.button("清除目前結果", use_container_width=True)

    if clear_btn:
        st.session_state["solve_result"] = None

info1, info2 = st.columns([1.35, 1])
with info1:
    st.markdown(
        """
        <div class="panel-card">
            <span class="pill">LP Optimization</span>
            <span class="pill">年負載發電</span>
            <span class="pill">Multi-asset</span>
            <div style="height:10px;"></div>
            <div class="small-text">
                以線性規劃直接求最佳解。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with info2:
    st.markdown(
        """
        <div class="panel-card">
            <div class="small-text">
                <b>必要 input file：</b><br>
                plant_load.csv<br>
                renewable_profiles.csv<br>
                candidate_config.csv
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab1, tab2, tab3 = st.tabs(["資料格式", "資料預覽", "求解結果"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### plant_load.csv")
        st.code("timestamp,Plant_A,Plant_B\n2025-01-01 00:00:00,120,90\n2025-01-01 01:00:00,115,88", language="csv")
    with c2:
        st.markdown("#### renewable_profiles.csv")
        st.code("timestamp,PV_1,PV_2,WIND_1,WIND_2\n2025-01-01 00:00:00,0,0,0.42,0.38\n2025-01-01 01:00:00,0,0,0.40,0.36", language="csv")
    with c3:
        st.markdown("#### candidate_config.csv")
        st.code("asset_name,asset_type,min_kw,max_kw,ppa_price_per_kwh\nPV_1,solar,0,5000,2.6\nWIND_1,wind,0,8000,3.1", language="csv")

with tab2:
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("#### 廠區負載檔")
        if load_file is not None:
            st.dataframe(uploaded_file_to_df(load_file).head(10), use_container_width=True)
        else:
            st.info("尚未上傳")
    with p2:
        st.markdown("#### 再生能源時序檔")
        if renewable_file is not None:
            st.dataframe(uploaded_file_to_df(renewable_file).head(10), use_container_width=True)
        else:
            st.info("尚未上傳")
    with p3:
        st.markdown("#### 候選容量與價格檔")
        if candidate_file is not None:
            st.dataframe(uploaded_file_to_df(candidate_file).head(10), use_container_width=True)
        else:
            st.info("尚未上傳")

with tab3:
    if run_btn:
        loading_placeholder = st.empty()
        try:
            with loading_placeholder.container():
                st.image("assets/snail.gif", width=120)
                st.caption("努力算最佳解中...")
            if mode == "weighted" and abs(weight_sum - 1.0) > 1e-9:
                st.error("加權模式下，CFE / RE / 成本 三個權重加總必須等於 1.00。")
                st.stop()

            if load_file is None or renewable_file is None or candidate_file is None:
                st.error("請先上傳三個 CSV 檔案。")
                st.stop()

            df_load = uploaded_file_to_df(load_file)
            df_re = uploaded_file_to_df(renewable_file)
            df_candidate = uploaded_file_to_df(candidate_file)

            plant_loads = load_plant_loads(df_load)
            renewable_assets = load_renewables(df_re, df_candidate)

            storage = None
            if use_storage and storage_energy_kwh > 0 and storage_power_kw > 0:
                storage = StorageAsset(
                    name="BESS_1",
                    energy_capacity_kwh=float(storage_energy_kwh),
                    power_capacity_kw=float(storage_power_kw),
                    round_trip_efficiency=float(storage_eff),
                    initial_soc_ratio=float(storage_init_soc),
                    min_soc_ratio=float(storage_min_soc),
                    max_soc_ratio=float(storage_max_soc),
                    annual_cost=float(storage_annual_cost),
                )

            result = solve_portfolio_lp(
                plant_loads=plant_loads,
                renewable_assets=renewable_assets,
                storage=storage,
                grid_price_per_kwh=float(grid_price_per_kwh),
                mode=mode,
                objective_weights=None if mode != "weighted" else {
                    "cfe": float(w_cfe),
                    "re": float(w_re),
                    "cost": float(w_cost),
                },
                cfe_target=cfe_target,
                re_target=re_target,
                solver_name=solver_name,
                time_limit_seconds=None if int(time_limit_seconds) == 0 else int(time_limit_seconds),
            )

            st.session_state["solve_result"] = {"result": result}

        except Exception as e:
            st.error(f"執行失敗：{e}")
            st.stop()
        finally:
            loading_placeholder.empty()
    if st.session_state["solve_result"] is not None:
        result = st.session_state["solve_result"]["result"]

        ts = result["timeseries"].copy().reset_index().rename(columns={"index": "timestamp"})
        caps_df = result["capacities"].copy()
        cost_df = result["ppa_cost_by_asset"].copy()

        st.markdown("### KPI 摘要")
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            make_kpi_card("CFE Ratio", f"{result['cfe_ratio']:.2%}", "全年 clean served / total load")
        with k2:
            make_kpi_card("CFE Hours", f"{result['cfe_hourly_compliance']:.2%}", "100% clean matching 的小時比例")
        with k3:
            make_kpi_card("RE Ratio", f"{result['re_ratio']:.2%}", "年度綠電比率")
        with k4:
            make_kpi_card("Total Cost", fmt_currency(result["total_cost"]), "PPA + Grid + Storage")

        k5, k6, k7, k8 = st.columns(4)
        with k5:
            make_kpi_card("Avg Cost", f"{result['avg_cost_per_kwh']:.2f}", "元 / kWh")
        with k6:
            make_kpi_card("Grid Ratio", f"{result['grid_ratio']:.2%}", "外購電占總負載比例")
        with k7:
            make_kpi_card("Curtailment", f"{result['curtailment_ratio']:.2%}", "棄電占總綠電比例")
        with k8:
            make_kpi_card("Mode / Status", f"{result['mode']} / {result['status']}", f"Objective={result['objective_value']:.4f}")

        left, right = st.columns([1.0, 1.15])
        with left:
            st.markdown("### 最佳固定簽約容量")
            st.dataframe(caps_df.sort_values(["asset_type", "asset_name"]).reset_index(drop=True), use_container_width=True, hide_index=True)
        with right:
            overview_df = pd.DataFrame(
                {
                    "指標": ["總用電量 (kWh)", "總綠電發電量 (kWh)", "總 clean served (kWh)", "外購電量 (kWh)", "棄電量 (kWh)", "PPA 成本", "Grid 成本", "Storage 成本"],
                    "數值": [round(result["total_load_kwh"], 2), round(result["total_renewable_generation_kwh"], 2), round(result["total_clean_served_kwh"], 2), round(result["total_grid_purchase_kwh"], 2), round(result["total_curtailment_kwh"], 2), fmt_currency(result["total_ppa_cost"]), fmt_currency(result["total_grid_cost"]), fmt_currency(result["total_storage_cost"])],
                }
            )
            st.markdown("### 結果概覽")
            st.dataframe(overview_df, use_container_width=True, hide_index=True)

        subtab1, subtab2, subtab3, subtab4, subtab5 = st.tabs(["Dispatch 圖", "Hourly Matching Map", "成本拆解", "PPA 成本明細", "時序明細"])

        with subtab1:
            st.markdown("#### Dispatch 圖（可指定日期）")
            ts_dispatch = ts.copy()
            ts_dispatch["date_only"] = pd.to_datetime(ts_dispatch["timestamp"]).dt.date
            available_dates = sorted(ts_dispatch["date_only"].unique())

            selected_date = st.selectbox(
                "選擇要查看的日期",
                options=available_dates,
                index=0,
                key="dispatch_date_select",
            )

            day_df = ts_dispatch[ts_dispatch["date_only"] == selected_date].copy()
            supply_cols = [c for c in day_df.columns if c.startswith("supply_")]

            fig_dispatch = go.Figure()
            color_pv = ["#FDB863", "#F46D43", "#FEE08B", "#FDAE61"]
            color_wind = ["#74ADD1", "#4575B4", "#A6CEE3", "#1F78B4"]
            pv_idx = 0
            wind_idx = 0

            for col in supply_cols:
                asset = col.replace("supply_", "")
                cap = float(caps_df.loc[caps_df["asset_name"] == asset, "contract_capacity_kw"].iloc[0])
                if "PV" in asset.upper():
                    color = color_pv[pv_idx % len(color_pv)]
                    pv_idx += 1
                else:
                    color = color_wind[wind_idx % len(color_wind)]
                    wind_idx += 1

                fig_dispatch.add_trace(
                    go.Bar(
                        x=day_df["timestamp"],
                        y=day_df[col],
                        name=f"{asset} ({cap:.0f} kW)",
                        marker=dict(color=color),
                    )
                )

            fig_dispatch.add_trace(
                go.Bar(
                    x=day_df["timestamp"],
                    y=day_df["storage_discharge"],
                    name="Storage",
                    marker=dict(color="#D73027"),
                )
            )
            fig_dispatch.add_trace(
                go.Bar(
                    x=day_df["timestamp"],
                    y=day_df["grid_purchase"],
                    name="Grid Purchase",
                    marker=dict(color="#9E9E9E", opacity=0.8),
                )
            )
            fig_dispatch.add_trace(
                go.Scatter(
                    x=day_df["timestamp"],
                    y=day_df["total_load"],
                    name="Total Load",
                    mode="lines",
                    line=dict(color="#2B2B2B", width=2),
                )
            )
            fig_dispatch.add_trace(
                go.Scatter(
                    x=day_df["timestamp"],
                    y=day_df["renewable_supply_total"],
                    name="Renewable Supply Total",
                    mode="lines",
                    line=dict(color="#66C2A5", dash="dash", width=2),
                )
            )

            fig_dispatch.update_layout(
                barmode="stack",
                height=520,
                xaxis_title="Time",
                yaxis_title="kWh",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_dispatch, use_container_width=True)

        with subtab2:
            st.markdown("#### Hourly Green Matching Map（Heatmap版）")
            ts_map = ts.copy()
            ts_map["datetime"] = pd.to_datetime(ts_map["timestamp"])
            ts_map["date"] = ts_map["datetime"].dt.date
            ts_map["hour"] = ts_map["datetime"].dt.hour
            ts_map["month"] = ts_map["datetime"].dt.strftime("%Y-%m")

            month_options = ["All"] + sorted(ts_map["month"].unique().tolist())
            selected_month = st.selectbox(
                "選擇月份",
                options=month_options,
                index=0,
                key="heatmap_month"
            )

            if selected_month != "All":
                ts_map = ts_map[ts_map["month"] == selected_month].copy()

            pivot = ts_map.pivot(index="date", columns="hour", values="hourly_clean_ratio").sort_index(ascending=True)

            fig = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale=[
                    [0.0, "#E6E6E6"],
                    [0.3, "#D4EFE8"],
                    [0.6, "#8ED1BF"],
                    [1.0, "#1FA37A"],
                ],
                zmin=0,
                zmax=1
            )

            fig.update_layout(
                height=500 if selected_month != "All" else 900,
                xaxis_title="Hour of Day",
                yaxis_title="Date",
                margin=dict(l=20, r=20, t=30, b=20),
                coloraxis_colorbar=dict(
                    title="",
                    tickvals=[0, 0.5, 1],
                    ticktext=["0%", "50%", "100%"],
                    len=0.7
                )
            )
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(24)),
                ticktext=[f"{h:02d}" for h in range(24)],
                showgrid=False
            )
            fig.update_yaxes(showgrid=False)

            st.plotly_chart(fig, use_container_width=True)

        with subtab3:
            st.markdown("#### 成本拆解")
            cost_breakdown_df = pd.DataFrame({
                "cost_type": ["PPA Cost", "Grid Cost", "Storage Cost"],
                "amount": [result["total_ppa_cost"], result["total_grid_cost"], result["total_storage_cost"]],
            })
            fig_cost = px.bar(cost_breakdown_df, x="cost_type", y="amount", text="amount")
            fig_cost.update_layout(height=380, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cost, use_container_width=True)
            st.dataframe(cost_breakdown_df, use_container_width=True, hide_index=True)

        with subtab4:
            st.markdown("#### PPA 成本明細")
            st.dataframe(cost_df, use_container_width=True, hide_index=True)

        with subtab5:
            st.markdown("#### 時序明細")
            st.dataframe(ts, use_container_width=True, hide_index=True)
    else:
        st.info("上傳資料後，按左側的「開始求解」。")
