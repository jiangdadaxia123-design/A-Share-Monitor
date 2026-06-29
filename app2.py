import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime, timedelta
import time

# ================= 配置页面 =================
st.set_page_config(page_title="A股 Top 100 动能监测仪", layout="wide", page_icon="📈")
st.title("📈 A股 Top 100 动能与趋势监测仪")

# ================= 数据获取 (含重试逻辑) =================
@st.cache_data(ttl=3600, show_spinner=False)
def get_top_100_active_stocks():
    for _ in range(3): # 尝试重试 3 次
        try:
            # 依然使用spot_em，如果云端仍报错，可尝试更换为 ak.stock_zh_a_spot()
            df = ak.stock_zh_a_spot_em()
            df = df.sort_values(by="成交额", ascending=False).head(100)
            df = df[['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率']]
            return df
        except Exception:
            time.sleep(2)
    return pd.DataFrame()

# ================= 手动计算指标 (去除了 pandas-ta 依赖) =================
def fetch_and_calc_indicators(row):
    time.sleep(0.2) # 降低请求频率
    code = row['代码']
    try:
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=(datetime.now()-timedelta(days=150)).strftime("%Y%m%d"), adjust="qfq")
        if hist.empty or len(hist) < 30: return None
        
        hist.rename(columns={'日期': 'date', '收盘': 'close', '最高': 'high', '最低': 'low'}, inplace=True)
        hist['date'] = pd.to_datetime(hist['date'])
        hist.set_index('date', inplace=True)
        
        # RSI
        delta = hist['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        hist['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        # MACD
        hist['MACD'] = hist['close'].ewm(span=12, adjust=False).mean() - hist['close'].ewm(span=26, adjust=False).mean()
        hist['MACDs'] = hist['MACD'].ewm(span=9, adjust=False).mean()
        hist['MACDh'] = hist['MACD'] - hist['MACDs']
        
        # Squeeze (简化逻辑)
        sma20 = hist['close'].rolling(20).mean()
        std20 = hist['close'].rolling(20).std()
        hist['SQZ_MOM'] = (hist['close'] - (sma20 + (hist['high'].rolling(20).max() + hist['low'].rolling(20).min())/2)/2).rolling(10).mean()
        
        latest = hist.iloc[-1]
        row.update({'RSI': round(latest['RSI'], 2), 'MACD柱': round(latest['MACDh'], 3), 'SQZ动能': round(latest['SQZ_MOM'], 3), '_hist_data': hist})
        return row
    except: return None

# ================= 主程序部分 =================
top_100_df = get_top_100_active_stocks()
if not top_100_df.empty:
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        final_df = pd.DataFrame([res for res in executor.map(fetch_and_calc_indicators, [row for _, row in top_100_df.iterrows()]) if res is not None])

    # --- 交互控制栏 ---
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1: sort_by = st.selectbox("成交额排序", ["成交额降序", "成交额升序"])
    with col2: filter_signal = st.selectbox("信号筛选", ["全部", "强买入", "风险"])
    with col3: search_query = st.text_input("🔍 搜索股票")

    # --- 数据处理 ---
    display_df = final_df.sort_values("成交额", ascending=(sort_by == "成交额升序"))
    if filter_signal != "全部": display_df = display_df[display_df['综合信号'] == filter_signal]
    if search_query: display_df = display_df[display_df['代码'].str.contains(search_query) | display_df['名称'].str.contains(search_query)]

    # --- 表格展示 ---
    st.subheader("📊 实时动能监测大盘")
    st.dataframe(display_df[['代码', '名称', '最新价', '成交额', 'RSI', 'MACD柱', 'SQZ动能']], use_container_width=True)

    # --- 深度透视 ---
    selected = st.selectbox("选择个股查看 K 线", display_df['名称'].tolist())
    if selected:
        hist = display_df[display_df['名称'] == selected].iloc[0]['_hist_data']
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2])
        fig.add_trace(plotly_go.Candlestick(x=hist.index, open=hist['open'], high=hist['high'], low=hist['low'], close=hist['close']), row=1, col=1)
        fig.add_trace(plotly_go.Bar(x=hist.index, y=hist['MACDh'], marker_color='red'), row=2, col=1)
        fig.add_trace(plotly_go.Scatter(x=hist.index, y=hist['RSI'], line=dict(color='purple')), row=3, col=1)
        fig.update_layout(height=800, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("数据获取中或接口被限流，请稍后刷新。")
