import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as plotly_go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime, timedelta

# ================= 配置页面 =================
st.set_page_config(page_title="A股 Top 100 动能监测仪", layout="wide", page_icon="📈")
st.title("📈 A股 Top 100 动能与趋势监测仪")
st.markdown("监控每日成交额前100的活跃个股，结合 **RSI**, **MACD**, **Squeeze Momentum** 寻找绝佳交易机会。*(纯净版：底层算法全手动实现)*")

# ================= 数据获取与缓存 =================
@st.cache_data(ttl=14400, show_spinner=False)
def get_top_100_active_stocks():
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.sort_values(by="成交额", ascending=False).head(100)
        df = df[['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率']]
        return df
    except Exception as e:
        st.error(f"基础数据获取失败: {e}")
        return pd.DataFrame()

def fetch_and_calc_indicators(row):
    code = row['代码']
    end_date = datetime.now().strftime("%Y%m%d")
    start_date = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
    
    try:
        hist = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
        if hist.empty or len(hist) < 30:
            return None
            
        hist.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume'}, inplace=True)
        hist['date'] = pd.to_datetime(hist['date'])
        hist.set_index('date', inplace=True)
        
        # ================= 手写原生指标算法 (脱离 pandas-ta) =================
        
        # 1. RSI (14)
        delta = hist['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. MACD (12, 26, 9)
        ema12 = hist['close'].ewm(span=12, adjust=False).mean()
        ema26 = hist['close'].ewm(span=26, adjust=False).mean()
        hist['MACD_12_26_9'] = ema12 - ema26
        hist['MACDs_12_26_9'] = hist['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        hist['MACDh_12_26_9'] = hist['MACD_12_26_9'] - hist['MACDs_12_26_9']
        
        # 3. Squeeze Momentum (20, 2.0, 1.5)
        # 真实波动幅度 (TR)
        tr1 = hist['high'] - hist['low']
        tr2 = (hist['high'] - hist['close'].shift(1)).abs()
        tr3 = (hist['low'] - hist['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        sma20 = hist['close'].rolling(window=20).mean()
        std20 = hist['close'].rolling(window=20).std(ddof=0)
        
        # 布林带 (BB)
        upperBB = sma20 + (2.0 * std20)
        lowerBB = sma20 - (2.0 * std20)
        
        # 肯特纳通道 (KC)
        atr20 = tr.rolling(window=20).mean()
        upperKC = sma20 + (1.5 * atr20)
        lowerKC = sma20 - (1.5 * atr20)
        
        # 挤压状态 (布林带收缩进肯特纳通道内)
        hist['SQZ_ON'] = (lowerBB > lowerKC) & (upperBB < upperKC)
        
        # 动能 (近似 TradingView 的线性回归动能)
        highest20 = hist['high'].rolling(window=20).max()
        lowest20 = hist['low'].rolling(window=20).min()
        donchian = (highest20 + lowest20) / 2
        hist['SQZ_MOM'] = (hist['close'] - (donchian + sma20) / 2).rolling(window=10).mean()
        
        # ================= 解析状态 =================
        latest = hist.iloc[-1]
        prev = hist.iloc[-2]
        
        macd_hist = latest['MACDh_12_26_9']
        prev_macd_hist = prev['MACDh_12_26_9']
        
        macd_status = "平"
        if macd_hist > 0 and prev_macd_hist <= 0:
            macd_status = "🚀 金叉"
        elif macd_hist < 0 and prev_macd_hist >= 0:
            macd_status = "🩸 死叉"
        elif macd_hist > prev_macd_hist:
            macd_status = "🟢 多头增强"
        else:
            macd_status = "🔴 空头增强/衰弱"

        is_sqz_on = latest['SQZ_ON']
        sqz_momentum = latest['SQZ_MOM']
        
        sqz_status = "🗜️ 挤压中 (蓄势)" if is_sqz_on else ("🔥 向上爆发" if sqz_momentum > 0 else "🧊 向下释放")
        
        rsi_val = latest['RSI']
        rsi_status = "超卖" if rsi_val < 30 else ("超买" if rsi_val > 70 else "震荡")

        signal = "观望"
        if ("金叉" in macd_status or "多头" in macd_status) and rsi_val < 50 and "向上爆发" in sqz_status:
            signal = "⭐ 强买入候选"
        elif "死叉" in macd_status or rsi_val > 70:
            signal = "⚠️ 风险警示"

        row['RSI'] = round(rsi_val, 2)
        row['RSI状态'] = rsi_status
        row['MACD柱'] = round(macd_hist, 3)
        row['MACD信号'] = macd_status
        row['SQZ动能'] = round(sqz_momentum, 3)
        row['SQZ状态'] = sqz_status
        row['综合信号'] = signal
        row['_hist_data'] = hist 
        
        return row
    except Exception as e:
        return None

@st.cache_data(ttl=14400, show_spinner=False)
def process_all_stocks(df):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_and_calc_indicators, row) for _, row in df.iterrows()]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
    return pd.DataFrame(results)

def color_signal(val):
    color = 'green' if '强买入' in str(val) else 'red' if '风险' in str(val) else 'gray'
    return f'color: {color}; font-weight: bold'

def color_macd_hist(val):
    try:
        color = 'red' if float(val) > 0 else 'green'
        return f'color: {color}'
    except:
        return ''

with st.spinner("🔄 正在抓取A股 Top 100 行情并计算技术指标 (多线程加速中)..."):
    top_100_df = get_top_100_active_stocks()
    if not top_100_df.empty:
        final_df = process_all_stocks(top_100_df)
    else:
        final_df = pd.DataFrame()

if not final_df.empty:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 手动强制刷新"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        filter_signal = st.selectbox("筛选信号", ["全部", "⭐ 强买入候选", "⚠️ 风险警示", "观望"])
    with col3:
        search_query = st.text_input("🔍 搜索股票代码或名称")

    display_df = final_df.copy()
    if filter_signal != "全部":
        display_df = display_df[display_df['综合信号'] == filter_signal]
    if search_query:
        display_df = display_df[display_df['代码'].str.contains(search_query) | display_df['名称'].str.contains(search_query)]

    show_cols = ['代码', '名称', '最新价', '涨跌幅', '综合信号', 'RSI', 'RSI状态', 'MACD柱', 'MACD信号', 'SQZ动能', 'SQZ状态']
    table_df = display_df[show_cols].copy()

    st.subheader("📊 实时动能监测大盘")
    styled_df = table_df.style\
        .applymap(color_signal, subset=['综合信号'])\
        .applymap(color_macd_hist, subset=['MACD柱'])\
        .background_gradient(subset=['RSI'], cmap='RdYlGn_r', vmin=20, vmax=80)\
        .format({'最新价': '{:.2f}', '涨跌幅': '{:.2f}%', 'MACD柱': '{:.3f}', 'SQZ动能': '{:.3f}'})

    st.dataframe(styled_df, use_container_width=True, height=500)

    st.markdown("---")
    st.subheader("🔎 个股深度透视 (K线 + MACD + RSI + Squeeze)")
    
    selected_stock = st.selectbox("选择要查看图表的股票", display_df['名称'].tolist() + ["请选择..."], index=len(display_df))
    
    if selected_stock != "请选择...":
        stock_data = display_df[display_df['名称'] == selected_stock].iloc[0]
        hist = stock_data['_hist_data']
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.02, 
                            row_heights=[0.5, 0.15, 0.15, 0.2],
                            subplot_titles=(f"{stock_data['名称']} - K线", "MACD", "RSI", "Squeeze Momentum"))

        fig.add_trace(plotly_go.Candlestick(x=hist.index, open=hist['open'], high=hist['high'], 
                                            low=hist['low'], close=hist['close'], name='K线',
                                            increasing_line_color='red', decreasing_line_color='green'), 
                      row=1, col=1)

        fig.add_trace(plotly_go.Bar(x=hist.index, y=hist['MACDh_12_26_9'], name='MACD柱',
                                    marker_color=['red' if val > 0 else 'green' for val in hist['MACDh_12_26_9']]), row=2, col=1)
        fig.add_trace(plotly_go.Scatter(x=hist.index, y=hist['MACD_12_26_9'], mode='lines', name='DIF', line=dict(color='yellow')), row=2, col=1)
        fig.add_trace(plotly_go.Scatter(x=hist.index, y=hist['MACDs_12_26_9'], mode='lines', name='DEA', line=dict(color='blue')), row=2, col=1)

        fig.add_trace(plotly_go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI', line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.add_trace(plotly_go.Bar(x=hist.index, y=hist['SQZ_MOM'], name='动能',
                                    marker_color=['darkred' if val > 0 else 'darkblue' for val in hist['SQZ_MOM']]), row=4, col=1)

        fig.update_layout(height=900, xaxis_rangeslider_visible=False, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("暂无数据，请检查网络或点击强制刷新。")