import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="A股 Top 100 动能监测仪", layout="wide", page_icon="📈")
st.title("📈 A股 Top 100 动能与趋势监测仪（已优化限流版）")

# ================= 带重试 + 缓存的单股票获取 =================
@st.cache_data(ttl=1800, show_spinner=False)  # 缓存30分钟
def fetch_stock_with_retry(code, max_retries=4):
    for attempt in range(max_retries):
        try:
            time.sleep(0.8 + attempt * 0.4)  # 指数退避
            hist = ak.stock_zh_a_hist(
                symbol=code, 
                period="daily", 
                start_date=(datetime.now() - timedelta(days=150)).strftime("%Y%m%d"), 
                adjust="qfq"
            )
            if hist.empty or len(hist) < 30:
                return None
            
            hist = hist.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close', 
                '最高': 'high', '最低': 'low'
            })
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
            
            # 简化 Squeeze 动能
            sma20 = hist['close'].rolling(20).mean()
            hist['SQZ_MOM'] = (hist['close'] - sma20).rolling(10).mean()
            
            latest = hist.iloc[-1]
            return {
                'RSI': round(latest['RSI'], 2),
                'MACD柱': round(latest['MACDh'], 3),
                'SQZ动能': round(latest['SQZ_MOM'], 3),
                '_hist_data': hist
            }
        except Exception as e:
            if attempt == max_retries - 1:
                st.warning(f"{code} 获取失败（可能限流）: {str(e)[:80]}")
                return None
            time.sleep(2 ** attempt)  # 指数退避
    return None

# ================= 获取 Top N（可调小测试） =================
@st.cache_data(ttl=3600, show_spinner=False)
def get_top_n_stocks(n=30):   # 先用30只测试，稳定后可改50
    interfaces = [
        lambda: ak.stock_zh_a_spot_em(),
        lambda: ak.stock_zh_a_spot(),   # 新浪备用接口
    ]
    
    for attempt in range(5):
        for interface_func in interfaces:
            try:
                df = interface_func()
                if not df.empty:
                    df = df.sort_values(by="成交额", ascending=False).head(n)
                    return df[['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率']].copy()
            except Exception as e:
                st.sidebar.warning(f"第{attempt+1}次尝试失败: {str(e)[:120]}")
                time.sleep(4 + attempt * 2)
    
    # 所有接口都失败
    st.error("⚠️ 东方财富/新浪接口持续限流，请等待5-10分钟后点击下方按钮重试")
    if st.button("🔄 强制刷新 Top 股票列表"):
        st.cache_data.clear()
        st.rerun()
    return pd.DataFrame()


# ================= 主程序 =================
top_df = get_top_n_stocks(30)

if not top_df.empty:
    # ================= 计算指标部分（保持你原来的逻辑） =================
    progress = st.progress(0, text="正在计算技术指标（低并发防限流）...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_stock_with_retry, row['代码']): row 
                   for _, row in top_df.iterrows()}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            row = futures[future]
            data = future.result()
            if data:
                row_dict = row.to_dict()
                row_dict.update(data)
                results.append(row_dict)
            progress.progress((i + 1) / len(futures))
    
    progress.empty()
    final_df = pd.DataFrame(results)
    
    if not final_df.empty:
        # 生成综合信号
        def get_signal(row):
            rsi = row.get('RSI', 50)
            macd = row.get('MACD柱', 0)
            if pd.isna(rsi):
                return "数据不足"
            if rsi < 35 and macd > 0:
                return "强买入"
            if rsi > 70:
                return "风险"
            return "中性"
        
        final_df['综合信号'] = final_df.apply(get_signal, axis=1)
        
        # ================= 筛选控件 =================
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_by = st.selectbox("排序方式", ["成交额降序", "RSI升序", "MACD柱降序"])
        with col2:
            filter_sig = st.selectbox("信号筛选", ["全部", "强买入", "风险", "中性"])
        with col3:
            kw = st.text_input("🔍 搜索代码或名称")
        
        display = final_df.copy()
        
        if sort_by == "成交额降序":
            display = display.sort_values("成交额", ascending=False)
        elif sort_by == "RSI升序":
            display = display.sort_values("RSI", ascending=True)
        else:
            display = display.sort_values("MACD柱", ascending=False)
        
        if filter_sig != "全部":
            display = display[display['综合信号'] == filter_sig]
        if kw:
            mask = display['代码'].astype(str).str.contains(kw) | display['名称'].str.contains(kw)
            display = display[mask]
        
        # ================= 表格展示 =================
        st.subheader(f"📊 Top {len(display)} 只股票实时动能监测")
        st.dataframe(
            display[['代码', '名称', '最新价', '成交额', 'RSI', 'MACD柱', 'SQZ动能', '综合信号']],
            use_container_width=True,
            hide_index=True
        )
        
        # ================= 单股K线图 =================
        if not display.empty:
            selected = st.selectbox("选择个股查看详细K线 + 指标", display['名称'].tolist())
            if selected:
                sel = display[display['名称'] == selected].iloc[0]
                hist = sel['_hist_data']
                
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=(f"{selected} K线", "MACD柱状图", "RSI指标"),
                    row_heights=[0.55, 0.25, 0.2]
                )
                fig.add_trace(go.Candlestick(
                    x=hist.index, open=hist['open'], high=hist['high'],
                    low=hist['low'], close=hist['close'], name="K线"
                ), row=1, col=1)
                fig.add_trace(go.Bar(
                    x=hist.index, y=hist['MACDh'], name="MACD柱", marker_color="orange"
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=hist.index, y=hist['RSI'], name="RSI", line=dict(color="purple")
                ), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3)
                fig.update_layout(height=780, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("指标计算失败（可能是单只股票限流），请稍后刷新。")
else:
    st.warning("Top股票列表获取失败，请稍后刷新。")
