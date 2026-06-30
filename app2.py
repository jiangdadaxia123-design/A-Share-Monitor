import streamlit as st
import akshare as ak
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from datetime import datetime, timedelta
import time

# ================== 页面配置 ==================
st.set_page_config(page_title="A股 Top 动能监测仪", layout="wide", page_icon="📈")
st.title("📈 A股 Top 25 动能与趋势监测仪（稳定版）")

# ================== 获取Top股票（最稳版） ==================
@st.cache_data(ttl=7200, show_spinner=False)   # 缓存2小时
def get_top_n_stocks(n=25):
    interfaces = [
        lambda: ak.stock_zh_a_spot(),        # 新浪接口（优先，相对稳定）
        lambda: ak.stock_zh_a_spot_em(),     # 东方财富（备用）
    ]
    
    for attempt in range(7):   # 最多尝试7次
        for interface_func in interfaces:
            try:
                wait_time = 6 + attempt * 3   # 第1次等6秒，后续越来越长
                time.sleep(wait_time)
                
                df = interface_func()
                if df is not None and not df.empty:
                    df = df.sort_values(by="成交额", ascending=False).head(n)
                    return df[['代码', '名称', '最新价', '涨跌幅', '成交额', '换手率']].copy()
            except Exception as e:
                st.sidebar.warning(f"第{attempt+1}次尝试失败: {str(e)[:120]}")
                time.sleep(5)
    
    # 全部失败
    st.error("⚠️ 接口持续被限流，请等待 **10-15分钟** 后再试")
    if st.button("🔄 清除缓存并强制重试"):
        st.cache_data.clear()
        st.rerun()
    return pd.DataFrame()


# ================== 单只股票指标计算（带缓存+重试） ==================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_with_retry(code, max_retries=5):
    for attempt in range(max_retries):
        try:
            time.sleep(1.2 + attempt * 0.6)   # 防止限流
            hist = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=(datetime.now() - timedelta(days=160)).strftime("%Y%m%d"),
                adjust="qfq"
            )
            if hist.empty or len(hist) < 35:
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
            hist['RSI'] = 100 - (100 / (1 + gain / loss))

            # MACD
            ema12 = hist['close'].ewm(span=12, adjust=False).mean()
            ema26 = hist['close'].ewm(span=26, adjust=False).mean()
            hist['MACD'] = ema12 - ema26
            hist['MACDs'] = hist['MACD'].ewm(span=9, adjust=False).mean()
            hist['MACDh'] = hist['MACD'] - hist['MACDs']

            # Squeeze 简化动能
            sma20 = hist['close'].rolling(20).mean()
            hist['SQZ_MOM'] = (hist['close'] - sma20).rolling(10).mean()

            latest = hist.iloc[-1]
            return {
                'RSI': round(latest['RSI'], 2),
                'MACD柱': round(latest['MACDh'], 3),
                'SQZ动能': round(latest['SQZ_MOM'], 3),
                '_hist_data': hist
            }
        except Exception:
            if attempt == max_retries - 1:
                return None
            time.sleep(3 + attempt * 2)
    return None


# ================== 主程序 ==================
top_df = get_top_n_stocks(25)

if not top_df.empty:
    st.success(f"成功获取 Top {len(top_df)} 只股票数据")
    
    progress = st.progress(0, text="正在计算技术指标（低并发模式）...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_stock_with_retry, code): code 
                   for code in top_df['代码']}
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            data = future.result()
            if data:
                idx = top_df[top_df['代码'] == futures[future]].index[0]
                row = top_df.loc[idx].to_dict()
                row.update(data)
                results.append(row)
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
                return "🟢 强买入"
            if rsi > 70:
                return "🔴 风险"
            return "🟡 中性"
        
        final_df['综合信号'] = final_df.apply(get_signal, axis=1)
        
        # ================== 筛选控件 ==================
        col1, col2, col3 = st.columns([1.2, 1.2, 2])
        with col1:
            sort_by = st.selectbox("排序方式", ["成交额降序", "RSI升序", "MACD柱降序"])
        with col2:
            filter_sig = st.selectbox("信号筛选", ["全部", "🟢 强买入", "🔴 风险", "🟡 中性"])
        with col3:
            kw = st.text_input("🔍 搜索代码或名称")
        
        display = final_df.copy()
        
        if sort_by == "成交额降序":
            display = display.sort_values("成交额", ascending=False)
        elif sort_by == "RSI升序":
            display = display.sort_values("RSI")
        else:
            display = display.sort_values("MACD柱", ascending=False)
        
        if filter_sig != "全部":
            display = display[display['综合信号'] == filter_sig]
        if kw:
            display = display[display['代码'].astype(str).str.contains(kw) | 
                              display['名称'].str.contains(kw)]
        
        # ================== 表格 ==================
        st.subheader("📊 实时动能监测表")
        st.dataframe(
            display[['代码', '名称', '最新价', '成交额', 'RSI', 'MACD柱', 'SQZ动能', '综合信号']],
            use_container_width=True,
            hide_index=True
        )
        
        # ================== K线图 ==================
        if not display.empty:
            selected = st.selectbox("选择股票查看K线 + 指标", display['名称'].tolist())
            if selected:
                sel_row = display[display['名称'] == selected].iloc[0]
                hist = sel_row['_hist_data']
                
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=(f"{selected} K线", "MACD柱", "RSI (30-70)"),
                    row_heights=[0.55, 0.25, 0.2]
                )
                fig.add_trace(go.Candlestick(
                    x=hist.index, open=hist['open'], high=hist['high'],
                    low=hist['low'], close=hist['close']
                ), row=1, col=1)
                fig.add_trace(go.Bar(x=hist.index, y=hist['MACDh'], name="MACD柱"), row=2, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name="RSI"), row=3, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3)
                fig.update_layout(height=780, template="plotly_dark", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("部分股票指标计算失败，请稍后刷新。")
else:
    st.warning("未能获取股票列表，请等待10-15分钟后点击上方按钮重试。")

# ================== 底部提示 ==================
st.caption("数据来源：新浪 + 东方财富 | 每2小时自动更新 | 低并发防限流设计")
