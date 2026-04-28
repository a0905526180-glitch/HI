import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import numpy as np
import sqlite3
import datetime
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. 系統初始化與資料庫設定
# ==========================================
st.set_page_config(page_title="TITAN Oracle 終極量化中樞", layout="wide", initial_sidebar_state="expanded")

def init_dbs():
    """初始化歷史訊號資料庫與 FinMind 快取資料庫"""
    conn_hist = sqlite3.connect('titan_history.db')
    conn_hist.execute('''CREATE TABLE IF NOT EXISTS signal_history 
                        (scan_date TEXT, ticker TEXT, close_price REAL, strategy TEXT, tech_status TEXT, chip_status TEXT)''')
    conn_hist.commit()
    conn_hist.close()

    # 建立 FinMind API 快取庫
    conn_cache = sqlite3.connect('titan_cache.db')
    conn_cache.execute('''CREATE TABLE IF NOT EXISTS cache_institutional 
                          (date TEXT, stock_id TEXT, foreign_buy REAL, it_buy REAL, PRIMARY KEY (date, stock_id))''')
    conn_cache.execute('''CREATE TABLE IF NOT EXISTS cache_revenue 
                          (revenue_month TEXT, stock_id TEXT, revenue REAL, mom REAL, yoy REAL, PRIMARY KEY (revenue_month, stock_id))''')
    conn_cache.commit()
    conn_cache.close()

init_dbs()

def load_history_data():
    """從 SQLite 讀取歷史掃描訊號"""
    conn = sqlite3.connect('titan_history.db')
    try:
        # 依照日期由新到舊排序撈出所有資料
        df = pd.read_sql_query("SELECT * FROM signal_history ORDER BY scan_date DESC", conn)
    except Exception:
        # 如果資料表還不存在或有錯誤，回傳空表格防呆
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def save_signal_to_db(ticker, price, strategy, tech_msg, chip_data):
    """將觸發訊號的標的存入 SQLite 歷史庫"""
    conn = sqlite3.connect('titan_history.db')
    cursor = conn.cursor()
    
    # 格式化今天的日期
    scan_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # 整合籌碼狀態字串
    chip_status = f"外資5日:{chip_data['外資近5日(張)']} | 投信連買:{chip_data['投信連買(天)']}d"
    
    try:
        cursor.execute("""
            INSERT INTO signal_history (scan_date, ticker, close_price, strategy, tech_status, chip_status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (scan_date, ticker, price, strategy, tech_msg, chip_status))
        conn.commit()
    except Exception as e:
        print(f"資料庫寫入失敗: {e}")
    finally:
        conn.close()

# ==========================================
# 2. 數據獲取引擎 (Data Engine - 真正全市場)
# ==========================================
@st.cache_data(ttl=86400)
def get_all_tw_tickers():
    """使用 twstock 真實獲取全台股上市與上櫃名單 (約 1700+ 檔)"""
    try:
        import twstock
        all_codes = twstock.codes
        tickers = []
        for code, info in all_codes.items():
            if info.type == '股票':
                if info.market == '上市':
                    tickers.append(f"{code}.TW")
                elif info.market == '上櫃':
                    tickers.append(f"{code}.TWO")
        return tickers
    except ImportError:
        st.error("⚠️ 系統偵測未安裝 twstock。請在終端機執行 `pip install twstock` 以解鎖全市場掃描。")
        return ["2330.TW", "2303.TW", "2454.TW"] # 備用防呆名單

def fetch_finmind_api(dataset, data_id, start_date):
    """底層 FinMind API 呼叫器 (加入 Google Chrome 瀏覽器偽裝防阻擋)"""
    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": dataset,
        "data_id": data_id,
        "start_date": start_date,
    }
    # 這是破解防火牆的關鍵：讓 API 以為我們是真人使用瀏覽器
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, params=parameter, headers=headers, timeout=10)
        data = resp.json()
        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            return pd.DataFrame(data["data"])
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def get_finmind_data(ticker):
    """雙引擎架構：整合 FinMind (營收/籌碼) 與 yfinance (財報/本益比)"""
    stock_id = ticker.replace(".TW", "").replace(".TWO", "")
    conn = sqlite3.connect('titan_cache.db')
    
    # --- 1. 處理營收數據 (FinMind) ---
    rev_lookback = (datetime.datetime.now() - datetime.timedelta(days=120)).strftime("%Y-%m-%d")
    df_rev_cache = pd.read_sql_query(f"SELECT * FROM cache_revenue WHERE stock_id = '{stock_id}' ORDER BY revenue_month DESC", conn)
    
    if df_rev_cache.empty or (datetime.datetime.now() - datetime.datetime.strptime(df_rev_cache['revenue_month'].iloc[0], "%Y-%m-%d")).days > 35:
        df_rev_raw = fetch_finmind_api("TaiwanStockMonthRevenue", stock_id, rev_lookback)
        if not df_rev_raw.empty:
            df_rev_raw = df_rev_raw.sort_values('date').reset_index(drop=True)
            df_rev_raw['mom'] = df_rev_raw['revenue'].pct_change() * 100
            df_rev_raw['yoy'] = df_rev_raw['revenue_YearOverYearRatio'] if 'revenue_YearOverYearRatio' in df_rev_raw.columns else 0.0
            df_final_rev = df_rev_raw[['date', 'stock_id', 'revenue', 'mom', 'yoy']].rename(columns={'date': 'revenue_month'}).tail(3)
            
            conn.execute(f"DELETE FROM cache_revenue WHERE stock_id = '{stock_id}'")
            df_final_rev.to_sql('cache_revenue', conn, if_exists='append', index=False)
            df_rev_cache = pd.read_sql_query(f"SELECT * FROM cache_revenue WHERE stock_id = '{stock_id}' ORDER BY revenue_month DESC", conn)
            time.sleep(0.3) # 緩衝時間
            
    # --- 2. 處理法人籌碼數據 (FinMind) ---
    chip_lookback = (datetime.datetime.now() - datetime.timedelta(days=15)).strftime("%Y-%m-%d")
    df_chip_cache = pd.read_sql_query(f"SELECT * FROM cache_institutional WHERE stock_id = '{stock_id}' ORDER BY date DESC", conn)
    
    if df_chip_cache.empty or str(df_chip_cache['date'].iloc[0]) < (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d"):
        df_chip_raw = fetch_finmind_api("TaiwanStockInstitutionalInvestorsBuySell", stock_id, chip_lookback)
        if not df_chip_raw.empty:
            
            # --- 👇 關鍵修復：手動把買進減去賣出，算出 'buy_sell' 欄位 ---
            if 'buy_sell' not in df_chip_raw.columns:
                # 確保欄位都是數字，避免字串相減報錯
                df_chip_raw['buy'] = pd.to_numeric(df_chip_raw['buy'], errors='coerce').fillna(0)
                df_chip_raw['sell'] = pd.to_numeric(df_chip_raw['sell'], errors='coerce').fillna(0)
                df_chip_raw['buy_sell'] = df_chip_raw['buy'] - df_chip_raw['sell']
            # -------------------------------------------------------------
                
            df_chip_clean = df_chip_raw.pivot_table(index='date', columns='name', values='buy_sell', aggfunc='sum').fillna(0).reset_index()
            
            foreign_buy = df_chip_clean['外陸資買賣超股數(不含外資自營商)'] / 1000 if '外陸資買賣超股數(不含外資自營商)' in df_chip_clean.columns else 0
            it_buy = df_chip_clean['投信買賣超股數'] / 1000 if '投信買賣超股數' in df_chip_clean.columns else 0
            
            df_final_chip = pd.DataFrame({'date': df_chip_clean['date'], 'stock_id': stock_id, 'foreign_buy': foreign_buy, 'it_buy': it_buy})            
            # 🚀 修復 2：先刪除該股票的舊快取再寫入，防止 SQLite 主鍵衝突導致後台當機
            conn.execute(f"DELETE FROM cache_institutional WHERE stock_id = '{stock_id}'")
            df_final_chip.to_sql('cache_institutional', conn, if_exists='append', index=False, method='multi')
            df_chip_cache = pd.read_sql_query(f"SELECT * FROM cache_institutional WHERE stock_id = '{stock_id}' ORDER BY date DESC", conn)
            time.sleep(0.3)

    conn.close()

    # --- 3. 處理缺失的財報數據與即時報價 (yfinance 引擎) ---
    profit_margin, inst_hold, pe_ratio, rt_price, prev_close = 0.0, 0.0, 0.0, 0.0, 0.0
    try:
        stock = yf.Ticker(ticker)
        
        # 🚀 修復 3：強制使用 stateless (無狀態) 的 fast_info 抓取報價，徹底解決 yfinance 抓錯股票的 Bug
        rt_price = stock.fast_info.last_price
        prev_close = stock.fast_info.previous_close
        
        info = stock.info
        profit_margin = round((info.get('profitMargins', 0) or 0) * 100, 2)
        inst_hold = round((info.get('heldPercentInstitutions', 0) or 0) * 100, 2)
        pe_ratio = round(info.get('trailingPE', 0) or 0, 2)
    except Exception:
        pass

    # --- 4. 結算最終數據 ---
    mom, yoy = 0.0, 0.0
    if not df_rev_cache.empty:
        mom = round(df_rev_cache['mom'].iloc[0], 2) if pd.notnull(df_rev_cache['mom'].iloc[0]) else 0.0
        yoy = round(df_rev_cache['yoy'].iloc[0], 2) if pd.notnull(df_rev_cache['yoy'].iloc[0]) else 0.0

    foreign_5d, it_streak = 0, 0
    if not df_chip_cache.empty:
        df_recent = df_chip_cache.head(5)
        foreign_5d = round(df_recent['foreign_buy'].sum(), 0)
        for val in df_chip_cache['it_buy']:
            if val > 0: it_streak += 1
            else: break

    return {
        "MoM(%)": mom,
        "YoY(%)": yoy,
        "外資近5日(張)": foreign_5d,
        "投信連買(天)": it_streak,
        "淨利率(%)": profit_margin, 
        "法人持股比例(%)": inst_hold, 
        "本益比": pe_ratio,
        "即時報價": rt_price,
        "昨日收盤": prev_close
    }

# ==========================================
# 3. 核心技術指標與策略運算 (Technical Engine)
# ==========================================
def process_technical_indicators(ticker):
    """計算所有高階指標 (加入除錯模式、yfinance 新版格式修復與欄位去重)"""
    try:
        # 1. 獲取數據
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, timeout=10)
        
        if df is None or df.empty:
            st.warning(f"⚠️ {ticker}：Yahoo Finance 回傳空資料，可能是代號錯誤或無交易。")
            return None

        # 2. 修復 yfinance 新版 MultiIndex 欄位問題
        if isinstance(df.columns, pd.MultiIndex):
            # 強制只保留第一層的欄位名稱
            df.columns = [col[0] for col in df.columns]

        # 確保欄位是單純的字串
        df.columns = df.columns.astype(str)

        # --- 👇 終極防護：剔除任何重複的欄位 (解決 Cannot set a DataFrame 錯誤) ---
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()
        # --------------------------------------------------------

        # 把沒有收盤價的「幽靈空行」直接刪除
        df = df.dropna(subset=['Close']) 

        if len(df) < 60: 
            st.warning(f"⚠️ {ticker}：資料筆數不足 60 天 ({len(df)} 筆)，無法計算季線，自動跳過。")
            return None

        # 3. 基礎指標運算
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.macd(fast=8, slow=17, signal=9, append=True)
        df.ta.stoch(k=9, d=3, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.atr(length=14, append=True)
        
        # 4. 均線與均量 (現在 df['Close'] 保證只會是單一欄位了)
        df['20MA'] = df['Close'].rolling(window=20).mean()
        df['60MA'] = df['Close'].rolling(window=60).mean()
        df['Vol_5MA'] = df['Volume'].rolling(window=5).mean()

        # 5. 進階：RSI 底背離與籌碼密集區 (POC)
        rsi_col = [col for col in df.columns if col.startswith('RSI_')][0]
        df['Price_Low_20'] = df['Low'].rolling(window=20).min()
        df['RSI_Low_20'] = df[rsi_col].rolling(window=20).min()
        df['Bullish_Div'] = (df['Low'] <= df['Price_Low_20']) & (df[rsi_col] > df['RSI_Low_20'] * 1.05)
        
        recent_df = df.tail(120)
        bins = np.linspace(recent_df['Low'].min(), recent_df['High'].max(), 20)
        price_cuts = pd.cut(recent_df['Close'], bins=bins)
        poc_interval = recent_df.groupby(price_cuts, observed=False)['Volume'].sum().idxmax()
        df['POC_Price'] = poc_interval.mid

        return df

    except Exception as e:
        # 萬一還有錯誤，直接顯示在側邊欄讓我們抓蟲
        st.sidebar.error(f"❌ 計算 {ticker} 時發生錯誤: {e}")
        return None

def evaluate_strategy(df, strategy):
    """擴充至 6 種高勝率多空策略，加入嚴格濾網"""
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 嚴格流動性防護：5日均量低於 500 張 (500,000股) 直接淘汰，避免騙線
    if latest['Vol_5MA'] < 500000: 
        return False, "" 
    
    bbb_col = [col for col in df.columns if col.startswith('BBB_')][0]
    bbu_col = [col for col in df.columns if col.startswith('BBU_')][0]
    bbl_col = [col for col in df.columns if col.startswith('BBL_')][0]
    macd_col = [col for col in df.columns if 'MACD' in col and 'MACDs' not in col and 'MACDh' not in col][0]
    signal_col = [col for col in df.columns if 'MACDs' in col][0]
    rsi_col = [col for col in df.columns if col.startswith('RSI_')][0]

    # --- 策略條件定義 ---
    # 多頭系列
    if strategy == "多頭：三指標共振 (極嚴格)":
        # 布林極限收斂 < 5%，且 MACD 剛好在零軸上金叉
        macd_gold = (latest[macd_col] > latest[signal_col]) and (prev[macd_col] <= prev[signal_col]) and (latest[macd_col] > 0)
        if (latest[bbb_col] < 5.0) and macd_gold and (latest['Close'] > latest['60MA']):
            return True, "零軸上金叉 + 布林極度擠壓"
            
    elif strategy == "多頭：均線糾結後突破":
        # 20MA 與 60MA 距離極近 (< 2%)，且今日帶量突破
        ma_diff = abs(latest['20MA'] - latest['60MA']) / latest['60MA'] * 100
        if ma_diff < 2.0 and latest['Close'] > latest['20MA'] and latest['Volume'] > latest['Vol_5MA'] * 1.5:
            return True, "均線糾結 + 帶量長紅突破"

    elif strategy == "多頭：隱藏底背離 (摸底)":
        if latest['Bullish_Div']:
            return True, "價格創新低但 RSI 底部墊高"

    # 空頭系列 (防套牢/做空)
    elif strategy == "空頭：高檔頭部背離 (逃頂)":
        # 價格創 20 日新高，但 RSI 低於前波高點
        price_high_20 = df['High'].rolling(window=20).max().iloc[-1]
        rsi_high_20 = df[rsi_col].rolling(window=20).max().iloc[-1]
        if (latest['High'] >= price_high_20) and (latest[rsi_col] < rsi_high_20 * 0.95) and (latest[rsi_col] > 70):
            return True, "高檔 RSI 頂背離，動能衰竭"

    elif strategy == "空頭：跌破生命線 (做空)":
        # 帶量跌破季線 (60MA)，且 MACD 在零軸下死叉
        macd_dead = (latest[macd_col] < latest[signal_col]) and (latest[macd_col] < 0)
        if (latest['Close'] < latest['60MA']) and (prev['Close'] >= prev['60MA']) and macd_dead:
            return True, "帶量跌破季線 + MACD 死叉"

    elif strategy == "空頭：布林下軌破位":
        # 帶寬放大且沿著下軌下跌
        if latest['Close'] < latest[bbl_col] and latest[bbb_col] > prev[bbb_col]:
            return True, "布林開口向下，空頭發動"
            
    return False, ""

# ==========================================
# 4. 分析報告引擎 (Report Generator)
# ==========================================
def generate_comprehensive_report(ticker, df_tech, fund_data):
    """更高維度與深度的市場分析報告 (即時報價校正版)"""
    if df_tech is None or df_tech.empty: 
        st.error(f"無法分析 {ticker}：技術面數據獲取失敗。")
        return

    latest = df_tech.iloc[-1]
    kline_close = latest['Close'] 
    poc = latest['POC_Price']
    
    # --- 整合即時報價系統 ---
    rt_price = fund_data.get('即時報價', 0)
    prev_close = fund_data.get('昨日收盤', 0)
    
    # 判斷是否使用即時報價 (如果抓不到即時，才退回使用 K 線收盤價)
    display_price = rt_price if rt_price > 0 else kline_close
    
    # 計算漲跌幅字串
    if rt_price > 0 and prev_close > 0:
        change = rt_price - prev_close
        change_pct = (change / prev_close) * 100
        # 格式化為: 72.70 元 (-1.80, -2.42%)
        price_str = f"{display_price:.2f} 元 ({change:+.2f}, {change_pct:+.2f}%)"
    else:
        price_str = f"{display_price:.2f} 元"
        
    try:
        atr_col = [col for col in df_tech.columns if col.startswith('ATRr_')][0]
        atr_val = latest[atr_col]
    except IndexError:
        atr_val = display_price * 0.02

    # 趨勢位階與乖離率計算，全面改用最新報價
    trend_status = "多頭排列" if latest['20MA'] > latest['60MA'] else "空頭排列"
    dist_to_ma20 = (display_price - latest['20MA']) / latest['20MA'] * 100
    
    # 多空評分也改用最新報價計算，避免盤中跌破卻沒給出警告
    score = 0
    if display_price > latest['60MA']: score += 20
    else: score -= 20
    if display_price > latest['20MA']: score += 20
    else: score -= 20
        
    if fund_data['YoY(%)'] > 15: score += 30
    elif fund_data['YoY(%)'] < 0: score -= 20
        
    if fund_data['法人持股比例(%)'] > 40: score += 10

    # 精準判斷市場狀態
    if score >= 70: 
        state = "🔥 強勢抬轎區：基本面強勁且技術面多頭，順勢偏多操作。"
        action = f"下檔防守線設於 20MA 或 {display_price - (1.5 * atr_val):.2f} (1.5倍 ATR)。"
    elif 20 <= score < 70:
        state = "👀 震盪洗盤區：多空拉鋸，若布林帶寬收斂則代表主力正在默默吸籌。"
        action = f"不宜追高，可於 POC 籌碼密集區 ({poc:.2f}) 附近逢低佈局。"
    elif -30 <= score < 20:
        state = "⚠️ 轉弱出貨區：跌破短期均線，法人可能正在高檔調節出貨。"
        action = f"嚴格控管資金，若跌破季線 ({latest['60MA']:.2f}) 應果斷停損。"
    else:
        state = "🧊 空頭探底區：基本面衰退且均線蓋頭反壓，極容易被套牢。"
        action = "嚴禁摸底做多，積極者可尋找反彈至均線時的放空機會。"

    st.markdown(f"## 📊 {ticker} 深度市場分析報告")
    
    st.markdown("### 1. 基本面與籌碼穩定度")
    # 狀態提示
    if fund_data['本益比'] > 0:
        st.success("✅ 市場基本面、籌碼動能與技術分析數據皆已成功載入。")
    elif fund_data['本益比'] == 0.0:
        st.warning("⚠️ Yahoo Finance 暫時未提供此標的之財報資訊 (本益比/淨利率)，但籌碼與營收數據已由 FinMind 正常載入。")
        
    st.write(f"- **營收動能**：年增率 (YoY) {fund_data['YoY(%)']}%，淨利率 {fund_data['淨利率(%)']}%。")
    st.write(f"- **籌碼結構**：法人持股比例達 {fund_data['法人持股比例(%)']}%，本益比 {fund_data['本益比']} 倍。")
    
    st.markdown("### 2. 技術面結構")
    # 將股價放大並高亮顯示，加入即時漲跌幅
    st.markdown(f"#### 💰 最新/即時報價：**{price_str}**")
    st.write(f"- **趨勢位階**：目前屬於【{trend_status}】，乖離月線 {dist_to_ma20:.2f}%。")
    st.write(f"- **支撐壓力**：120 日最大籌碼密集區 (POC) 落於 {poc:.2f} 元。")

    st.markdown("### 3. 綜合診斷與操作規劃")
    st.info(f"**TITAN 多空戰力分數：{score} / 100**\n\n**狀態判定**：{state}\n\n**行動規劃**：{action}")

# ==========================================
# 5. UI 與主程式 (Streamlit App)
# ==========================================
st.title("🦅 TITAN Oracle 專業量化決策中樞")

# --- 側邊欄：手動查股功能 ---
st.sidebar.markdown("### 🔎 單一個股深度診斷")
st.sidebar.caption("支援隨時手動輸入代號進行快速分析")

# 🚀 修復 1：使用 Form 表單鎖定輸入狀態，保證每次點擊抓到的絕對是最新的代號
with st.sidebar.form("search_form"):
    manual_ticker = st.text_input("輸入股票代號 (上市加 .TW / 上櫃加 .TWO)", "2330.TW")
    submit_search = st.form_submit_button("執行個股診斷")

if submit_search:
    # 終極防呆：強制清除所有空白
    clean_ticker = manual_ticker.replace(" ", "").replace("　", "").upper()
    
    with st.spinner(f"正在深度運算 {clean_ticker} 的市場數據..."):
        df_tech = process_technical_indicators(clean_ticker)
        chip_data = get_finmind_data(clean_ticker)
        
        st.success(f"已完成 {clean_ticker} 的診斷！")
        generate_comprehensive_report(clean_ticker, df_tech, chip_data)

st.sidebar.markdown("---")

# --- 主畫面：三大核心功能分頁 ---
tab_radar, tab_portfolio, tab_history = st.tabs(["📡 全市場策略雷達", "🛡️ 持倉風控戰情室", "🗄️ 歷史回測資料庫"])

with tab_radar:
    st.markdown("### ⚙️ 全市場雷達掃描 (約 1,700 檔)")
    st.caption("⚠️ 全市場掃描將耗時數分鐘，系統已啟動多執行緒加速。")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        strategy = st.selectbox("選擇今日策略", ["多頭：三指標共振 (極嚴格)", "多頭：均線糾結後突破", "多頭：隱藏底背離 (摸底)", "空頭：高檔頭部背離 (逃頂)", "空頭：跌破生命線 (做空)", "空頭：布林下軌破位"])
    with col2:
        st.write("")
        st.write("")
        run_scan = st.button("🌎 啟動全市場真實掃描")
    
    if run_scan:
        tickers = get_all_tw_tickers()
        st.info(f"系統正在運用多執行緒並行掃描 {len(tickers)} 檔標的，請稍候...")
        
        results = []
        progress_bar = st.progress(0)
        
        # 使用多執行緒提升全市場掃描速度 (建議 max_workers 設為 10-15 避免被 Yahoo 封鎖 IP)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(process_technical_indicators, t): t for t in tickers}
            completed = 0
            for future in as_completed(future_to_ticker):
                completed += 1
                ticker = future_to_ticker[future]
                df = future.result()
                if df is not None:
                    is_match, reason = evaluate_strategy(df, strategy)
                    if is_match:
                        chip = get_finmind_data(ticker)
                        
                        # --- 🚀 關鍵修復：把抓到的股票寫入歷史資料庫 ---
                        current_price = chip.get('即時報價', df['Close'].iloc[-1])
                        save_signal_to_db(ticker, current_price, strategy, reason, chip)
                        # ---------------------------------------------
                        
                        results.append({"股票代號": ticker, "收盤價": round(df['Close'].iloc[-1], 2), 
                                        "觸發原因": reason, "投信連買(天)": chip['投信連買(天)']})
                
                # 每掃描完成部分進度更新 UI
                if completed % 10 == 0 or completed == len(tickers):
                    progress_bar.progress(completed / len(tickers))
                
        if results:
            df_res = pd.DataFrame(results)
            st.success(f"掃描完成！共發現 {len(df_res)} 檔符合策略的潛力標的，已同步存入歷史庫。")
            st.dataframe(df_res, use_container_width=True)
        else:
            st.warning("今日全市場無符合該策略條件之標的。")

with tab_portfolio:
    st.markdown("### 💼 實戰庫存與 ATR 動態防守")
    st.caption("於此處監控你的庫存標的，嚴守紀律，防守線跌破即亮紅燈。")
    # 範例庫存
    portfolio = pd.DataFrame({"股票代號": ["2303.TW", "2454.TW"], "成本": [48.5, 950.0], "股數": [3000, 1000]})
    status_list = []
    
    for idx, row in portfolio.iterrows():
        df_p = process_technical_indicators(row["股票代號"])
        if df_p is not None:
            close = df_p['Close'].iloc[-1]
            atr_col = [col for col in df_p.columns if col.startswith('ATRr_')][0]
            stop_price = close - (1.5 * df_p[atr_col].iloc[-1])
            warn = "🚨 跌破防守線" if close < stop_price else "✅ 續抱"
            status_list.append({"代號": row["股票代號"], "最新報價": round(close,2), "防守價": round(stop_price,2), "狀態": warn})
            
    st.dataframe(pd.DataFrame(status_list), use_container_width=True)

# --- 歷史回測資料庫區塊 ---
with tab_history:  # 請確認變數名稱與你的 tabs 定義一致
    st.markdown("## 📚 歷史訊號覆盤與追蹤")
    st.caption("查看過去雷達掃描所捕捉到的潛力標的，驗證策略勝率並追蹤後續走勢。")

    # 載入歷史資料
    df_history = load_history_data()

    if df_history.empty:
        st.info("📭 目前資料庫中尚無歷史訊號紀錄。請先到「全市場策略雷達」執行掃描，系統會自動將符合條件的標的存入這裡！")
    else:
        # 實用功能 1：動態過濾器 (利用 columns 並排顯示)
        col1, col2 = st.columns(2)
        with col1:
            date_list = ["全部"] + list(df_history['scan_date'].unique())
            filter_date = st.selectbox("📅 選擇掃描日期", date_list)
        with col2:
            strategy_list = ["全部"] + list(df_history['strategy'].unique())
            filter_strategy = st.selectbox("🎯 選擇觸發策略", strategy_list)

        # 實用功能 2：資料篩選邏輯
        df_display = df_history.copy()
        if filter_date != "全部":
            df_display = df_display[df_display['scan_date'] == filter_date]
        if filter_strategy != "全部":
            df_display = df_display[df_display['strategy'] == filter_strategy]

        # 顯示統計資訊
        st.write(f"🔍 篩選結果：共找到 **{len(df_display)}** 筆紀錄")

        # 實用功能 3：美化資料表顯示 (使用 column_config 讓數字和欄位更漂亮)
        st.dataframe(
            df_display,
            column_config={
                "scan_date": "掃描日期",
                "ticker": "股票代號",
                "close_price": st.column_config.NumberColumn("觸發時股價", format="%.2f 元"),
                "strategy": "觸發策略",
                "tech_status": "技術面狀態",
                "chip_status": "籌碼面狀態"
            },
            use_container_width=True,
            hide_index=True,
            height=400
        )

        # 實用功能 4：一鍵匯出 CSV (加入 utf-8-sig 確保 Excel 打開中文不會亂碼)
        csv = df_display.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下載目前篩選的歷史訊號 (CSV Excel 格式)",
            data=csv,
            file_name=f"TITAN_歷史訊號_{filter_date}.csv",
            mime="text/csv",
        )
