import pandas as pd
import sqlite3
import yfinance as yf
from time import sleep
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

def create_pricing_db():
    """
    Main function to build the pricing database.
    It reads tickers from a CSV, downloads their historical prices,
    and inserts them into an SQLite database.
    """
    
    print("[INFO] Starting to build the pricing database...")
    # === Step 1: Load Ticker List ===
    df = pd.read_csv(config.METADATA_CSV_FILEPATH)
    tickers = df['Symbol'].unique()

    # === Step 2: Connect to SQLite DB ===
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    # === Step 3: Create Table (if not exists) ===
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {config.PRICING_TABLE_NAME} (
        ticker TEXT,
        date DATE,
        adjusted_close REAL,
        PRIMARY KEY (ticker, date)
    );
    """)
    conn.commit()

    # === Step 4: Download & Insert Per Ticker ===
    for ticker in tickers:
        print(f"Processing {ticker}...")

        try:
            data = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, progress=False)
            if data.empty:
                print(f" - No data for {ticker}")
                continue

            # Use 'Close' because it's now auto-adjusted by default
            df_prices = data.reset_index()[['Date', 'Close']]
            df_prices.columns = ['date', 'adjusted_close']
            df_prices['ticker'] = ticker
            df_prices = df_prices[['ticker', 'date', 'adjusted_close']]
            df_prices['date'] = pd.to_datetime(df_prices['date'])

            # === NEW FILTER: Only accept if exactly 3269 rows ===
            if len(df_prices) != config.REQUIRED_ROW_COUNT:
                print(f" - Skipped: has {len(df_prices)} rows (requires {config.REQUIRED_ROW_COUNT})")
                continue

            # Remove already-inserted dates
            existing_dates = pd.read_sql_query(
                f"SELECT date FROM {config.PRICING_TABLE_NAME} WHERE ticker = ?",
                conn, params=(ticker,)
            )
            existing_dates = pd.to_datetime(existing_dates['date']).dt.date
            df_prices = df_prices[~df_prices['date'].dt.date.isin(existing_dates)]

            if df_prices.empty:
                print(" - All records already inserted.")
                continue

            # Insert new rows
            df_prices.to_sql(config.PRICING_TABLE_NAME, conn, if_exists='append', index=False)
            print(f" - Inserted {len(df_prices)} new rows.")

            sleep(1)  # polite delay to avoid throttling

        except Exception as e:
            print(f" - Error with {ticker}: {e}")

    # Also get SPY ticker data for comparison if its not already in the database
    # first check if its already in the database
    cursor.execute(f"SELECT COUNT(*) FROM {config.PRICING_TABLE_NAME} WHERE ticker = 'SPY'")
    count = cursor.fetchone()[0]
    if count == 0:
        print("Processing SPY ticker...")
        try:
            spy_data = yf.download('SPY', start=config.START_DATE, end=config.END_DATE, progress=False)
            if spy_data.empty:
                print(" - No data for SPY")
            else:
                df_spy_prices = spy_data.reset_index()[['Date', 'Close']]
                df_spy_prices.columns = ['date', 'adjusted_close']
                df_spy_prices['ticker'] = 'SPY'
                df_spy_prices = df_spy_prices[['ticker', 'date', 'adjusted_close']]
                df_spy_prices['date'] = pd.to_datetime(df_spy_prices['date'])

                # Insert SPY data
                df_spy_prices.to_sql(config.PRICING_TABLE_NAME, conn, if_exists='append', index=False)
                print(" - Inserted SPY data.")
        except Exception as e:
            print(f" - Error with SPY: {e}")
    else:
        print("SPY ticker data already exists in the database, skipping...")
        

    # === Done ===
    conn.close()
    print("All tickers processed.")

def look_up_pricing(
    ticker: str,
    start_date: str,
    end_date: str,
    announce_date: str = None,
    short_window_end: int = 5,
    medium_window_end: int = 60
):
    """
    Plot historical prices for `ticker` and SPY (rescaled) between start_date and end_date.
    If `announce_date` is provided, draw:
      - vertical dashed line at the announcement
      - dashed best-fit lines for short and medium-term windows after the announcement.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        announce_date: Date of earnings announcement (YYYY-MM-DD), optional.
        short_window_end: Number of trading days from announce_date for short-term window.
        medium_window_end: Number of trading days from announce_date for medium-term window.
    """
    import sqlite3
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
    import config

    # Connect and load
    conn = sqlite3.connect(config.DB_PATH)
    try:
        query = f"""
            SELECT date, adjusted_close
            FROM {config.PRICING_TABLE_NAME}
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date;
        """
        df_ticker = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
        df_spy = pd.read_sql_query(query, conn, params=("SPY", start_date, end_date))
    finally:
        conn.close()

    if df_ticker.empty:
        print(f"No data found for {ticker} between {start_date} and {end_date}.")
        return

    # Prep
    for df in (df_ticker, df_spy):
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True, ignore_index=True)

    # Rescale SPY
    have_spy = not df_spy.empty
    if have_spy:
        t_min, t_max = df_ticker["adjusted_close"].min(), df_ticker["adjusted_close"].max()
        s_min, s_max = df_spy["adjusted_close"].min(), df_spy["adjusted_close"].max()
        t_rng = t_max - t_min
        s_rng = s_max - s_min
        if s_rng > 0:
            scale = (t_rng / s_rng) if t_rng > 0 else 1.0
            shift = t_min - scale * s_min
            df_spy["adjusted_close_rescaled"] = df_spy["adjusted_close"] * scale + shift
        else:
            df_spy["adjusted_close_rescaled"] = t_min

    # Helper to plot fitted dotted line
    def plot_window_fit(ax, df, start_idx, end_idx, label):
        end_idx = min(end_idx, len(df) - 1)
        if start_idx is None or start_idx >= end_idx:
            return False
        window = df.iloc[start_idx:end_idx + 1].copy()
        x = np.arange(len(window), dtype=float)
        y = window["adjusted_close"].values.astype(float)
        if len(x) < 2:
            return False
        a, b = np.polyfit(x, y, 1)
        y_fit = a * x + b
        ax.plot(window["date"].values, y_fit, linestyle="--", linewidth=1.5, label=label)
        return True

    # Find announcement index
    announce_idx = None
    if announce_date:
        try:
            ad = pd.to_datetime(announce_date)
            pos = df_ticker.index[df_ticker["date"] >= ad]
            if len(pos) > 0:
                announce_idx = int(pos[0])
            else:
                print("Announcement date is after available price series; skipping slope lines.")
                ad = None
        except Exception:
            print(f"Warning: could not parse announce_date '{announce_date}'")
            ad = None
    else:
        ad = None

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_ticker["date"], df_ticker["adjusted_close"], label=f"{ticker} Adjusted Close")
    if have_spy:
        ax.plot(df_spy["date"], df_spy["adjusted_close_rescaled"], label="SPY (rescaled)")

    if ad is not None:
        ax.axvline(ad, linestyle="--", color="red", label="Announcement date")
        if announce_idx is not None:
            short_ok = plot_window_fit(ax, df_ticker, announce_idx, announce_idx + short_window_end, f"Short-term slope (0–{short_window_end}d)")
            med_ok = plot_window_fit(ax, df_ticker, announce_idx + short_window_end, announce_idx + medium_window_end, f"Medium-term slope ({short_window_end}–{medium_window_end}d)")
            if not short_ok:
                print(f"Not enough data for short-term window (0–{short_window_end}d).")
            if not med_ok:
                print(f"Not enough data for medium-term window ({short_window_end}–{medium_window_end}d).")

    ax.set_title(f"Historical Prices: {ticker} vs SPY (rescaled)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted close")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()


def look_up_pricing_abnormal(
    ticker: str,
    start_date: str,
    end_date: str,
    announce_date: str = None,
    short_window_end: int = 5,
    medium_window_end: int = 60
):
    """
    Plot cumulative abnormal return (CAR) for `ticker` vs SPY between start_date and end_date.
    If announce_date is provided, draw:
      - vertical dashed line at the announcement
      - dashed best-fit lines for short (0–short_window_end) and medium (short_window_end–medium_window_end) windows.

    Notes:
      - Abnormal return = r_ticker - r_SPY (daily log returns)
      - CAR is cumulative sum of abnormal returns
    """
    import sqlite3
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import config

    # Load prices
    conn = sqlite3.connect(config.DB_PATH)
    try:
        query = f"""
            SELECT date, adjusted_close
            FROM {config.PRICING_TABLE_NAME}
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date;
        """
        df_t = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))
        df_m = pd.read_sql_query(query, conn, params=("SPY", start_date, end_date))
    finally:
        conn.close()

    if df_t.empty:
        print(f"No data for {ticker} between {start_date} and {end_date}.")
        return
    if df_m.empty:
        print(f"No SPY data between {start_date} and {end_date}. Cannot compute abnormal returns.")
        return

    # Prep and align on common trading days
    for df in (df_t, df_m):
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True, ignore_index=True)
        df.rename(columns={"adjusted_close": "px"}, inplace=True)

    df = pd.merge(df_t[["date", "px"]].rename(columns={"px": "px_t"}),
                  df_m[["date", "px"]].rename(columns={"px": "px_m"}),
                  on="date", how="inner")

    if len(df) < 3:
        print("Not enough overlapping days to compute returns.")
        return

    # Daily log returns
    df["r_t"] = np.log(df["px_t"] / df["px_t"].shift(1))
    df["r_m"] = np.log(df["px_m"] / df["px_m"].shift(1))
    df = df.dropna(subset=["r_t", "r_m"]).reset_index(drop=True)

    # Abnormal return and CAR
    df["abn"] = df["r_t"] - df["r_m"]
    df["car"] = df["abn"].cumsum()

    # Find announcement index on/after announce_date
    ad = None
    ad_idx = None
    if announce_date:
        try:
            ad = pd.to_datetime(announce_date)
            pos = df.index[df["date"] >= ad]
            if len(pos) > 0:
                ad_idx = int(pos[0])
                ad = df.loc[ad_idx, "date"]  # snap to trading day
            else:
                print("Announcement date is after the available series; skipping slope lines.")
                ad = None
        except Exception:
            print(f"Warning: could not parse announce_date '{announce_date}'")
            ad = None

    # Helper: draw anchored least-squares fit that passes through the start point of the window
    def plot_window_fit(ax, yseries, dates, start_idx, end_idx, label):
        end_idx = min(end_idx, len(yseries) - 1)
        if start_idx is None or start_idx >= end_idx:
            return False, None
        y = yseries[start_idx:end_idx + 1].astype(float)
        x = np.arange(len(y), dtype=float)
        if len(x) < 2:
            return False, None
        # Unanchored slope then anchor at window start (so line begins exactly at first CAR point)
        a, b = np.polyfit(x, y, 1)
        y0 = y[0]
        y_fit = a * x + (y0 - a * 0.0)
        ax.plot(dates[start_idx:end_idx + 1], y_fit, linestyle="--", linewidth=1.5, label=label)
        return True, a  # return slope per day

    # Plot CAR
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df["date"], df["car"], label=f"{ticker} CAR vs SPY")

    # Announcement marker and slopes
    if ad is not None:
        ax.axvline(ad, linestyle="--", color="red", label="Announcement date")
        ok_short, slope_short = plot_window_fit(
            ax, df["car"].values, df["date"].values,
            ad_idx, ad_idx + short_window_end,
            f"Short-term slope (0–{short_window_end}d)"
        )
        ok_med, slope_med = plot_window_fit(
            ax, df["car"].values, df["date"].values,
            ad_idx + short_window_end, ad_idx + medium_window_end,
            f"Medium-term slope ({short_window_end}–{medium_window_end}d)"
        )
        if not ok_short:
            print(f"Not enough data for short window 0–{short_window_end}d.")
        if not ok_med:
            print(f"Not enough data for medium window {short_window_end}–{medium_window_end}d.")
        # Optional: print daily slope in bps
        if slope_short is not None:
            print(f"Short-term daily CAR slope: {slope_short*10000:.2f} bps/day")
        if slope_med is not None:
            print(f"Medium-term daily CAR slope: {slope_med*10000:.2f} bps/day")

    ax.set_title(f"Cumulative Abnormal Return (CAR) for {ticker} vs SPY")
    ax.set_xlabel("Date")
    ax.set_ylabel("CAR (log-return units)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()

    

def make_dates(date:str):
    """
    Convert a date string to a datetime object, get the initial date by subtracting 15 days, and the final date by adding 90 days.
    Return all dates as a list.
    """
    from datetime import datetime, timedelta

    # Convert the date string to a datetime object
    initial_date = datetime.strptime(date, '%Y-%m-%d')
    
    # Calculate the initial and final dates
    initial_date = initial_date - timedelta(days=15)
    final_date = initial_date + timedelta(days=90)
    
    # Create a list of dates in the required format
    date_list = [initial_date.strftime('%Y-%m-%d'), final_date.strftime('%Y-%m-%d')]
    
    return date_list


def plot_average_pead_from_csv(
    surprises_csv: str = "eps_surprises_quarterly_2012_2024.csv",
    short_window_end: int = 5,
    medium_window_end: int = 60,
    pre_event_days: int = 20,
    save_path: str = "Plots/PEAD_demo/average_pead_plot.png"
):
    """
    Event-study PEAD plot:
      - Builds average cumulative abnormal return (CAR) vs SPY for positive- and negative-surprise cohorts.
      - Shades 0–short_window_end and short_window_end–medium_window_end windows.
      - Annotates slopes (bps/day) on the plot.
      - Saves figure to save_path.

    CSV must contain columns:
      - symbol
      - surprise or surprisePercent
      - at least one date-like column: announce_date (preferred), date, or period
    """
    import os
    import sqlite3
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["text.usetex"] = True
    import config

    # ---------------- Load events ----------------
    events = pd.read_csv(surprises_csv)
    date_col = next((c for c in ["announce_date", "date", "period"] if c in events.columns), None)
    if date_col is None:
        raise ValueError("No date-like column found in CSV. Add 'announce_date' or 'period'.")
    if "symbol" not in events.columns:
        raise ValueError("CSV must contain a 'symbol' column.")
    if "surprise" in events.columns:
        surprise_series = events["surprise"]
    elif "surprisePercent" in events.columns:
        surprise_series = events["surprisePercent"]
    else:
        raise ValueError("CSV must contain 'surprise' or 'surprisePercent'.")

    events = events.loc[~surprise_series.isna()].copy()
    events["event_date"] = pd.to_datetime(events[date_col], errors="coerce")
    events = events.loc[~events["event_date"].isna()].copy()
    if events.empty:
        raise ValueError("No valid events after filtering for date and surprise.")

    # ---------------- DB helper ----------------
    def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
        q = f"""
            SELECT date, adjusted_close
            FROM {config.PRICING_TABLE_NAME}
            WHERE ticker = ? AND date BETWEEN ? AND ?
            ORDER BY date;
        """
        with sqlite3.connect(config.DB_PATH) as conn:
            dfp = pd.read_sql_query(q, conn, params=(ticker, start, end))
        dfp["date"] = pd.to_datetime(dfp["date"])
        dfp = dfp.sort_values("date").rename(columns={"adjusted_close": "px"}).reset_index(drop=True)
        return dfp

    # ---------------- Build CAR per event ----------------
    rel_min = -pre_event_days
    rel_max = medium_window_end
    rel_index = np.arange(rel_min, rel_max + 1)

    car_pos, car_neg = [], []
    spy_cache = {}

    def get_spy_span(start, end):
        key = (start, end)
        if key not in spy_cache:
            spy_cache[key] = load_prices("SPY", start, end)
        return spy_cache[key].copy()

    skipped, processed = 0, 0
    for _, row in events.iterrows():
        sym = str(row["symbol"]).upper()
        s_val = row["surprise"] if "surprise" in row else row["surprisePercent"]
        ev_date = row["event_date"]

        span_start = (ev_date - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
        span_end   = (ev_date + pd.Timedelta(days=180)).strftime("%Y-%m-%d")

        df_t = load_prices(sym, span_start, span_end)
        df_m = get_spy_span(span_start, span_end)
        if df_t.empty or df_m.empty:
            skipped += 1
            continue

        df = pd.merge(df_t, df_m, on="date", how="inner", suffixes=("_t", "_m"))
        if df.empty:
            skipped += 1
            continue

        pos = df.index[df["date"] >= ev_date]
        if len(pos) == 0:
            skipped += 1
            continue
        e_idx = int(pos[0])

        if e_idx + medium_window_end >= len(df) or e_idx - pre_event_days < 1:
            skipped += 1
            continue

        df["r_t"] = np.log(df["px_t"] / df["px_t"].shift(1))
        df["r_m"] = np.log(df["px_m"] / df["px_m"].shift(1))
        df = df.dropna(subset=["r_t", "r_m"]).reset_index(drop=True)

        pos2 = df.index[df["date"] >= ev_date]
        if len(pos2) == 0:
            skipped += 1
            continue
        e_idx = int(pos2[0])

        start_i = e_idx - pre_event_days
        end_i = e_idx + medium_window_end
        if start_i < 0 or end_i >= len(df):
            skipped += 1
            continue

        win = df.iloc[start_i:end_i + 1].copy()
        win["abn"] = win["r_t"] - win["r_m"]
        win["car"] = win["abn"].cumsum()

        car_arr = win["car"].to_numpy()
        if len(car_arr) != len(rel_index):
            skipped += 1
            continue

        if s_val > 0:
            car_pos.append(car_arr)
        elif s_val < 0:
            car_neg.append(car_arr)
        processed += 1

    # ---------------- Aggregate and plot ----------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvline(0, linestyle="--", color="grey", linewidth=1, label=r"Announcement (Day 0)")

    # Shade windows (use light greys so lines stay clear)
    ax.axvspan(0, short_window_end, alpha=0.10, color="lightblue",
            label=rf"Short window (0–{short_window_end}d)")
    ax.axvspan(short_window_end, medium_window_end, alpha=0.10, color="lightgreen",
            label=rf"Medium window ({short_window_end}–{medium_window_end}d)")

    def slope_anchored(x, y, i0, i1):
        i1 = min(i1, len(x) - 1)
        i0 = max(i0, 0)
        if i1 <= i0:
            return np.nan
        xw = x[i0:i1 + 1].astype(float)
        yw = y[i0:i1 + 1].astype(float)
        if len(xw) < 2:
            return np.nan
        a, b = np.polyfit(xw - xw[0], yw, 1)
        return a  # per trading day

    # Helper to annotate a slope in bps/day at a chosen x position
    def annotate_slope(ax, x, y, i0, i1, colour, text_prefix):
        s = slope_anchored(x, y, i0, i1)
        if not np.isfinite(s):
            return
        mid = int((i0 + i1) / 2)
        mid = max(min(mid, len(x) - 1), 0)
        y_mid = y[mid]
        txt = f"{text_prefix}: {s*10000:.2f} bps/day"
        ax.text(x[mid], y_mid, txt, fontsize=9, color=colour, ha="center", va="bottom")

    legend_labels = []
    # Positive cohort
    if car_pos:
        pos_mean = np.nanmean(np.vstack(car_pos), axis=0)
        line_pos, = ax.plot(rel_index, pos_mean, label=rf"Positive surprise (N={len(car_pos)})")
        c_pos = line_pos.get_color()
        # Annotate slopes on averaged line
        i0_s = pre_event_days + 0
        i1_s = pre_event_days + short_window_end
        i0_m = pre_event_days + short_window_end
        i1_m = pre_event_days + medium_window_end
        annotate_slope(ax, rel_index, pos_mean, i0_s, i1_s, c_pos, r"Short slope")
        annotate_slope(ax, rel_index, pos_mean, i0_m, i1_m, c_pos, r"Medium slope")

    # Negative cohort
    if car_neg:
        neg_mean = np.nanmean(np.vstack(car_neg), axis=0)
        line_neg, = ax.plot(rel_index, neg_mean, label=rf"Negative surprise (N={len(car_neg)})")
        c_neg = line_neg.get_color()
        i0_s = pre_event_days + 0
        i1_s = pre_event_days + short_window_end
        i0_m = pre_event_days + short_window_end
        i1_m = pre_event_days + medium_window_end
        annotate_slope(ax, rel_index, neg_mean, i0_s, i1_s, c_neg, r"Short slope")
        annotate_slope(ax, rel_index, neg_mean, i0_m, i1_m, c_neg, r"Medium slope")

    ax.set_title(r"Average PEAD: Cumulative Abnormal Return (CAR) by Surprise Sign")
    ax.set_xlabel(r"Trading days relative to announcement")
    ax.set_ylabel(r"CAR (log-return units)")
    ax.grid(True)
    ax.legend()

    # Add a small note explaining bps
    ax.text(0.01, -0.12, r"bps = basis points (1 bps = 0.01\%)",
            transform=ax.transAxes, fontsize=9, ha="left", va="top")

    plt.tight_layout()

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot to: {save_path}")

    plt.show()




if __name__ == "__main__":
    create_pricing_db()
    # print("[INFO] Pricing database build complete.")
    # Example usage of look_up_pricing

    # look_up_pricing(ticker='AMZN', 
    #                 start_date='2020-06-15', 
    #                 end_date='2020-08-31',
    #                 announce_date='2020-06-30')
    TICKER = 'AAPL'
    ANNOUNCE_DATE = '2023-03-31'

    # date_range = make_dates(ANNOUNCE_DATE)
    # look_up_pricing_abnormal(ticker=TICKER, 
    #                 start_date=date_range[0], 
    #                 end_date=date_range[1],
    #                 announce_date=ANNOUNCE_DATE,
    #                 short_window_end=5,
    #                 medium_window_end=60)

    # plot_average_pead_from_csv(
    #     surprises_csv="Data/eps_surprises_quarterly_2012_2024.csv",
    #     short_window_end=10,
    #     medium_window_end=60,
    #     pre_event_days=20,
    #     save_path="Plots/PEAD_demo/average_pead_plot.png"
    # )

