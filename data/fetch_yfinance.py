import argparse, pandas as pd
import yfinance as yf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, required=True)
    ap.add_argument("--start", type=str, default="2015-01-01")
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--interval", type=str, default="1d")
    ap.add_argument("--out", type=str, default="data/stock.csv")
    args = ap.parse_args()

    df = yf.download(args.ticker, start=args.start, end=args.end, interval=args.interval)
    df = df.reset_index()
    df.rename(columns={"Date":"date", "Open":"open", "High":"high", "Low":"low", "Close":"close", "Adj Close":"adj_close", "Volume":"volume"}, inplace=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] wrote {args.out} with {len(df)} rows for {args.ticker}")

if __name__ == "__main__":
    main()
