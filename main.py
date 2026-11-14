import argparse

from src.model_pipeline import AlternativeDataAlphaEngine


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Does Wikipedia attention predict next-day ETF realized volatility?")
    p.add_argument("--ticker", default="QQQ", help="Yahoo Finance ticker (default: QQQ)")
    p.add_argument("--article", default="Nvidia", help="en.wikipedia article title (default: Nvidia)")
    p.add_argument("--start", default="2025-01-01", help="start date YYYY-MM-DD")
    p.add_argument("--end", default="2026-06-01", help="end date YYYY-MM-DD")
    p.add_argument("--out", default=None, help="optional CSV path for the per-fold metrics")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    engine = AlternativeDataAlphaEngine(
        ticker=args.ticker,
        wiki_page=args.article,
        start_date=args.start,
        end_date=args.end,
    )
    X, y = engine.construct_design_matrix()
    results = engine.run_backtest_pipeline(X, y)
    if args.out:
        results.to_csv(args.out, index=False)
        print(f"Per-fold metrics written to {args.out}")


if __name__ == "__main__":
    main()
