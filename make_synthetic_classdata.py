"""Generate SYNTHETIC stand-in classdata for the three notebooks here.

The original course datasets (Compustat/CRSP/IBES extracts) are licensed
and cannot be redistributed, so they were never committed to this repo.
This script builds small synthetic files with the exact schemas the
notebooks read, so every notebook executes top-to-bottom:

  classdata/AnnualTotalAsset.csv  LPERMNO, datadate (YYYYMMDD), fyear, at
  classdata/IBES.csv              CUSIP, STATPERS (YYYYMMDD), NUMEST,
                                  MEANEST, STDEV
  classdata/ActualPrice.csv       CUSIP, STATPERS, PRICE
  classdata/cusip.csv             PERMNO, NCUSIP
  classdata/return.csv            permno, yyyymm (quarter code, e.g.
                                  201412), prc, fret1
  classdata/benchmark.csv         yyyymm, benchmark

classdata/Signal1.csv and classdata/Signal2.csv are NOT written here:
they are the outputs the Asset Growth and Forecast Dispersion notebooks
themselves produce from the files above (the executed notebooks write
them; copies are kept in classdata/ because the Portfolio Construction
notebook reads them from there).

Details mirrored from the original course data so the notebooks'
walkthrough examples behave as written:
  * PERMNO 93433 and 32548 exist; 32548 reports total assets of 0 for
    its first three fiscal years, which is what produces the inf
    asset-growth signal the notebook discusses.
  * PERMNO 10421 changed its fiscal year end from March to December in
    2001, so two of its signals land on Date=200206 -- the duplicate
    the notebook resolves with tail(1).
  * CUSIP 87482X10 exists and has Q4-2014 observations.
  * 300 firms per quarter, so the backtest's ">=200 stocks" filter has
    quarters to work with. A mild asset-growth effect is baked into
    fret1 (high asset growth -> slightly lower next-quarter returns),
    as in the paper the course is based on.

ALL DATA IS SYNTHETIC. Backtest statistics computed from it are
illustrative only and are not real market results.

Run:  python make_synthetic_classdata.py
"""

import os

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "classdata")
rng = np.random.default_rng(42)

N_FIRMS = 300
SPECIALS = [93433, 32548, 10421]


def quarter_codes(y0=2000, y1=2021):
    return [y * 100 + q * 3 for y in range(y0, y1 + 1) for q in (1, 2, 3, 4)]


def build_universe():
    pool = set()
    while len(pool) < N_FIRMS - len(SPECIALS):
        pool.add(int(rng.integers(10000, 100000)))
    permnos = SPECIALS + sorted(pool - set(SPECIALS))
    chars = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    cusips, seen = {}, set()
    for i, p in enumerate(permnos):
        if i == 0:
            c = "87482X10"          # the CUSIP the Forecast notebook inspects
        else:
            while True:
                c = "".join(rng.choice(chars, size=8)) + str(int(rng.integers(0, 10)))
                if c not in seen:
                    break
        seen.add(c)
        cusips[p] = c
    prices = {p: float(rng.uniform(5, 350)) for p in permnos}
    return permnos, cusips, prices


def build_annual_total_asset(permnos):
    rows = []
    for p in permnos:
        if p == 10421:
            # Fiscal year end changes from March to December in 2001.
            dates = [(y, 3, 31, y - 1) for y in range(1999, 2002)]
            dates += [(y, 12, 31, y) for y in range(2001, 2020)]
        elif p == 32548:
            # January fiscal year end; at = 0 for the first three years
            # (this is what makes its 4th-year growth signal inf).
            dates = [(y, 1, 31, y - 1) for y in range(2001, 2020)]
        elif p == 93433:
            dates = [(y, 12, 31, y) for y in range(1998, 2020)]
        else:
            fye = [(12, 31), (12, 31), (12, 31), (12, 31),
                   (6, 30), (3, 31), (1, 31)][int(rng.integers(0, 7))]
            y0 = int(rng.integers(1996, 2001))
            dates = []
            for y in range(y0, 2020):
                fy = y if fye == (12, 31) else y - 1 if fye in ((3, 31), (1, 31)) else y
                dates.append((y, fye[0], fye[1], fy))
        at = float(np.exp(rng.normal(6.5, 1.6)))
        for j, (y, m, d, fy) in enumerate(dates):
            if j > 0:
                at = max(at * (1 + rng.normal(0.05, 0.15)), 0.5)
            val = round(at, 3)
            if p == 32548 and j < 3:
                val = 0.0
            rows.append((p, y * 10000 + m * 100 + d, float(fy), val))
    df = pd.DataFrame(rows, columns=["LPERMNO", "datadate", "fyear", "at"])
    return df.sort_values(["LPERMNO", "datadate"]).reset_index(drop=True)


def asset_growth_signals(annual):
    """Replicates the Asset Growth notebook's signal construction."""
    df = annual.copy()
    df["datadate"] = pd.to_datetime(df["datadate"], format="%Y%m%d")
    df = df[["LPERMNO", "datadate", "fyear", "at"]]
    df.sort_values(by=["LPERMNO", "datadate"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["Signal"] = df.groupby("LPERMNO")["at"].pct_change() + 1
    df["Date"] = [(s + relativedelta(months=6)).year * 100
                  + (s + relativedelta(months=6)).quarter * 3 for s in df["datadate"]]
    df.replace(np.inf, np.nan, inplace=True)
    df.dropna(how="any", axis=0, inplace=True)
    base = df.copy()
    for i in (9, 12, 15):
        tmp = base.copy()
        tmp["Date"] = [(s + relativedelta(months=i)).year * 100
                       + (s + relativedelta(months=i)).quarter * 3 for s in tmp["datadate"]]
        df = pd.concat([df, tmp])
    df.sort_values(by=["LPERMNO", "datadate"], inplace=True)
    df = df.groupby(["LPERMNO", "Date"]).tail(1)
    return {(int(r.LPERMNO), int(r.Date)): float(r.Signal) for r in df.itertuples()}


def build_ibes_and_prices(permnos, cusips, prices):
    ibes_rows, price_rows = [], []
    months = pd.date_range("2013-01-18", "2016-12-18", freq="MS") + pd.Timedelta(days=17)
    for p in permnos:
        c = cusips[p]
        for dt in months:
            sp = int(dt.strftime("%Y%m%d"))
            ibes_rows.append((c, sp, int(rng.integers(3, 26)),
                              round(float(rng.uniform(0.3, 2.5)), 2),
                              round(float(rng.uniform(0.01, 0.09)), 2)))
            price_rows.append((c, sp, round(prices[p] * (1 + rng.normal(0, 0.03)), 2)))
    ibes = pd.DataFrame(ibes_rows, columns=["CUSIP", "STATPERS", "NUMEST", "MEANEST", "STDEV"])
    actual = pd.DataFrame(price_rows, columns=["CUSIP", "STATPERS", "PRICE"])
    return ibes, actual


def build_returns(permnos, prices, signals):
    # The asset-growth effect varies by quarter (multiplier ~ N(1, 1.2)),
    # so it is strong in some quarters and reverses in others -- as in
    # real data, where the anomaly is not a constant.
    effect = {qc: float(rng.normal(1.0, 1.2)) for qc in quarter_codes()}
    firm_vol = {p: float(rng.uniform(0.05, 0.20)) for p in permnos}
    rows = []
    for p in permnos:
        for qc in quarter_codes():
            sig = signals.get((p, qc))
            mu = 0.008 - (0.08 * effect[qc] * (sig - 1.0) if sig is not None else 0.0)
            fret1 = float(np.clip(rng.normal(mu, firm_vol[p]), -0.85, 1.50))
            prc = round(prices[p] * (1 + rng.normal(0, 0.05)), 4)
            rows.append((p, qc, prc, round(fret1, 6)))
    return pd.DataFrame(rows, columns=["permno", "yyyymm", "prc", "fret1"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    permnos, cusips, prices = build_universe()

    annual = build_annual_total_asset(permnos)
    annual.to_csv(os.path.join(OUT_DIR, "AnnualTotalAsset.csv"), index=False)

    signals = asset_growth_signals(annual)

    ibes, actual = build_ibes_and_prices(permnos, cusips, prices)
    ibes.to_csv(os.path.join(OUT_DIR, "IBES.csv"), index=False)
    actual.to_csv(os.path.join(OUT_DIR, "ActualPrice.csv"), index=False)

    pd.DataFrame({"PERMNO": permnos, "NCUSIP": [cusips[p] for p in permnos]}
                 ).to_csv(os.path.join(OUT_DIR, "cusip.csv"), index=False)

    rets = build_returns(permnos, prices, signals)
    rets.to_csv(os.path.join(OUT_DIR, "return.csv"), index=False)

    bench = pd.DataFrame({"yyyymm": quarter_codes(),
                          "benchmark": np.round(rng.normal(0.018, 0.05, len(quarter_codes())), 6)})
    bench.to_csv(os.path.join(OUT_DIR, "benchmark.csv"), index=False)

    for f in ["AnnualTotalAsset.csv", "IBES.csv", "ActualPrice.csv",
              "cusip.csv", "return.csv", "benchmark.csv"]:
        n = sum(1 for _ in open(os.path.join(OUT_DIR, f))) - 1
        print(f"Wrote classdata/{f}: {n} rows (synthetic)")


if __name__ == "__main__":
    main()
