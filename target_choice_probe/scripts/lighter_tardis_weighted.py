"""Cost per side on Lighter weighted by how often F2b strategies trade each asset (Bybit ∩ Lighter, frozen Q4 variant).

Run after lighter_tardis_liquidity.py: python scripts/lighter_tardis_weighted.py
Cost per side: half time-weighted spread (small order) and taker impact vs mid for $10k / $100k (includes half spread).
Rebalance hours: 4 h bars on UTC boundaries, execution at the open of the next bar -> hours 0, 4, ..., 20 UTC.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W = ROOT / "data" / "work" / "lighter_tardis"
REP = ROOT / "reports"
REBAL = {0, 4, 8, 12, 16, 20}


def load(prefix):
    df = pd.concat([pd.read_csv(p) for p in W.glob(f"{prefix}_*.csv") if p.stat().st_size > 1], ignore_index=True)
    df["utc_hour"] = df.hour % 24
    return df


def md(df):
    from src.common import md as _md
    return _md(df.round(2), digits=6)


def main():
    q, b = load("quotes"), load("book_snapshot_25")
    hourly = q.merge(b, on=["symbol", "market_id", "day", "hour", "utc_hour"], how="outer")
    hourly.to_csv(REP / "lighter_tardis_hourly.csv", index=False)
    hourly["rebal"] = hourly.utc_hour.isin(REBAL)
    cost = lambda d: pd.DataFrame({"half_spread_bp": d.groupby("symbol").spread_tw_bp.mean() / 2,
                                   "imp10k_bp": d.groupby("symbol").imp10k.median(),
                                   "imp100k_bp": d.groupby("symbol").imp100k.median(),
                                   "imp100k_missing": d.groupby("symbol").imp100k_missing.mean(),
                                   "depth10_kusd": d.groupby("symbol").depth10.median() / 1e3,
                                   "hours": d.groupby("symbol").size()})
    per_all, per_reb = cost(hourly), cost(hourly[hourly.rebal])
    share = pd.read_csv(REP / "f2b_trade_share_lighter.csv", index_col=0)
    cols = [c for c in share if c.startswith("lighter_all|")]
    metrics = ["half_spread_bp", "imp10k_bp", "imp100k_bp"]
    rows = []
    for c in cols:
        w = share[c].reindex(per_reb.index).fillna(0)
        for label, per in (("все часы", per_all), ("часы ребалансировки", per_reb)):
            r = {"weights": c.removeprefix("lighter_all|"), "hours": label}
            for m in metrics:
                ok = per[m].notna()
                r[m] = (per[m][ok] * w[ok]).sum() / w[ok].sum()
                r[f"{m}_weight_covered"] = w[ok].sum() / w.sum()
            rows.append(r)
    Wt = pd.DataFrame(rows)
    # stability: weighted cost per day and per UTC hour (X:T1 dev weights)
    w0 = share["lighter_all|X:T1|dev"]

    def wavg(d, m):
        v = d.set_index("symbol")[m]
        ww = w0.reindex(v.index).fillna(0)[v.notna()]
        return (v.dropna() * ww).sum() / ww.sum()

    by = lambda key: pd.DataFrame({m: hourly.groupby([key, "symbol"])[m].mean().reset_index().groupby(key)
                                   .apply(lambda d: wavg(d, m)) for m in ("spread_tw_bp", "imp10k", "imp100k")})
    Day, Hour = by("day"), by("utc_hour")
    per_reb.to_csv(REP / "lighter_tardis_cost_by_symbol.csv")
    Wt.to_csv(REP / "lighter_tardis_weighted_cost.csv", index=False)
    (REP / "lighter_tardis_cost.md").write_text(f"""# Lighter: цена стороны сделки, взвешенная по торговле F2b (Tardis, 7 дней)

Дни: первое число месяца, май–ноябрь 2026 (5 будних, 2 выходных), полные сутки. Полуспред — среднее по времени по всем
изменениям лучших цен (`quotes`); импакт — медиана по сетке 10 с снимков 25 уровней (`book_snapshot_25`), худшая сторона.
Веса — доля оборота актива у замороженного Q4-варианта на наборе `lighter_all` (64 актива).

## Взвешенная цена одной стороны, б.п.

{md(Wt.set_index(['weights', 'hours']))}

## По дням (веса X:T1, разработка; спред — полный, не половина)

{md(Day)}

## По часам UTC

{md(Hour)}

## По активам (часы ребалансировки)

{md(per_reb.sort_values('imp10k_bp'))}
""", encoding="utf-8")
    print(md(Wt.set_index(["weights", "hours"])))
    print(md(Day))


if __name__ == "__main__":
    main()
