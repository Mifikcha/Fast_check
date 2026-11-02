"""F2b on the Bybit ∩ Lighter universe: same F2 forecasts, only assets also listed on Lighter may be traded.

Run from target_choice_probe/: python scripts/run_f2b_lighter.py
F2/F2b code, config and results are read only; outputs go to reports/*_lighter.*
Models are not re-fitted: features and selection stay on the full 80-asset universe, only the traded cross-section shrinks.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean import load_panels  # noqa: E402
from src.common import ROOT, REPORTS, load_config, md, now_utc  # noqa: E402
from src.common_F2b import load_f2b  # noqa: E402
from src.portfolio import simulate  # noqa: E402
from src.q1_F2b import ept_bp, inputs, period_daily, strategy_dec  # noqa: E402
from src.stats import sharpe  # noqa: E402
from src.strategies_F2b import asset_betas, band_path, beta_neutral  # noqa: E402

LIGHTER = "https://mainnet.zklighter.elliot.ai/api/v1/orderBookDetails"
RENAMES = {"PUMPFUN": "PUMP", "CL": "WTI", "BZ": "BRENTOIL",     # same underlying, different ticker
           "SKHYNIX": "SKHYNIXUSD", "SAMSUNG": "SAMSUNGUSD"}     # Lighter's SKHYNIX/SAMSUNG markets are inactive


def lighter_overlap(symbols: list[str]) -> pd.DataFrame:
    raw = ROOT / "data" / "raw" / f"lighter_orderBookDetails_{pd.Timestamp.utcnow():%Y%m%d}.json"
    if not raw.exists():
        with urllib.request.urlopen(LIGHTER, timeout=30) as r:
            raw.write_bytes(r.read())
    perps = {d["symbol"] for d in json.loads(raw.read_bytes())["order_book_details"]
             if d["market_type"] == "perp" and d["status"] == "active"}
    rows = []
    for s in symbols:
        base = s.removesuffix("USDT")
        exact, renamed = base in perps, RENAMES.get(base) in perps
        rows.append({"bybit": s, "lighter": base if exact else RENAMES[base] if renamed else "",
                     "match": "exact" if exact else "renamed" if renamed else "none"})
    return pd.DataFrame(rows)


def metrics(inp, comp, cfg, label, strat, period):
    d = period_daily(inp, comp, period)
    g = d["price"] + d["funding"]
    return {"universe": label, "strategy": strat, "period": period, "days": len(d), "gross_sharpe_S0": sharpe(g),
            "net_sharpe_S1": sharpe(g - d["cost_S1"]), "turnover_day": d["turnover"].mean(),
            "ept_bp": ept_bp(d), "ept_price_only_bp": ept_bp(d, False), "gross_ann": g.mean() * 365}


def asset_share(comp_dec, inp, period, cols):
    lo, hi = inp["win"][period]
    held = np.vstack([np.zeros((1, comp_dec.shape[1])), comp_dec[:-1]])
    prev = np.vstack([np.zeros((1, comp_dec.shape[1])), held[:-1]])
    tv = np.abs(held - prev)[lo:hi].sum(axis=0)
    return pd.Series(tv / tv.sum(), index=cols)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    cfg, _ = load_config()
    fb, _ = load_f2b()
    cols = list(load_panels(["close"])["close"].columns)
    ov = lighter_overlap(cols)
    ov.to_csv(REPORTS / "universe_lighter_overlap.csv", index=False)
    universes = {"all_80": np.ones(len(cols), bool),
                 "lighter_exact": (ov.match == "exact").to_numpy(),
                 "lighter_all": (ov.match != "none").to_numpy()}
    base = inputs(cfg)
    beta = asset_betas(base, fb)
    sel = json.loads((ROOT / "data" / "work" / "f2b" / "q4_selected.json").read_text(encoding="utf-8"))
    rows, q4rows, shares = [], [], []
    for label, keep in universes.items():
        inp = dict(base, dec_ok=base["dec_ok"] & keep)
        # Q1 view: every h = 1 strategy, mark-price returns, no band
        for br, tgts in (("X", cfg["targets"]["branch_X"]), ("D", cfg["targets"]["branch_D"])):
            for tgt in tgts:
                comp = simulate(strategy_dec(inp, cfg, br, tgt, 1), inp["R_M"], inp["fs"], cfg)
                rows += [metrics(inp, comp, cfg, label, f"{br}:{tgt}", p) for p in inp["win"]]
        # Q4 view: beta-neutral + no-trade band; F2b-frozen (band_mult, lam), median |dw| recomputed on this universe
        for key, s in sel.items():
            br, tgt = key.split(":")
            dec, _ = beta_neutral(strategy_dec(inp, cfg, br, tgt, 1), beta, 1)
            lo, hi = inp["win"]["dev"]
            held = np.vstack([np.zeros((1, dec.shape[1])), dec[:-1]])
            dw = np.abs(np.diff(held[lo - 1:hi], axis=0))
            med = float(np.median(dw[dw > 1e-12]))
            for bm in fb["q4"]["band_mult"]:
                for lam in fb["q4"]["lam"]:
                    w = band_path(dec, bm * med, lam)
                    comp = simulate(w, inp["R_M"], inp["fs"], cfg)
                    frozen = bm == s["band_mult"] and lam == s["lam"]
                    for p in inp["win"]:
                        q4rows.append(dict(metrics(inp, comp, cfg, label, key, p), band_mult=bm, lam=lam, frozen=frozen))
                    if frozen:
                        for p in inp["win"]:
                            shares.append(asset_share(w, inp, p, cols).rename(f"{label}|{key}|{p}"))
        print(f"{now_utc()} {label}: {keep.sum()} assets", flush=True)
    A = pd.DataFrame(rows)
    Q = pd.DataFrame(q4rows)
    S = pd.concat(shares, axis=1)
    A.to_csv(REPORTS / "f2b_h1_lighter.csv", index=False)
    Q.to_csv(REPORTS / "f2b_q4_lighter.csv", index=False)
    S.to_csv(REPORTS / "f2b_trade_share_lighter.csv", index_label="symbol")
    fr = Q[Q.frozen].drop(columns=["frozen"])
    (REPORTS / "f2b_lighter.md").write_text(f"""# F2b на пересечении Bybit ∩ Lighter

Прогнозы F2 не переобучались: отбор признаков и модели — на всех 80 активах; торговать разрешено только активам из набора.
Доходности — марк-цена Bybit (как в F2b Q1/Q4), фандинг — Bybit. Контроль уже открывался в F2: на нём только знак и порядок величины.
Сопоставление тикеров: `reports/universe_lighter_overlap.csv` (exact {int(universes['lighter_exact'].sum())}, с переименованиями {int(universes['lighter_all'].sum())}).

## Q4-вариант (бета-нейтрально, зона без сделок с замороженными F2b параметрами)

{md(fr.round(3), index=False)}

## Все h = 1 стратегии без зоны (марк-цена)

{md(A.round(3), index=False)}

## Максимум EPT по сетке Q4 (разработка)

{md(Q[Q.period == 'dev'].groupby(['universe', 'strategy']).ept_bp.max().round(3).reset_index(), index=False)}
""", encoding="utf-8")
    print(md(fr.round(3), index=False))


if __name__ == "__main__":
    main()
