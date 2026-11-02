"""F2b on liquid Lighter markets (frozen in configs/stageF2b_lighter_liquid.yaml): one attempt, per-asset Lighter costs.

Run from target_choice_probe/: python scripts/run_f2b_lighter_liquid.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean import load_panels  # noqa: E402
from src.common import LOGS, REPORTS, ROOT, load_config, md, sha256_file  # noqa: E402
from src.common_F2b import HOLDOUT_NOTE, log_attempt_f2b  # noqa: E402
from src.portfolio import simulate  # noqa: E402
from src.q1_F2b import ept_bp, inputs, period_daily, strategy_dec  # noqa: E402
from src.report import boot_sharpe_ci  # noqa: E402
from src.stats import sharpe  # noqa: E402
from src.strategies_F2b import asset_betas, band_path, beta_neutral  # noqa: E402

CFG = ROOT / "configs" / "stageF2b_lighter_liquid.yaml"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if sha256_file(CFG) not in (LOGS / "frozen_hash_F2b_lighter_liquid.txt").read_text(encoding="utf-8"):
        raise SystemExit("config differs from the frozen one")
    fz = yaml.safe_load(CFG.read_text(encoding="utf-8"))["stageF2b_lighter_liquid"]
    bs = fz["bootstrap"]
    cfg, _ = load_config()
    fb = {"q3": {"beta_window_bars": 180, "beta_min_bars": 90}}       # F2b Q3 values
    cols = list(load_panels(["close"])["close"].columns)
    cost = pd.read_csv(ROOT / fz["universe"]["source"], index_col=0)["imp10k_bp"].reindex(cols)
    keep = (cost <= 2.0).to_numpy()
    c_bp = np.nan_to_num(cost.to_numpy())
    base = inputs(cfg)
    inp = dict(base, dec_ok=base["dec_ok"] & keep)
    beta = asset_betas(base, fb)
    alive_dev = inp["dec_ok"][inp["win"]["dev"][0]:inp["win"]["dev"][1]].sum(1)
    rng = np.random.default_rng(20261102)
    rows = []
    for key in fz["strategies"]:
        br, tgt = key.split(":")
        raw = strategy_dec(inp, cfg, br, tgt, 1)
        bn, _ = beta_neutral(raw, beta, 1)
        lo, hi = inp["win"]["dev"]
        held = np.vstack([np.zeros((1, bn.shape[1])), bn[:-1]])
        dw = np.abs(np.diff(held[lo - 1:hi], axis=0))
        med = float(np.median(dw[dw > 1e-12]))
        for var, w in (("raw_h1", raw), ("q4_frozen", band_path(bn, 1.0 * med, 0.25))):
            comp = simulate(w, inp["R_M"], inp["fs"], cfg)
            held = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
            prev = np.vstack([np.zeros((1, w.shape[1])), held[:-1]])
            comp["cost_L"] = (np.abs(held - prev) * c_bp).sum(1) / 1e4
            for p in inp["win"]:
                d = period_daily(inp, comp, p)
                g = d["price"] + d["funding"]
                nL = g - d["cost_L"]
                ci = boot_sharpe_ci(nL.to_numpy(), bs["reps"], bs["block_days"], bs["ci_level"], rng)
                rows.append({"strategy": key, "variant": var, "period": p, "days": len(d),
                             "turnover_day": d["turnover"].mean(), "ept_bp": ept_bp(d),
                             "lighter_cost_side_bp": d["cost_L"].sum() / d["turnover"].sum() * 1e4,
                             "gross_sharpe": sharpe(g), "net_sharpe_lighter": sharpe(nL), "ci_lo": ci[0], "ci_hi": ci[1],
                             "net_ann_lighter": nL.mean() * 365, "net_sharpe_bybit_S1": sharpe(g - d["cost_S1"])})
    T = pd.DataFrame(rows)
    ok = {}
    for (k, v), g in T.groupby(["strategy", "variant"]):
        dv, hv = g[g.period == "dev"].iloc[0], g[g.period == "holdout"].iloc[0]
        ok[f"{k} {v}"] = bool(dv.net_sharpe_lighter > 0 and dv.ci_lo > 0 and hv.net_sharpe_lighter > 0)
    verdict = "кандидат" if any(ok.values()) else "NO-GO"
    log_attempt_f2b("Lighter-liquid", "X:T1, D:T5 on Lighter markets with imp10k<=2bp", len(ok), "dev+holdout", verdict)
    T.to_csv(REPORTS / "f2b_lighter_liquid.csv", index=False)
    sel = [c for c, k in zip(cols, keep) if k]
    (REPORTS / "f2b_lighter_liquid.md").write_text(f"""# F2b на ликвидных рынках Lighter (одна попытка, правило заморожено)

Конфиг `configs/stageF2b_lighter_liquid.yaml`, хеш `logs/frozen_hash_F2b_lighter_liquid.txt`.
Набор ({len(sel)}): {', '.join(sel)}. Активов в срезе на разработке: медиана {np.median(alive_dev):.0f}, мин {alive_dev.min()}, макс {alive_dev.max()}.
Издержки: на каждую сторону — `imp10k_bp` этого актива на Lighter (Tardis, часы ребалансировки).

**Вердикт: {verdict}.** Правило: {fz['verdict']['candidate']}.
{json.dumps(ok, ensure_ascii=False)}

{md(T.round(3), index=False)}

Оговорка: {HOLDOUT_NOTE}.
""", encoding="utf-8")
    print(md(T.round(3), index=False))
    print(verdict, ok, "alive dev median/min/max", np.median(alive_dev), alive_dev.min(), alive_dev.max())


if __name__ == "__main__":
    main()
