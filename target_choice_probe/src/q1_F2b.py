"""Q1: bid-ask bounce check. Same F2 forecasts and weights; returns by last-trade open (P) vs mark open (M);
PM = features and targets on mark OHLC, model re-fitted (X:T1, X:T5, D:T1 at h = 1).
"""

from __future__ import annotations

import copy
import json
import pickle
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from .clean import load_panels
from .common import CLEAN, FIG, LOGS, REPORTS, ROOT, md, now_utc
from .common_F2b import HOLDOUT_NOTE, WORK_F2B, log_attempt_f2b
from .data_F2b import load_mark
from .features import apply_cs, base_series, candidates, placebos, targets
from .portfolio import KEY, assemble, daily, decisions, market_inputs, simulate, windows
from .report import boot_diff, boot_sharpe_ci
from .stats import rowwise_spearman, sharpe
from .walkforward import _score_chunk, blocks, context, fit_family, load_refits, score_inputs, select

PM_MM = WORK_F2B / "features_pm.f32"
PM_DIR = WORK_F2B / "pm"
PM_PL = WORK_F2B / "placebo_pm.f32"


# --------------------------------------------------------------------------- shared inputs (also used by Q3-Q5)

def inputs(cfg: dict[str, Any]) -> dict[str, Any]:
    """F2 context + P and M bar returns on common coverage, EW market (M), funding, F2 refits and theta."""
    ctx = context(cfg)
    ctx["until_ts"] = pd.Timestamp(cfg["period"]["data_end_utc"])
    R_P, fs, _ = market_inputs(ctx)
    mk = {k: v.reindex(ctx["bars"]) for k, v in load_mark().items() if k != "mark_open_1h"}
    mo = mk["mark_open"].to_numpy()
    R_M = np.vstack([mo[1:] / mo[:-1] - 1, np.full((1, mo.shape[1]), np.nan)])
    cov = ~np.isnan(R_P) & ~np.isnan(R_M)                 # both returns of holding bar b exist
    R_P, R_M = np.where(cov, R_P, np.nan), np.where(cov, R_M, np.nan)
    mkt = np.nan_to_num(np.nanmean(np.where(ctx["alive"], R_M, np.nan), axis=1))
    refits = load_refits()
    theta = json.loads((LOGS / "theta_selected_F2.json").read_text(encoding="utf-8"))["theta"]
    # decision bar t is usable only if the asset has mark data for its first holding bar t+1
    dec_ok = np.vstack([cov[1:], np.zeros((1, cov.shape[1]), bool)])
    return {"ctx": ctx, "R_P": R_P, "R_M": R_M, "fs": fs, "mkt_M": mkt, "refits": refits, "theta": theta,
            "dec_ok": dec_ok, "win": windows(ctx, refits), "mark": mk}


def strategy_dec(inp: dict, cfg: dict, br: str, tgt: str, h: int, refits=None, score=None) -> np.ndarray:
    ctx = inp["ctx"]
    s = assemble(refits or inp["refits"], (br, tgt, h), ctx["alive"].shape) if score is None else score.copy()
    s[~inp["dec_ok"]] = np.nan
    return decisions(s, br, h, cfg["portfolio"]["X"]["quantile"], inp["theta"].get(KEY(br, tgt, h), 0.0))


def period_daily(inp: dict, comp: dict, period: str) -> pd.DataFrame:
    lo, hi = inp["win"][period]
    d = daily(comp, inp["ctx"]["bars"], lo, hi)
    b = inp["ctx"]["bars"][lo:hi]
    d["market"] = pd.Series(inp["mkt_M"][lo:hi], index=b).groupby(b.floor("D")).sum()
    return d


def ept_bp(d: pd.DataFrame, funding: bool = True) -> float:
    g = d["price"].sum() + (d["funding"].sum() if funding else 0.0)
    return float(g / d["turnover"].sum() * 1e4) if d["turnover"].sum() > 0 else np.nan


def mark_targets(cfg: dict, inp: dict) -> dict:
    mk = inp["mark"]
    pan = {"open": mk["mark_open"], "close": mk["mark_close"]}
    return targets(cfg, pan)


def oos_ic(inp: dict, refits: list, fam: tuple, tg: dict, period: str, score=None) -> float:
    br, tgt, h = fam
    rs = [r for r in refits if r["block"]["period"] == period]
    rows = np.unique(np.concatenate([np.arange(r["block"]["T0"], r["block"]["T1"]) for r in rs]))
    s = (assemble(refits, fam, inp["ctx"]["alive"].shape) if score is None else score)[rows].copy()
    s[~inp["dec_ok"][rows]] = np.nan
    y = tg[(tgt, h)][rows]
    if br == "X":
        return float(np.nanmean(rowwise_spearman(s, y)))
    ok = ~np.isnan(s) & ~np.isnan(y)
    return float(spearmanr(s[ok], y[ok])[0])


# --------------------------------------------------------------------------- PM: features and models on mark prices

def pm_panels(cfg) -> dict[str, pd.DataFrame]:
    pan = load_panels()
    mk = load_mark()
    out = dict(pan)
    for c in ("open", "high", "low", "close"):
        out[c] = mk[f"mark_{c}"].reindex(pan["close"].index)
    return out


def pm_features(cfg: dict, fb: dict) -> None:
    from .features import open_features
    _, meta = open_features()
    pan = pm_panels(cfg)
    cands = candidates(cfg, pan)
    assert [c[0] for c in cands] == meta["names"], "grammar differs from F2"
    b = base_series(pan)
    kept = meta["kept"]
    mm = np.memmap(PM_MM, dtype=np.float32, mode="w+", shape=(len(kept), meta["shape"][1], meta["shape"][2]))
    for n, i in enumerate(kept):
        _, _, var, fn = cands[i]
        mm[n] = apply_cs(fn(b), var)
    mm.flush()
    print(f"{now_utc()} PM features: {len(kept)} kept F2 features recomputed on mark OHLC")


def pm_context(cfg, inp) -> dict:
    ctx = dict(inp["ctx"])
    n = len(ctx["kept"])
    ctx["feats"] = np.memmap(PM_MM, dtype=np.float32, mode="r", shape=(n,) + tuple(ctx["meta"]["shape"][1:]))
    ctx["kept"] = np.arange(n)
    tg = dict(ctx["tg"])
    tg.update({k: v for k, v in mark_targets(cfg, inp).items()})
    ctx["tg"] = tg
    ctx["alive"] = ctx["alive"] & ~np.isnan(inp["mark"]["mark_close"].to_numpy())
    return ctx


def pm_walkforward(cfg: dict, fb: dict) -> None:
    PM_DIR.mkdir(parents=True, exist_ok=True)
    inp = inputs(cfg)
    ctx = pm_context(cfg, inp)
    c1 = copy.deepcopy(cfg)
    c1["horizons_bars"] = [fb["q1"]["horizon"]]
    fams = [(br, t, fb["q1"]["horizon"]) for br, t in fb["q1"]["pm_retrain"]]
    n_real = len(ctx["kept"])
    n_pl = int(round(n_real * cfg["placebo"]["share_of_real"]))
    for k, blk in enumerate(blocks(cfg, ctx)):
        path = PM_DIR / f"{blk['label']}.pkl"
        if path.exists():
            continue
        t0 = pd.Timestamp.now()
        seed = int(cfg["seed"]) + k                                     # same seeds as F2
        pl, pl_names = placebos(n_pl, ctx["kept"], ctx["feats"], ctx["alive"], blk["T1"], cfg, seed)
        mm = np.memmap(PM_PL, dtype=np.float32, mode="w+", shape=pl.shape)
        mm[:] = pl
        mm.flush()
        del mm
        sc = score_inputs(c1, ctx, blk["T0"], seed)
        shapes = {"real": (str(PM_MM), tuple(ctx["feats"].shape)), "placebo": (str(PM_PL), pl.shape)}
        src = [("real", int(j)) for j in ctx["kept"]] + [("placebo", i) for i in range(n_pl)]
        res = Parallel(n_jobs=14, backend="loky", max_nbytes="1M")(
            delayed(_score_chunk)(src[i:i + 40], shapes, ctx["a"], blk["T0"], sc) for i in range(0, len(src), 40))
        scores = [r for part in res for r in part]
        names = [ctx["meta"]["names"][j] for j in ctx["meta"]["kept"]] + pl_names
        sels = {f: select(c1, scores, names, n_real, f) for f in fams}
        fits = {f: fit_family(c1, ctx, blk, f, sels[f], pl) for f in fams}
        path.write_bytes(pickle.dumps({"block": blk, "selection": sels, "fits": fits, "utc": now_utc()}))
        print(f"{now_utc()} PM {blk['label']} {(pd.Timestamp.now() - t0).total_seconds():.0f}s", flush=True)
    log_attempt_f2b("Q1", "PM: features+targets on mark OHLC, F2 walk-forward re-fitted", len(fams), "dev+holdout",
                    "X:T1, X:T5, D:T1 at h=1; F2 seeds, BH, K, alpha grid")


def load_pm() -> list[dict]:
    out = [pickle.loads(p.read_bytes()) for p in sorted(PM_DIR.glob("*.pkl"))]
    return sorted(out, key=lambda r: r["block"]["T0"])


# --------------------------------------------------------------------------- Q1 report

def per_asset(dec: np.ndarray, R: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    held = np.vstack([np.zeros((1, dec.shape[1])), dec[:-1]])
    prev = np.vstack([np.zeros((1, dec.shape[1])), held[:-1]])
    return np.nansum((held * np.nan_to_num(R))[lo:hi], axis=0), np.abs(held - prev)[lo:hi].sum(axis=0)


def run_q1(cfg: dict, fb: dict) -> dict:
    rng = np.random.default_rng(fb["seed"])
    bs = fb["bootstrap"]
    inp = inputs(cfg)
    ctx = inp["ctx"]
    h = fb["q1"]["horizon"]
    tg_M = mark_targets(cfg, inp)
    pm = load_pm()
    # liquidity terciles by dev mean turnover; half median L1 spread from Stage 4
    turn = load_panels(["turnover"])["turnover"].reindex(ctx["bars"])
    lo_d, hi_d = inp["win"]["dev"]
    liq = turn.iloc[lo_d:hi_d].mean()
    terc = pd.qcut(liq.rank(method="first"), 3, labels=["низкая", "средняя", "высокая"])
    spread = pd.read_csv(ROOT / cfg["universe"]["stage4_spread_csv"]).set_index("symbol")["median_spread_bps"] / 2
    rows, trows = [], []
    strategies = [("X", t) for t in cfg["targets"]["branch_X"]] + [("D", t) for t in cfg["targets"]["branch_D"]]
    variants = [(br, t, "P") for br, t in strategies] + [(br, t, "M") for br, t in strategies] + \
               [(br, t, "PM") for br, t in fb["q1"]["pm_retrain"]]
    daily_store = {}
    for br, tgt, var in variants:
        if var == "PM":
            score = assemble(pm, (br, tgt, h), ctx["alive"].shape)
            dec = strategy_dec(inp, cfg, br, tgt, h, score=score)
        else:
            score = None
            dec = strategy_dec(inp, cfg, br, tgt, h)
        R = inp["R_P"] if var == "P" else inp["R_M"]
        comp = simulate(dec, R, inp["fs"], cfg)
        for period in inp["win"]:
            d = period_daily(inp, comp, period)
            daily_store[(br, tgt, var, period)] = d
            g = d["price"] + d["funding"]
            ic = oos_ic(inp, pm if var == "PM" else inp["refits"], (br, tgt, h), tg_M if var != "P" else ctx["tg"], period,
                        score=score)
            rows.append({"period": period, "branch": br, "target": tgt, "variant": var, "gross_sharpe_S0": sharpe(g),
                         "ci_lo": np.nan, "ci_hi": np.nan, "ept_bp": ept_bp(d), "ept_price_only_bp": ept_bp(d, False),
                         "turnover_day": d["turnover"].mean(), "ic": ic, "days": len(d)})
            if var == "M":
                rows[-1]["ci_lo"], rows[-1]["ci_hi"] = boot_sharpe_ci(g.to_numpy(), bs["reps"], bs["block_days"], bs["ci_level"], rng)
            if var in ("P", "M"):
                lo, hi = inp["win"][period]
                pnl, tv = per_asset(dec, R, lo, hi)
                for tc in terc.cat.categories:
                    m = (terc == tc).to_numpy()
                    trows.append({"period": period, "branch": br, "target": tgt, "variant": var, "tercile": tc,
                                  "ept_bp": pnl[m].sum() / tv[m].sum() * 1e4 if tv[m].sum() > 0 else np.nan,
                                  "half_spread_bp": float(spread.reindex(liq.index[m]).median())})
    T = pd.DataFrame(rows)
    T.to_csv(REPORTS / "q1_bounce_F2b.csv", index=False)
    # P - M differences (paired)
    diffs = []
    for br, tgt in strategies:
        for period in inp["win"]:
            a = (daily_store[(br, tgt, "P", period)][["price", "funding"]].sum(1)).to_numpy()
            b = (daily_store[(br, tgt, "M", period)][["price", "funding"]].sum(1)).to_numpy()
            d = boot_diff(a, b, bs["reps"], bs["block_days"], bs["ci_level"], rng)
            eP = T[(T.period == period) & (T.branch == br) & (T.target == tgt) & (T.variant == "P")].ept_bp.iloc[0]
            eM = T[(T.period == period) & (T.branch == br) & (T.target == tgt) & (T.variant == "M")].ept_bp.iloc[0]
            diffs.append({"period": period, "branch": br, "target": tgt, "sharpe_P_minus_M": d["diff"], "lo": d["lo"], "hi": d["hi"],
                          "ept_P": eP, "ept_M": eM, "bounce_share": (eP - eM) / eP if eP > 0 else np.nan, "ept_M_over_P": eM / eP if eP > 0 else np.nan})
    Dd = pd.DataFrame(diffs)
    Tt = pd.DataFrame(trows).pivot_table(index=["period", "branch", "target", "tercile", "half_spread_bp"], columns="variant",
                                         values="ept_bp", observed=True).reset_index()
    Tt["bounce_share"] = (Tt["P"] - Tt["M"]) / Tt["P"]
    # verdict per strategy on dev, overall on the three best F2 strategies (open question 1)
    rule = fb["q1"]["rule"]
    verd = []
    for br, tgt in strategies:
        dd = Dd[(Dd.period == "dev") & (Dd.branch == br) & (Dd.target == tgt)].iloc[0]
        lo_m = T[(T.period == "dev") & (T.branch == br) & (T.target == tgt) & (T.variant == "M")].ci_lo.iloc[0]
        r = dd.ept_M_over_P
        if not r > rule["artifact"]["ept_m_over_p_max"] or lo_m <= rule["artifact"]["or_sharpe_m_ci_lo_max"]:
            v = "артефакт"
        elif r >= rule["real"]["ept_m_over_p_min"] and lo_m > rule["real"]["sharpe_m_ci_lo_min"]:
            v = "сигнал настоящий"
        else:
            v = f"частично артефакт (отскок {1 - r:.0%})"
        verd.append({"branch": br, "target": tgt, "ept_M_over_P": r, "sharpe_M_ci_lo": lo_m, "verdict": v})
    V = pd.DataFrame(verd)
    best = V[V.apply(lambda x: [x.branch, x.target] in fb["q1"]["pm_retrain"], axis=1)]
    kinds = set(v.split(" (")[0] for v in best.verdict)
    overall = "артефакт" if kinds == {"артефакт"} else "сигнал настоящий" if kinds == {"сигнал настоящий"} else \
        f"частично артефакт (доля отскока по EPT у трёх лучших: {', '.join(f'{b.branch}:{b.target} {1 - b.ept_M_over_P:.0%}' for b in best.itertuples())})"
    (WORK_F2B / "q1_result.json").write_text(json.dumps({"overall": overall, "per_strategy": verd}, ensure_ascii=False, default=float, indent=1), encoding="utf-8")
    log_attempt_f2b("Q1", "P vs M vs PM at h=1", len(variants), "dev+holdout", f"overall: {overall}")
    figure_q1(T)
    write_q1(cfg, fb, T, Dd, Tt, V, overall)
    return {"overall": overall}


def figure_q1(T) -> None:
    from .report import C1, C2, C3, _style
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for i, period in enumerate(("dev", "holdout")):
        d = T[T.period == period]
        labels = (d[d.variant == "P"].branch + ":" + d[d.variant == "P"].target).tolist()
        x = np.arange(len(labels))
        for j, (col, title) in enumerate((("gross_sharpe_S0", "валовой Шарп (S0)"), ("ept_bp", "EPT, б.п. (пунктир — 6,5 б.п. издержек S1)"))):
            ax = axes[i, j]
            for k, (var, c) in enumerate((("P", C1), ("M", C2), ("PM", C3))):
                dv = d[d.variant == var].assign(lbl=lambda z: z.branch + ":" + z.target).set_index("lbl").reindex(labels)
                ax.bar(x + (k - 1) * 0.27, dv[col].to_numpy(dtype=float), 0.26, color=c, label={"P": "P: цена сделки", "M": "M: марк-цена", "PM": "PM: признаки и доходность по марк-цене"}[var])
            if col == "ept_bp":
                ax.axhline(6.5, color="#999", ls="--", lw=0.8)
            ax.set_xticks(x, labels, fontsize=8)
            ax.set_title(f"{'разработка' if period == 'dev' else 'контроль'}: {title}", fontsize=9)
            _style(ax)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Q1. Горизонт 4 часа: доходность по цене сделки (P) против марк-цены (M)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "f2b_fig1_bounce.png", dpi=130)
    plt.close(fig)


def write_q1(cfg, fb, T, Dd, Tt, V, overall) -> None:
    cov = pd.read_csv(CLEAN / "mark_coverage.csv", index_col=0)
    low = cov[cov.common_bars < 0.99 * cov.last_bars]
    (REPORTS / "q1_bounce.md").write_text(f"""# Q1. Проверка на отскок между bid и ask (h = 1, 4 часа)

**Отскок между bid и ask** — цена последней сделки случайно попадает то на лучшую цену покупки, то на лучшую цену продажи,
и цена «прыгает» без реального движения. **Марк-цена** — справедливая цена контракта, которую считает биржа; на ней этого нет.
**EPT (выигрыш на единицу оборота)** — валовая прибыль (цена + фандинг) / оборот, в б.п.; сравнивается с ценой одной стороны сделки (6,5 б.п. в S1).

Варианты: **P** — как в F2 (доходность по открытию последней сделки); **M** — те же прогнозы и веса F2, доходность по марк-цене
открытия следующего бара; **PM** — признаки и цель обучения по марк-цене (OHLC; оборот, объём, OI, фандинг, премиум — как в F2),
модель переобучена тем же walk-forward F2 (X:T1, X:T5, D:T1). Сравнение на общем покрытии: актив исключается на баре, если
нет марк-цены для его ближайшего бара удержания. Активы с покрытием марк-цен < 99% от покрытия цен сделок: {', '.join(low.index) or 'нет'}
(общее покрытие {cov.common_bars.sum() / cov.last_bars.sum():.2%} баров).

## Вывод (правило заморожено; решающие числа — разработка)

**{overall}.**

Правило: «артефакт», если EPT_M ≤ 50% EPT_P или нижняя граница 90% интервала Шарпа M ≤ 0; «сигнал настоящий», если EPT_M ≥ 80% EPT_P и
нижняя граница > 0; иначе «частично артефакт». Общий вывод — по трём лучшим стратегиям F2 (X:T1, X:T5, D:T1), см. `open_questions_F2b.md`.

{md(V.round(3), index=False)}

## Метрики

`gross_sharpe_S0` — валовой Шарп (цена + фандинг, без издержек); `ci_lo/ci_hi` — 90% интервал для M (блочный бутстреп 7 дней, 2000 повторов);
`ept_bp` — EPT; `ept_price_only_bp` — без фандинга; `ic` — IC прогноза с целью (P — цель по ценам сделок, M и PM — по марк-цене).

{md(T.round(3), index=False)}

![bounce](figures/f2b_fig1_bounce.png)

## Разность P − M (парная, блочный бутстреп)

`bounce_share` = (EPT_P − EPT_M) / EPT_P — доля сигнала, объяснённая отскоком.

{md(Dd.round(3), index=False)}

## По терцилям ликвидности (средний 4-часовой оборот актива на разработке)

`half_spread_bp` — медиана половины медианного L1-спреда Stage 4 по активам терциля (48 ч записи, октябрь 2026).

{md(Tt.round(3), index=False)}

Оговорка: {HOLDOUT_NOTE}.
""", encoding="utf-8")
    print(f"Q1: {overall}")
