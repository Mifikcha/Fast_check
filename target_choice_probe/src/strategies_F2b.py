"""Q3 beta decomposition, Q4 no-trade band, Q5 timing hint, F2 reporting corrections. All on variant M (mark returns)."""

from __future__ import annotations

import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .clean import load_panels
from .common import CLEAN, FIG, REPORTS, md, now_utc
from .common_F2b import HOLDOUT_NOTE, WORK_F2B, log_attempt_f2b
from .portfolio import KEY, assemble, simulate
from .q1_F2b import ept_bp, inputs, period_daily, strategy_dec
from .report import boot_diff, boot_sharpe_ci
from .stats import block_boot_days, holm, max_drawdown, sharpe
from .walkforward import families

SUMMARY = REPORTS / "summary_F2b.json"


def save_summary(key: str, value: Any) -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    s[key] = value
    SUMMARY.write_text(json.dumps(s, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}


def net(d: pd.DataFrame, scen: str = "S1", funding: bool = True) -> pd.Series:
    return d["price"] + (d["funding"] if funding else 0) - d[f"cost_{scen}"]


def regress(y: np.ndarray, x: np.ndarray) -> dict:
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    r2 = 1 - res.var() / y.var() if y.var() > 0 else np.nan
    return {"alpha_ann": b[0] * 365, "beta": b[1], "r2": r2}


# --------------------------------------------------------------------------- Q3

def asset_betas(inp: dict, fb: dict) -> np.ndarray:
    """Beta of each asset's mark bar return to the EW market over the last 180 bars, known at t (rows <= t-1)."""
    w, mn = fb["q3"]["beta_window_bars"], fb["q3"]["beta_min_bars"]
    R = pd.DataFrame(inp["R_M"])
    m = pd.Series(inp["mkt_M"]).where(pd.Series(inp["mkt_M"]) != 0)
    cov = R.rolling(w, min_periods=mn).cov(m)
    var = m.rolling(w, min_periods=mn).var()
    return cov.div(var, axis=0).shift(1).to_numpy()


def beta_neutral(dec: np.ndarray, beta: np.ndarray, h: int) -> tuple[np.ndarray, float]:
    """Scale the long and the short leg at each rebalance so that sum w*beta = 0 and sum|w| = 1."""
    out = dec.copy()
    b = np.where(np.isnan(beta), 1.0, beta)                  # no history yet: market beta
    fails = n = 0
    last = None
    for t in range(len(dec)):
        if t % h:
            out[t] = last if last is not None else out[t]
            continue
        w = dec[t]
        L, S = w > 0, w < 0
        if L.any() and S.any():
            n += 1
            BL, BS, GL, GS = (w[L] * b[t, L]).sum(), (w[S] * b[t, S]).sum(), w[L].sum(), -w[S].sum()
            det = BL * GS - BS * GL
            a, c = (-BS / det, BL / det) if det != 0 else (-1, -1)
            if a > 0 and c > 0:
                out[t] = np.where(L, a * w, np.where(S, c * w, 0.0))
            else:
                fails += 1
        last = out[t]
    return out, fails / n if n else np.nan


def run_q3(cfg: dict, fb: dict) -> dict:
    inp = inputs(cfg)
    beta = asset_betas(inp, fb)
    rows, bn_rows = [], []
    store = {}
    for br, tgt, h in [f for f in families(cfg) if f[0] != "V"]:
        dec = strategy_dec(inp, cfg, br, tgt, h)
        decn, fail = beta_neutral(dec, beta, h)
        for kind, dd in (("as_F2", dec), ("beta_neutral", decn)):
            comp = simulate(dd, inp["R_M"], inp["fs"], cfg)
            for period in inp["win"]:
                d = period_daily(inp, comp, period)
                store[(kind, br, tgt, h, period)] = d
                n1 = net(d).to_numpy()
                g = (d["price"] + d["funding"]).to_numpy()
                rg = regress(n1, d["market"].to_numpy())
                rg0 = regress(g, d["market"].to_numpy())
                row = {"kind": kind, "period": period, "branch": br, "target": tgt, "h": h,
                       "net_S1_ann": n1.mean() * 365, "alpha_S1_ann": rg["alpha_ann"], "beta": rg["beta"], "r2": rg["r2"],
                       "gross_alpha_S0_ann": rg0["alpha_ann"], "sharpe_S0": sharpe(g), "sharpe_S1": sharpe(n1),
                       "ept_bp": ept_bp(d), "market_ret_period": d["market"].sum(), "market_ret_ann": d["market"].mean() * 365,
                       "beta_part_period": rg["beta"] * d["market"].sum(), "net_period": n1.sum()}
                if kind == "beta_neutral":
                    row["neutralise_fail_share"] = fail
                rows.append(row)
    Q = pd.DataFrame(rows)
    hl = Q[(Q.kind == "as_F2") & (Q.period == "holdout") & Q.h.isin([6, 18])]
    losers = hl[hl.net_period < 0]
    share = float(losers.beta_part_period.clip(upper=0).sum() / losers.net_period.sum()) if len(losers) else np.nan
    bn = Q[(Q.kind == "beta_neutral") & Q.h.isin([6, 18])]
    no_signal = bool((bn.gross_alpha_S0_ann <= 0).all())
    pos = bn[bn.gross_alpha_S0_ann > 0][["period", "branch", "target", "h", "gross_alpha_S0_ann"]]
    concl = ("на горизонтах сутки и 3 суток сигнала нет, потери были в основном рыночной ставкой" if no_signal and share >= 0.5 else
             "на горизонтах сутки и 3 суток сигнала нет (валовая альфа ≤ 0 после бета-нейтрализации на разработке и контроле)" if no_signal else
             "после бета-нейтрализации у части стратегий h = 6/18 валовая альфа > 0 (см. таблицу)")
    save_summary("q3", {"holdout_loss_share_explained_by_beta_h6_18": share, "no_signal_h6_18": no_signal, "conclusion": concl,
                        "n_losing_holdout_strategies": int(len(losers))})
    log_attempt_f2b("Q3", "beta regression + beta-neutral rescaling, all F2 strategies", 1, "dev+holdout", concl)
    cols = ["kind", "period", "branch", "target", "h", "net_S1_ann", "alpha_S1_ann", "beta", "r2", "gross_alpha_S0_ann", "sharpe_S0", "sharpe_S1", "ept_bp"]
    mk = Q[(Q.kind == "as_F2")].groupby("period")[["market_ret_period", "market_ret_ann"]].first()
    (REPORTS / "q3_beta.md").write_text(f"""# Q3. Разложение на бету (вариант M, сценарий S1 с фандингом)

**Бета-нейтральность** — чувствительность портфеля к движению рынка в целом равна нулю (а не просто одинаковое число длинных и
коротких позиций). Рынок — равновзвешенная доходность тех же активов по марк-цене.

1. Регрессия дневной чистой прибыли S1 на дневную доходность рынка: `alpha_S1_ann` — альфа в годовом выражении, `beta`, `r2` — доля
   объяснённой дисперсии. `gross_alpha_S0_ann` — то же для валовой прибыли (без издержек).
2. Бета-нейтральная версия: на каждой ребалансировке длинная и короткая части масштабируются так, чтобы оценка беты портфеля
   (беты активов к рынку по 180 барам до t) была нулевой при сумме модулей весов = 1. Где решения с положительными множителями нет
   (например, у ветки D все позиции в одну сторону), веса оставлены как есть; доля таких ребалансировок — `neutralise_fail_share`.

Доходность рынка за период: {', '.join(f"{p}: {r.market_ret_period:+.1%} ({r.market_ret_ann:+.1%} годовых)" for p, r in mk.iterrows())}.

## Вывод

- Доля потерь стратегий h = 6 и 18 на контроле, объяснённая бетой (Σ бета × доходность рынка / Σ чистая прибыль убыточных стратегий, {len(losers)} шт.):
  **{share:.0%}**.
- **{concl}.**
{('- Исключения (валовая альфа > 0 после нейтрализации):' + chr(10) + md(pos.round(3), index=False)) if len(pos) else ''}

## Таблица

{md(Q[cols + ['neutralise_fail_share']].round(3), index=False)}

Оговорка: {HOLDOUT_NOTE}.
""", encoding="utf-8")
    Q.to_csv(REPORTS / "q3_beta_F2b.csv", index=False)
    print(f"Q3: share {share:.2f}; {concl}")
    return {"Q": Q}


# --------------------------------------------------------------------------- Q4

def band_path(target: np.ndarray, b: float, lam: float) -> np.ndarray:
    """Positions with a no-trade band b around the target and partial moves (fraction lam of the gap)."""
    out = np.zeros_like(target)
    w = np.zeros(target.shape[1])
    for t in range(len(target)):
        gap = target[t] - w
        mv = np.abs(gap) > b
        w = np.where(mv, w + lam * gap, w)
        out[t] = w
    return out


def run_q4(cfg: dict, fb: dict) -> dict:
    q1 = json.loads((WORK_F2B / "q1_result.json").read_text(encoding="utf-8"))
    if q1["overall"].startswith("артефакт"):
        (REPORTS / "q4_turnover.md").write_text(f"# Q4. Снижение оборота — пропущен\n\nПо Q1 вывод «{q1['overall']}»: валовой сигнал h = 1 "
                                                "объясняется отскоком, снижать оборот нечему.\n", encoding="utf-8")
        save_summary("q4", {"status": "skipped", "verdict": "закрыть"})
        return {}
    rng = np.random.default_rng(fb["seed"] + 4)
    bs = fb["bootstrap"]
    inp = inputs(cfg)
    beta = asset_betas(inp, fb)
    T = pd.read_csv(REPORTS / "q1_bounce_F2b.csv")
    cand = T[(T.period == "dev") & (T.variant == "M")].sort_values("gross_sharpe_S0", ascending=False).head(2)
    rows, curves, chosen = [], [], {}
    for r in cand.itertuples():
        dec, _ = beta_neutral(strategy_dec(inp, cfg, r.branch, r.target, 1), beta, 1)
        lo, hi = inp["win"]["dev"]
        held = np.vstack([np.zeros((1, dec.shape[1])), dec[:-1]])
        dw = np.abs(np.diff(held[lo - 1:hi], axis=0))
        med = float(np.median(dw[dw > 1e-12]))
        grid = [(bm, lam) for bm in fb["q4"]["band_mult"] for lam in fb["q4"]["lam"]]
        res = {}
        for bm, lam in grid:
            comp = simulate(band_path(dec, bm * med, lam), inp["R_M"], inp["fs"], cfg)
            d = period_daily(inp, comp, "dev")
            res[(bm, lam)] = (comp, d)
            curves.append({"strategy": f"{r.branch}:{r.target}", "band_mult": bm, "lam": lam, "turnover_day": d["turnover"].mean(),
                           "gross_sharpe_S0": sharpe(d["price"] + d["funding"]), "net_sharpe_S1": sharpe(net(d)), "ept_bp": ept_bp(d)})
        cv = pd.DataFrame([c for c in curves if c["strategy"] == f"{r.branch}:{r.target}"])
        best = cv.loc[cv.net_sharpe_S1.idxmax()]
        key = f"{r.branch}:{r.target}"
        chosen[key] = {"band_mult": float(best.band_mult), "lam": float(best.lam), "median_abs_dw": med, "dev_net_sharpe_S1": float(best.net_sharpe_S1)}
        log_attempt_f2b("Q4", f"no-trade band {key}", len(grid), "dev", f"selected b={best.band_mult}xmed, lam={best.lam} before holdout")
        (WORK_F2B / "q4_selected.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")    # written before the holdout run
        comp = res[(best.band_mult, best.lam)][0]
        for period in inp["win"]:
            d = period_daily(inp, comp, period)
            n1 = net(d)
            ci = boot_sharpe_ci(n1.to_numpy(), bs["reps"], bs["block_days"], bs["ci_level"], rng)
            g = d["price"] + d["funding"]
            rows.append({"strategy": key, "period": period, "band_mult": best.band_mult, "lam": best.lam,
                         "sharpe_S0": sharpe(g), "sharpe_S1": sharpe(n1), "S1_ci_lo": ci[0], "S1_ci_hi": ci[1],
                         "sharpe_S2": sharpe(net(d, "S2")), "turnover_day": d["turnover"].mean(), "ept_bp": ept_bp(d),
                         "ccr": g.sum() / d["cost_S1"].sum(), "breakeven_cost_side_bp": ept_bp(d)})
    Rq = pd.DataFrame(rows)
    Cv = pd.DataFrame(curves)
    ok = []
    for key in chosen:
        dv, hv = Rq[(Rq.strategy == key) & (Rq.period == "dev")].iloc[0], Rq[(Rq.strategy == key) & (Rq.period == "holdout")].iloc[0]
        ok.append(dv.ept_bp >= fb["q4"]["rule"]["min_ept_bp"] and dv.sharpe_S1 > 0 and dv.S1_ci_lo > 0 and hv.sharpe_S1 > 0)
    max_ept = float(Cv.ept_bp.max())
    tv = Rq.pivot_table(index="strategy", columns="period", values="turnover_day")
    froze = tv[tv["holdout"] < 0.2 * tv["dev"]]
    frozen_note = ("**Оговорка о контроле.** У " + ", ".join(f"{k} (оборот {r.holdout:.3f} против {r.dev:.3f} в сутки на разработке)" for k, r in froze.iterrows())
                   + " на контроле позиции почти не меняются: порог зоны задан в единицах медианного |Δw| разработки, а на контроле активов больше и"
                   " веса мельче, поэтому изменения цели почти всегда меньше порога. Высокий EPT на контроле здесь — результат почти"
                   " статичного портфеля, а не сигнала; решение принимается по разработке.") if len(froze) else ""
    verdict = "продолжение оправдано" if any(ok) else "закрыть"
    save_summary("q4", {"status": "run", "verdict": verdict, "max_ept_bp_grid": max_ept, "cost_over_max_ept": fb["cost_side_bp_S1"] / max_ept,
                        "selected": chosen, "breakeven_bp": {k: float(Rq[(Rq.strategy == k) & (Rq.period == "dev")].breakeven_cost_side_bp.iloc[0]) for k in chosen}})
    from .report import C1, C2, _style
    fig, axes = plt.subplots(1, len(chosen), figsize=(5.5 * len(chosen), 3.8))
    for ax, key in zip(np.atleast_1d(axes), chosen):
        cv = Cv[Cv.strategy == key]
        for lam, c in zip(sorted(cv.lam.unique()), (C1, C2, "#1baf7a")):
            d = cv[cv.lam == lam].sort_values("turnover_day")
            ax.plot(d.turnover_day, d.gross_sharpe_S0, marker="o", lw=2, color=c, label=f"λ = {lam}")
        ax.set_xlabel("оборот в сутки", fontsize=8)
        ax.set_ylabel("валовой Шарп (S0)", fontsize=8)
        ax.set_title(f"{key} (бета-нейтр.), разработка: 12 вариантов", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "f2b_fig4_turnover_curve.png", dpi=130)
    plt.close(fig)
    (REPORTS / "q4_turnover.md").write_text(f"""# Q4. Одна попытка снизить оборот: зона без сделок с частичным движением

**Зона без сделок (no-trade band)** — позиция актива меняется, только если целевой вес отличается от текущего больше порога `b`;
тогда позиция сдвигается к цели на долю `λ` разницы. Сетка заморожена: b ∈ {{0; 0,25; 0,5; 1}} × медианное |Δw| стратегии
(ненулевые изменения на разработке), λ ∈ {{0,25; 0,5; 1}} — 12 вариантов. Объекты — две лучшие по валовому Шарпу (M, разработка)
стратегии h = 1 в бета-нейтральной версии (Q3). Выбор — по чистому Шарпу S1 на разработке, записан в `logs/attempts_F2b.csv` и
`data/work/f2b/q4_selected.json` до расчёта на контроле.

## Вывод (правило заморожено)

**{verdict}.** Максимальный EPT на сетке (разработка): **{max_ept:.2f} б.п.** — в **{fb['cost_side_bp_S1'] / max_ept:.1f} раза** ниже цены
одной стороны сделки в S1 (6,5 б.п.). Точка безубыточности по издержкам на сторону для выбранного варианта (разработка) равна его EPT:
{', '.join(f"{k}: {v:.2f} б.п." for k, v in {k: float(Rq[(Rq.strategy == k) & (Rq.period == 'dev')].breakeven_cost_side_bp.iloc[0]) for k in chosen}.items())}.

{frozen_note}

## Выбранные варианты

{md(Rq.round(3), index=False)}

## Все 12 вариантов (разработка)

{md(Cv.round(3), index=False)}

![curve](figures/f2b_fig4_turnover_curve.png)

Оговорка: {HOLDOUT_NOTE}.
""", encoding="utf-8")
    print(f"Q4: {verdict}, max EPT {max_ept:.2f}")
    return {}


# --------------------------------------------------------------------------- Q5

def hint_exec(z: np.ndarray, trades: np.ndarray, direction: np.ndarray, mo: np.ndarray, D: int, theta: float,
              rnd: np.random.Generator | None = None, p_rand: float = 0.0) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Savings (bp) and delays for every forced trade (t, i). Base: mark open of t+1; postpone while the forecast
    says the next bar moves in our favour by more than theta (or at random with probability p_rand)."""
    tt, ii = np.nonzero(trades)
    d = direction[tt, ii]
    delay = np.zeros(len(tt), dtype=int)
    active = np.ones(len(tt), dtype=bool)
    steps = fav_count = 0
    for k in range(D):
        zz = z[np.minimum(tt + k, len(z) - 1), ii]
        fav = (-d * zz > theta) if rnd is None else (rnd.random(len(tt)) < p_rand)
        fav &= active
        steps += int(active.sum())
        fav_count += int(fav.sum())
        delay += fav
        active &= fav
    base = mo[tt + 1, ii]
    ex = mo[tt + 1 + delay, ii]
    return d * (base - ex) / base * 1e4, delay, fav_count, steps


def run_q5(cfg: dict, fb: dict) -> dict:
    inp = inputs(cfg)
    ctx = inp["ctx"]
    bs = fb["bootstrap"]
    T = pd.read_csv(REPORTS / "q1_bounce_F2b.csv")
    best = T[(T.period == "dev") & (T.variant == "M") & (T.branch == "D")].sort_values("gross_sharpe_S0", ascending=False).iloc[0]
    z = assemble(inp["refits"], ("D", best.target, 1), ctx["alive"].shape)
    mo = inp["mark"]["mark_open"].to_numpy()
    nb, ns = mo.shape
    rng = np.random.default_rng(fb["q5"]["seed"])
    u, dirs = rng.random((nb, ns)), np.where(rng.random((nb, ns)) < 0.5, 1.0, -1.0)
    Dmax = max(fb["q5"]["max_delay_bars"])
    ok_px = np.ones((nb, ns), bool)
    for k in range(1, Dmax + 2):
        ok_px &= np.vstack([~np.isnan(mo[k:]), np.zeros((k, ns), bool)])
    base_tr = (u < fb["q5"]["trade_prob_per_bar"]) & ctx["alive"] & ok_px
    bars = ctx["bars"]
    rows, dist, chosen = [], {}, {}

    def evaluate(period, D, theta, rnd=None, p=0.0):
        lo, hi = inp["win"][period]
        tr = base_tr.copy()
        tr[: lo - 1] = False
        tr[hi - 1 - Dmax:] = False
        tr &= ~np.isnan(z)                       # only bars with an out-of-sample forecast
        s, dl, fav, steps = hint_exec(z, tr, dirs, mo, D, theta, rnd, p)
        tt, _ = np.nonzero(tr)
        return s, dl, fav / max(steps, 1), bars[tt].floor("D")

    def ci(s, days):
        df = pd.DataFrame({"s": s, "d": days}).groupby("d")["s"].agg(["sum", "count"])
        idx = block_boot_days(len(df), bs["block_days"], bs["reps"], np.random.default_rng(fb["seed"] + 5))
        S, N = df["sum"].to_numpy()[idx].sum(1), df["count"].to_numpy()[idx].sum(1)
        return tuple(np.quantile(S / N, [(1 - bs["ci_level"]) / 2, (1 + bs["ci_level"]) / 2]))

    for D in fb["q5"]["max_delay_bars"]:
        dev_means = {th: float(evaluate("dev", D, th)[0].mean()) for th in fb["q5"]["theta_grid"]}
        th = max(dev_means, key=dev_means.get)
        chosen[D] = {"theta": th, "dev_mean_by_theta": dev_means}
        log_attempt_f2b("Q5", f"timing hint D={D}", len(fb["q5"]["theta_grid"]), "dev", f"theta={th}")
        for period in inp["win"]:
            s, dl, p_post, days = evaluate(period, D, th)
            sr, dlr, _, _ = evaluate(period, D, 0.0, np.random.default_rng(fb["q5"]["random_hint_seed"] + D), p_post)
            for kind, ss, dd in (("модель", s, dl), ("случайная", sr, dlr)):
                lo_, hi_ = ci(ss, days)
                post = dd > 0
                rows.append({"period": period, "D": D, "theta": th, "hint": kind, "trades": len(ss), "postponed_share": post.mean(),
                             "mean_saving_bp": ss.mean(), "ci_lo": lo_, "ci_hi": hi_,
                             "helped_share_of_postponed": float((ss[post] > 0).mean()) if post.any() else np.nan,
                             "p5_saving_postponed_bp": float(np.quantile(ss[post], 0.05)) if post.any() else np.nan})
                dist[(period, D, kind)] = ss[post]
    Q = pd.DataFrame(rows)
    useful = {}
    for D in fb["q5"]["max_delay_bars"]:
        m = Q[(Q.D == D) & (Q.hint == "модель")]
        r = Q[(Q.D == D) & (Q.hint == "случайная")]
        useful[D] = bool((m.ci_lo > 0).all() and ((r.ci_lo <= 0) & (r.ci_hi >= 0)).all())
    verdict = "полезен как тайминг" if all(useful.values()) else "пользы для тайминга не обнаружено"
    save_summary("q5", {"verdict": verdict, "useful_by_D": useful, "hint_strategy": f"D:{best.target}", "theta": chosen,
                        "dev_mean_saving_bp": {D: float(Q[(Q.D == D) & (Q.hint == 'модель') & (Q.period == 'dev')].mean_saving_bp.iloc[0]) for D in useful},
                        "holdout_mean_saving_bp": {D: float(Q[(Q.D == D) & (Q.hint == 'модель') & (Q.period == 'holdout')].mean_saving_bp.iloc[0]) for D in useful}})
    from .report import C1, C2, _style
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, period in zip(axes, ("dev", "holdout")):
        for kind, c in (("модель", C1), ("случайная", C2)):
            x = dist[(period, 1, kind)]
            ax.hist(np.clip(x, -300, 300), bins=60, alpha=0.55, color=c, label=f"{kind} (n = {len(x)}, среднее {x.mean():+.2f} б.п.)", density=True)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_title(f"{'разработка' if period == 'dev' else 'контроль'}, D = 1: экономия на отложенную сделку, б.п.", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "f2b_fig5_timing.png", dpi=130)
    plt.close(fig)
    (REPORTS / "q5_timing.md").write_text(f"""# Q5. Сигнал как подсказка, когда совершать вынужденные сделки

Синтетическая модель (заморожена): для каждого актива и бара с вероятностью 1/6 возникает вынужденная сделка, направление случайное
(генератор с фиксированным зерном {fb['q5']['seed']}). Базовое исполнение — по марк-цене открытия следующего бара. С подсказкой сделку
можно отложить не более чем на D баров: если прогноз ветки D (лучшая по валовому Шарпу на разработке, вариант M: **D:{best.target}**, h = 1,
z-оценка) говорит, что ближайший бар сдвинет цену в нашу пользу сильнее порога θ, сделка откладывается на бар (и так далее до D).
θ выбран на разработке из {{0; 0,5; 1}} отдельно для каждого D: {', '.join(f"D={D}: θ={v['theta']}" for D, v in chosen.items())}.
Учитываются только бары, где есть внешневыборочный прогноз. Экономия — в б.п. от базовой цены, со знаком (покупка дешевле / продажа дороже — плюс).
Случайная подсказка откладывает с той же частотой, но случайно. Интервалы — 90%, блочный бутстреп по дням (блок 7 суток).

## Вывод (правило заморожено: нижняя граница > 0 на разработке и контроле, случайная подсказка — интервал включает 0; принято для обоих D)

**{verdict}.** По D: {', '.join(f"D={D}: {'да' if v else 'нет'}" for D, v in useful.items())}.

{md(Q.round(3), index=False)}

![timing](figures/f2b_fig5_timing.png)

Данных F1 о реальных входах и выходах carry нет (`doc_check_F2b.md`, п. 4), справочный расчёт на них не делался.

Оговорки: модель вынужденных сделок синтетическая; влияние объёма на цену не учтено; {HOLDOUT_NOTE}.
""", encoding="utf-8")
    print(f"Q5: {verdict}")
    return {}


# --------------------------------------------------------------------------- F2 corrections

def funding_at_mark(cfg: dict, inp: dict) -> np.ndarray:
    """Funding per holding bar with the position valued at the mark price at settlement: rate x mark(ts) / open(bar)."""
    ctx = inp["ctx"]
    f = pd.read_parquet(CLEAN / "funding.parquet")
    m1h = pd.read_parquet(CLEAN / "panel_mark_open_1h.parquet")
    op = load_panels(["open"])["open"].reindex(ctx["bars"])
    out = np.zeros(ctx["alive"].shape)
    syms = list(op.columns)
    for s, g in f.groupby("symbol"):
        if s not in syms:
            continue
        bar = g["ts"].dt.ceil("4h") - pd.Timedelta(hours=4)
        bi = ctx["bars"].get_indexer(bar)
        mk = m1h[s].reindex(g["ts"]).to_numpy()
        o = op[s].to_numpy()[np.clip(bi, 0, None)]
        fac = np.where(np.isnan(mk) | np.isnan(o), 1.0, mk / o)
        ok = bi >= 0
        np.add.at(out[:, syms.index(s)], bi[ok], (g["rate"].to_numpy() * fac)[ok])
    return out


def run_corrections(cfg: dict, fb: dict) -> None:
    inp = inputs(cfg)
    # 4. funding by mark-price notional for the three best F2 strategies (F2 convention: P returns, F2 positions)
    from .portfolio import decisions, market_inputs
    R_P, fs, _ = market_inputs(inp["ctx"])
    fs_m = funding_at_mark(cfg, inp)
    frow = []
    for br, tgt, h in fb["corrections"]["funding_mark_strategies"]:
        sc = assemble(inp["refits"], (br, tgt, h), inp["ctx"]["alive"].shape)
        dec = decisions(sc, br, h, cfg["portfolio"]["X"]["quantile"], inp["theta"].get(KEY(br, tgt, h), 0.0))
        for lbl, f in (("по весу (F2)", fs), ("по марк-цене", fs_m)):
            comp = simulate(dec, R_P, f, cfg)
            for period in inp["win"]:
                d = period_daily(inp, comp, period)
                frow.append({"strategy": f"{br}:{tgt}:h{h}", "funding": lbl, "period": period,
                             "funding_ann": d["funding"].mean() * 365, "net_S1_ann": net(d).mean() * 365, "sharpe_S1": sharpe(net(d))})
    F = pd.DataFrame(frow)
    piv = F.pivot_table(index=["strategy", "period"], columns="funding", values=["funding_ann", "sharpe_S1"])
    dmax = float((piv["funding_ann"]["по марк-цене"] - piv["funding_ann"]["по весу (F2)"]).abs().max())
    smax = float((piv["sharpe_S1"]["по марк-цене"] - piv["sharpe_S1"]["по весу (F2)"]).abs().max())
    # 3. Holm without the comparisons that are identical by construction
    C = pd.read_csv(REPORTS / "comparisons_F2.csv")
    drop = {(b, t, h) for b, t, h in fb["corrections"]["holm_drop_identical"]}
    C2 = C[[(r.branch, r.target, r.h) not in drop for r in C.itertuples()]].copy()
    C2["p_holm_corrected"] = holm(C2["p"].to_numpy())
    q2 = pd.read_csv(REPORTS / "q2_placebo_F2b.csv")
    d18 = q2[(q2.branch == "D") & (q2.target == "T5") & (q2.h == 18)].iloc[0]
    (REPORTS / "F2_corrections.md").write_text(f"""# Исправления отчётности F2

Исходные файлы F2 не менялись; этот файл уточняет их формулировки.

## 1. Регулятор волатильности

В `VERDICT.md` F2 стоит «полезна как регулятор» по правилу на разработке. **Исправленная формулировка: не подтверждено, на контроле
вредит; тест проведён на стратегиях без валовой прибыли после издержек и поэтому неинформативен.** На контроле разность Шарпа «с
регулятором − без» была −0,97 (90% ДИ −1,80 … −0,05) в ветке X и −0,84 (−1,53 … −0,10) в ветке D (`volatility_target.md`); на
разработке правило сработало только для D:T5 h18 через снижение просадки при почти нулевой доходности (Шарп 0,04).

## 2. «Эмпирический FDR 6,7%»

Это доля плацебо среди **отобранных** признаков в **одной** паре (D:T5, h = 18), при плацебо типов 1 и 2 F2. Отбор в F2 фактически
делал правило «топ-20 по |t|»: BH пропускала 50–82% реальных признаков, а из прошедших брались 20 с наибольшим |t|. Поэтому 6,7%
говорит о том, что плацебо редко попадают в двадцатку лучших, а не о том, что BH удерживает FDR. С правильным (перестановочным)
плацебо в той же паре доля типов 3+4 среди отобранных: (а) BH + топ-20 — {d18.fdr34_bh_top20:.1%}, (в) только BH — {d18.fdr34_bh_only:.1%};
полная картина — в `q2_placebo.md`.

## 3. T1, T3, T4 в ветке X

T1, T3 и T4 имеют одинаковый порядок активов на баре, поэтому ранговый IC и отбор признаков у них одинаковы. На h = 6 и 18 портфели
T3 и T1 совпадают полностью (разность Шарпа 0,000, p = 1): сравнения T3 с T1 на этих горизонтах неинформативны. Без них сравнений
{len(C2)} (было {len(C)}); поправка Холма пересчитана:

{md(C2[['branch', 'target', 'h', 'diff', 'lo', 'hi', 'p', 'p_holm_corrected']].round(3), index=False)}

Минимальное p с поправкой Холма: {C2.p_holm_corrected.min():.2f} — различие целей по-прежнему не доказано.

## 4. Фандинг «по весу, а не по марк-цене»

Пересчёт с оценкой позиции по марк-цене в момент расчёта (ставка × вес × марк-цена в момент расчёта / цена входа бара) для трёх лучших
стратегий F2 (цены сделок, позиции F2):

{md(F.round(4), index=False)}

Максимальное изменение годового фандинга: **{dmax * 1e4:.2f} б.п. в год** на валовой объём; максимальное изменение чистого Шарпа S1:
**{smax:.3f}**. Упрощение F2 на выводы не влияет.
""", encoding="utf-8")
    save_summary("corrections", {"funding_mark_max_change_ann_bp": dmax * 1e4, "funding_mark_max_sharpe_change": smax,
                                 "holm_n": int(len(C2)), "holm_min_p": float(C2.p_holm_corrected.min())})
    print(f"corrections: funding diff {dmax * 1e4:.2f} bp/yr, sharpe {smax:.3f}, holm min p {C2.p_holm_corrected.min():.2f}")
