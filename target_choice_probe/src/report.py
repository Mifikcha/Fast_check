"""Reports: price-level demo (T0), walk-forward summary, placebo, targets comparison, volatility, figures, VERDICT.md."""

from __future__ import annotations

import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .clean import load_panels
from .common import FIG, HOLDOUT_LOG, LOGS, REPORTS, ROOT, WORK, holdout_open, last_usable, md, now_utc, periods
from .download import universe
from .features import cut_panels
from .portfolio import KEY, assemble, decisions, daily, market_inputs, simulate, windows
from .stats import adf, block_boot_days, holm, max_drawdown, rowwise_spearman, sharpe
from .walkforward import X_CLASS, context, families, load_refits

C1, C2, C3, GRID, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#e4e3df", "#0b0b0b", "#52514e"
SIMPLE = ["T1", "T2", "T3", "T4", "T8", "T5", "T6"]          # tie-break "simpler first"
NAMES = {"T1": "сырая доходность", "T2": "знак", "T3": "отн. рынка", "T4": "ранг", "T5": "доходность/волатильность",
         "T6": "ранг T5", "T7": "волатильность", "T8": "метка «после издержек»"}


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbb")
    ax.tick_params(colors=INK2, labelsize=8)


# --------------------------------------------------------------------------- section 6: price level demo

def price_demo(cfg: dict[str, Any]) -> None:
    uni = universe(cfg)
    syms = list(uni["symbol"][:5])
    p = periods(cfg)
    pan = cut_panels(load_panels(["close"]), last_usable(cfg))["close"].loc[p["start"]:, syms]
    rows = []
    for s in syms:
        lp = np.log(pan[s].dropna().to_numpy())
        r = np.diff(lp)
        r2 = lambda x, y: np.corrcoef(x, y)[0, 1] ** 2
        rows.append({"актив": s, "баров": len(lp), "R² lnP(t+1) по lnP(t)": r2(lp[:-1], lp[1:]),
                     "R² r(t+1) по r(t)": r2(r[:-1], r[1:]),
                     "ADF lnP": adf(lp)["stat"], "ADF r": adf(r)["stat"]})
    tab = pd.DataFrame(rows)
    s = syms[0]
    lp = np.log(pan[s].dropna())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].scatter(lp.to_numpy()[:-1], lp.to_numpy()[1:], s=2, color=C1, alpha=0.4)
    ax[0].set_title(f"{s}: ln P(t+1) против ln P(t), R² = {tab.iloc[0, 2]:.4f}", fontsize=9, color=INK)
    r = np.diff(lp.to_numpy())
    ax[1].scatter(r[:-1], r[1:], s=2, color=C2, alpha=0.4)
    ax[1].set_title(f"{s}: доходность бара против прошлой, R² = {tab.iloc[0, 3]:.4f}", fontsize=9, color=INK)
    for a in ax:
        _style(a)
    fig.tight_layout()
    fig.savefig(FIG / "price_level_demo.png", dpi=130)
    plt.close(fig)
    text = f"""# Почему не предсказываем уровень цены (T0)

Данные: 5 самых ликвидных активов, 4-часовые бары, только период разработки ({p['start']:%Y-%m-%d} … {p['dev_end']:%Y-%m-%d}).

{md(tab, index=False)}

**Как читать.** R² «предсказания» логарифма цены следующего бара по текущему почти равен 1, но это не умение
предсказывать: ряд цены просто «помнит» своё положение (он нестационарен — тест Дики–Фуллера не отвергает
единичный корень, статистика выше критического значения −2,86 для 5%). Такая модель говорит «завтра цена будет
примерно как сегодня» и ничего не даёт для торговли. Как только переходим к изменению цены (доходности), R²
падает почти до нуля, а ряд становится стационарным (ADF сильно ниже −3,43). Вся торгуемая информация — в
доходности, а её предсказуемость мала.

Критические значения ADF (с константой, асимптотические MacKinnon): 1% −3,43; 5% −2,86; 10% −2,57.

![price demo](figures/price_level_demo.png)

**Вывод:** цену не предсказываем; дальше это не обсуждается.
"""
    (REPORTS / "price_level_demo.md").write_text(text, encoding="utf-8")
    print(text)


# --------------------------------------------------------------------------- metrics

def strat_daily(dly: pd.DataFrame, br: str, tgt: str, h: int, theta: dict, period: str) -> pd.DataFrame:
    d = dly[(dly.branch == br) & (dly.target == tgt) & (dly.h == h) & (dly.period == period)]
    if br == "D":
        d = d[np.isclose(d.theta, theta[KEY(br, tgt, h)])]
    return d.set_index("date").sort_index()


def net(d: pd.DataFrame, scen: str = "S1", funding: bool = True) -> pd.Series:
    return d["price"] + (d["funding"] if funding else 0) - d[f"cost_{scen}"]


def metrics(d: pd.DataFrame, scen: str, funding: bool) -> dict:
    n = net(d, scen, funding)
    gross = d["price"] + (d["funding"] if funding else 0)
    cost = d["cost_S1"].sum()
    mon = n.groupby(n.index.to_period("M")).sum()
    beta = np.cov(n, d["market"])[0, 1] / np.var(d["market"], ddof=1) if d["market"].std() > 0 else np.nan
    return {"ann_ret": n.mean() * 365, "sharpe": sharpe(n), "mdd": max_drawdown(n), "turnover_day": d["turnover"].mean(),
            "cost_day": d[f"cost_{scen}"].mean(), "ccr": gross.sum() / cost if cost > 0 else np.nan,
            "pos_months": float((mon > 0).mean()), "beta": beta, "days": len(n)}


def boot_sharpe_ci(x: np.ndarray, reps: int, block: int, level: float, rng) -> tuple[float, float]:
    idx = block_boot_days(len(x), block, reps, rng)
    b = x[idx]
    s = b.mean(1) / b.std(1, ddof=1) * np.sqrt(365)
    return tuple(np.nanquantile(s, [(1 - level) / 2, (1 + level) / 2]))


def boot_diff(a: np.ndarray, b: np.ndarray, reps: int, block: int, level: float, rng) -> dict:
    idx = block_boot_days(len(a), block, reps, rng)
    sa = a[idx].mean(1) / a[idx].std(1, ddof=1) * np.sqrt(365)
    sb = b[idx].mean(1) / b[idx].std(1, ddof=1) * np.sqrt(365)
    d = sa - sb
    lo, hi = np.nanquantile(d, [(1 - level) / 2, (1 + level) / 2])
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return {"diff": sharpe(a) - sharpe(b), "lo": lo, "hi": hi, "p": min(p, 1.0)}


# --------------------------------------------------------------------------- selection-side summaries

def selection_summary(refits: list[dict], fam: tuple) -> dict:
    s = [r["selection"][fam] for r in refits]
    npl = sum(x["n_placebo"] for x in s)
    nsel = sum(len(x["selected"]) for x in s)
    sets = [set(x["selected_names"]) for x in s]
    jac = [len(a & b) / len(a | b) for a, b in zip(sets[:-1], sets[1:]) if a | b]
    return {"months": len(s), "no_find": float(np.mean([len(x["selected"]) == 0 for x in s])),
            "placebo_pass_rate": sum(x["placebo_pass"] for x in s) / npl,
            "placebo_fdr": sum(x["selected_placebo"] for x in s) / nsel if nsel else np.nan,
            "placebo_naive_rate": sum(x["placebo_naive"] for x in s) / npl,
            "real_pass_rate": sum(x["real_pass"] for x in s) / sum(x["n_real"] for x in s),
            "real_naive_rate": sum(x["real_naive"] for x in s) / sum(x["n_real"] for x in s),
            "type1_pass_rate": sum(x["type1_pass"] for x in s) / sum(x["type1_n"] for x in s),
            "type2_pass_rate": sum(x["type2_pass"] for x in s) / sum(x["type2_n"] for x in s),
            "type2_naive_rate": sum(x["type2_naive"] for x in s) / sum(x["type2_n"] for x in s),
            "type1_fdr": sum(x["type1_selected"] for x in s) / nsel if nsel else np.nan,
            "type2_fdr": sum(x["type2_selected"] for x in s) / nsel if nsel else np.nan,
            "mean_selected": nsel / len(s), "jaccard": float(np.mean(jac)) if jac else np.nan,
            "alphas": [r["fits"][fam]["alpha"] for r in refits]}


def oos_ic(ctx, refits, fam) -> float:
    br, tgt, h = fam
    score = assemble(refits, fam, ctx["alive"].shape)
    rows = np.unique(np.concatenate([np.arange(r["block"]["T0"], r["block"]["T1"]) for r in refits]))
    y = ctx["tg"][(tgt, h)][rows]
    f = score[rows]
    if br == "X":
        return float(np.nanmean(rowwise_spearman(f, y)))
    ok = ~np.isnan(f) & ~np.isnan(y)
    return float(spearmanr(f[ok], y[ok])[0]) if ok.sum() > 100 else np.nan


# --------------------------------------------------------------------------- driver

def write_all(cfg: dict[str, Any]) -> None:
    rng = np.random.default_rng(cfg["seed"])
    cmp_cfg = cfg["comparison"]
    reps, blkd, lvl = cmp_cfg["bootstrap_reps"], cmp_cfg["bootstrap_block_days"], cmp_cfg["ci_level"]
    dly = pd.read_parquet(WORK / "daily.parquet")
    theta = json.loads((LOGS / "theta_selected_F2.json").read_text(encoding="utf-8"))["theta"]
    ctx = context(cfg)
    ctx["until_ts"] = last_usable(cfg)
    refits_all = load_refits()
    periods_ = ["dev"] + (["holdout"] if holdout_open() else [])
    ref = {p_: [r for r in refits_all if r["block"]["period"] == p_] for p_ in periods_}
    fams = [f for f in families(cfg) if f[0] != "V"]

    # ---- metrics table
    rows = []
    for br, tgt, h in fams:
        for period in periods_:
            d = strat_daily(dly, br, tgt, h, theta, period)
            sel = selection_summary(ref[period], (br, tgt, h))
            ic = oos_ic(ctx, ref[period], (br, tgt, h))
            for scen in ("S0", "S1", "S2"):
                for fund in (True, False):
                    m = metrics(d, scen, fund)
                    rows.append({"period": period, "branch": br, "target": tgt, "h": h, "scenario": scen, "funding": fund,
                                 **m, "oos_ic": ic, **{k: v for k, v in sel.items() if k != "alphas"}})
    M = pd.DataFrame(rows)
    M.to_csv(REPORTS / "metrics_F2.csv", index=False)
    main = M[(M.scenario == "S1") & M.funding]
    dev = main[main.period == "dev"].set_index(["branch", "target", "h"])
    hold = main[main.period == "holdout"].set_index(["branch", "target", "h"]) if "holdout" in periods_ else None

    # ---- Sharpe CIs and comparisons vs T1 (dev OOS, S1 + funding)
    ci, comps = {}, []
    for br, tgt, h in fams:
        x = net(strat_daily(dly, br, tgt, h, theta, "dev")).to_numpy()
        ci[(br, tgt, h)] = boot_sharpe_ci(x, reps, blkd, lvl, rng)
        if tgt != "T1":
            b = net(strat_daily(dly, br, "T1", h, theta, "dev")).to_numpy()
            comps.append({"branch": br, "target": tgt, "h": h, **boot_diff(x, b, reps, blkd, lvl, rng),
                          "corr_with_T1": float(np.corrcoef(x, b)[0, 1])})
    C = pd.DataFrame(comps)
    C["p_holm"] = holm(C["p"].to_numpy())
    C.to_csv(REPORTS / "comparisons_F2.csv", index=False)

    # ---- placebo table
    P = dev.reset_index()[["branch", "target", "h", "placebo_pass_rate", "placebo_fdr", "placebo_naive_rate",
                           "type1_pass_rate", "type2_pass_rate", "type2_naive_rate", "type1_fdr", "type2_fdr",
                           "real_pass_rate", "real_naive_rate", "no_find", "mean_selected", "jaccard"]]

    # ---- volatility
    vol = volatility(cfg, ctx, refits_all, dly, dev, theta, periods_, rng)

    # ---- decisions
    verdicts = decide(cfg, dev, hold, C, P, periods_)
    figures(cfg, dev, ci, P, vol)
    write_reports(cfg, M, dev, hold, ci, C, P, vol, verdicts, ref, periods_)


def volatility(cfg, ctx, refits, dly, dev, theta, periods_, rng) -> dict:
    out = {"fit": [], "reg": []}
    for period in periods_:
        rs = [r for r in refits if r["block"]["period"] == period]
        rows = np.unique(np.concatenate([np.arange(r["block"]["T0"], r["block"]["T1"]) for r in rs]))
        for h in cfg["horizons_bars"]:
            y = ctx["tg"][("T7", h)][rows]
            for name, field in (("HAR", "har_pred"), ("HAR+признаки", "pred")):
                f = assemble(rs, ("V", "T7", h), ctx["alive"].shape, field)[rows]
                ok = ~np.isnan(f) & ~np.isnan(y)
                e = y[ok] - f[ok]
                out["fit"].append({"period": period, "h": h, "model": name, "r2_oos": 1 - np.sum(e ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2),
                                   "mae": float(np.mean(np.abs(e))), "n": int(ok.sum())})
            out["fit"][-1]["mean_selected"] = float(np.mean([len(r["selection"][("V", "T7", h)]["selected"]) for r in rs]))
    # regulator on the best strategy of each branch (chosen by dev Sharpe, S1 + funding)
    R, fs, mkt = market_inputs(ctx)
    win = windows(ctx, refits)
    lo_clip, hi_clip = cfg["volatility"]["multiplier_clip"]
    for br in ("X", "D"):
        best = dev.xs(br, level="branch")["sharpe"].idxmax()
        tgt, h = best
        score = assemble(refits, (br, tgt, h), ctx["alive"].shape)
        dec = decisions(score, br, h, cfg["portfolio"]["X"]["quantile"], theta.get(KEY(br, tgt, h), 0.0))
        vf = np.exp(assemble(refits, ("V", "T7", h), ctx["alive"].shape) / 2)          # forecast vol = sqrt(RV)
        with np.errstate(all="ignore"):
            m = np.clip(np.nanmedian(vf, axis=1, keepdims=True) / vf, lo_clip, hi_clip)
        m = np.where(np.isnan(m), 1.0, m)
        reb = np.arange(len(m)) % h == 0
        m = pd.DataFrame(np.where(reb[:, None], m, np.nan)).ffill().fillna(1.0).to_numpy()   # fixed within the holding
        base, adj = simulate(dec, R, fs, cfg), simulate(dec, R, fs, cfg, mult=m)
        for period, (lo, hi) in win.items():
            a = net(daily(base, ctx["bars"], lo, hi)).to_numpy()
            b = net(daily(adj, ctx["bars"], lo, hi)).to_numpy()
            dd = boot_diff(b, a, cfg["comparison"]["bootstrap_reps"], cfg["comparison"]["bootstrap_block_days"], cfg["comparison"]["ci_level"], rng)
            out["reg"].append({"period": period, "branch": br, "target": tgt, "h": h,
                               "sharpe_base": sharpe(a), "sharpe_reg": sharpe(b), "diff": dd["diff"], "lo": dd["lo"], "hi": dd["hi"],
                               "mdd_base": max_drawdown(a), "mdd_reg": max_drawdown(b),
                               "ret_base": a.mean() * 365, "ret_reg": b.mean() * 365})
    return out


def decide(cfg, dev, hold, C, P, periods_) -> list[dict]:
    dc = cfg["decision"]
    Pi = P.set_index(["branch", "target", "h"])
    out = []
    for br in ("X", "D"):
        for h in cfg["horizons_bars"]:
            tg = cfg["targets"]["branch_X" if br == "X" else "branch_D"]
            rows = []
            for t in tg:
                k = (br, t, h)
                s_dev = dev.loc[k, "sharpe"]
                s_hold = hold.loc[k, "sharpe"] if hold is not None else np.nan
                p1 = s_dev > dc["P1"]["sharpe_gt"] and (hold is not None and np.sign(s_hold) == np.sign(s_dev))
                p2 = dev.loc[k, "ccr"] >= dc["P2"]["min_ccr"]
                fdr = Pi.loc[k, "placebo_fdr"]
                p3 = (not np.isnan(fdr)) and fdr <= dc["P3"]["max_placebo_fdr"]
                p4 = Pi.loc[k, "no_find"] <= dc["P4"]["max_no_find_share"]
                c = C[(C.branch == br) & (C.target == t) & (C.h == h)]
                rows.append({"target": t, "P1": p1, "P2": p2, "P3": p3, "P4": p4, "all": p1 and p2 and p3 and p4,
                             "sharpe_dev": s_dev, "sharpe_hold": s_hold, "ccr": dev.loc[k, "ccr"], "fdr": fdr,
                             "no_find": Pi.loc[k, "no_find"], "turnover": dev.loc[k, "turnover_day"],
                             "ci_lo_vs_T1": float(c["lo"].iloc[0]) if len(c) else np.nan})
            T = pd.DataFrame(rows)
            ok = T[T["all"]]
            if "holdout" not in periods_:
                v, rec = "ОЖИДАЕТ КОНТРОЛЯ", None
            elif ok.empty:
                v, rec = "НИ ОДНА ЦЕЛЬ НЕ ПРОХОДИТ", None
            else:
                better = ok[ok.ci_lo_vs_T1 > 0]
                if len(better) == 1 or (len(ok) == 1 and ok.target.iloc[0] == "T1"):
                    v, rec = "ЦЕЛЬ РЕКОМЕНДОВАНА", (better if len(better) == 1 else ok).target.iloc[0]
                else:
                    v = "РАЗНИЦА НЕ ДОКАЗАНА"
                    ok = ok.assign(simple=ok.target.map(SIMPLE.index)).sort_values(["fdr", "turnover", "simple"])
                    rec = ok.target.iloc[0]
            out.append({"branch": br, "h": h, "verdict": v, "recommended": rec, "table": T})
    return out


def figures(cfg, dev, ci, P, vol) -> None:
    hs = cfg["horizons_bars"]
    hl = {1: "4 часа", 6: "сутки", 18: "3 суток"}
    # 1. placebo
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, h in zip(axes, hs):
        q = P[P.h == h].assign(lbl=lambda d: d.branch + ":" + d.target)
        x = np.arange(len(q))
        ax.bar(x - 0.27, q.placebo_fdr.fillna(0), 0.25, color=C1, label="эмп. FDR (доля плацебо среди выбранных)")
        ax.bar(x, q.placebo_pass_rate, 0.25, color=C3, label="доля плацебо, прошедших BH")
        ax.bar(x + 0.27, q.placebo_naive_rate, 0.25, color=C2, label="доля плацебо с |t| > 2 (без поправки)")
        ax.axhline(0.10, color="#999", ls="--", lw=0.8)
        ax.set_xticks(x, q.lbl, rotation=60, fontsize=7)
        ax.set_title(f"горизонт {hl[h]}", fontsize=9)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Ложные открытия по целям (плацебо, разработка); пунктир — порог P3 = 10%", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_placebo_fdr.png", dpi=130)
    plt.close(fig)
    # 2-4: Sharpe with CI, CCR, no-find
    for name, col, title, ref in (("fig2_sharpe", "sharpe", "Чистый Шарп (S1, с фандингом, разработка), 90% ДИ", 0.0),
                                  ("fig3_ccr", "ccr", "CCR = валовая прибыль / издержки (S1); пунктир — порог 1,2", 1.2),
                                  ("fig4_nofind", "no_find", "Доля месяцев «нет находок»; пунктир — порог 50%", 0.5)):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
        for i, br in enumerate(("X", "D")):
            for j, h in enumerate(hs):
                ax = axes[i, j]
                d = dev.xs(br, level="branch").xs(h, level="h")
                x = np.arange(len(d))
                vals = d[col].to_numpy(dtype=float) if col != "no_find" else d["no_find"].to_numpy(dtype=float)
                ax.bar(x, np.nan_to_num(vals), 0.6, color=C1 if br == "X" else C2)
                if col == "sharpe":
                    lohi = np.array([ci[(br, t, h)] for t in d.index])
                    ax.errorbar(x, vals, yerr=[vals - lohi[:, 0], lohi[:, 1] - vals], fmt="none", ecolor=INK, lw=1, capsize=3)
                ax.axhline(ref, color="#999", ls="--", lw=0.8)
                ax.set_xticks(x, d.index, fontsize=8)
                ax.set_title(f"ветка {br}, горизонт {hl[h]}", fontsize=9)
                _style(ax)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"{name}.png", dpi=130)
        plt.close(fig)
    # 5. volatility regulator
    r = pd.DataFrame(vol["reg"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    lbl = r.period + "\n" + r.branch + ":" + r.target + " h" + r.h.astype(str)
    x = np.arange(len(r))
    axes[0].bar(x - 0.2, r.sharpe_base, 0.38, color=C1, label="без регулятора")
    axes[0].bar(x + 0.2, r.sharpe_reg, 0.38, color=C3, label="с регулятором")
    axes[0].set_title("Шарп (S1, с фандингом)", fontsize=9)
    axes[1].bar(x - 0.2, r.mdd_base, 0.38, color=C1)
    axes[1].bar(x + 0.2, r.mdd_reg, 0.38, color=C3)
    axes[1].set_title("Максимальная просадка (доля валового объёма)", fontsize=9)
    for ax in axes:
        ax.set_xticks(x, lbl, fontsize=7)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Регулятор размера позиции по прогнозу волатильности", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_vol_regulator.png", dpi=130)
    plt.close(fig)


def _fmt(df, cols, digits=3):
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(digits)
    return d


def write_reports(cfg, M, dev, hold, ci, C, P, vol, verdicts, ref, periods_) -> None:
    p = periods(cfg)
    n_dev_days = int(dev["days"].iloc[0])
    n_hold_days = int(hold["days"].iloc[0]) if hold is not None else 0
    se = lambda T: np.sqrt(1 / (T / 365))
    mde = lambda T, rho: 2.8 * se(T) * np.sqrt(2 * (1 - rho))
    rho_med = float(C["corr_with_T1"].median())
    power = pd.DataFrame([{"период": nm, "дней": T, "SE Шарпа": se(T), "мин. различие (ρ=0)": mde(T, 0),
                           f"мин. различие (ρ={rho_med:.2f}, медиана)": mde(T, rho_med), "мин. различие (ρ=0.8)": mde(T, 0.8)}
                          for nm, T in (("разработка (OOS)", n_dev_days), ("контроль", n_hold_days)) if T])

    # walkforward summary
    wf = []
    for period in periods_:
        for r in ref[period]:
            for fam, s in r["selection"].items():
                wf.append({"period": period, "block": r["block"]["label"][len(period) + 1:], "family": f"{fam[0]}:{fam[1]}:h{fam[2]}",
                           "real_pass": s["real_pass"], "placebo_pass": s["placebo_pass"], "selected": len(s["selected"]),
                           "selected_placebo": s["selected_placebo"], "alpha": r["fits"][fam]["alpha"],
                           "top3": "; ".join(s["selected_names"][:3])})
    W = pd.DataFrame(wf)
    W.to_csv(REPORTS / "walkforward_refits_F2.csv", index=False)
    piv = W.pivot_table(index="block", columns="family", values="selected", aggfunc="first")
    top = W[W.selected > 0].groupby("family")["top3"].agg(lambda s: pd.Series("; ".join(s).split("; ")).value_counts().head(3).to_dict())
    (REPORTS / "walkforward_summary.md").write_text(f"""# Walk-forward: отчёт по переобучениям

Переобучение раз в месяц, расширяющееся окно, первое окно 12 месяцев (с {p['start']:%Y-%m-%d}), purge = h баров, embargo
перед контролем не используется никогда. Отбор: BH при q = 5% внутри каждой пары (цель, h) по реальным признакам
**вместе с** плацебо, затем не более K = 20 с наибольшим |t|. α выбирается по последним 3 месяцам окна (MSE для Ridge,
log-loss для логистической регрессии). Полная таблица по каждому переобучению (сколько прошло, сколько плацебо,
α, первые 3 признака): `walkforward_refits_F2.csv`.

**Важно про ветку X.** T1, T3 и T4 имеют одинаковый порядок активов на каждом баре (T3 = T1 минус общее среднее,
T4 = ранг T1), поэтому ранговые IC признаков с ними совпадают, и отбор признаков у них одинаковый; различаются только
коэффициенты модели (на что учится Ridge). То же для пары T5/T6.

## Число выбранных признаков по месяцам (0 = «нет находок»)

{md(piv.fillna(-1).astype(int))}

## Самые частые признаки среди первых трёх (по числу месяцев)

{md(top.rename('признак: месяцев').to_frame())}
""", encoding="utf-8")

    # placebo
    Pt = _fmt(P, list(P.columns))
    conf = P[P.branch == "X"].groupby("target")["placebo_pass_rate"].mean()
    t1_vs = "выше" if conf.get("T1", 0) > conf.drop("T1", errors="ignore").mean() else "не выше"
    (REPORTS / "placebo.md").write_text(f"""# Плацебо и ложные открытия

На каждом переобучении к {dev['n_real'].iloc[0] if 'n_real' in dev else ''} реальным признакам добавлялись свежие плацебо (20% от числа реальных):
половина — реальные признаки, сдвинутые по времени внутри каждого актива на случайное смещение ≥ 60 суток (циклически),
половина — шум AR(1) с φ ∈ {{0,5; 0,9; 0,98}}. Плацебо проходили **тот же** отбор. Цифры по всем месяцам разработки:

- `placebo_pass_rate` — доля плацебо, прошедших BH (эмпирическая доля ложных срабатываний);
- `placebo_fdr` — доля плацебо среди **выбранных** признаков (эмпирический FDR; пусто, если ничего не выбиралось);
- `placebo_naive_rate` — доля плацебо с |t| > 2 без поправки;
- `type1_*`, `type2_*` — то же отдельно для сдвинутых реальных признаков (тип 1) и шума AR(1) (тип 2);
- `real_pass_rate`, `real_naive_rate` — то же для реальных признаков (для сравнения);
- `no_find` — доля месяцев без находок; `jaccard` — средний индекс Жаккара наборов признаков соседних месяцев.

{md(Pt, index=False)}

![placebo](figures/fig1_placebo_fdr.png)

**Проверка ожидания «у T1 (тяжёлые хвосты) ложных срабатываний больше».** В ветке X средняя доля плацебо, прошедших BH:
{', '.join(f'{k} {v:.3f}' for k, v in conf.items())}. У T1 она **{t1_vs}** среднего по остальным целям ветки X.
Замечание: в ветке X ранговый IC не зависит от хвостов (T1, T3, T4 дают одинаковый отбор), поэтому эффект хвостов
может проявиться только через ветку D (пул по всем парам) и через модель, а не через отбор.
""", encoding="utf-8")

    # targets comparison
    cols = ["ann_ret", "sharpe", "mdd", "turnover_day", "cost_day", "ccr", "pos_months", "oos_ic", "no_find", "beta", "jaccard"]
    blocks_txt = []
    for period in periods_:
        for scen in ("S1", "S2", "S0"):
            for fund in (True, False):
                if scen != "S1" and not fund:
                    continue
                t = M[(M.period == period) & (M.scenario == scen) & (M.funding == fund)]
                blocks_txt.append(f"### {period}, {scen}, {'с фандингом' if fund else 'без фандинга'}\n\n" + md(_fmt(t, ["branch", "target", "h"] + cols), index=False))
    cdisp = _fmt(C, ["branch", "target", "h", "diff", "lo", "hi", "p", "p_holm", "corr_with_T1"])
    cit = pd.DataFrame([{"strategy": f"{k[0]}:{k[1]}:h{k[2]}", "sharpe": dev.loc[k, "sharpe"], "ci_lo": v[0], "ci_hi": v[1]} for k, v in ci.items()])
    spread = pd.read_csv(ROOT / cfg["universe"]["stage4_spread_csv"])
    half = spread["median_spread_bps"] / 2
    (REPORTS / "targets_comparison.md").write_text(f"""# Сравнение целей прогноза: деньги после издержек

Портфели: ветка X — длинные верхние 20% и короткие нижние 20% активов по оценке модели, равные веса; ветка D — по каждому
активу +1/−1/0 по порогу θ на z-оценке (θ выбран на разработке, `logs/theta_selected_F2.json`). Пересчёт раз в h баров,
вход по открытию следующего бара. Доходность — на валовой объём (сумма модулей весов = 1). Издержки начисляются на
каждое изменение позиции. Фандинг — по расчётным ставкам, стоимость позиции ≈ вес. `beta` — бета к равновзвешенному рынку.
`ann_ret` — годовая чистая доходность на валовой объём; `mdd` — максимальная просадка (в долях валового объёма, сумма дневных доходностей).

Основное доказательство — объединённые внешневыборочные результаты walk-forward на разработке
({n_dev_days} дней); контроль ({n_hold_days} дней) — проверка знака и порядка величины.

**Проскальзывание 1 б.п.** Половина медианного L1-спреда по записям Stage 4 (48 ч, 80 символов): медиана по символам
{half.median():.2f} б.п., 90-й перцентиль {half.quantile(0.9):.2f} б.п., максимум {half.max():.2f} б.п. Это нижняя оценка
стоимости пересечения спреда для малого объёма; принятый 1 б.п. её покрывает для большинства символов (влияние объёма на цену не учтено).

## Метрики

{chr(10).join(blocks_txt)}

## Чистый Шарп с 90% доверительным интервалом (разработка, S1, с фандингом; блочный бутстреп 7 дней, {cfg['comparison']['bootstrap_reps']} повторов)

{md(_fmt(cit, list(cit.columns)), index=False)}

![sharpe](figures/fig2_sharpe.png)
![ccr](figures/fig3_ccr.png)
![nofind](figures/fig4_nofind.png)

## Разность чистых Шарпа с T1 (в своей ветке и горизонте), парный блочный бутстреп

`diff` — Шарп цели минус Шарп T1; `lo`, `hi` — 90% интервал; `p` — двустороннее бутстреп-p; `p_holm` — с поправкой Холма
на все {len(C)} сравнений; `corr_with_T1` — корреляция дневной прибыли с T1.

{md(cdisp, index=False)}

## Мощность

Ошибка годового Шарпа SE ≈ √(1/T) (T в годах, Шарп около нуля). Минимальное обнаружимое различие Шарпа двух стратегий
(двусторонний тест 5%, мощность 80%) ≈ 2,8 · SE · √(2(1−ρ)), где ρ — корреляция их дневной прибыли.

{md(_fmt(power, list(power.columns), 2), index=False)}
""", encoding="utf-8")

    # volatility
    F = pd.DataFrame(vol["fit"])
    Rg = pd.DataFrame(vol["reg"])
    vrules = cfg["decision"]["vol_regulator"]
    rd = Rg[Rg.period == "dev"]
    useful = []
    for r in rd.itertuples():
        mdd_red = 1 - r.mdd_reg / r.mdd_base if r.mdd_base > 0 else 0
        loss = 1 - r.ret_reg / r.ret_base if r.ret_base > 0 else np.nan
        ok = r.lo > vrules["ci_lower_gt"] or (mdd_red >= vrules["or_mdd_reduction"] and (loss <= vrules["max_return_loss"] if not np.isnan(loss) else False))
        useful.append({"branch": r.branch, "strategy": f"{r.target} h{r.h}", "mdd_reduction": mdd_red, "return_loss": loss, "useful": ok})
    U = pd.DataFrame(useful)
    vol_verdict = "ПОЛЕЗНА КАК РЕГУЛЯТОР" if U["useful"].any() else "НЕ ПОКАЗАЛА ПОЛЬЗЫ КАК РЕГУЛЯТОР"
    rh = Rg[Rg.period == "holdout"]
    if len(rh):     # the frozen rule is judged on dev; the holdout check is reported next to it, not hidden
        vol_verdict += (" по правилу на разработке (" + ", ".join(f"{r.branch}:{r.strategy}" for r in U[U.useful].itertuples()) + ")"
                        if U["useful"].any() else "")
        vol_verdict += "; на контроле разность Шарпа «с регулятором − без»: " + ", ".join(
            f"{r.branch} {r.diff:+.2f} (90% ДИ {r.lo:+.2f} … {r.hi:+.2f})" for r in rh.itertuples())
    (REPORTS / "volatility_target.md").write_text(f"""# Волатильность (T7) как цель и как регулятор размера позиции

T7 = ln RV, RV — сумма квадратов часовых лог-доходностей на окне от открытия бара t+1 до открытия бара t+1+h.
Базовая модель HAR — линейная регрессия на логарифмы прошлой RV за 6, 42 и 180 баров (сутки, неделя, месяц).
Модель с признаками — Ridge на HAR + признаки, отобранные тем же BH-отбором (пул по парам, как ветка D).
R² внешней выборки = 1 − SSE / сумма квадратов отклонений от среднего внешней выборки.

{md(_fmt(F, list(F.columns)), index=False)}

## Регулятор размера

Лучшая стратегия каждой ветки (по Шарпу на разработке, S1 с фандингом). Позиция актива умножается на
(медиана прогноза волатильности по активам на баре / прогноз волатильности актива), множитель ограничен [0,5; 2],
фиксируется на время удержания. `diff`, `lo`, `hi` — разность Шарпа «с регулятором − без» и её 90% интервал (парный блочный бутстреп).

{md(_fmt(Rg, list(Rg.columns)), index=False)}

Правило «полезна как регулятор» (по разработке): нижняя граница интервала > 0, или просадка ниже на ≥ 20% при потере
чистой доходности ≤ 10%.

{md(_fmt(U, list(U.columns)), index=False)}

**Вердикт по волатильности: {vol_verdict}.**

![vol](figures/fig5_vol_regulator.png)
""", encoding="utf-8")

    verdict_md(cfg, dev, hold, ci, C, P, verdicts, vol_verdict, F, n_dev_days, n_hold_days, power)


def verdict_md(cfg, dev, hold, ci, C, P, verdicts, vol_verdict, F, n_dev, n_hold, power) -> None:
    lines = []
    for v in verdicts:
        T = v["table"]
        passed = ", ".join(T.target[T["all"]]) or "—"
        lines.append(f"| {v['branch']} | {v['h']} | **{v['verdict']}** | {v['recommended'] or '—'} | {passed} |")
    best = dev["sharpe"].idxmax()
    b_ci = ci[best]
    b_fdr = P.set_index(["branch", "target", "h"]).loc[best, "placebo_fdr"]
    b_hold = hold.loc[best, "sharpe"] if hold is not None else float("nan")
    gloss = """- **Цель прогноза (target)** — что именно модель пытается угадать: цену, доходность, порядок активов и т. д.
- **Признак (feature)** — число, которое считается из прошлых данных и может подсказывать будущее (например, «рост за последние сутки»).
- **IC (информационный коэффициент)** — насколько порядок признака совпадает с порядком будущей доходности (ранговая корреляция Спирмена): 0 — как монетка, 1 — идеально.
- **t-статистика** — сила связи относительно шума; |t| около 2 обычно считают «значимым» для одной проверки.
- **Множественные проверки** — если проверить тысячи признаков, «лучшие» окажутся случайными; нужна поправка.
- **FDR (доля ложных открытий)** — какая часть «найденных» признаков на самом деле шум. **Метод Бенджамини–Хохберга (BH)** удерживает FDR на заданном уровне (здесь q = 5%).
- **Плацебо-признак** — заведомо бесполезный признак, пропускаемый через тот же поиск, чтобы увидеть, как часто метод «находит» то, чего нет.
- **Walk-forward** — обучение «вперёд по времени»: модель учится только на прошлом и предсказывает следующий месяц, затем окно сдвигается.
- **Purge и embargo** — вырезание пересекающихся по времени данных между обучением и проверкой, чтобы не было утечки из будущего.
- **Коэффициент Шарпа** — средняя дневная прибыль, делённая на её разброс, в годовом выражении.
- **Рыночная нейтральность** — портфель с одинаковыми длинными и короткими позициями, почти не зависящий от направления рынка.
- **CCR (коэффициент покрытия издержек)** — валовая прибыль / суммарные издержки; меньше 1 — «издержки съели всё».
- **Контрольный период (holdout)** — последний кусок данных, который открывается один раз."""
    any_rec = [v for v in verdicts if v["recommended"]]
    answer = ("Ни одна цель не дала торгового результата после издержек по сценарию S1: на этой библиотеке признаков и этих горизонтах "
              "автоматический поиск не окупает комиссии taker. Выбор цели меняет число ложных находок и оборот, но не знак итога."
              if not any_rec else
              "Есть цели, прошедшие все условия: " + "; ".join(f"ветка {v['branch']}, h={v['h']}: {v['recommended']} ({NAMES[v['recommended']]}), {v['verdict'].lower()}" for v in any_rec) + ".")
    text = f"""# VERDICT — Stage F2: что предсказывать

**Мини-словарь**

{gloss}

**Вердикты (сценарий S1: taker 5,5 б.п. + 1 б.п. проскальзывания за сторону, с фандингом).** Разработка — {n_dev} дней
внешневыборочного walk-forward, контроль — {n_hold} дней ({cfg['period']['holdout_days']} дн.).

| ветка | h (баров по 4 ч) | вердикт | рекомендуемая цель | прошли P1–P4 |
|---|---|---|---|---|
{chr(10).join(lines)}

**Три главных числа.** Лучшая по Шарпу на разработке стратегия: ветка {best[0]}, цель {best[1]} ({NAMES[best[1]]}), h = {best[2]};
чистый Шарп {dev.loc[best, 'sharpe']:.2f} (90% ДИ {b_ci[0]:.2f} … {b_ci[1]:.2f}), на контроле {b_hold:.2f};
эмпирический FDR по плацебо {b_fdr if b_fdr == b_fdr else float('nan'):.1%}. CCR {dev.loc[best, 'ccr']:.2f}.

**Что предсказывать.** Цену — нет (ряд нестационарен, «R² ≈ 1» — иллюзия, `reports/price_level_demo.md`). {answer}
Волатильность как цель предсказывается заметно лучше доходности (R² HAR на разработке: {', '.join(f"h={r.h}: {r.r2_oos:.2f}" for r in F[(F.period=='dev') & (F.model=='HAR')].itertuples())});
как регулятор размера: **{vol_verdict}** (`reports/volatility_target.md`).

**Ограничения.** Смещение выживших (80 нынешних ликвидных контрактов, делистинги недоступны); нет симуляции очереди
лимитных заявок (S2 справочный); низкая мощность контроля (минимальное обнаружимое различие Шарпа на контроле ≈ {power.iloc[-1, 3]:.1f});
результат зависит от библиотеки признаков; не учтено влияние объёма на цену; фандинг по весу, а не по марк-цене.

**Неизвестное.** Формулы A101 сверены по вторичному источнику; поведение при другом размере портфеля и других биржах;
устойчивость к режимам рынка до 2022 года; см. `reports/open_questions_F2.md`.
"""
    (ROOT / "VERDICT.md").write_text(text, encoding="utf-8")
    print(text)
