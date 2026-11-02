"""Q2: correct (permutation) placebos and the role of BH. Dev only, same walk-forward windows as F2.

Every dev refit re-scores the 890 F2 features plus fresh placebos of types 1-4 (20% of real in total, equal split),
with the F2 scoring code. Only the statistics are kept (WORK_F2B/q2/<block>.pkl); selection regimes are compared after.
"""

from __future__ import annotations

import pickle
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

from .common import FIG, REPORTS, WORK, md, now_utc
from .common_F2b import HOLDOUT_NOTE, WORK_F2B, log_attempt_f2b
from .features import cs_zscore, placebos
from .stats import bh, block_corr, nw_tstat, p_two_sided, rowwise_spearman
from .walkforward import D_TARGETS, X_CLASS, blocks, context, score_inputs

Q2DIR = WORK_F2B / "q2"
PL_MM = WORK_F2B / "placebo_q2.f32"
FAMS = None  # filled from the F2 config: X and D families (T7 is not part of Q2)


def permute_across_assets(a: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Values of each row moved to a random order of the row's present assets (NaN pattern unchanged).

    keys[t, i]: random sort keys; equal keys for every row of a block give one permutation per block (type 4).
    """
    nan = np.isnan(a)
    k = np.where(nan, 2.0, keys)                                # present assets first, random order
    src = np.argsort(k, axis=1, kind="stable")
    dst = np.argsort(nan, axis=1, kind="stable")                # present assets first, natural order
    out = np.full_like(a, np.nan)
    rows = np.arange(a.shape[0])[:, None]
    out[rows, dst] = a[rows, src]
    return out


def make_placebos(cfg, fb, ctx, T1: int, seed: int) -> tuple[np.ndarray, list[str], list[int]]:
    n_real = len(ctx["kept"])
    n_tot = int(round(n_real * fb["q2"]["placebo_share_of_real"]))
    n_each = [n_tot // 4 + (1 if i < n_tot % 4 else 0) for i in range(4)]       # types 1, 2, 3, 4
    p12, names12 = placebos(n_each[0] + n_each[1], ctx["kept"], ctx["feats"], ctx["alive"], T1, cfg, seed)
    if sum(n.startswith("placebo1") for n in names12) != n_each[0]:              # F2 splits half/half
        n_each[0], n_each[1] = sum(n.startswith("placebo1") for n in names12), sum(n.startswith("placebo2") for n in names12)
    rng = np.random.default_rng(seed + 7)
    nb, ns = ctx["alive"].shape
    blk = cfg["bar_hours"]
    per_block = fb["q2"]["type4_block_days"] * 24 // blk
    extra, names = [], []
    for typ, n in ((3, n_each[2]), (4, n_each[3])):
        for _ in range(n):
            j = int(rng.choice(ctx["kept"]))
            src = np.asarray(ctx["feats"][j, :T1], dtype=float)
            if typ == 3:
                keys = rng.random(src.shape)
            else:
                keys = np.repeat(rng.random((int(np.ceil(T1 / per_block)), ns)), per_block, axis=0)[:T1]
            out = np.full((nb, ns), np.nan, dtype=np.float32)
            out[:T1] = cs_zscore(permute_across_assets(src, keys))
            extra.append(out)
            names.append(f"placebo{typ}:{'perm_bar' if typ == 3 else 'perm_block'}:{j}")
    pl = np.concatenate([p12, np.stack(extra)]) if extra else p12
    allnames = names12 + names
    types = [int(n[7]) for n in allnames]
    return pl, allnames, types


def _score_chunk_q2(src, shapes, a, T0, sc, lag_mult) -> list[dict]:
    """F2 _score_chunk with extra Newey-West lags for branch X."""
    mms = {k: np.memmap(path, dtype=np.float32, mode="r", shape=shape) for k, (path, shape) in shapes.items()}
    out = []
    for kind, j in src:
        F = np.asarray(mms[kind][j, a:T0], dtype=float)
        res = {}
        for h, d in sc.items():
            for cls, Y in d["X"].items():
                ic = rowwise_spearman(F, Y)
                for m in lag_mult:
                    res[("X", cls, h, m)] = float(nw_tstat(ic[None, :], m * h)[1][0])
            x = np.nan_to_num(F[d["mask"]], nan=0.0)
            c, sd = block_corr(rankdata(x), d["Yr"], d["blk"], d["nblk"], d["C"])
            for k, tgt in enumerate(D_TARGETS):
                res[("D", tgt, h)] = float(c[k] / sd[k]) if sd[k] > 0 else np.nan
        out.append(res)
    return out


def run(cfg: dict[str, Any], fb: dict[str, Any]) -> None:
    Q2DIR.mkdir(parents=True, exist_ok=True)
    ctx = context(cfg)
    bl = [b for b in blocks(cfg, ctx) if b["period"] == "dev"]
    for k, blk in enumerate(bl):
        path = Q2DIR / f"{blk['label']}.pkl"
        if path.exists():
            continue
        t0 = pd.Timestamp.now()
        seed = int(cfg["seed"]) + int(fb["q2"]["seed_offset"]) + k
        pl, names, types = make_placebos(cfg, fb, ctx, blk["T1"], seed)
        mm = np.memmap(PL_MM, dtype=np.float32, mode="w+", shape=pl.shape)
        mm[:] = pl
        mm.flush()
        del mm
        # type 1: cross-sectional similarity between the shifted placebo and its source on the training bars
        a, T0 = ctx["a"], blk["T0"]
        t1corr = {i: float(np.nanmean(rowwise_spearman(np.asarray(ctx["feats"][int(n.split(":")[-1]), a:T0], dtype=float),
                                                       pl[i, a:T0].astype(float))))
                  for i, n in enumerate(names) if types[i] == 1}
        sc = score_inputs(cfg, ctx, T0, seed)
        shapes = {"real": (str(WORK / "features_cand.f32"), tuple(ctx["meta"]["shape"])), "placebo": (str(PL_MM), pl.shape)}
        src = [("real", int(j)) for j in ctx["kept"]] + [("placebo", i) for i in range(len(pl))]
        chunks = [src[i:i + 40] for i in range(0, len(src), 40)]
        res = Parallel(n_jobs=14, backend="loky", max_nbytes="1M")(
            delayed(_score_chunk_q2)(c, shapes, a, T0, sc, fb["q2"]["nw_lag_multipliers"]) for c in chunks)
        scores = [r for part in res for r in part]
        keys = list(scores[0])
        t = {key: np.array([s[key] for s in scores]) for key in keys}
        path.write_bytes(pickle.dumps({"block": blk, "names": names, "types": [0] * len(ctx["kept"]) + types,
                                       "t": t, "t1corr": t1corr, "utc": now_utc()}))
        print(f"{now_utc()} q2 {blk['label']} done in {(pd.Timestamp.now() - t0).total_seconds():.0f}s", flush=True)
    log_attempt_f2b("Q2", "re-score F2 features + placebo types 1-4, regimes a-d, NW lags h/2h/4h", 4, "dev",
                    f"{len(bl)} refits; no model fitting")
    report(cfg, fb)


# --------------------------------------------------------------------------- analysis

def families(cfg) -> list[tuple[str, str, int]]:
    t = cfg["targets"]
    return [("X", x, h) for x in t["branch_X"] for h in cfg["horizons_bars"]] + \
           [("D", x, h) for x in t["branch_D"] for h in cfg["horizons_bars"]]


def regimes(t: np.ndarray, q: float, k: int, naive: float) -> dict[str, np.ndarray]:
    p = p_two_sided(t)
    passed = bh(p, q)
    at = np.nan_to_num(np.abs(t), nan=-1)
    order = np.argsort(-at)
    top_bh = np.zeros_like(passed)
    top_bh[[i for i in order if passed[i]][:k]] = True
    top = np.zeros_like(passed)
    top[order[:k]] = True
    return {"bh_top20": top_bh, "top20_only": top, "bh_only": passed, "naive_t2": at > naive}


def report(cfg: dict[str, Any], fb: dict[str, Any]) -> dict:
    runs = [pickle.loads(p.read_bytes()) for p in sorted(Q2DIR.glob("*.pkl"))]
    q, K, naive = cfg["walkforward"]["bh_q"], cfg["walkforward"]["top_k"], cfg["walkforward"]["naive_abs_t"]
    rows, calib, t1rows = [], [], []
    for br, tgt, h in families(cfg):
        key = ("X", X_CLASS[tgt], h, 1) if br == "X" else ("D", tgt, h)
        acc = {}
        for r in runs:
            typ = np.array(r["types"])
            t = r["t"][key]
            reg = regimes(t, q, K, naive)
            for name, sel in reg.items():
                for ty in (0, 1, 2, 3, 4):
                    acc[(name, "sel", ty)] = acc.get((name, "sel", ty), 0) + int((sel & (typ == ty)).sum())
                acc[(name, "n")] = acc.get((name, "n"), 0) + int(sel.sum())
            for ty in (0, 1, 2, 3, 4):
                acc[("n_type", ty)] = acc.get(("n_type", ty), 0) + int((typ == ty).sum())
                acc[("bh_pass", ty)] = acc.get(("bh_pass", ty), 0) + int((reg["bh_only"] & (typ == ty)).sum())
            if br == "X" and tgt in ("T1", "T5"):
                for i, c in r["t1corr"].items():
                    t1rows.append({"family": f"X:{tgt}:h{h}", "h": h, "corr": c, "passed": bool(reg["bh_only"][len(typ) - len(r["names"]) + i])})
        row = {"branch": br, "target": tgt, "h": h}
        for ty, lbl in ((0, "real"), (1, "t1"), (2, "t2"), (3, "t3"), (4, "t4")):
            row[f"bh_pass_{lbl}"] = acc[("bh_pass", ty)] / acc[("n_type", ty)]
        for name in ("bh_top20", "top20_only", "bh_only", "naive_t2"):
            n = acc[(name, "n")]
            row[f"fdr34_{name}"] = (acc[(name, "sel", 3)] + acc[(name, "sel", 4)]) / n if n else np.nan
            row[f"fdr234_{name}"] = (acc[(name, "sel", 2)] + acc[(name, "sel", 3)] + acc[(name, "sel", 4)]) / n if n else np.nan
        row["bh_only_selected_per_refit"] = acc[("bh_only", "n")] / len(runs)
        rows.append(row)
    R = pd.DataFrame(rows)
    # NW calibration on type 2 with phi = 0.98, branch X (T1 class; T5 class reported too)
    for h in cfg["horizons_bars"]:
        for cls in ("r", "z"):
            for m in fb["q2"]["nw_lag_multipliers"]:
                hits = n = 0
                for r in runs:
                    idx = [len(r["types"]) - len(r["names"]) + i for i, nm in enumerate(r["names"]) if nm == "placebo2:ar1:0.98"]
                    t = r["t"][("X", cls, h, m)][idx]
                    hits += int(np.sum(np.abs(t) > 2))
                    n += int(np.sum(~np.isnan(t)))
                calib.append({"h": h, "target_class": "T1/T3/T4" if cls == "r" else "T5/T6", "nw_lag": f"{m}h = {m * h}",
                              "share_abs_t_gt_2": hits / n, "n": n, "se": np.sqrt(0.046 * 0.954 / n)})
    Cb = pd.DataFrame(calib)
    T1 = pd.DataFrame(t1rows)
    T1["corr_bin"] = pd.qcut(T1["corr"], 5, duplicates="drop")
    t1tab = T1.groupby(["h", "corr_bin"], observed=True).agg(n=("passed", "size"), pass_rate=("passed", "mean"), corr_mid=("corr", "median")).reset_index()
    figures(R, T1, t1tab)
    write(cfg, fb, R, Cb, t1tab, T1, len(runs))
    return {"R": R, "Cb": Cb}


def figures(R, T1, t1tab) -> None:
    from .report import C1, C2, C3, _style
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, h in zip(axes, sorted(R.h.unique())):
        d = R[R.h == h]
        x = np.arange(len(d))
        for k, (col, c, lbl) in enumerate((("bh_pass_t1", "#999999", "тип 1 (сдвиг)"), ("bh_pass_t2", C3, "тип 2 (AR(1))"),
                                            ("bh_pass_t3", C1, "тип 3 (перестановка по бару)"), ("bh_pass_t4", C2, "тип 4 (перестановка блоками)"))):
            ax.bar(x + (k - 1.5) * 0.2, d[col], 0.19, color=c, label=lbl)
        ax.axhline(0.05, color="#555", ls="--", lw=0.8)
        ax.set_xticks(x, d.branch + ":" + d.target, rotation=60, fontsize=7)
        ax.set_title(f"h = {h}", fontsize=9)
        _style(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Доля плацебо, прошедших BH, по типам (разработка); пунктир — 5%", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "f2b_fig2_bh_pass_by_type.png", dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for h, c in zip(sorted(t1tab.h.unique()), (C1, C2, C3)):
        d = t1tab[t1tab.h == h]
        ax.plot(d.corr_mid, d.pass_rate, marker="o", color=c, lw=2, label=f"h = {h}")
    ax.set_xlabel("средняя корреляция по срезу активов: исходный и сдвинутый признак", fontsize=8)
    ax.set_ylabel("доля прохождения BH", fontsize=8)
    ax.set_title("Плацебо типа 1: чем больше сдвиг похож на исходный признак, тем чаще «проходит»\n(ветка X, цели T1 и T5, квинтили корреляции)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "f2b_fig3_type1_corr.png", dpi=130)
    plt.close(fig)


def write(cfg, fb, R, Cb, t1tab, T1, n_runs) -> None:
    rule = fb["q2"]["rule"]
    a_ok = bool((R["fdr34_bh_top20"].fillna(0) <= rule["max_fdr_selected"]).all())
    c_ok = bool((R["fdr34_bh_only"].fillna(0) <= rule["max_fdr_selected"]).all())
    t34 = float(max(R.bh_pass_t3.max(), R.bh_pass_t4.max()))
    t1m = float(R.bh_pass_t1.mean())
    lines = [f"- «Отбор контролирует ложные находки» (режим (а), доля плацебо типов 3+4 среди отобранных ≤ 10% во всех парах): **{'да' if a_ok else 'нет'}** "
             f"(максимум {R['fdr34_bh_top20'].max():.1%}).",
             f"- «BH сама по себе контролирует» (режим (в) ≤ 10% во всех парах): **{'да' if c_ok else 'нет'}** (максимум {R['fdr34_bh_only'].max():.1%}; "
             f"BH оставляет в среднем {R.bh_only_selected_per_refit.mean():.0f} признаков из ~1070 на переобучение)."]
    if t34 <= rule["max_bh_pass_types34"] and t1m > 2 * t34:
        lines.append(f"- **Сдвиг по времени не годится как плацебо для сравнения активов**: типы 3 и 4 проходят BH не чаще {t34:.1%} "
                     f"(максимум по парам), а тип 1 — в среднем {t1m:.1%}.")
    else:
        lines.append(f"- Типы 3 и 4 проходят BH до {t34:.1%} (максимум по парам), тип 1 — в среднем {t1m:.1%}.")
    rec = []
    for (h, cls), g in Cb.groupby(["h", "target_class"]):
        ok = g[g.share_abs_t_gt_2 <= 0.046 + 1.96 * g.se]
        rec.append(f"h={h} ({cls}): доля |t|>2 при лаге h = {g.share_abs_t_gt_2.iloc[0]:.1%}; "
                   + (f"к номиналу возвращает лаг {ok.nw_lag.iloc[0]}" if len(ok) else "ни один лаг до 4h не возвращает к номиналу"))
    inflated = bool((Cb[Cb.nw_lag.str.startswith("1h")].share_abs_t_gt_2 > 0.046 + 1.96 * Cb[Cb.nw_lag.str.startswith("1h")].se).any())
    lines.append(("- t-статистика при лаге Ньюи–Уэста = h **завышена** для медленного шума (φ = 0,98): " if inflated else
                  "- t-статистика при лаге = h на шуме φ = 0,98 не завышена сверх случайного разброса: ") + "; ".join(rec) + ".")
    cols = ["branch", "target", "h", "bh_pass_real", "bh_pass_t1", "bh_pass_t2", "bh_pass_t3", "bh_pass_t4"]
    cols2 = ["branch", "target", "h", "fdr34_bh_top20", "fdr34_top20_only", "fdr34_bh_only", "fdr34_naive_t2", "fdr234_bh_only", "bh_only_selected_per_refit"]
    (REPORTS / "q2_placebo.md").write_text(f"""# Q2. Правильное плацебо и роль поправки BH

Разработка, те же {n_runs} помесячных переобучений, что в F2 (контроль в Q2 не используется). На каждом переобучении 890
реальных признаков F2 оценивались заново тем же кодом F2 вместе со свежими плацебо — 20% от числа реальных, поровну 4 типа:

- **тип 1** — сдвиг по времени ≥ 60 суток (как в F2; только для демонстрации, в итоговый FDR не входит);
- **тип 2** — шум AR(1), φ ∈ {{0,5; 0,9; 0,98}} (как в F2);
- **тип 3** — **перестановочное плацебо**: реальный признак, значения которого на каждом баре случайно переставлены между активами
  (распределение по срезу сохраняется, связь с конкретным активом ломается);
- **тип 4** — то же, но одна перестановка на блок 30 суток (сохраняет память признака во времени).

Для ветки D перестановка делается так же по барам; но там признак оценивается одной корреляцией по пулу (время × актив) с
блочным бутстрепом, поэтому перестановка между активами может вести себя иначе, чем в ветке X (по срезу активов).

## Вывод

{chr(10).join(lines)}

## Доля прохождения BH по типам

`real` — реальные признаки. Все плацебо и реальные признаки проходят BH вместе, как в F2.

{md(R[cols].round(3), index=False)}

![bh by type](figures/f2b_fig2_bh_pass_by_type.png)

## Доля плацебо среди отобранных при четырёх режимах

`fdr34_*` — доля плацебо **типов 3 и 4** среди отобранных; режимы: (а) `bh_top20` — BH + не более 20 по |t| (как в F2);
(б) `top20_only` — 20 лучших по |t| без BH; (в) `bh_only` — все прошедшие BH; (г) `naive_t2` — |t| > 2.
`fdr234_bh_only` — то же для типов 2+3+4 в режиме (в). `bh_only_selected_per_refit` — сколько признаков (реальных и плацебо) в среднем
проходит BH на одно переобучение (из ~1070). Плацебо типов 3+4 составляют ~{0.5 * fb['q2']['placebo_share_of_real'] / (1 + fb['q2']['placebo_share_of_real']):.1%} всех проверяемых признаков,
поэтому доля среди отобранных сравнима с этим числом «при полной слепоте» отбора.

{md(R[cols2].round(3), index=False)}

## Калибровка t-статистики (ветка X, плацебо типа 2 с φ = 0,98)

Номинальная доля |t| > 2 для честного теста — 4,6%. `se` — стандартная ошибка доли при номинале.

{md(Cb.round(4), index=False)}

## Тип 1: почему сдвиг по времени «проходит»

Для каждого сдвинутого признака — средняя по барам корреляция Спирмена по срезу активов между исходным и сдвинутым значением
на обучающем окне; ниже — доля прохождения BH по квинтилям этой корреляции (ветка X, цели T1 и T5, все переобучения).
Средняя корреляция: {T1['corr'].mean():.2f}; доля сдвигов с корреляцией > 0,3: {(T1['corr'] > 0.3).mean():.1%}.

{md(t1tab.assign(corr_bin=t1tab.corr_bin.astype(str)).round(3), index=False)}

![type1](figures/f2b_fig3_type1_corr.png)

Оговорка: Q2 считается только на разработке; {HOLDOUT_NOTE} — здесь не применяется.
""", encoding="utf-8")
    R.to_csv(REPORTS / "q2_placebo_F2b.csv", index=False)
    Cb.to_csv(REPORTS / "q2_nw_calibration_F2b.csv", index=False)
    from .strategies_F2b import save_summary
    save_summary("q2", {"selection_controls": a_ok, "bh_alone_controls": c_ok, "max_bh_pass_t34": t34, "mean_bh_pass_t1": t1m,
                        "max_fdr34_bh_top20": float(R["fdr34_bh_top20"].max()), "max_fdr34_bh_only": float(R["fdr34_bh_only"].max()),
                        "nw_inflated_at_h": inflated, "nw_recommendation": rec, "mean_bh_pass_real": float(R.bh_pass_real.mean())})
    print("\n".join(lines))
