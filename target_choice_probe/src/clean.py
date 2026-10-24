"""Raw API answers -> clean hourly parquet -> 4-hour wide panels, data quality report and criterion C1.

Parsing covers the whole range (storage, not analysis). The report only counts rows (coverage), it computes
no statistic on values, so it may use all data before the holdout is opened.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import CLEAN, FIG, RAW, REPORTS, md, periods
from .download import read_raw, universe

H = pd.Timedelta(hours=1)


def _ms(x) -> pd.Series:
    return pd.to_datetime(pd.to_numeric(x), unit="ms", utc=True)


def _rows(path) -> list:
    return [x for rec in read_raw(path) for x in json.loads(rec["body"])["result"]["list"]] if path.exists() else []


def parse(uni: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, list] = {"last": [], "premium": [], "oi": [], "funding": []}
    dups = []
    for sym in uni["symbol"]:
        d = {"symbol": sym}
        for series, cols in (("last", ["ts", "open", "high", "low", "close", "volume", "turnover"]),
                             ("premium", ["ts", "open", "high", "low", "close"])):
            rows = _rows(RAW / "kline60" / series / f"{sym}.jsonl.gz")
            df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
            df = df.astype({c: float for c in cols[1:]}).assign(ts=lambda x: _ms(x["ts"]), symbol=sym)
            d[f"{series}_dups"] = int(df.duplicated("ts").sum())
            out[series].append(df.drop_duplicates("ts"))
        rows = _rows(RAW / "oi1h" / f"{sym}.jsonl.gz")
        df = pd.DataFrame({"ts": _ms([r["timestamp"] for r in rows]), "oi": [float(r["openInterest"]) for r in rows], "symbol": sym})
        d["oi_dups"] = int(df.duplicated("ts").sum())
        out["oi"].append(df.drop_duplicates("ts"))
        rows = _rows(RAW / "funding" / f"{sym}.jsonl.gz")
        df = pd.DataFrame({"ts": _ms([r["fundingRateTimestamp"] for r in rows]), "rate": [float(r["fundingRate"]) for r in rows]})
        d["funding_dups"] = int(df.duplicated("ts").sum())
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        gap = df["ts"].diff().dt.total_seconds().div(3600)
        df["interval_h"] = gap.fillna(gap.iloc[1] if len(df) > 1 else 8.0)   # first settlement: next gap
        df["rate_per_h"] = df["rate"] / df["interval_h"]
        out["funding"].append(df.assign(symbol=sym))
        dups.append(d)
    res = {k: pd.concat(v, ignore_index=True).sort_values(["symbol", "ts"]).reset_index(drop=True) for k, v in out.items()}
    for k, v in res.items():
        v.to_parquet(CLEAN / f"{k}_1h.parquet" if k != "funding" else CLEAN / "funding.parquet", index=False)
    res["dups"] = pd.DataFrame(dups)
    return res


def panels(cfg: dict[str, Any], raw: dict[str, pd.DataFrame], syms: list[str]) -> dict[str, pd.DataFrame]:
    """4-hour bars [t, t+4h) by UTC; a bar exists only if all 4 hourly candles exist."""
    p = periods(cfg)
    hours = pd.date_range(p["data_start"], p["end"] - H, freq="h")
    bars = pd.date_range(p["data_start"], p["end"] - pd.Timedelta(hours=4), freq="4h")
    k = raw["last"]
    wide = {c: k.pivot(index="ts", columns="symbol", values=c).reindex(index=hours, columns=syms) for c in
            ("open", "high", "low", "close", "volume", "turnover")}
    grp = lambda f: f.groupby(f.index.floor("4h"))
    full = grp(wide["close"]).count().reindex(bars) == 4
    pan = {"open": grp(wide["open"]).first(), "high": grp(wide["high"]).max(), "low": grp(wide["low"]).min(),
           "close": grp(wide["close"]).last(), "volume": grp(wide["volume"]).sum(), "turnover": grp(wide["turnover"]).sum()}
    prem = raw["premium"].pivot(index="ts", columns="symbol", values="close").reindex(index=hours, columns=syms)
    pan["premium"] = grp(prem).mean()                          # mean of hourly closes inside the bar
    pan = {c: v.reindex(bars).where(full) for c, v in pan.items()}
    # OI and funding: last value strictly inside the bar (timestamp < bar end), known at the decision time
    oi = raw["oi"].pivot(index="ts", columns="symbol", values="oi").reindex(columns=syms)
    pan["oi"] = _last_before(oi, bars + pd.Timedelta(hours=4)).where(full)
    f = raw["funding"].pivot(index="ts", columns="symbol", values="rate_per_h").reindex(columns=syms)
    pan["funding"] = _last_before(f, bars + pd.Timedelta(hours=4)).where(full)
    # funding paid by a position held over bar b (settlements in (start, end] of the holding bar)
    fr = raw["funding"].pivot(index="ts", columns="symbol", values="rate").reindex(columns=syms)
    pan["funding_settled"] = fr.groupby(fr.index.ceil("4h") - pd.Timedelta(hours=4)).sum().reindex(bars).fillna(0.0)
    # hourly log returns inside each candle, for realised volatility
    hr = np.log(wide["close"] / wide["open"])
    hr.to_parquet(CLEAN / "hourly_logret.parquet")
    for c, v in pan.items():
        v.astype("float64").to_parquet(CLEAN / f"panel_{c}.parquet")
    return pan


def _last_before(wide: pd.DataFrame, ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Value of the last row with index < end, for every end (strict, conservative)."""
    wide = wide.sort_index().ffill()
    pos = wide.index.searchsorted(ends, side="left") - 1
    out = wide.to_numpy()[np.clip(pos, 0, None)]
    out[pos < 0] = np.nan
    return pd.DataFrame(out, index=ends - pd.Timedelta(hours=4), columns=wide.columns)


def load_panels(names=None) -> dict[str, pd.DataFrame]:
    names = names or ["open", "high", "low", "close", "volume", "turnover", "premium", "oi", "funding", "funding_settled"]
    return {n: pd.read_parquet(CLEAN / f"panel_{n}.parquet") for n in names}


def clean_all(cfg: dict[str, Any]) -> dict[str, Any]:
    CLEAN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    uni = universe(cfg)
    syms = list(uni["symbol"])
    raw = parse(uni)
    pan = panels(cfg, raw, syms)
    p = periods(cfg)
    a = pan["close"].loc[p["start"]:]
    alive = a.notna().sum(axis=1)
    months = (p["end"] - p["start"]).days / 30.44
    first_full = alive.index[alive > 0][0] if (alive > 0).any() else None
    c1 = cfg["data_requirement_C1"]
    ok = months >= c1["min_months"] and alive.mean() >= c1["min_mean_assets_per_bar"]
    # per-asset coverage
    rows = []
    for s in syms:
        life = a[s].dropna()
        lo, hi = (life.index[0], life.index[-1]) if len(life) else (None, None)
        inside = a[s].loc[lo:hi] if lo is not None else a[s].iloc[:0]
        cov = lambda n: float(pan[n][s].loc[lo:hi].notna().mean()) if lo is not None else 0.0
        d = raw["dups"].set_index("symbol").loc[s]
        rows.append({"symbol": s, "first_bar": str(lo)[:10], "last_bar": str(hi)[:10], "bars": len(life),
                     "missing_inside": int(inside.isna().sum()), "premium_cov": cov("premium"), "oi_cov": cov("oi"),
                     "funding_cov": cov("funding"), "dups": int(d.sum())})
    cov = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(alive.index, alive.to_numpy(), lw=0.8, color="#3b6ea5")
    ax.axvline(p["dev_end"], color="#999", ls="--", lw=0.8)
    ax.axvline(p["holdout_start"], color="#c44", ls="--", lw=0.8)
    ax.set_ylabel("активов на баре")
    ax.set_title("Сколько активов «живёт» на каждом 4-часовом баре (пунктир: конец разработки / начало контроля)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "assets_alive.png", dpi=130)
    plt.close(fig)
    by_year = alive.groupby(alive.index.year).agg(["mean", "min", "max"]).round(1)
    text = f"""# Качество данных (Stage F2)

Источник: публичный REST Bybit, часовые свечи последней цены и премиум-индекса, часовой открытый интерес (OI),
история фандинга. 4-часовые бары собраны из часовых по границам UTC; бар есть, только если есть все 4 часа.

- Период анализа: {p['start']:%Y-%m-%d} … {p['end']:%Y-%m-%d} (данные качались с {p['data_start']:%Y-%m-%d} для разогрева окон).
- Длина: {months:.1f} мес. Контрольный период: последние **{p['holdout_days']} дней** ({p['holdout_start']:%Y-%m-%d} … {p['end']:%Y-%m-%d}), embargo 3 дня, разработка до {p['dev_end']:%Y-%m-%d}.
- Символов в списке Stage 4: {len(syms)}; с данными: {int((cov.bars > 0).sum())}.
- Активов на бар: среднее **{alive.mean():.1f}**, минимум {alive.min()}, максимум {alive.max()}. Первый бар с данными: {first_full}.

## Критерий C1

Нужно ≥ {c1['min_months']} мес. истории и ≥ {c1['min_mean_assets_per_bar']} активов в среднем на бар: **{'выполнен' if ok else 'НЕ выполнен'}**.

## Активов на бар по годам

{md(by_year)}

![assets alive](figures/assets_alive.png)

## Покрытие по активам

`missing_inside` — 4-часовые бары без полного набора часовых свечей внутри «жизни» актива; `*_cov` — доля баров жизни,
где есть значение (OI и фандинг берутся последним значением строго внутри бара); `dups` — повторы строк в сырых ответах (удалены).

{md(cov, index=False, digits=3)}

## Смещение выживших

Список из 80 контрактов взят из Stage 4 (снимок 2026-10-13: топ по обороту среди торгующихся). Контракты, которые
за 2022–2026 годы были сняты с торгов или упали в обороте, в выборку не попали, а исторические данные по
делистингованным контрактам через этот API недоступны. Значит, в начале периода выборка смещена к «выжившим и
выросшим» активам. Для сравнения целей между собой это смещение общее, но абсолютные цифры доходности могут быть
завышены (или искажены в любую сторону для стратегий, ставящих против слабых активов).
"""
    (REPORTS / "data_quality_F2.md").write_text(text, encoding="utf-8")
    print(text[:1500])
    return {"c1_ok": ok}
