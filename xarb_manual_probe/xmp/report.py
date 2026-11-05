"""Weekly description and the final verdict, in plain Russian for the owner."""
import datetime as dt
import json

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import store
from .common import DERIVED, HOLDOUT, ROOT, SELECTED

REPORTS = ROOT / "reports"
GLOSSARY = """## Мини-словарь

- **Спред входа (s_in)** — насколько продать на бирже шорта дороже, чем купить на бирже лонга, по средней цене заявки нужного размера.
- **Стоимость выхода (s_out)** — то же для закрытия: сколько теряется на откупе шорта и продаже лонга.
- **VWAP** — средняя цена заявки, прошедшей по стакану на нужную сумму.
- **Фандинг** — регулярная плата между лонгами и шортами перпа; положительная ставка: лонги платят шортам.
- **Задержка (D)** — время от сигнала до действия человека; ноги ставятся по очереди, вторая ещё через 20 с.
- **Проклятие победителя** — самые большие спреды часто самые плохие: за ними стоит фандинг, закрытый вывод или шум.
- **Капитал на биржах** — все деньги, разложенные по биржам, включая те, что лежат без дела; доходность считается на всю сумму.
- **Блочный бутстреп** — оценка точности, где выборка перетасовывается целыми днями, потому что сделки одного дня связаны.
"""


def md(df):
    if df.empty:
        return "нет данных\n"
    df = df.copy()
    for c in df.select_dtypes("float"):
        df[c] = df[c].map(lambda x: f"{x:.2f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).to_numpy()) + "\n"


def conv_table(c):
    rows = []
    for cat, g in [*c.groupby("cat"), ("все", c)]:
        rows.append({"категория": cat, "эпизодов": len(g), "длительность, медиана ч": g["dur_h"].median(), "сошлось за 1 ч, %": (g["conv_h"] <= 1).mean() * 100,
                     "за 24 ч, %": (g["conv_h"] <= 24).mean() * 100, "за 72 ч, %": (g["conv_h"] <= 72).mean() * 100, "не сошлось, %": np.isinf(g["conv_h"]).mean() * 100})
    return pd.DataFrame(rows)


def weekly(cfg):
    """Episodes, convergence, gaps. Paper trades of the control period never appear here."""
    days = store.days_recorded()
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    full = days[1:]
    c = ev.convergence(cfg, full) if full else pd.DataFrame()
    g = store.gaps_table()
    text = [f"# X2 · неделя на {today}\n", f"Записано дней: {len(days)} (полных {len(full)}). Доля времени с пропусками: {store.gap_share(full):.1%}.\n"]
    text += ["## Кандидаты и схождение\n", md(conv_table(c)) if not c.empty else "кандидатов пока нет\n"]
    text += ["## Пропуски по биржам\n", md(g.assign(минут=(g["end"] - g["start"]) / 60).groupby("venue", as_index=False)["минут"].sum().sort_values("минут", ascending=False)) if len(g) else "нет\n"]
    path = REPORTS / f"X2_weekly_{today}.md"
    path.write_text("\n".join(text), encoding="utf-8")
    return path


def verdict(cfg):
    if not (DERIVED / "holdout_trades.parquet").exists():
        raise SystemExit("control not computed yet")
    p = cfg["paper"]
    t = pd.read_parquet(DERIVED / "holdout_trades.parquet")
    h = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    sel = json.loads(SELECTED.read_text(encoding="utf-8"))
    days = h["days"]
    n = len(days)
    gap = store.gap_share(sel["dev_days"] + days)
    alive = len({v for d in days for v in (x.name.split(".")[0] for x in (store.RAW / "snaps" / d).glob("*"))})
    base = t[(t["q"] == p["main_q_usd"]) & (t["capital"] == p["main_capital_usd"])]
    cell = lambda name, delay=p["main_delay_s"]: base[(base["venues"] == name) & (base["delay"] == delay)]
    names = ["all"] + (["owner"] if p["owner_venues"] else [])
    res = {name: ev.criteria_table(cell(name), None, cfg, n, gap, alive) for name in names}
    s, marks, overall = res[names[-1]]
    c = ev.convergence(cfg, days)
    conv24 = (c["conv_h"] <= 24).mean() if len(c) else float("nan")
    main = cell(names[-1])
    done = main[main["status"] == "done"]

    out = ["# Вердикт X2A: приносят ли спреды скринера деньги при ручном входе\n"]
    out.append(f"**Итог: {overall}.** Проверено на бумаге: контроль {days[0]} … {days[-1]} ({n} дн.), вариант выбран на разработке до контроля: θ = {sel['theta_bp']:g} б.п., выход {sel['exit']}, T_max = {sel['t_max_h']:g} ч.\n")
    out.append(f"## Три главных числа\n\n- Доходность на капитал ${p['main_capital_usd']}, D = {p['main_delay_s']} с: **{s.get('apr', 0) * 100:.1f}% годовых** (порог {cfg['criteria']['a2_min_apr'] * 100:.0f}%).")
    out.append(f"- Спредов, сошедшихся за 24 ч: **{conv24 * 100:.0f}%**.")
    out.append(f"- Средний чистый итог сделки: **{s.get('mean_bp', float('nan')):.1f} б.п.**, 90% интервал {s.get('ci_lo', float('nan')):.1f} … {s.get('ci_hi', float('nan')):.1f}; сделок {s.get('trades', 0)}.\n")
    out.append(GLOSSARY)

    rows = []
    for name in names:
        sm, mk, v = res[name]
        rows.append({"набор бирж": "биржи владельца" if name == "owner" else "все", "категория": "все", **{k: "да" if x else "нет" for k, x in mk.items()}, "сделок": sm.get("trades", 0), "итог": v})
        for cat, (cs, cm, cv) in ev.category_verdicts(cell(name), cfg, n, gap, alive).items():
            rows.append({"набор бирж": "биржи владельца" if name == "owner" else "все", "категория": cat, **{k: "да" if x else "нет" for k, x in cm.items()}, "сделок": cs.get("trades", 0), "итог": cv})
    out += ["## Критерии A1–A4\n", md(pd.DataFrame(rows))]
    if not p["owner_venues"]:
        out.append("Владелец не назвал свои биржи, поэтому «биржи владельца» совпадают со «всеми».\n")

    attr = []
    for cat, g in [*done.groupby("cat"), ("все", done)]:
        attr.append({"категория": cat, "цена": g["price_bp"].mean(), "комиссии": -g["fees_bp"].mean(), "фандинг": g["funding_bp"].mean(), "переводы": -g["transfer_bp"].mean(), "чистый итог": g["net_bp"].mean(), "в плюсе, %": (g["net_bp"] > 0).mean() * 100, "сделок": len(g)})
    out += ["## Откуда деньги, б.п. номинала на сделку\n", md(pd.DataFrame(attr))]

    dl = [{"D, с": d, **{k: ev.summary(cell(names[-1], d).to_dict("records"), p["main_capital_usd"], n, cfg["criteria"]).get(k) for k in ("trades", "not_done_share", "mean_bp", "apr")}} for d in [p["robot_delay_s"], *p["delays_s"]]]
    out += ["## Человек против робота\n", md(pd.DataFrame(dl).rename(columns={"trades": "сделок", "not_done_share": "доля «не состоялась»", "mean_bp": "средний итог, б.п.", "apr": "годовых, доля"}))]
    ladder = []
    for cap in p["capital_usd"]:
        for q in p["q_usd"]:
            g = t[(t["capital"] == cap) & (t["q"] == q) & (t["delay"] == p["main_delay_s"]) & (t["venues"] == names[-1])]
            sm = ev.summary(g.to_dict("records"), cap, n, cfg["criteria"])
            ladder.append({"капитал, $": cap, "Q, $": q, "сделок": sm.get("trades", 0), "нет капитала": sm.get("no_capital", 0), "годовых, доля": sm.get("apr", 0.0)})
    out += ["## Лесенка капитала\n", md(pd.DataFrame(ladder))]
    out += ["## Проклятие победителя: итог по децилям спреда на входе\n", md(ev.winner_curse(main))]
    if "rank" in done:
        top = done["rank"] <= p["top_n_screener"]
        out += [f"## Верх списка скринера\n\nСкринер не выгружает данные, поэтому ранг восстановлен: порядок по спреду лучших цен среди кандидатов. Топ-{p['top_n_screener']}: средний итог {done.loc[top, 'net_bp'].mean():.1f} б.п., {int(top.sum())} сделок; остальные: {done.loc[~top, 'net_bp'].mean():.1f} б.п., {int((~top).sum())} сделок.\n"]
    wo = ev.without_top(done)
    out.append(f"Сумма итогов без 3 лучших сделок: {wo:.0f} б.п.; без 3 лучших пар: {s.get('sum_wo_top3_pairs', float('nan')):.0f} б.п.; просадка {s.get('max_dd_of_capital', 0) * 100:.1f}% капитала; худшая сделка {s.get('worst_of_capital', 0) * 100:.2f}% капитала; ликвидаций при L = 2: {s.get('liquidations', 0)}, порог L = 3 задет у {s.get('liq_ref', 0)} сделок.\n")

    robot = ev.summary(cell(names[-1], p["robot_delay_s"]).to_dict("records"), p["main_capital_usd"], n, cfg["criteria"])
    if overall == "бумажно выгодно":
        rec = "Переходить к X2B: реальные сделки малым размером по протоколу, записанному до первой сделки (отдельное задание)."
    elif overall == "мало данных":
        rec = "Одно продление записи на 14 дней с тем же конфигом, дальше без продлений."
    elif robot.get("apr", 0) > 0:
        rec = "Деньги есть только у быстрых: при D = 0 итог положительный. Ручной арбитраж закрыт."
    else:
        rec = "Закрыто. Источник большого спреда виден в таблице «Откуда деньги»: фандинг, закрытый вывод или синтетика без схождения."
    out += ["## Что делать дальше\n", rec + "\n"]
    out += ["## Ограничения\n", "\n".join(f"- {x}" for x in [
        "Короткий период и один режим рынка.", "Стакан записывался только у кандидатов; выход после конца эпизода считан по лучшим ценам.",
        "Сигналы Hyperliquid идут по ценам удара (нет массового BBO), спот Hyperliquid не записан.", "Лучшие цены пишутся при их изменении, объём на лучшей цене может отставать до часа.",
        "Статус ввода/вывода известен только у WhiteBIT и XT.", "Правила расчёта K-SYN при листинге компании не моделировались.",
        "Комиссии «не подтверждено» там, где так указано в `reports/X2_doc_check.md`; перевод маржи — оценка $1.",
        "У Bitfinex, Paradex и Bitunix нет публичной истории фандинга в удобном виде: ставка взята из снимка перед расчётом.",
        "Ранг скринера восстановлен, а не выгружен.", "Часы владельца действуют и на выход: сработавшее ночью условие ждёт 10:00 МСК (ликвидация не ждёт)."]) + "\n"]
    path = ROOT / "VERDICT_X2A.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path
