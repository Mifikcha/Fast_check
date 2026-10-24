"""VERDICT_F2b.md: one page, plain language, numbers pulled from the F2b reports."""

from __future__ import annotations

import json

import pandas as pd

from .common import REPORTS, ROOT
from .common_F2b import WORK_F2B
from .strategies_F2b import load_summary

GLOSS = """- **Отскок между bid и ask** — цена последней сделки случайно попадает то на лучшую цену покупки, то на лучшую цену продажи; цена «прыгает» без реального движения и создаёт ложный эффект «цена вернулась».
- **Марк-цена** — справедливая цена контракта, которую считает биржа (по индексу и базису); на ней нет такого «прыгания».
- **Выигрыш на единицу оборота (EPT)** — валовая прибыль / объём сделок, в б.п.; сравнивается с ценой одной стороны сделки (6,5 б.п. в S1).
- **Бета-нейтральность** — чувствительность портфеля к движению рынка в целом равна нулю.
- **Зона без сделок** — позиция меняется, только если новая цель отличается от текущей больше порога; мелкие изменения игнорируются.
- **Перестановочное плацебо** — признак, у которого значения перемешаны между активами внутри каждого бара: распределение сохраняется, связь с активом ломается.
- Остальные термины (IC, BH, FDR, плацебо, walk-forward, Шарп, CCR, контроль) — как в `VERDICT.md` F2."""


def run_verdict(cfg: dict, fb: dict) -> None:
    s = load_summary()
    q1 = json.loads((WORK_F2B / "q1_result.json").read_text(encoding="utf-8"))
    T1 = pd.read_csv(REPORTS / "q1_bounce_F2b.csv")
    q2, q3, q4, q5, corr = s["q2"], s["q3"], s["q4"], s["q5"], s["corrections"]
    dm = T1[(T1.period == "dev") & (T1.variant == "M")].sort_values("gross_sharpe_S0", ascending=False).iloc[0]
    hm = T1[(T1.period == "holdout") & (T1.variant == "M") & (T1.branch == dm.branch) & (T1.target == dm.target)].iloc[0]
    dp = T1[(T1.period == "dev") & (T1.variant == "P") & (T1.branch == dm.branch) & (T1.target == dm.target)].iloc[0]
    M = pd.read_csv(REPORTS / "metrics_F2.csv")
    ic = M[(M.period == "dev") & (M.scenario == "S1") & M.funding & (M.branch == "X")].pivot_table(index="target", columns="h", values="oos_ic")
    Q3 = pd.read_csv(REPORTS / "q3_beta_F2b.csv")
    bn = Q3[(Q3.kind == "beta_neutral") & (Q3.branch == "X")].pivot_table(index="target", columns=["period", "h"], values="gross_alpha_S0_ann")
    q4v = q4["verdict"]
    q5v = q5["verdict"]
    branch = ("ОДНО ПРОДОЛЖЕНИЕ" if q4v == "продолжение оправдано" else
              "ЗАКРЫТЬ КАК СТРАТЕГИЮ, ОСТАВИТЬ КАК ТАЙМИНГ" if q5v == "полезен как тайминг" else "ЗАКРЫТЬ")
    q4line = (f"**{q4v}** — максимальный EPT на сетке {q4['max_ept_bp_grid']:.2f} б.п., в {q4['cost_over_max_ept']:.1f} раза ниже издержек 6,5 б.п.; "
              f"безубыточность при издержках {', '.join(f'{k} {v:.2f}' for k, v in q4['breakeven_bp'].items())} б.п. на сторону"
              if q4.get("status") == "run" else "**пропущен** (Q1 — артефакт) → закрыть")
    text = f"""# VERDICT F2b — закрывающие проверки ветки F2

**Словарь (новые термины)**

{GLOSS}

**Ответы** (решающие числа — внешневыборочная разработка; контроль уже открывался в F2, поэтому на нём подтверждаем только знак и порядок величины)

1. **Q1, отскок:** {q1['overall']}. Лучшая по валовому Шарпу стратегия h = 4 ч на марк-цене — {dm.branch}:{dm.target}: Шарп S0 {dm.gross_sharpe_S0:.2f}
   (90% ДИ {dm.ci_lo:.2f} … {dm.ci_hi:.2f}; на контроле {hm.gross_sharpe_S0:.2f}), EPT {dm.ept_bp:.2f} б.п. против {dp.ept_bp:.2f} б.п. по ценам сделок.
2. **Q2, плацебо:** отбор BH + топ-20 {'контролирует' if q2['selection_controls'] else 'не контролирует'} ложные находки (максимум доли перестановочных плацебо
   среди отобранных {q2['max_fdr34_bh_top20']:.1%}); BH без ограничения {'контролирует' if q2['bh_alone_controls'] else 'не контролирует'} ({q2['max_fdr34_bh_only']:.1%}).
   Перестановочные плацебо проходят BH не чаще {q2['max_bh_pass_t34']:.1%}, сдвиг по времени — в среднем {q2['mean_bh_pass_t1']:.1%}.
   Лаг Ньюи–Уэста = h {'занижает ошибку' if q2['nw_inflated_at_h'] else 'не занижает ошибку сверх разброса'}: {'; '.join(r for r in q2['nw_recommendation'] if not r.startswith('h=1 '))}.
3. **Q3, бета:** бета объясняет {q3['holdout_loss_share_explained_by_beta_h6_18']:.0%} потерь стратегий h = 6/18 на контроле; {q3['conclusion'].replace('(см. таблицу)', '(`reports/q3_beta.md`)')}.
4. **Q4, оборот:** {q4line}.
5. **Q5, тайминг:** **{q5v}** (средняя экономия на сделку {', '.join(f'D={k}: разработка {v:+.2f}, контроль {q5["holdout_mean_saving_bp"][k]:+.2f} б.п.' for k, v in q5['dev_mean_saving_bp'].items())}).

**Итог ветки F2: {branch}.**

**Три результата для диссертации**

- *Сигнал меньше издержек на масштабе часов.* После проверки на отскок лучший валовой сигнал на 4 часа даёт {dm.ept_bp:.1f} б.п. на единицу
  оборота, а одна сторона сделки taker стоит 6,5 б.п.; даже снижение оборота зоной без сделок поднимает EPT максимум до
  {q4.get('max_ept_bp_grid', float('nan')):.1f} б.п.
- *Высокий IC не означает больше денег.* В ветке X у ранговой цели T4 самый высокий внешневыборочный IC (h = 1/6/18: {ic.loc['T4', 1]:.3f}/{ic.loc['T4', 6]:.3f}/{ic.loc['T4', 18]:.3f}
  против {ic.loc['T1', 1]:.3f}/{ic.loc['T1', 6]:.3f}/{ic.loc['T1', 18]:.3f} у T1), но после бета-нейтрализации её валовая альфа отрицательна
  (h = 6: разработка {bn.loc['T4', ('dev', 6)]:+.1%}, контроль {bn.loc['T4', ('holdout', 6)]:+.1%} годовых), а у T1 — {bn.loc['T1', ('dev', 6)]:+.1%} и {bn.loc['T1', ('holdout', 6)]:+.1%}.
- *Плацебо для сравнения активов должно ломать связь с активом, а не со временем.* Сдвиг по времени ≥ 60 суток проходит BH в {q2['mean_bh_pass_t1']:.0%}
  случаев (устойчивые свойства активов переживают сдвиг), перестановка между активами — не чаще {q2['max_bh_pass_t34']:.1%}.

**Поправки к F2** (`reports/F2_corrections.md`): регулятор волатильности — не подтверждён, на контроле вредит; FDR 6,7% — одна пара, фильтр
«топ-20»; T3 = T1 на h = 6/18 (Холм по {corr['holm_n']} сравнениям, минимум p = {corr['holm_min_p']:.2f}); фандинг по марк-цене меняет годовой
фандинг не более чем на {corr['funding_mark_max_change_ann_bp']:.1f} б.п.

**Ограничения.** Контроль уже открывался в F2; смещение выживших; синтетическая модель вынужденных сделок (Q5); не учтено влияние объёма на цену;
исполнение лимитными заявками не моделируется.
"""
    (ROOT / "VERDICT_F2b.md").write_text(text, encoding="utf-8")
    print(text)
