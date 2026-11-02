# F2b на ликвидных рынках Lighter (одна попытка, правило заморожено)

Конфиг `configs/stageF2b_lighter_liquid.yaml`, хеш `logs/frozen_hash_F2b_lighter_liquid.txt`.
Набор (14): BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, HYPEUSDT, XAUUSDT, NVDAUSDT, CLUSDT, SNDKUSDT, MUUSDT, BNBUSDT, XAGUSDT, BZUSDT, DRAMUSDT. Активов в срезе на разработке: медиана 5, мин 5, макс 9.
Издержки: на каждую сторону — `imp10k_bp` этого актива на Lighter (Tardis, часы ребалансировки).

**Вердикт: NO-GO.** Правило: dev net Sharpe (Lighter costs) > 0 and its 90% CI lower bound > 0 and holdout net Sharpe > 0.
{"D:T5 q4_frozen": false, "D:T5 raw_h1": false, "X:T1 q4_frozen": false, "X:T1 raw_h1": false}

| strategy | variant | period | days | turnover_day | ept_bp | lighter_cost_side_bp | gross_sharpe | net_sharpe_lighter | ci_lo | ci_hi | net_ann_lighter | net_sharpe_bybit_S1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X:T1 | raw_h1 | dev | 1209 | 8.899 | 0.835 | 1.1 | 0.662 | -0.211 | -1.293 | 0.793 | -0.086 | -4.548 |
| X:T1 | raw_h1 | holdout | 180 | 9.078 | 1.003 | 1.292 | 0.909 | -0.263 | -1.927 | 1.58 | -0.096 | -5.057 |
| X:T1 | q4_frozen | dev | 1209 | 0.768 | -3.518 | 1.095 | -0.553 | -0.725 | -1.573 | 0.137 | -0.129 | -1.577 |
| X:T1 | q4_frozen | holdout | 180 | 0.032 | -149.9 | 1.107 | -2.091 | -2.107 | -4.643 | -0.218 | -0.176 | -2.182 |
| D:T5 | raw_h1 | dev | 1209 | 8.483 | 2.401 | 1.103 | 1.352 | 0.731 | -0.17 | 1.599 | 0.402 | -2.306 |
| D:T5 | raw_h1 | holdout | 180 | 9.154 | 0.577 | 1.29 | 0.507 | -0.625 | -3.469 | 1.783 | -0.238 | -5.185 |
| D:T5 | q4_frozen | dev | 1209 | 1.07 | 0.785 | 1.121 | 0.098 | -0.042 | -0.915 | 0.886 | -0.013 | -0.711 |
| D:T5 | q4_frozen | holdout | 180 | 0.596 | 22.27 | 1.329 | 1.836 | 1.726 | -0.288 | 3.599 | 0.455 | 1.3 |

Оговорка: контроль уже открывался в F2: подтверждаем только знак и порядок величины.
