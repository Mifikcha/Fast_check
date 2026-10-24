# Stage F2 — что предсказывать: сравнение целей прогноза в автоматическом поиске сигналов

Отдельный проект-ветка исследования `microstructure_alpha_probe`, параллельный Stage F1. Код, данные, конфиг и журналы —
только здесь; файлы Stage 1–5 и F1 не меняются. Из Stage 4 взяты снимки `universe.csv` и `spread_depth.csv`
(`data/universe_stage4.csv`, `data/spread_depth_stage4.csv`, SHA-256 в конфиге).

Итог: [`VERDICT.md`](VERDICT.md). Отчёты: `reports/`. Правила заморожены в `configs/stageF2_frozen.yaml`
(хеш — `logs/frozen_hash_F2.txt`).

## Запуск (из этой папки, Python общего `.venv` репозитория)

```
python scripts/run_f2.py freeze           # один раз: хеш конфига -> logs/frozen_hash_F2.txt
python scripts/run_f2.py download         # публичный REST Bybit, возобновляемо (готовые файлы пропускаются)
python scripts/run_f2.py clean            # parquet, 4-часовые панели, reports/data_quality_F2.md (критерий C1)
python scripts/run_f2.py price_demo       # раздел 6: почему не уровень цены
python scripts/run_f2.py features         # библиотека признаков (до конца разработки), дубли, PCA
python scripts/run_f2.py walkforward      # помесячный walk-forward на разработке (кэш в data/work/refits)
python scripts/run_f2.py portfolios       # дневная прибыль, выбор и заморозка θ (logs/theta_selected_F2.json)
python scripts/run_f2.py open_holdout_F2  # ОДИН раз: logs/holdout_opened_F2.log
python scripts/run_f2.py features         # пересчёт на всех данных (тот же список признаков; разработка не меняется)
python scripts/run_f2.py walkforward      # продолжение walk-forward на контрольные месяцы
python scripts/run_f2.py portfolios
python scripts/run_f2.py report           # все отчёты, графики и VERDICT.md
python -m pytest -q tests                 # тест усечения (причинность), статистика, издержки
```

Сырые ответы API (`data/raw/`), parquet (`data/clean/`) и рабочие файлы (`data/work/`, ~4 ГБ) в git не входят.
