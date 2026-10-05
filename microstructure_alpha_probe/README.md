# Microstructure Alpha Feasibility Probe

Небольшой исследовательский проект для одного вопроса: существует ли в
публичной микроструктуре BTCUSDT perpetual воспроизводимый краткосрочный сигнал,
который оправдывает дальнейшее исследование. Это не торговый бот и не попытка
максимизировать backtest PnL.

Проект намеренно консервативен: финальные 20% времени изолированы, все пороги и
гиперпараметры фиксируются до test, плохие строки остаются в полной секундной
сетке, а короткий запуск всегда получает `SMOKE ONLY`, независимо от красивых
метрик.

## Быстрый старт (Windows PowerShell)

Из каталога `microstructure_alpha_probe`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Перед длинным сбором синхронизируйте системные часы Windows: откройте
**Параметры → Время и язык → Дата и время**, включите автоматическую установку
времени и нажмите **Синхронизировать сейчас**. Состояние службы можно проверить
без изменения системы:

```powershell
w32tm /query /status
```

Сильное расхождение часов не ломает порядок по локальному receive-time, но делает
оценку exchange-to-receive latency недостоверной и принудительно проваливает
quality gate. Сначала полезно сделать короткий pilot и проверить latency в
`data_quality_report.md`.

Собрать 12 часов и затем выполнить анализ:

```powershell
.\.venv\Scripts\python.exe scripts\run_pilot.py collect --hours 12
.\.venv\Scripts\python.exe scripts\run_pilot.py analyze
```

Для 24 часов замените `12` на `24`. Команда `analyze` выбирает последнюю
завершённую сессию и, если нужно, сначала строит processed dataset. Конкретную
сессию можно указать явно:

```powershell
.\.venv\Scripts\python.exe scripts\run_pilot.py build --session <SESSION_ID>
.\.venv\Scripts\python.exe scripts\run_pilot.py analyze --session <SESSION_ID>
```

Единая команда тоже доступна:

```powershell
.\.venv\Scripts\python.exe scripts\run_pilot.py all --hours 12
```

Во время сбора CLI каждые 30 секунд показывает прошедшее и оставшееся время,
процент выполнения, число wire/book/ticker сообщений и сделок, скорость потока,
число и объём уже сохранённых Parquet parts, строки в буфере, состояние стакана и
reconnects. После сбора выводятся стадии `BUILD [n/8]` и `ANALYZE [n/12]` с
прошедшим временем. Интервал можно изменить, например:

```powershell
.\.venv\Scripts\python.exe scripts\run_pilot.py all --hours 24 --progress-seconds 10
```

`book OK` означает, что получен snapshot и стакан можно уверенно восстановить;
`book UNSYNCED` во время подключения или после потери транспорта означает, что
коллектор ожидает новый snapshot и не выдаёт этот интервал за пригодные данные.

Непрерывный сбор до `Ctrl+C`:

```powershell
.\.venv\Scripts\python.exe scripts\run_pilot.py collect
```

Повторно сгенерировать только Markdown-отчёт из уже сохранённых таблиц, не
переобучая модели:

```powershell
.\.venv\Scripts\python.exe scripts\run_pilot.py report --run <RUN_ID>
```

## Что собирается

Используется публичный Bybit V5 linear WebSocket
`wss://stream.bybit.com/v5/public/linear`, без API-ключа:

- `orderbook.50.BTCUSDT` — snapshot/delta depth 50;
- `publicTrade.BTCUSDT` — пакеты публичных сделок, развёрнутые в отдельные строки;
- `tickers.BTCUSDT` — разреженные ticker delta для независимого L1, OI и funding.

Исходный UTF-8 JSON, локальное receive-time, exchange timestamps, IDs и typed
поля пишутся в новые ZSTD Parquet parts. Временный part публикуется атомарным
rename; существующий raw-файл никогда не заменяется. Манифест и `events.jsonl`
показывают connect/reconnect, heartbeat, parse/write failures и завершение.

Bybit документирует `u`, `seq` и `cts`, но не обещает, что `u` или `seq`
увеличиваются строго на единицу. Поэтому forward jump — диагностическая аномалия,
а не выдуманная потеря. Доказанная локальная/транспортная потеря, конфликтующий
identity, убывающий ID, malformed/crossed book или reconnect переводят стакан в
`UNSYNCED` до нового snapshot.

## Причинная таблица

`data/processed/<SESSION_ID>/research_table.parquet` содержит каждую UTC-секунду
сессии. Стакан выбирается только по `recv_utc_ns <= timestamp`; exchange-time не
используется для доступности. Никакого future interpolation или backward fill.

Таблица включает L1–L20, spread/mid/microprice, OBI 1/5/10/20, event OFI,
trade-flow окна 1/5/10/30/60 с coverage, прошлые returns/realized volatility,
каузально carried-forward OI/funding с возрастом и точные self-join targets
`t+10/30/60s`. Invalid/stale/gap строки не удаляются.

## Оценивание

- Outer split 60/20/20 строится на полной сетке до фильтрации.
- У каждой границы общий purge и leading embargo не меньше максимального horizon.
- Внутри development — expanding walk-forward без shuffle.
- Imputer/scaler обучаются только на training-части соответствующего fold.
- Test не выбирает features, alpha/C, LightGBM grid, threshold, quantile edges,
  regimes, extreme cutoff или модель.
- `decision_lock.json` и его SHA-256 записываются до test evaluation.
- Предзаявленная confirmatory пара — Ridge, 30 секунд; лучший test-результат
  показывается лишь описательно и не подменяет её.

Модели: zero, historical mean, previous-return control, OLS, небольшой Ridge
grid, multinomial logistic regression и один небольшой LightGBM grid. Нейросетей,
live execution и автоматической торговли нет.

Экономическая диагностика использует неперекрывающиеся позиции. Mid вариант
помечен optimistic. Taker вариант для long использует
`log(future_bid/current_ask)`, для short — `log(current_bid/future_ask)`.
Quoted spread уже включён; 0/2/5/10/15 bps — дополнительная round-trip
fee/slippage гипотеза из YAML, а не утверждение о фактических комиссиях.

## Что смотреть после реального запуска

В каждом уникальном `runs/<RUN_ID>/` сначала откройте:

1. `data_quality_report.md` и `data_quality.json` — пригоден ли capture вообще;
2. `verdict.json` — все флаги, значения и предзаявленные пороги;
3. `metrics_summary.csv` и `fold_metrics.csv` — untouched-test и каждый fold;
4. `quantile_analysis.csv` — монотонность frozen buckets;
5. `bootstrap_confidence_intervals.csv` и `multiple_testing.csv` — uncertainty/FDR;
6. `trading_metrics.csv`, `cost_sensitivity.csv` и `concentration_metrics.csv`;
7. `ablation_metrics.csv` — добавляет ли микроструктура что-либо к price history;
8. `predictions_test.parquet` и `trades_test.parquet` — аудит на уровне строк.

`report.md` собирает краткий вывод и 20 PNG, но CSV/Parquet являются более
полным источником доказательств. `LEAKAGE_CHECKLIST.md` — ручной audit contract.

## GO / WEAK GO / NO-GO

Все численные границы находятся в `configs/pilot.yaml`. GO требует совместно:
положительный confirmatory test IC, устойчивый знак по fold, монотонные quantiles,
отсутствие концентрации, прирост к price-only, положительную time-series CI и
выживание при ненулевой дополнительной стоимости. WEAK GO допускает слабую
экономику/неопределённость при устойчивой статистической структуре. Иначе NO-GO.

Capture короче 12 часов, слишком маленький test или критическая проблема качества
принудительно дают `NO-GO — INSUFFICIENT EVIDENCE (SMOKE ONLY)`.

## Ресурсы

Ориентир для обычного рынка: 0.08–0.30 GB raw/час, burst до 0.6 GB/час. Типично
1 час занимает 0.08–0.30 GB, 12 часов 1.0–3.6 GB, 24 часа 1.9–7.2 GB. С учётом
processed/runs и запаса рекомендуется иметь около 1/10/20 GB соответственно.

На ноутбуке с 4–8 ядрами ожидаемый analysis runtime примерно 2–8 минут (1 h),
10–35 минут (12 h), 20–70 минут (24 h); RAM около 1–1.5/1.5–3/2–4 GB. Реальная
нагрузка зависит от активности рынка и числа bootstrap resamples.

## Происхождение решений

До реализации был read-only изучен `Transformer_RV_Crypto` на commit
`933ddbb57b5a987e3f0313e0483611d04d915bbc`. Заимствованы хорошие малые паттерны
(UTC, Parquet, causal rolling, train-only preprocessing, walk-forward,
абляции), но WebSocket collector, book state machine, секундная таблица,
экономическая диагностика и verdict написаны отдельно. Зафиксированный план —
`docs/implementation_plan.md`, первичные источники —
`docs/reference_research.md`.

## Stage 2: многодневная проверка воспроизводимости

Запуск из каталога `microstructure_alpha_probe` в PowerShell:

```powershell
..\.venv\Scripts\python.exe scripts\run_pilot.py collect --days 7
```

Цель — 7 **полных** UTC-дней; для 10–14 дней замените число. Перед запуском
показываются фактические размеры первого 24-часового прогона: примерно 0.82 GB
raw + 0.05 GB processed в день, ~6.0 GB за 7 дней или ~12.1 GB за 14 дней;
свободное место проверяется с резервом 3 GB. `collect` без аргументов создаёт
14-дневный эксперимент, который можно останавливать Ctrl+C. Старый 24-часовой
`collect --hours 24` и `all --hours 24` остаются доступны.

В другом терминале:

```powershell
..\.venv\Scripts\python.exe scripts\run_pilot.py status
..\.venv\Scripts\python.exe scripts\run_pilot.py daily
```

После Ctrl+C повторный запуск **того же** эксперимента:

```powershell
..\.venv\Scripts\python.exe scripts\run_pilot.py collect --days 7 --resume
```

Для выбора определённого эксперимента добавьте `--experiment-id <ID>` ко всем
этим командам. Каждый restart создаёт новый session ID; уже закрытые Parquet parts
не изменяются. Промежуток между сессиями остаётся пустым и показан в
`recording_sessions.csv`, а секундная таблица помечает его как invalid.
Если процесс упал, выполните `status`, затем `collect --days 7 --resume`;
манифест незавершённой сессии учитывается, а последний timestamp извлекается
из опубликованных Parquet частей. Временные `.tmp` файлы не считаются
опубликованными частями; не удаляйте их до проверки причины сбоя.

Манифест эксперимента: `data/long/<ID>/experiment.json`; raw-сессии остаются в
`data/raw/{orderbook,trades,aux}/<SESSION_ID>/`. Чанки закрываются каждые
15 минут либо после 100 тысяч строк. Отчёты по завершённым UTC-дням находятся
в `runs/long_<ID>/daily/<YYYY-MM-DD>/`. Неполный live smoke отдельно помечается
`partial` и не входит в вердикт.

Когда собраны дни:

```powershell
..\.venv\Scripts\python.exe scripts\run_pilot.py analyze-long
..\.venv\Scripts\python.exe scripts\run_pilot.py report-long
```

Итог в уникальном `runs/long_<ID>/analyses/<TIMESTAMP>/`; сохранены все
подробные CSV, `longitudinal_summary.json`, `longitudinal_report.md` и 16 PNG.
Сначала смотрите `data_quality_by_day.csv`, `daily_confirmatory_metrics.csv`,
`nonoverlap_ic_by_phase.csv`, `latency_stress.csv`, `cost_sensitivity.csv`, затем
`plots/daily_ic.png`. Научный и торговый вердикты отделены. Замороженная
Ridge@30s использует только прежний development для обучения и прежние пороги;
остальные горизонты и абляции явно exploratory.

Для долгого Windows-сбора отключите сон и гибернацию в параметрах питания,
обеспечьте стабильную сеть и не закрывайте окно PowerShell. `status` показывает
последнее сообщение, валидность стакана, скорость сообщений, reconnects и диск.
Остановка Ctrl+C безопасно закрывает текущую сессию; при жёстком падении уже
опубликованные chunks остаются неизменными. Для теста без многодневного запуска
можно использовать `collect --days 7 --smoke-minutes 5`, затем
`collect --days 7 --resume --smoke-minutes 5` и
`analyze-long --include-partial`.
