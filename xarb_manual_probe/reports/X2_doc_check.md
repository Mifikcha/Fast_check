# X2 · сверка документации: комиссии, лимиты, доступ

Проверено 2026-11-06. Базовый уровень, без VIP и скидок за токен биржи. Статусы: **офиц.** — страница или документация биржи; **вторичн.** — сторонние источники; **не подтверждено** — по памяти или прошлым этапам.
В расчёте — **taker** на всех четырёх сделках; при расхождении источников берётся большая комиссия. Значения заморожены в `configs/stageX2_frozen.yaml` (`fees_bp`).

## Комиссии, б.п.

| биржа | перп maker / taker | спот taker | в расчёте перп / спот | статус | расхождения |
|---|---|---|---|---|---|
| OKX | 2 / 5 | 10 | 5 / 10 | вторичн. (X1, F3) | уровни Regular зависят от OKB |
| Bybit | 2 / 5,5 | 10 | 5,5 / 10 | перп подтверждён автором (F1); спот не подтверждено | — |
| BingX | 2 / 5 | 10 | 5 / 10 | перп офиц.; спот не подтверждено | VIP-таблицы источников расходятся, VIP0 совпадает |
| Hyperliquid | 1,5 / 4,5 | спот не записан | 4,5 / — | офиц. (X1) | — |
| Aster | 0 / 4 | 10 | 4 / 10 | перп: расхождение (X1, F3); спот **не подтверждено** | документация 3,5/1, с 02.2026 — 4/0; спот-тариф найти не удалось, взято осторожное 10 |
| WhiteBIT | 1 / 5,5 | 10 | 5,5 / 10 | офиц. API: 1 / 3,5 и спот 10; справка и сторонние: 5,5 | взято большее |
| Bitfinex | 0 / 0 | 0 | 0 / 0 | офиц.: без комиссий с 17.12.2025 | старая таблица 2 / 6,5 устарела |
| Blofin | 2 / 6 | — | 6 / — | офиц. (справка и Academy) | у новых листингов временные скидки — не учитываются |
| XT | 4 / 6 | 20 | 6 / 20 | вторичн. | страницу xt.com открыть не удалось |
| Paradex | −0,5 / 3 | — | 3 / — | офиц. документация; **источники расходятся** | сторонние: 0 / 0 для розницы через интерфейс, Pro/API 0,2–0,3 / 2. Взято большее |
| Lighter | 0 / 0 | 0 | 0 / 0 | офиц. (standard-аккаунт, задержка taker 300 мс) | premium 0,4 / 2,8 |
| Bitunix | 2 / 6 | — | 6 / — | вторичн. (цитируют таблицу биржи, проверка 05.2026) | гайд 01.2025 даёт 2 / 5 — вероятно, VIP1 |
| Pionex | 2 / 5 | 5 | 5 / 5 | офиц. (справка Pionex) | один обзор ошибочно пишет, что фьючерсов нет |

## Перевод маржи

Как в X1 и F3: **$1 за перевод**, **30 минут** до начала сделки (по заданию). Не подтверждено по каждой бирже: комиссии вывода без ключа публично отдают только WhiteBIT и XT. Комиссии вывода самой монеты не нужны: спот-ноги считаются купленными на месте, перевозятся только стейблкоины.

## Лимиты запросов (заложено в `recording.book_rps`)

Опрос стаканов: один поток на биржу, запас к опубликованному лимиту. Массовые тикеры — один запрос раз в 15 с на тип инструмента.

| биржа | лимит | основание | стаканов в секунду |
|---|---|---|---|
| OKX | 40 запросов за 2 с на эндпоинт | документация, по памяти | 8 |
| Bybit | 600 запросов за 5 с на IP | документация, по памяти | 20 |
| BingX | заголовок `X-RateLimit-Requests-Remain`: 499 за 10 с | ответ API, 2026-11-06 | 5 |
| Hyperliquid | вес 1200 в минуту, `l2Book` = 2 | документация, по памяти | 8 |
| Aster | вес 2400 в минуту, `depth` 100 = 5 | ответ `exchangeInfo` | 5 |
| WhiteBIT | 2000 за 10 с | документация, по памяти | 10 |
| Bitfinex | 90 в минуту на `book` | документация, по памяти | 1 |
| Blofin | 500 в минуту | документация, по памяти | 5 |
| XT | 10 в секунду, запас 20 | заголовки ответа, 2026-11-06 | 5 |
| Paradex | 50 в секунду | заголовок `X-Ratelimit-Limit` | 10 |
| Lighter | 60 в минуту (standard) | документация, заметка 2026-11-02 | 0,7 |
| Bitunix | 10 в секунду | не подтверждено | 5 |
| Pionex | 30 токенов | заголовок `X-RateLimit-Tokens` | 3 |

При ответе 429 или 418 рекордер ждёт `Retry-After` (нет заголовка — 60 с) и пишет строку в `logs/gaps.csv`.

## Что отдают биржи без ключа

| что | есть | нет |
|---|---|---|
| лучшие цены пачкой | OKX, Bybit, BingX, Aster, WhiteBIT, Bitfinex, Blofin, XT, Paradex, Pionex | Hyperliquid — только цены удара; Lighter и Bitunix — только через WebSocket; объёма на лучшей цене нет у WhiteBIT, XT-перпов, Hyperliquid |
| история фандинга | OKX, Bybit, BingX, Hyperliquid, Aster, WhiteBIT, Blofin, XT, Lighter, Pionex | Bitfinex, Paradex, Bitunix — ставка из снимка перед расчётом |
| статус ввода/вывода | WhiteBIT, XT | все остальные требуют ключ или не публикуют — «неизвестно» |
| единицы ставки | доля за интервал | Bitunix — проценты; Lighter `/fundings` — проценты с `direction`; `/funding-rates` у Lighter приведён к 8 ч |

## Ссылки

- [BingX: комиссии перпетуалов](https://bingx.com/en/support/articles/360046487573-perpetual-futures-fee-schedule)
- [Blofin: как считаются комиссии](https://blofin.zendesk.com/hc/en-us/articles/9049331029775-How-Are-Futures-Trading-Fees-Calculated), [Blofin Academy](https://blofin.com/academy/guides/blofin-fee-structure)
- [XT: обзор TradersUnion](https://tradersunion.com/brokers/crypto/view/xt.com/fees/)
- [Bitunix: обзор Cryptodaily 2026](https://cryptodaily.co.uk/2026/06/bitunix-fees-spot-futures-vip-withdrawals), [Trade Reclaim](https://trade-reclaim.com/en/blog/bitunix-fees)
- [Paradex: документация](https://docs.paradex.trade/documentation/trading/trading-fees)
- [Pionex: обзор комиссий](https://intercom.help/pionex/en/articles/14993683-pionex-trading-fee-overview)
- [Bitfinex: какие комиссии](https://support.bitfinex.com/hc/en-us/articles/213919589-What-fees-do-you-charge-)
- [WhiteBIT: API market fee](https://docs.whitebit.com/api-reference/trading/query-market-fee), [справка](https://help.whitebit.com/hc/en-gb/articles/25029308319005)
- Остальное — `xarb_probe/reports/doc_check.md`, `funding_carry_probe/reports/F3_doc_check.md`
