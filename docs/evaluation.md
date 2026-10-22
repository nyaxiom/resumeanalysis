# Выполненная оценка

Дата: 22 октября 2026 года. Это exploratory smoke-эксперимент на синтетических текстах,
а не подтверждение качества решений о найме. Числа приведены из фактического запуска;
никакая LLM не использовалась как оценщик. Машиночитаемые результаты:
[cpu_smoke.json](experiments/cpu_smoke.json), [skillspan.json](experiments/skillspan.json).

## Данные и обучение

Скачаны реальные parquet Candidate Matching Synthetic: 10 000 резюме, 2 500 jobs,
2 500 positive lists. Из них отобрано 250 jobs, по возможности два positive и два
hard negative на job. Получено 994 пары: 696 train, 150 validation, 148 test.
Все классы сбалансированы; test содержит 74 positive и 74 negative.
ID и нормализованные summary/description обеих сторон не пересекаются между splits.
Hashes, точный seed, исходный revision и negative policy находятся в JSON.

Разметка имеет ограниченную достоверность: исходные тексты Qwen-generated,
positive lists эвристические, наши negatives — слабые labels по покрытию навыков
меньше 60% после объединения structured и text evidence. Это не подтверждённые
отказы работодателей. Более близкие допустимые sub-threshold profiles выбираются
как hard negatives; произвольные непромаркированные пары не объявляются отрицательными.

Encoder: настоящий pretrained `intfloat/multilingual-e5-small`, frozen,
revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, query/profile + passage/job.
Обучались только MLP и Logistic Regression. Модель MLP: 1 553 → 128 → 32 → 1,
GELU, Dropout 0.2, AdamW lr 0.001, weight decay 0.001, batch 64, seed 42,
два CPU threads, BCEWithLogitsLoss. Лучший validation loss на эпохе 63,
patience 10, обучение остановлено после 73 эпох. Веса действительно изменились.
MLP и LR имеют одинаковое входное представление; LR scaler обучается только на train.

Порог выбирается по validation Macro-F1: rule 0.5, logistic 0.325, neural 0.575.
MLP checkpoint восстанавливается по validation loss; test не участвует в этом коде.
В процессе разработки была устранена противоречивая разметка отрицательных примеров
без учёта текста. Поэтому итоговая выборка — исследовательская проверка pipeline,
не запечатанный внешний benchmark для model selection.

## Итоговый test

| Метод | Precision (positive) | Recall (positive) | Macro-F1 | ROC-AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---|
| Rule-based | 0.9024 | 1.0000 | 0.9458 | 0.9956 | 66 / 8 / 0 / 74 |
| Logistic Regression | 0.8625 | 0.9324 | 0.8917 | 0.9680 | 63 / 11 / 5 / 69 |
| ResumeAnalysisNet | 0.9733 | 0.9865 | 0.9797 | 0.9996 | 72 / 2 / 1 / 73 |

MLP лучше обоих baseline на этой выборке. Преимущество не проверялось на других
seed, на независимом реальном fit dataset или доверительных интервалах.
Высокий ROC-AUC ожидаем для синтетической задачи, построенной вокруг skill coverage.
Его нельзя переносить на кадровые решения. Калибровка не выполнялась; sigmoid
может насыщаться до 1.0, что не означает достоверность 100%.

Сохранённые MLP/LR повторно загружены и оценены: все test-метрики точно совпали.
CLI inference на `examples/pair.json` выполнен с `HF_HUB_OFFLINE=1` и
`TRANSFORMERS_OFFLINE=1`: Python/SQL подтверждены, Airflow неизвестен, опыт соответствует.
Публичный Python `predict()` также проверен. Reserved fields равны null.

## Анализ ошибок

MLP: два false positive с required coverage 0.5 и семантикой 0.8580 / 0.8454.
Один из этих профилей покрывает optional skill, оба соответствуют указанному seniority;
их score 0.6849 / 0.6789 превысил validation threshold 0.575.
Один false negative: required coverage 0.6, optional coverage 0, seniority ниже требования,
semantic similarity 0.8509 и score 0.5256. Исходный positive list всё же считает его relevant.
Это расхождение skill-rule labels и многопризнакового ранжирования.

Rule baseline допускает 8 false positive, LR — 11 false positive и 5 false negative.
По сравнению с простой формулой дополнительная линейная модель на высокоразмерных
векторах не улучшила результат при малой выборке; точную причинную роль каждого
блока признаков без ablation утверждать нельзя.

Файлы `artifacts/run/errors_{rule,logistic,neural}.json` содержат IDs, labels, scores,
provenance и все признаки. Полные тексты резюме в error logs не записываются.
Ошибки не исправлялись подстройкой test threshold или генеративным объяснением.

## Skill extraction

На полном SkillSpan test (3 569 строк) объединены gold SKILL и KNOWLEDGE BIO spans.
Словарь не расширялся по test. Метрики finite dictionary fallback:

| Уровень | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---|
| Exact span | 0.6065 | 0.0830 | 0.1460 | 188 / 122 / 2077 |
| Token overlap | 0.7906 | 0.0427 | 0.0810 | 268 / 71 / 6012 |

Низкая полнота ожидаема для технологического словаря против открытой NER-задачи:
теряются длинные профессиональные и soft-skill spans, языки и неизвестные технологии.
Это существенный недостаток MVP, а не готовый NER уровня Nesta. Token F1 и span F1
имеют разные знаменатели и не взаимозаменяемы. Русский extractor проверен unit tests,
но количественного русского annotated benchmark нет.

## Проверки разработки

* `uv sync --frozen --extra dev --extra data`: повторяемая установка CPU dependencies.
* `RESUMEANALYSIS_RUN_E5_TESTS=1 pytest -q`: 47 выполненных тестов, включая pretrained E5.
* `ruff check .`, `ruff format --check .`: lint/format.
* `uv build`: созданы wheel и sdist; пакет содержит все Python-подмодули.
* Prepare повторён: hashes одинаковы. Проверены entity/content leakage и отсутствие
  отрицательных примеров с evidence coverage ≥60%.
* Проверены `evaluate` после reload, CLI offline inference и публичный `predict`.

CI описывает Python 3.11/3.12; удалённый GitHub Actions результат нужно смотреть
отдельно. На текущей облачной машине проверен Python 3.12. Arch Linux и целевой
Ryzen 5700U отдельно не тестировались. Полный датасет, NER train, med2425 payload,
калибровка, latency на целевом CPU и восстановление нового cloud task не проверены.
