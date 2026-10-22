# Исследование исходников и данных

Проверено 22 октября 2026 года. Временные клоны находятся вне checkout проекта,
в `/workspace/reference-repos/`. Исходный код ниже не копировался: реализация пакета
написана заново; используются архитектурные идеи и публичные библиотеки.

| Проект | Проверенный commit | Изученные файлы и решение |
|---|---|---|
| [ResuMatch](https://github.com/JananiV07/resumatch) | `6eb8d49b6a5de28222d538f5c9350c639caa380b` | `backend/resumatch/features.py`, `skill_matcher.py`, `ranker.py`, `explain.py`, `api.py`, `backend/scripts/train.py`, requirements, tests. Референс разделения extraction → features → ranker → explanation. Три признака: семантика, пересечение навыков, опыт. Обучение на случайных beta-признаках и эвристических метках не переносилось. |
| [Nesta OJD DAPS Skills](https://github.com/nestauk/ojd_daps_skills) | `e73c2b5045793d891678ad387b2c7fda1756d236` | `extract_skills/extract_skills.py`, `extract_skills_utils.py`, `multiskill_rules.py`, `map_skills/skill_mapper.py`, ESCO config и таблица. NER в spaCy, разделение составных span, семантический mapper и пороги уровней таксономии. Использована идея разделять извлечение, нормализацию и taxonomy mapping. |
| [SkillAlign](https://github.com/Y4SSERk/SkillAlign) | `cf0681df995838b14823825f0a44ec18c3903108` | `ml_pipeline/data_processing.py`, `embedding_generator.py`, `neo4j_etl.py`, `app/api/services/occupations.py`, routes, requirements, LICENSE/NOTICE. Референс ESCO concept URI, essential/optional и skill gap. Neo4j, FAISS, frontend и occupation-based inference не переносились. |
| [SkillSpan](https://github.com/kris927b/SkillSpan) | `2ccf3de5b5af7a5409b8dd814fb1315dd6e0ae1b` | README, LICENSE, JSONL train/dev/test, BIO skill/knowledge tags и evaluation scripts. Выполнена независимая проверка словаря на полном test split, без обучения/расширения словаря по test. |

## Совместимость и лицензии

Все три репозитория имеют MIT для кода. Их зависимости не устанавливаются целиком.
ResuMatch использует sentence-transformers 3.1.1 и scikit-learn 1.5.2.
Nesta 3.0.0 декларирует Python ^3.9, spaCy ~3.7.4, torch ^1.13.1,
transformers 4.41.2; torch 1.13 несовместим с выбранным Python 3.12.
Nesta также автоматически устанавливает модель через shell и загружает pickle;
этот механизм не используется. В MVP работает собственный словарный fallback RU/EN.
Nesta NER не обучался и не запускался; полноценное выделение произвольных компетенций
остаётся отдельной задачей.

Лицензии model cards проверены отдельно через Hugging Face metadata:

| Модель | Лицензия | Использование |
|---|---|---|
| `intfloat/multilingual-e5-small` | MIT | Скачана и проверена SHA256, frozen CPU encoder; revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`. |
| `nestauk/en_skillner` | MIT | Исследована metadata, не загружалась и не используется. |
| `nestauk/multiskill-classifier` | MIT | Исследована metadata, не загружалась и не используется. |
| `sentence-transformers/all-MiniLM-L6-v2` | Apache-2.0 | Модель референса ResuMatch, не используется. |
| `sentence-transformers/all-mpnet-base-v2` | Apache-2.0 | Модель референса SkillAlign, не используется. |

ESCO: четыре concept URI сверены с форматированной таблицей ESCO v1.1.1 в Nesta.
Русские алиасы и веса связей составлены вручную; это не официальные ESCO relations.
SkillAlign NOTICE указывает CC BY 4.0 для ESCO. Ссылка на официальный download page
во время проверки возвращала HTTP 403; полный официальный CSV не скачивался.
Есть импорт `Taxonomy.import_esco()` для CSV с `conceptUri/preferredLabel/altLabels`.
Атрибуция и уведомления находятся в `THIRD_PARTY_NOTICES.md`.

## Проверенные схемы датасетов

### Candidate Matching Synthetic

Источник: `michaelozon/candidate-matching-synthetic`, MIT, revision
`178ab864dcad9910c5670d43e4bdbbb901a11f18`.
Загружены и проверены checksum трёх parquet: 10 000 резюме, 2 500 вакансий,
2 500 списков соответствий. В отличие от обычного pair-classification датасета,
таблица matches содержит **только положительные resume IDs**, примерно 30 на вакансию.

* `resumes`: `resume_id`, `role`, `seniority`, `years_experience`, `industry`,
  `education`, `skills` (список), `summary`, `experience_bullets` (список).
* `jobs`: `job_id`, `job_title`, `seniority`, `industry`, `must_have_skills`,
  `nice_to_have_skills`, `description`, `responsibilities`, `requirements`.
* `matches`: `job_id`, `relevant_resume_ids` (список).

Профили/вакансии сгенерированы Qwen, соответствия основаны на эвристиках навыков.
Это загруженные текстовые данные, а не случайные числовые признаки; метки при этом
не являются решениями работодателей. Предварительно рассчитанные upstream embeddings
не используются: E5 считает собственные векторы.

Отрицательные примеры — **дополнительная слабая синтетическая разметка**:
кандидат не находится в опубликованном positive list, и покрытие обязательных
навыков меньше 60% с учётом списков **и текста**. Из допустимых кандидатов выбираются
самые близкие по пересечению навыков. Одно отсутствие в positive list не даёт метку 0.
Критерий нужен только для эксперимента и не трактуется как доказательство
отсутствия компетенций у реального человека.

### Resume-Job Fit Merged

Источник: `med2425/resume-job-fit-merged-v1`, revision
`f46f355acddab1368a7386cbe94d42fd42767d88`. Изучены model/dataset card и metadata
схемы; payload не загружался и обучение на нём не выполнялось.

Поля: `resume`, `jd`, `label`, `source`, `resume_domain`, `jd_domain`.
Метки — `Good Fit`, `Potential Fit`, `No Fit`, а не бинарный integer.
Card указывает Qwen2.5-32B как источник меток и смешение двух других датасетов.
Лицензия в metadata/card не заявлена; использовать её как MIT нельзя.

Адаптер `prepare_med2425` / CLI `prepare-med2425` требует reviewed manifest
с `license` и `source_license_urls`, после получения подходящих прав на все исходные
компоненты. `Good Fit → 1`, `No Fit → 0`, `Potential Fit` по умолчанию исключается;
другую политику нужно выбрать явно. Текущая реализация адаптера проверена на отказе
при отсутствующей лицензии; полноценный запуск на upstream payload не проверен.

### SkillSpan

README явно выпускает код и данные, корневой LICENSE — MIT, copyright Mike Zhang.
JSON-файлы на деле JSONL, одна строка на предложение:
`idx`, `tokens`, `tags_skill`, `tags_knowledge`, `source`.
Это NER-разметка, **не** fit labels; не конвертировалась в решения о найме.
Оценка использует объединение skill/knowledge spans на всех 3 569 test-предложениях.

## Предотвращение утечек

Для Candidate Matching независимые кандидаты и вакансии распределяются до sampling
пар: seed + hash нормализованного summary/description, доли 70/15/15.
Одинаковые тексты получают один split даже при разных IDs/metadata. Положительные
рёбра между разными partitions отбрасываются, отрицательные ищутся только внутри split.
После подготовки проверяются пересечения IDs и нормализованных текстов обеих сторон.
Из-за этого эксперимент меньше исходных 75 000 положительных пар.

Для произвольных готовых пар используется объединение связных компонент двудольного
графа кандидатов/вакансий, включая связи дублирующих текстов. Если компонент меньше
трёх, подготовка завершается ошибкой, без fallback на случайный split строк.
Near-duplicates и перефразировки остаются риском; template-based синтетические данные
не дают независимой проверки качества на живых резюме.
