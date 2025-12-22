# AGENTS.md

## Назначение проекта
ChartReader — это пайплайн для «derendering» графиков и ответов на вопросы по графикам.
Он состоит из двух частей:
1) извлечение структурированных компонентов графика (keypoints + группировка),
2) генерация ответов на вопросы (T5) по табличному представлению.

## Ключевые директории и файлы
- `config/` — JSON-конфиги `KPDetection.json` и `KPGrouping.json` для модели извлечения.
- `train_extraction.py` — обучение извлечения компонент (KP Detection/Grouping).
- `model_factory.py` — сборка модели и оптимизатора по `system_configs`.
- `db/` — COCO-совместимый датасет и кэширование аннотаций.
- `sampling_function.py` — аугментации, генерация heatmap/targets.
- `run_t5.py` — fine-tuning T5 для вопросов/ответов.
- `val_extraction.py`, `infer_chart.py`, `demo.py` — валидация и демо.
- `requirements.yaml` — окружение conda.

## Данные и формат
- Датасет ожидается в COCO-формате внутри `data/` (см. README).
- Аннотации: `data/annotations/{train,val,test}.json`.
- Изображения: `data/images/{train,val,test}/`.
- Кэш детекций создается в `cache/` и/или `data/cache/` (см. `config.py`).

## Как происходит обучение модели

### 1) Извлечение компонент графика (KP Detection/Grouping)
Обучение построено вокруг двух конфигураций в `config/`:
- `KPDetection.json` обучает детектор ключевых точек/компонент.
- `KPGrouping.json` группирует найденные ключевые точки в структурные элементы.

Базовый скрипт: `train_extraction.py`.
Он делает следующее:
1. Загружает конфиг `config/<cfg_file>.json`.
2. Заполняет `system_configs` (см. `config.py`) путями `data_dir`, `cache_path`,
   именем снапшота, гиперпараметрами (LR, stepsize, max_iter).
3. Создает `Network()` из `model_factory.py`, который:
   - динамически импортирует модель из `models/<snapshot_name>.py`,
   - создает оптимизатор (Adam/SGD) по конфигу,
   - использует loss из модуля модели.
4. Инициализирует датасет `Chart` из `db/coco.py`:
   - читает COCO-аннотации,
   - фильтрует категории `self._cat_ids`,
   - строит кэш в `cache_dir`.
5. Основной training loop:
   - выборка батча через `sample_data` из `sampling_function.py`,
   - перенос на GPU/CPU, mixed precision (`autocast` + `GradScaler`),
   - `train_step()` -> forward + loss,
   - сохранение лучшей модели по `val_loss`.

Важно:
- Для `KPGrouping` используется предобученная `KPDetection` модель
  (параметр `--pretrained_model`).
- Снапшоты пишутся в `cache_path/nnet/<snapshot_name>/`.

Пример запуска (см. README):
```
python train_extraction.py \
  --cfg_file KPDetection \
  --data_dir "/path/to/data" \
  --cache_path "/path/to/cache"
```

### 2) Вопросы-ответы по графику (T5 fine-tuning)
Скрипт `run_t5.py` — это адаптация стандартного HuggingFace
Seq2Seq тренера под CSV/JSONL.
Пайплайн:
1. Загружает CSV/JSONL с колонками `Input` и `Output`.
2. Токенизирует вход/цель, применяет max length.
3. Обучает `AutoModelForSeq2SeqLM` (обычно `t5-base`)
   через `Seq2SeqTrainer`.
4. Сохраняет лучшие чекпоинты в `--output_dir`.

Пример (см. README):
```
torchrun --nproc_per_node=1 run_t5.py \
  --model_name_or_path t5-base \
  --train_file "/path/to/train.csv" \
  --validation_file "/path/to/val.csv" \
  --test_file "/path/to/test.csv" \
  --text_column Input \
  --summary_column Output \
  --output_dir "/path/to/t5_output"
```

## Валидация и инференс
- `val_extraction.py` — проверка качества извлечения компонент.
- `infer_chart.py` — инференс на изображениях.
- `demo.py` — UI демо (нужно вручную задать пути в файле).

## Что важно помнить агенту
- Часть с COCO-датасетом чувствительна к структуре директорий.
  Перед любыми изменениями проверьте `data_dir` и `cache_path`.
- Модель извлечения использует конфиги из `config/`, а модель QA —
  стандартный HuggingFace pipeline.
- `db/coco.py` сейчас фильтрует категории через `self._cat_ids` —
  это влияет на набор компонент, которые реально учатся.

## Как тестировать изменения
Автоматических тестов нет. Для проверки:
1) запуск мини-обучения `train_extraction.py` на небольшом сэмпле;
2) прогон `val_extraction.py` на валид-сете;
3) запуск `run_t5.py` на маленьком CSV для sanity-check.
