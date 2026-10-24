# ML web-сервис на FastAPI

Web-сервис для онлайн-предсказания вероятности совершения пользователем целевого действия по параметрам сессии на основе данных Google Analytics.
Результат: ROC-AUC = 0.70 на отложенной тестовой выборке при ориентире технического задания ≈ 0.65. Итоговая модель — RandomForestClassifier.

В рамках данного проекта приняты следующие термины:
- model — чистая ML-модель
- preprocessor — пайплайн подготовки данных для модели
- pipeline — единый пайплайн, включающий preprocessor и model
- bundle — результат обучения, сериализованный артефакт, содержащий обученный pipeline и метаданные.

Файл eda_design_optimization.ipynb содержит все исследования, разработку preprocessor с обоснованиями всех решений, оптимизацию пайплайна с тремя разными моделями, выбор оптимальной модели и её оценку на тестовой выборке. Для запуска ноутбука необходимы следующие файлы:
- /data/raw/ga_sessions.pkl (~ 4 Гб)
- /data/raw/ga_hits.pkl (~ 380 Мб)

Они отсутствуют в репозитории из-за ограничений на размер. Перед запуском ноутбука необходимо скачать их по ссылке: `https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw`.

## Необходимые файлы

Для запуска сервиса требуется файл `/models/predict_target_pipeline.pkl`, содержащий bundle. Его можно извлечь из архива `/models/predict_target_pipeline.zip` или создать с помощью `train.py`.

Для `train.py` необходим файл `/data/processed/sessions_target.pkl`, который создаётся исследовательским ноутбуком `eda_design_optimization.ipynb`.

## Запустить сервис можно двумя способами

### 1. Локально

```bash
pip install -r requirements.txt  # при необходимости
uvicorn main:app
```

### 2. С помощью Docker

```bash
docker build -t service .
docker run -p 8000:8000 service
```

## Интерфейс сервиса

### GET /status
Проверка доступности сервиса.

### GET /info
Возвращает метаданные модели.

### POST /predict
Основной endpoint для получения предсказаний.

- Вход: JSON с признаками сессии типа utm_*, device_*, geo_*.
- Выход: prediction, probability, threshold

## Документация и тестирование

После запуска сервиса доступен Swagger UI:

http://localhost:8000/docs

Через Swagger UI можно:
- просмотреть описание всех доступных endpoints,
- увидеть формат входных и выходных данных,
- протестировать работу сервиса без использования сторонних инструментов (Postman, curl и т.п.).

## Структура проекта

```
final_project/
│
├─ data/
│  ├─ processed/
│  │  ├─ df_test_raw.pkl (в репо нет)      # тестовая часть таблицы sessions без обработки (создаёт ноутбук только для исследования)
│  │  ├─ df_train_raw.pkl (в репо нет)     # обучающая часть таблицы sessions без обработки (создаёт ноутбук только для исследования)
│  │  ├─ df_train.pkl (в репо нет)         # обработанные обучающие данные (создаёт ноутбук только для исследования)
│  │  └─ sessions_target.pkl (в репо нет)  # sessions без изменения + столбец target (создаёт ноутбук для train.py)
│  │
│  └─ raw/
│     ├─ ga_sessions.pkl (в репо нет)      # для запуска нотбука необходимо скачать оба файла по ссылке
│     └─ ga_hits.pkl (в репо нет)          # https://drive.google.com/drive/folders/1rA4o6KHH-M2KMvBLHp5DZ5gioF2q7hZw
│
├─ models/
│  └─ predict_target_pipeline.zip          # архив bundle, перед запуском сервиса необходимо распаковать
│
├─ test_jsons/
│  ├─ make_test_jsons.py                   # создаёт два json-файла для тестирования сервиса
│  ├─ predict_class0.json
│  └─ predict_class1.json
│
├─ .dockerignore
├─ Dockerfile
├─ eda_design_optimization.ipynb           # EDA + разработка и оптимизация пайплайна + evaluation
├─ main.py                                 # FastAPI app + загрузка модели + /predict
├─ make_requirements.py                    # создаёт файл requirements.txt
├─ pipeline.py                             # make_pipeline + кастомные функции
├─ requirements.txt
├─ README.md
└─ train.py                                # кросс-валидация + обучение + сохранение .pkl
```