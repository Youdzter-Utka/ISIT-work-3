# Задача № 3 — самолёты и Hugging Face

30 фото, 15 военных и 15 гражданских, включая 20 сложных случаев. Две VLM: Qwen2.5-VL-72B-Instruct и GLM-4.5V.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Youdzter-Utka/ISIT-work-3/blob/main/report.ipynb)

**Отчёт: [report.ipynb](report.ipynb). Данные готовы, 30 фото проверены. Метрики пока отсутствуют:**
Hugging Face принял токен, но вернул HTTP 402 — доступные кредиты закончились.
После пополнения установите `RUN_INFERENCE=True` в notebook и выполните ячейки.
Входящие в notebook таблицы и фотографии получены реально; предсказаний моделей в них нет.

## Запуск
1. Откройте report.ipynb в Jupyter. Для Colab загрузите notebook через «Файл → Загрузить блокнот».
2. В Colab загрузите ZIP проекта в Files и выполните `!unzip -q aircraft-hf-comparison.zip -d /content`.
3. Для просмотра галереи токен не нужен. Для эксперимента добавьте HF_TOKEN в Colab Secrets, разрешите доступ notebook. Локально используйте скрытый ввод.
4. После пополнения HF установите `RUN_INFERENCE=True`. Выполните ячейки по порядку: установка, подготовка/проверка фото, 60 запросов, таблицы и графики.
5. Сохраните notebook с выводами. GPU не требуется. Бесплатного кредита может не хватить: проверьте баланс HF.

Можно открыть GitHub notebook через https://colab.research.google.com/github/; исходники проекта
дополнительно клонируйте перед выполнением ячеек:
```python
!git clone https://github.com/Youdzter-Utka/ISIT-work-3.git /content/aircraft-hf-comparison
%cd /content/aircraft-hf-comparison
```
При повторном запуске клонировать ещё раз не нужно.

## Содержимое
- report.ipynb — методика, галерея, код запуска и отчёт на русском.
- src/experiment.py — загрузка, запросы, метрики, сохранение и возобновление.
- config.json — две модели и параметры; перед запуском проверяется доступность провайдеров.
- data/selection.csv — 30 типов, метки и причины сложности.
- data/manifest.csv и data/images — источники, лицензии, JPEG; уже включены и доступны офлайн.
- results/predictions.jsonl — реальные ответы; metrics.csv, errors.csv, summary.json и график — после полного прогона.

## Командная строка
```bash
pip install -r requirements.txt
python src/experiment.py prepare
# Проверьте фото, подтвердите reviewed=1 в manifest.csv, задайте HF_TOKEN через окружение.
python src/experiment.py run --limit 2
python src/experiment.py run
python src/experiment.py report
```
Повторный запуск пропускает готовые ответы. Invalid считается ошибкой, ошибки API останавливают запуск.
После изменения моделей, провайдера, промпта или данных архивируйте прежние результаты.

## GitHub
Распакуйте архив и загрузите содержимое папки в репозиторий через **Add file → Upload files**.
Notebook должен быть в корне. После выполнения обновите notebook, изображения, manifest и results.
Токен не публикуйте. Все авторы и лицензии фотографий — в manifest; код лицензирован MIT отдельно от фото.

Особые случаи: фото 09 и 10 показывают дозаправку; классифицируется самый крупный основной самолёт. Фото 21 — гражданская конструкция L-100 у военного оператора: по принятому правилу это civilian. Проверены фото и описания источников до запросов моделей.
