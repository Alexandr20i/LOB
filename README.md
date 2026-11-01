# LOB — распознавание стакана из видео

Полуавтоматический пайплайн для извлечения данных стакана (Limit Order Book) из видео и стриминга с записью в Parquet.

## Установка

    uv sync

## Быстрый старт

Всё сразу одной командой — калибровка и обучение выполнятся только если файлов ещё нет:

    uv run python scripts/run.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1

Чтобы принудительно перекалибровать и переобучить, добавьте флаги:

    uv run python scripts/run.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --recalibrate --retrain

## Пошаговый запуск

### 1. Калибровка

Если layout уже есть и подходит — пропустите этот шаг.

    uv run python scripts/calibrate.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp

Обведите стакан мышью и нажмите **Enter**. Проверьте результат в `configs/layouts/metascalp_overlay.png`.

### 2. Обучение шаблонов цифр

    uv run python scripts/train_glyphs.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1

На вопрос введите цену самой верхней строки с `configs/glyphs/train_frame.png`.

### 3. Запуск стриминга

Базовый запуск:

    uv run python scripts/stream_lob.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1

С окном видео и разметкой (выход по **q**):

    uv run python scripts/stream_lob.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --show-video

С записью данных в Parquet:

    uv run python scripts/stream_lob.py MetaScalp_2026-10-30_15-45-22.mp4 --terminal metascalp --tick 0.1 --save out/session1


## Полезные замечания

- Остановить стриминг можно в любой момент через **Ctrl+C**.
- Печатать каждое изменение стакана, а не раз в 250 мс, помогает флаг `--print-interval-ms 0`.
