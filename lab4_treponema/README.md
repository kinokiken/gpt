# Лабораторная работа № 4 — Treponema pallidum

Комплект работы Андреева А. К., группа 4299, по прикладным методам компьютерного зрения на изображениях и видео бледной трепонемы.

В комплект входят:

- основное изображение бледной трепонемы из задания и шесть дополнительных открытых изображений;
- открытый исторический микроскопический фильм Жана Командона (около 1910 года), используемый для поиска в Streamlit;
- подготовка цветовой маски, скелета и трёх ключевых точек видимого фрагмента;
- графики фактически полученных метрик трекинга, сверхразрешения, поиска, MLflow и YOLO-pose;
- локальный CLIP-индексатор видео и Streamlit-интерфейс;
- отдельный код и инструкция для самостоятельной работы с RuTube;
- подробный отчёт с алгоритмами, листингами, входными и выходными данными и рисунками:
  `deliverables/Andreev_Lab4_Treponema_Report.docx` и `.pdf`.

Основное изображение является цветной иллюстрацией без физического масштаба. Шесть связных областей на нём нельзя считать шестью отдельными биологическими клетками. Ключевые точки обозначают начало, середину и конец видимого фрагмента, а не подтверждают вид или клинический диагноз.

## Основное изображение

Файл `data/raw/treponema_primary.jpg` соответствует изображению:
<https://pcr.news/upload/iblock/3a5/3a5ebdd65880badcc702e1bed5c07618.jpg>.

## Проверенный запуск обработки

```bash
python scripts/fetch_open_data.py
python scripts/prepare_and_visualize.py
python scripts/verify_checkpoint.py
```

В контрольном запуске получено изображение 870×540, доля маски 0,225481 и шесть крупных связных областей. Результаты сохранены в `results/`.

## Поиск в видео Streamlit

Локальный поиск использует файл `data/raw/t07.ogv` — открытый фильм Жана Командона о спирохетах сифилиса. Постройте индекс и запустите интерфейс:

```bash
python scripts/video_index.py data/raw/t07.ogv --out data/video_index --step 2
streamlit run scripts/app.py
```

В поле каталога оставьте `data/video_index`. Пример запроса:
`Treponema pallidum spirochetes moving in a microscope field`.

## RuTube

Сетевой запуск RuTube не выполнялся. Готовый код и поля, которые нужно заполнить после самостоятельного запуска, приведены в [RUN_RUTUBE.md](RUN_RUTUBE.md) и в разделе 7.1 отчёта.

## Пересборка отчёта

```bash
python scripts/report_content.py
REPORT_ROOT="$PWD" \
REPORT_BASENAME=Andreev_Lab4_Treponema_Report \
REPORT_HEADING='Распознавание и анализ Treponema pallidum\nна изображениях и видео' \
REPORT_SUBJECT='Компьютерное зрение для изображений и видео бледной трепонемы' \
python ../lab4/scripts/render_report.py
```

## Источники данных

- Основное изображение: <https://pcr.news/upload/iblock/3a5/3a5ebdd65880badcc702e1bed5c07618.jpg>
- Дополнительные изображения и видео: Wikimedia Commons; точные страницы, авторы и лицензии записаны в `data/raw/sources.json`.
- Видео: Jean Comandon, *Ultramicroscope time-lapse of syphilis parasite*, около 1910 года, public domain.
