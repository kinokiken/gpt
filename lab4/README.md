# Скачать материалы

**Андреев А. К., группа 4299 — лабораторная работа № 4.**

- [Отчёт Word — DOCX](deliverables/Andreev_Lab4_Report.docx?raw=true)
- [Отчёт PDF — 42 страницы](deliverables/Andreev_Lab4_Report.pdf)
- [Архив кода, моделей и результатов](deliverables/Andreev_Lab4_Code_Results.zip?raw=true)
- [Архив исходных данных](deliverables/Andreev_Lab4_Input_Data.zip?raw=true)
- [Видео сравнения трекеров](deliverables/Andreev_Lab4_Tracking.mp4?raw=true)
- [Предварительные аннотации CVAT](deliverables/Andreev_Lab4_CVAT_Annotations.zip?raw=true)
- [Изображения для задачи CVAT](deliverables/Andreev_Lab4_CVAT_Images.zip?raw=true)

Если GitHub открывает страницу файла, нажмите **Download raw file**. Два основных архива распакуйте в одну папку. Код и данные доступны также без архивов в каталогах ниже.

Состояние: эксперименты выполнены на открытых данных. Для завершения индивидуального варианта нужны исходный JPG/PNG пользователя, онлайн-проверка CVAT и доступ к RuTube. Эти ограничения отражены в отчёте.

---

# Лабораторная работа № 4 — Андреев А. К., группа 4299

Выполненные вычислительные эксперименты на открытых данных. Отчёт: `deliverables/Andreev_Lab4_Report.docx` и `.pdf`.

## Что остаётся для индивидуального варианта

1. Добавить исходное фото пользователя в JPG/PNG и выполнить обработку именно этого файла. Картинка из переписки в вычисления не входила; вид бактерии не установлен.
2. Войти в онлайн CVAT, импортировать локальные предварительные аннотации, проверить и исправить их. Наличие валидного XML не подтверждает импорт на сервер.
3. Проверить RuTube в доступной среде: два публичных endpoint вернули HTTP 403. Поисковый механизм и Streamlit функционально проверены на видео Wikimedia Commons.

## Что выполнено

- Девять публичных снимков с авторством и лицензиями, 45 вырезок; разделение по исходным снимкам.
- 33 изолированных учебных примера для пяти точек, обучение YOLOv8n-pose 30 эпох, тест на 7 фрагментах. Это упрощённые изображения с созданным серым фоном, не естественный датасет сцен.
- MIL/KCF/CSRT на реальном видео, 5 ручных проверочных рамок после инициализации; Farneback и Lucas–Kanade.
- Калибровка отдельной камеры OpenCV (10/3 снимка), Real-ESRGAN/интерполяция, Tesseract OCR.
- CLIP-поиск, triplet-проекция, YOLO-World и CLIPSeg, 27 кадров поискового индекса.
- CVAT XML, 7 рамок и 35 точечных фигур, проверка структуры и координат.
- MLflow: 7 запусков, 3 версии моделей, проверенная повторная загрузка Scikit-learn/PyTorch; локальный HTTP API вернул 200.
- Streamlit AppTest: загрузка приложения, нажатие кнопки, три результата без исключений.

## Воспроизведение

Нужен Python 3.12, FFmpeg и Tesseract с английским языковым пакетом. На Linux MediaPipe может требовать `libgles2`. Создайте venv, затем установите зависимости из `requirements.txt`. Для CLIP-пакета YOLO-World используется официальный форк Ultralytics.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Ultralytics может подтянуть второй поставщик cv2. Оставьте только contrib:
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless opencv-contrib-python
pip install opencv-contrib-python==4.11.0.86
python scripts/download_assets.py
python scripts/prepare_data.py
python scripts/download_models.py
python scripts/vision_baselines.py
python scripts/calibrate_camera.py
python scripts/train_pose.py
python scripts/semantic_experiments.py
python scripts/zero_shot_and_controls.py
python scripts/export_cvat.py
python scripts/log_experiments.py
streamlit run scripts/app.py
```

В Windows активация: `.venv\Scripts\activate`. При переносе папки запускайте `prepare_data.py`: он пересоздаёт абсолютные пути в метаданных и YAML. Скрипты находят корень проекта относительно собственного расположения. Индекс видео пересоздаётся `semantic_experiments.py`. `download_assets.py` получает недостающий полный оригинал видео и контрольные веса из открытых источников; тяжёлые предобученные модели не включены в архив. Включены обученная `best.pt` и `metric_head.pt`.

`zero_shot_and_controls.py` сохраняет отрицательные результаты и диагностические ошибки, поэтому проверяйте `results/zero_shot_controls.json`. На текущей машине дополнительно выполнены отдельные `superresolution.py` и `mediapipe_control.py` после исправления совместимости RRDB и установки системной GLES-библиотеки. Финальные JSON содержат реально проверенные результаты.

Для отчёта: `report_content.py`, затем `render_report.py`. PDF требует DejaVu в `/usr/share/fonts/truetype/dejavu`; этот путь следует изменить на своей ОС. Код полного проекта предоставляется отдельно от отчёта, чтобы не включать сотни строк в основной текст.

## Разметка и методические ограничения

`prepare_data.py` содержит визуально заданные геометрические точки. Приближённая полоса вокруг оси используется только для создания изолированной обучающей вырезки; это не экспертная сегментация клеточной мембраны. Test и validation состоят из источников, не входящих в train. Высокий mAP на маленьких изолированных фрагментах не означает готовности к произвольной микроскопии: на полном снимке модель дала лишние рамки.

Для CVAT создайте метки из `results/cvat/labels.json`, загрузите `cvat_task_images.zip`, затем `cvat_annotations.zip` как CVAT for images 1.1. Точки p1–p5 и рамка объединены group_id; это не native skeleton. Проверьте их вручную.

Числа трекеров относятся к одному объекту и 3 секундам видео. Общая доля маски не является точностью. Верификация 36 пар использует повторяющиеся фрагменты в разных парах, поэтому пары статистически зависимы. Синтетический аудиоконтроль явно отделён от видео без аудиодорожки.

## Источники и лицензии

Авторы снимков и лицензии приведены в отчёте и `data/raw/sources.json`. Видео: Alexander Klepnev, CC BY 4.0; изменение — масштабирование и наложение результатов. Изображения NIAID: CC BY 2.0; другие выбранные снимки CDC опубликованы как public domain согласно Commons. Сохраняйте указание авторов при распространении. Материалы одногруппника использованы только для структуры; его PDF и методичка не включены в архив. Предобученные модели имеют собственные лицензии, указанные в соответствующих репозиториях. `models/RRDBNet_arch.py` — исходник проекта xinntao/ESRGAN.
