# Самостоятельный запуск RuTube

Запустите команды из каталога `lab4_treponema`. Нужны Python, FFmpeg и зависимости из `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python scripts/rutube_download.py "https://rutube.ru/video/ВАШ_32-СИМВОЛЬНЫЙ_ID/" --out data/rutube --step 2
streamlit run scripts/app.py
```

В Streamlit укажите каталог `data/rutube/index`. Скрипт сохраняет ролик и `source.json` в `data/rutube/download`, кадры и `index.json` в `data/rutube/index`, а признаки CLIP — в `vectors.npy`.

Если доступ к загрузке ограничен, сначала сохраните доступный вам ролик вручную, затем выполните:

```bash
python scripts/rutube_download.py "https://rutube.ru/video/ВАШ_32-СИМВОЛЬНЫЙ_ID/" --local-video "C:/Videos/video.mp4" --out data/rutube --step 2
```

При `--local-video` сетевое скачивание ролика не выполняется. После запуска запишите в отчёт название и URL, число проиндексированных кадров, запрос, три найденных времени и оценки сходства, а также добавьте скриншот Streamlit.
