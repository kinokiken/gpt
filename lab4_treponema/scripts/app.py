"""Streamlit search over a prepared Treponema pallidum video index."""
from pathlib import Path
import json, numpy as np, streamlit as st, torch
from PIL import Image
from video_index import model
import torch.nn.functional as F
@st.cache_resource
def get_model():return model()
st.set_page_config(page_title='Поиск Treponema pallidum в видео', layout='wide')
st.title('Поиск кадров Treponema pallidum')
st.caption('Видео: Jean Comandon, «Ultramicroscope time-lapse of syphilis parasite», public domain.')
directory=Path(st.text_input('Каталог индекса','data/video_index'))
query=st.text_input('Запрос','Treponema pallidum spirochetes moving in a microscope field')
if st.button('Найти'):
    index_path, vectors_path = directory/'index.json', directory/'vectors.npy'
    if not index_path.exists() or not vectors_path.exists():
        st.error('Индекс не найден. Сначала выполните: python scripts/video_index.py data/raw/t07.ogv --out data/video_index --step 2')
        st.stop()
    metadata=json.loads(index_path.read_text(encoding='utf-8'));vectors=np.load(vectors_path);net,proc=get_model()
    with torch.no_grad():q=F.normalize(net.get_text_features(**proc(text=[query],return_tensors='pt',padding=True)),dim=1)[0].numpy()
    for i in np.argsort(-(vectors@q))[:3]:
        row=metadata['records'][i]
        st.image(str(directory/row['path']),width=480)
        st.write(f"Время: {row['seconds']:.2f} с; сходство: {(vectors@q)[i]:.3f}")
        if row.get('source_url'):
            st.link_button('Источник видео', row['source_url'])
