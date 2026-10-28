"""Streamlit search over a prepared local index."""
from pathlib import Path
import json, numpy as np, streamlit as st, torch
from PIL import Image
from video_index import model
import torch.nn.functional as F
@st.cache_resource
def get_model():return model()
st.title('Поиск кадров Treponema pallidum')
directory=Path(st.text_input('Каталог индекса','data/video_index'));query=st.text_input('Запрос','spiral spirochetes moving among blood cells')
if st.button('Найти'):
    metadata=json.loads((directory/'index.json').read_text());vectors=np.load(directory/'vectors.npy');net,proc=get_model()
    with torch.no_grad():q=F.normalize(net.get_text_features(**proc(text=[query],return_tensors='pt',padding=True)),dim=1)[0].numpy()
    for i in np.argsort(-(vectors@q))[:3]:
        row=metadata['records'][i];st.image(str(directory/row['path']),width=480);st.write(f"Время: {row['seconds']:.2f} с; сходство: {(vectors@q)[i]:.3f}")
