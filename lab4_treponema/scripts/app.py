"""Streamlit search by text or example image in a prepared local index."""
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
import streamlit as st
from PIL import Image
from video_index import model, search

@st.cache_resource
def get_model():
    return model()

st.set_page_config(page_title='Поиск кадров в микроскопическом видео',layout='wide')
st.title('Поиск кадров в микроскопическом видео')
directory=Path(st.text_input('Каталог индекса','data/video_index'))
query=st.text_input('Запрос','Treponema pallidum spirochetes moving in a microscope field')
uploaded=st.file_uploader('Или изображение для поиска',type=['jpg','jpeg','png'])
gap=st.slider('Минимальный интервал между результатами, с',0,15,6)
if st.button('Найти'):
    if not (directory/'index.json').exists() or not (directory/'vectors.npy').exists():
        st.error('Индекс не найден. Укажите каталог с index.json и vectors.npy.')
        st.stop()
    net,processor=get_model()
    image=Image.open(uploaded).convert('RGB') if uploaded is not None else None
    hits=search(directory,query,net,processor,image=image,gap=gap)
    if not hits:
        st.info('Для заданных условий подходящие кадры отсутствуют.')
    for row in hits:
        st.image(str(directory/row['path']),width=480)
        st.write(f"Время: {row['seconds']:.2f} с; сходство: {row['cosine']:.3f}")
        if row.get('source_url'):
            parts=urlsplit(row['source_url'])
            if parts.hostname in {'rutube.ru','www.rutube.ru'}:
                params=dict(parse_qsl(parts.query));params['t']=str(int(row['seconds']))
                url=urlunsplit((parts.scheme,parts.netloc,parts.path,urlencode(params),parts.fragment))
            else:
                url=row['source_url']
            st.link_button('Открыть источник',url)
