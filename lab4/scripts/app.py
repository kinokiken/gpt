"""Search precomputed microscopy frames; RuTube adapter accepts public video IDs."""
from pathlib import Path
import os,json,re
P=Path(__file__).resolve().parents[1];os.environ['HF_HOME']=str(P/'models/hf');os.environ['HF_HUB_DISABLE_XET']='1'
import streamlit as st,numpy as np,torch
from transformers import CLIPModel,CLIPProcessor
from PIL import Image
import torch.nn.functional as F
@st.cache_resource
def load():
 torch.set_num_threads(2)
 return CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True).eval(),CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
def search(query=None,uploaded=None,k=3,gap=6):
 model,processor=load();matrix=np.load(P/'results/video_embeddings.npy');index=json.loads((P/'results/video_index.json').read_text())
 with torch.no_grad():
  if uploaded is not None:z=model.get_image_features(**processor(images=Image.open(uploaded).convert('RGB'),return_tensors='pt'))
  else:z=model.get_text_features(**processor(text=[query],return_tensors='pt',padding=True))
  z=F.normalize(z,dim=1)[0].numpy()
 scores=matrix@z;picked=[]
 for i in np.argsort(-scores):
  if all(abs(index[i]['seconds']-r['seconds'])>=gap for r in picked):picked.append(dict(index[i],score=float(scores[i])))
  if len(picked)==k:break
 return picked
st.set_page_config(page_title='Поиск в микроскопическом видео',layout='wide')
st.title('Поиск момента в микроскопическом видео')
st.write('Открытая запись Alexander Klepnev, CC BY 4.0. Поиск по тексту CLIP лучше работает с английскими описаниями.')
q=st.text_input('Описание кадра',value='thin curved bacteria under a microscope');uploaded=st.file_uploader('Или изображение для поиска',type=['png','jpg','jpeg']);gap=st.slider('Минимальный интервал между результатами, с',0,15,6)
if st.button('Найти'):
 for row in search(q,uploaded,gap=gap):
  st.image(str(P/'data/frames'/Path(row['path']).name),width=480);st.write(f"Время: {row['seconds']:.1f} с; сходство: {row['score']:.3f}");st.link_button('Источник записи',row['source_url'])
