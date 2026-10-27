import os,pathlib,json,time
P=pathlib.Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(P/'models/hf');os.environ['YOLO_CONFIG_DIR']=str(P/'models/yolo_config');os.environ['HF_HUB_DISABLE_XET']='1';os.environ['TORCH_HOME']=str(P/'models/torch')
import torch;torch.set_num_threads(3)
from transformers import CLIPModel,CLIPProcessor
print('Downloading CLIP',flush=True)
m=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True);p=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32');print('CLIP ready',flush=True)
from ultralytics import YOLO
os.chdir(P/'models')
for f in ['yolov8n-pose.pt','yolov8s-worldv2.pt']:
 try: m=YOLO(f);print(f,'ready',flush=True)
 except Exception as e: print(f,'ERROR',str(e),flush=True)
