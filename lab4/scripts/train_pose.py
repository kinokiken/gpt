from pathlib import Path
import os,json,time
P=Path(__file__).resolve().parents[1];os.environ['YOLO_CONFIG_DIR']=str(P/'models/yolo_config');os.environ['OMP_NUM_THREADS']='3';os.environ['MKL_NUM_THREADS']='3'
import torch;torch.set_num_threads(3)
from ultralytics import YOLO,settings
settings.update({'mlflow':False,'sync':False})
model=YOLO(str(P/'models/yolov8n-pose.pt'))
t=time.perf_counter()
model.train(data=str(P/'data/pose/dataset.yaml'),epochs=30,imgsz=256,batch=8,workers=0,device='cpu',project=str(P/'results/pose'),name='train',exist_ok=True,seed=42,deterministic=True,pretrained=True,optimizer='AdamW',lr0=.001,patience=12,mosaic=0.0,mixup=0.0,fliplr=0.0,flipud=0.0,degrees=0.0,translate=.08,scale=.2,plots=True)
best=YOLO(str(P/'results/pose/train/weights/best.pt'))
r=best.val(data=str(P/'data/pose/dataset.yaml'),split='test',imgsz=256,batch=4,workers=0,device='cpu',project=str(P/'results/pose'),name='test',exist_ok=True,plots=True)
metrics={k:float(v) for k,v in r.results_dict.items()};metrics['elapsed_s']=time.perf_counter()-t
(P/'results/pose_metrics.json').write_text(json.dumps(metrics,indent=2));print(metrics,flush=True)
best.predict(source=str(P/'data/pose/images/test'),conf=.05,imgsz=256,device='cpu',save=True,project=str(P/'results/pose'),name='predictions',exist_ok=True)
# Probe on full scenes. This is a shift from isolated teaching crops.
best.predict(source=str(P/'data/raw/s04_640.png'),conf=.1,imgsz=640,device='cpu',save=True,project=str(P/'results/pose'),name='full_scene',exist_ok=True)
