from pathlib import Path
import os,json,time,sys
P=Path(__file__).resolve().parents[1];R=P/'results';os.environ['YOLO_CONFIG_DIR']=str(P/'models/yolo_config');os.environ['HF_HOME']=str(P/'models/hf');os.environ['HF_HUB_DISABLE_XET']='1';os.environ['TORCH_HOME']=str(P/'models/torch');os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,numpy as np,cv2
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(2);cv2.setNumThreads(2);out={}
from ultralytics import YOLO,YOLOWorld
pose=YOLO(str(P/'models/yolov8n-pose.pt'))
for source in ['bus.jpg','s06_640.png']:
 r=pose.predict(str(P/'data/raw'/source),conf=.25,imgsz=640,device='cpu',verbose=False)[0];r.save(str(R/f'control_pose_{source}.jpg'));out['pose_control_'+source]={'detections':len(r.boxes),'keypoints_per_detection':17}
try:
 import mediapipe as mp
 base=mp.tasks.BaseOptions;options=mp.tasks.vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(P/'models/pose_landmarker_lite.task')),num_poses=4)
 with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
  img=mp.Image.create_from_file(str(P/'data/raw/bus.jpg'));res=detector.detect(img);im=cv2.imread(str(P/'data/raw/bus.jpg'));hh,ww=im.shape[:2]
  for person in res.pose_landmarks:
   for point in person:cv2.circle(im,(round(point.x*ww),round(point.y*hh)),4,(0,255,255),-1)
  cv2.imwrite(str(R/'mediapipe_pose.jpg'),im);out['MediaPipe']={'people':len(res.pose_landmarks),'landmarks_per_person':33}
except Exception as e:out['MediaPipe']={'error':str(e)}
try:
 model=YOLOWorld(str(P/'models/yolov8s-worldv2.pt'));model.set_classes(['bacterium'])
 detects=[]
 for sid in ['s03','s04','s06']:
  for conf in [.1,.25]:
   t=time.perf_counter();r=model.predict(str(P/'data/raw'/f'{sid}_640.png'),conf=conf,imgsz=640,device='cpu',verbose=False)[0];r.save(str(R/f'world_{sid}_{conf}.jpg'));detects.append({'source':sid,'confidence_threshold':conf,'detections':len(r.boxes),'time_s':time.perf_counter()-t,'boxes_xyxy':r.boxes.xyxy.cpu().tolist(),'confidences':r.boxes.conf.cpu().tolist()})
 out['YOLO_World']=detects
except Exception as e:out['YOLO_World']={'error':str(e)}
try:
 from transformers import CLIPSegProcessor,CLIPSegForImageSegmentation
 proc=CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined');model=CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined',use_safetensors=True).eval();image=Image.open(P/'data/raw/s06_640.png').convert('RGB');prompts=['bacteria','rod shaped bacteria','background']
 with torch.no_grad():logits=model(**proc(text=prompts,images=[image]*3,padding=True,return_tensors='pt')).logits
 pred=logits.sigmoid().numpy();fig,axes=plt.subplots(1,4,figsize=(12,3));axes[0].imshow(image);axes[0].set_title('Source');rows=[]
 for i,pr in enumerate(prompts):
  prob=cv2.resize(pred[i],image.size);mask=(prob>=.5).astype('uint8')*255;cv2.imwrite(str(R/f'clipseg_{i}.png'),mask);axes[i+1].imshow(image);axes[i+1].imshow(prob,cmap='magma',alpha=.6,vmin=0,vmax=1);axes[i+1].set_title(pr);rows.append({'prompt':pr,'threshold':.5,'foreground_fraction':float((mask>0).mean())})
 for a in axes:a.axis('off')
 fig.tight_layout();fig.savefig(R/'clipseg.png',dpi=150);plt.close(fig);out['CLIPSeg']=rows
except Exception as e:out['CLIPSeg']={'error':str(e)}
# SR: actual RRDB super-resolution model. Files are from named public repositories.
try:
 sys.path.insert(0,str(P/'models'));from RRDBNet_arch import RRDBNet
 net=RRDBNet(3,3,64,23,gc=32);state=torch.load(P/'models/RRDB_ESRGAN_x4.pth',map_location='cpu',weights_only=True)
 if 'params_ema' in state:state=state['params_ema']
 elif 'params' in state:state=state['params']
 mapping={'body':'RRDB_trunk','conv_body':'trunk_conv','conv_up1':'upconv1','conv_up2':'upconv2','conv_hr':'HRconv'}
 state={'.'.join([mapping.get(k.split('.')[0],k.split('.')[0])]+[part.upper() if part.startswith('rdb') else part for part in k.split('.')[1:]]):v for k,v in state.items()}
 net.load_state_dict(state,strict=True);net.eval();im=cv2.imread(str(R/'sr_lowres.png'));inp=torch.from_numpy(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()[None]/255
 with torch.no_grad():sr=net(inp).clamp(0,1)[0].permute(1,2,0).numpy()
 sr=np.round(sr*255).astype('uint8');cv2.imwrite(str(R/'sr_neural.png'),cv2.cvtColor(sr,cv2.COLOR_RGB2BGR));ref=np.array(Image.open(R/'sr_reference.png').convert('RGB'))
 from skimage.metrics import peak_signal_noise_ratio,structural_similarity
 out['Real_ESRGAN']={'PSNR':float(peak_signal_noise_ratio(ref,sr,data_range=255)),'SSIM':float(structural_similarity(ref,sr,channel_axis=2,data_range=255)),'model_source':'https://huggingface.co/ai-forever/Real-ESRGAN','architecture':'RRDB 23 blocks, x4'}
except Exception as e:out['Real_ESRGAN']={'error':str(e)}
(R/'zero_shot_controls.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
