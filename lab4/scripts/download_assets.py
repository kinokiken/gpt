"""Download missing public input assets and control weights over verified HTTPS."""
from pathlib import Path
import json,hashlib,urllib.request,subprocess
P=Path(__file__).resolve().parents[1];D=P/'data/raw';D.mkdir(parents=True,exist_ok=True)
UA='Lab4ComputerVision/1.0 (educational image analysis; https://github.com/kinokiken/gpt)'
def fetch(url,dest):
 if dest.exists():return
 req=urllib.request.Request(url,headers={'User-Agent':UA})
 with urllib.request.urlopen(req,timeout=90) as r:data=r.read()
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);print(dest.name,len(data),flush=True)
for i,row in enumerate(json.loads((D/'sources.json').read_text()),1):
 if row.get('metadata'):fetch(row['metadata']['url'],D/f's{i:02d}.jpg')
video=json.loads((D/'video_metadata.json').read_text());fetch(video['url'],D/'bacteria.webm')
fetch('https://ultralytics.com/images/bus.jpg',D/'bus.jpg')
from PIL import Image
for f in D.glob('s[0-9][0-9].jpg'):
 im=Image.open(f).convert('RGB');im.resize((640,round(im.height*640/im.width))).save(D/(f.stem+'_640.png'))
if not (D/'bacteria_12s.mp4').exists():subprocess.run(['ffmpeg','-y','-i',str(D/'bacteria.webm'),'-t','12','-vf','scale=960:-2','-an','-c:v','libx264','-preset','fast','-crf','20',str(D/'bacteria_12s.mp4')],check=True)
fetch('https://raw.githubusercontent.com/xinntao/ESRGAN/master/RRDBNet_arch.py',P/'models/RRDBNet_arch.py')
fetch('https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth',P/'models/RRDB_ESRGAN_x4.pth')
fetch('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',P/'models/pose_landmarker_lite.task')
print('Input assets ready')
