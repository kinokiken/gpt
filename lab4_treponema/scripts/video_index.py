"""Build and search a local CLIP video index; the module makes no hosting requests."""
from pathlib import Path
import argparse, json, cv2, numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F

def model():
    return (CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True).eval(),
            CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'))
def index_video(video, out, step=2.0, source_url=''):
    out=Path(out); frames=out/'frames'; frames.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video)); fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); net,proc=model(); rows=[]; vectors=[]
    for number in range(0,count,max(1,round(fps*step))):
        cap.set(cv2.CAP_PROP_POS_FRAMES,number);ok,frame=cap.read()
        if not ok:continue
        path=frames/f'frame_{number:06d}.jpg';cv2.imwrite(str(path),frame)
        with torch.no_grad():z=F.normalize(net.get_image_features(**proc(images=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)),return_tensors='pt')),dim=1)[0]
        vectors.append(z.numpy());rows.append({'frame':number,'seconds':number/fps,'path':str(path.relative_to(out)),'source_url':source_url})
    cap.release();np.save(out/'vectors.npy',np.asarray(vectors));(out/'index.json').write_text(json.dumps({'fps':fps,'count':count,'step_s':step,'records':rows},ensure_ascii=False,indent=2));return rows
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--out',default='data/video_index');p.add_argument('--step',type=float,default=2);a=p.parse_args();print('Indexed',len(index_video(a.video,a.out,a.step)))
