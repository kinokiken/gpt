"""Create and search a CLIP index by sequentially reading a local video."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import CLIPModel, CLIPProcessor

def model():
    torch.set_num_threads(2)
    return (CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True).eval(),
            CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32'))

def index_video(video, out, step=2., source_url=''):
    if step <= 0:
        raise ValueError('step must be positive')
    out=Path(out)
    frames=out/'frames'
    frames.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video))
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise ValueError('Video cannot be opened or has invalid FPS')
    net,processor=model()
    stride=max(1,round(fps*step))
    rows=[];vectors=[];number=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:
                break
            if number % stride == 0:
                path=frames/f'frame_{number:06d}.jpg'
                cv2.imwrite(str(path),frame)
                image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                with torch.no_grad():
                    vector=F.normalize(net.get_image_features(**processor(images=image,return_tensors='pt')),dim=1)[0]
                vectors.append(vector.numpy())
                rows.append({'frame':number,'seconds':number/fps,
                             'path':str(path.relative_to(out)),'source_url':source_url})
            number+=1
    finally:
        cap.release()
    if not rows:
        raise ValueError('Video contains no decoded frames')
    np.save(out/'vectors.npy',np.stack(vectors))
    metadata={'fps':fps,'count':number,'step_s':step,'source_url':source_url,'records':rows}
    (out/'index.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows

def search(directory, query, net, processor, image=None, gap=6., k=3):
    directory=Path(directory)
    metadata=json.loads((directory/'index.json').read_text(encoding='utf-8'))
    vectors=np.load(directory/'vectors.npy')
    if vectors.ndim!=2 or len(vectors)!=len(metadata['records']):
        raise ValueError('Index metadata and vectors disagree')
    with torch.no_grad():
        if image is None:
            encoded=net.get_text_features(**processor(text=[query],padding=True,return_tensors='pt'))
        else:
            encoded=net.get_image_features(**processor(images=image.convert('RGB'),return_tensors='pt'))
        vector=F.normalize(encoded,dim=1)[0].numpy()
    scores=vectors@vector
    chosen=[]
    for i in np.argsort(-scores):
        if any(abs(metadata['records'][i]['seconds']-metadata['records'][j]['seconds'])<gap
               or vectors[i]@vectors[j]>.98 for j in chosen):
            continue
        chosen.append(int(i))
        if len(chosen)==k:
            break
    return [dict(metadata['records'][i],cosine=float(scores[i])) for i in chosen]

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('video')
    parser.add_argument('--out',default='data/video_index')
    parser.add_argument('--step',type=float,default=2)
    parser.add_argument('--source-url',default='')
    args=parser.parse_args()
    print('Indexed',len(index_video(args.video,args.out,args.step,args.source_url)))
