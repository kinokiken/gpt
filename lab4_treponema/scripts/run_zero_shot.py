"""Measure five zero-shot approaches on the supplied Treponema illustration."""
from experiment_common import *
from PIL import Image
from ultralytics import YOLOWorld, FastSAM, settings
from transformers import (CLIPModel, CLIPProcessor, CLIPSegProcessor,
    CLIPSegForImageSegmentation, AutoProcessor, AutoModelForZeroShotObjectDetection,
    SamModel, SamProcessor)
from torch.nn import functional as F

def main():
    settings.update({'weights_dir':str(CACHE)})
    image=Image.open(ROOT/'data/raw/treponema_primary.jpg').convert('RGB')
    metrics={};figures=[np.array(image)];titles=['Исходное изображение']
    world=YOLOWorld(CACHE/'yolov8s-worldv2.pt')
    world.set_classes(['bacterium','spirochete','Treponema pallidum'])
    result=world(image,conf=.1,imgsz=640,device='cpu',verbose=False)[0]
    metrics['YOLO-World']={'threshold':.1,'boxes':len(result.boxes),
                          'confidence':result.boxes.conf.cpu().tolist()}
    figures.append(cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB));titles.append(f'YOLO-World: {len(result.boxes)} рамок')
    save('zero_actual.json',metrics)
    fast=FastSAM(CACHE/'FastSAM-s.pt')
    result=fast(image,texts='a spiral bacterium',conf=.25,imgsz=640,device='cpu',verbose=False)[0]
    metrics['FastSAM']={'text':'a spiral bacterium','masks':0 if result.masks is None else len(result.masks)}
    figures.append(cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB));titles.append('FastSAM + текст')
    save('zero_actual.json',metrics)
    clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True).eval()
    processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    prompts=['a spiral spirochete bacterium','a rod shaped bacterium','round cocci bacteria','a human person']
    with torch.no_grad():
        a=F.normalize(clip.get_image_features(**processor(images=image,return_tensors='pt')),dim=1)
        b=F.normalize(clip.get_text_features(**processor(text=prompts,padding=True,return_tensors='pt')),dim=1)
    metrics['CLIP']=[{'prompt':p,'cosine':float(s)} for p,s in zip(prompts,(a@b.T)[0])]
    processor=CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
    net=CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined',use_safetensors=True).eval()
    texts=['spiral bacterium','Treponema pallidum']
    with torch.no_grad():maps=net(**processor(text=texts,images=[image]*2,padding=True,return_tensors='pt')).logits.sigmoid().numpy()
    metrics['CLIPSeg']=[]
    for text,prob in zip(texts,maps):
        prob=cv2.resize(prob,image.size)
        metrics['CLIPSeg'].append({'prompt':text,'mask_fraction':float((prob>.5).mean())})
        heat=cv2.applyColorMap((prob*255).astype('uint8'),cv2.COLORMAP_JET)
        overlay=cv2.addWeighted(np.array(image),.55,cv2.cvtColor(heat,cv2.COLOR_BGR2RGB),.45,0)
        figures.append(overlay);titles.append('CLIPSeg: '+text)
    save('zero_actual.json',metrics)
    processor=AutoProcessor.from_pretrained('IDEA-Research/grounding-dino-tiny')
    net=AutoModelForZeroShotObjectDetection.from_pretrained('IDEA-Research/grounding-dino-tiny',use_safetensors=True).eval()
    inputs=processor(images=image,text='a spiral bacterium. a spirochete.',return_tensors='pt')
    with torch.no_grad():outputs=net(**inputs)
    result=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,
             box_threshold=.20,text_threshold=.20,target_sizes=[image.size[::-1]])[0]
    boxes=result['boxes'].tolist()
    metrics['Grounding DINO']={'threshold':.20,'boxes':boxes,'scores':result['scores'].tolist()}
    shown=np.array(image)
    for box in boxes:
        x1,y1,x2,y2=map(int,box);cv2.rectangle(shown,(x1,y1),(x2,y2),(0,255,0),2)
    figures.append(shown);titles.append(f'Grounding DINO: {len(boxes)} рамок')
    if boxes:
        processor=SamProcessor.from_pretrained('facebook/sam-vit-base')
        net=SamModel.from_pretrained('facebook/sam-vit-base',use_safetensors=True).eval()
        inputs=processor(image,input_boxes=[boxes],return_tensors='pt')
        with torch.no_grad():outputs=net(**inputs,multimask_output=False)
        masks=processor.image_processor.post_process_masks(outputs.pred_masks,inputs['original_sizes'],
                                                  inputs['reshaped_input_sizes'])[0][:,0].numpy()
        shown=np.array(image)
        union=masks.any(axis=0)
        shown[union]=(shown[union]*.5+np.array([0,255,0])*.5).astype('uint8')
        figures.append(shown);titles.append('Grounding DINO → SAM')
        metrics['SAM']={'masks':len(masks),'foreground_fraction':float(union.mean())}
    else:
        metrics['SAM']={'masks':0,'input_boxes':0}
    panel('zero_actual.png',figures,titles,cols=2)
    save('zero_actual.json',metrics)
    print(metrics,flush=True)

if __name__=='__main__':
    main()
