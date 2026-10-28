"""Train and evaluate two pose models on grouped geometric Treponema fragments."""
from experiment_common import *
from ultralytics import YOLO, settings
import shutil

def main():
    settings.update({'mlflow':False})
    data=ROOT/'data/final_pose'
    source=cv2.imread(str(ROOT/'data/raw/treponema_primary.jpg'))
    mask=cv2.imread(str(ROOT/'results/primary_mask.png'),0)
    count,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype('uint8'))
    objects=[i for i in range(1,count) if stats[i,4]>450]
    records=[]
    for index,label in enumerate(objects):
        x,y,w,h,area=map(int,stats[label]); yy,xx=np.nonzero(labels==label)
        kp,_=skeleton_points(labels==label)
        split='train' if index<4 else ('val' if index==4 else 'test')
        crop=source[y:y+h,x:x+w].copy()
        crop[labels[y:y+h,x:x+w]!=label]=114
        for variant in range(8 if split=='train' else 4):
            scale=(130+variant*4)/max(w,h)
            ww,hh=round(w*scale),round(h*scale)
            dx=(192-ww)//2+variant-3;dy=(192-hh)//2
            canvas=np.full((192,192,3),114,np.uint8)
            canvas[dy:dy+hh,dx:dx+ww]=cv2.resize(crop,(ww,hh))
            points=(kp-[x,y])*[ww/w,hh/h]+[dx,dy]
            box=[(dx+ww/2)/192,(dy+hh/2)/192,ww/192,hh/192]
            line=[0,*box]
            for point in points:line.extend([point[0]/192,point[1]/192,2])
            folder=data/'images'/split;folder.mkdir(parents=True,exist_ok=True)
            target=data/'labels'/split;target.mkdir(parents=True,exist_ok=True)
            name=f'object_{index}_variant_{variant}'
            cv2.imwrite(str(folder/(name+'.jpg')),canvas)
            (target/(name+'.txt')).write_text(' '.join(map(str,line))+'\n')
            records.append({'file':str((folder/(name+'.jpg')).relative_to(ROOT)),
                            'object':index,'split':split,'keypoints':points.tolist(),'box':box})
    yaml=data/'dataset.yaml'
    yaml.write_text(f'path: {data}\ntrain: images/train\nval: images/val\ntest: images/test\n'
                    'names: [treponema]\nkpt_shape: [3, 3]\nflip_idx: [2, 1, 0]\n')
    save('pose_dataset.json',records)
    gallery=[rgb(ROOT/r['file']) for r in records if r['file'].endswith('variant_0.jpg')]
    panel('pose_dataset.png',gallery,[f"Object {i}: {'train' if i<4 else ('val' if i==4 else 'test')}" for i in range(6)])
    metrics=[];predictions=[];titles=[]
    for weights,name in [('yolov8s-pose.pt','v8s'),('yolo11s-pose.pt','11s')]:
        net=YOLO(CACHE/weights)
        net.train(data=str(yaml),epochs=20,imgsz=192,batch=8,device='cpu',workers=0,
                  project=str(CACHE/'pose_training'),name=name,exist_ok=True,seed=7,
                  degrees=0,flipud=0,fliplr=.5,mosaic=0,translate=.1,scale=.1,
                  plots=True,verbose=False,optimizer='AdamW',lr0=.001)
        best=YOLO(net.trainer.best)
        result=best.val(data=str(yaml),split='test',imgsz=192,batch=4,device='cpu',workers=0,plots=True)
        errors=[];pck=[];misses=0
        for record in [r for r in records if r['split']=='test']:
            r=best(ROOT/record['file'],conf=.01,imgsz=192,verbose=False)[0]
            predictions.append(cv2.cvtColor(r.plot(),cv2.COLOR_BGR2RGB));titles.append(name+' '+Path(record['file']).stem)
            if not len(r.boxes):misses+=1;continue
            kp=r.keypoints.xy[0].cpu().numpy();truth=np.array(record['keypoints'])
            distance=np.linalg.norm(kp-truth,axis=1)
            errors.extend(distance.tolist())
            pck.extend((distance<.1*np.linalg.norm(truth[2]-truth[0])).tolist())
        metrics.append({'model':name,'epochs':20,'pose_map50':float(result.pose.map50),
                        'pose_map':float(result.pose.map),'box_map':float(result.box.map),
                        'mean_keypoint_error_px':float(np.mean(errors)) if errors else None,
                        'PCK_at_01':float(np.mean(pck)) if pck else None,'missed':misses})
        shutil.copy2(net.trainer.save_dir/'results.csv',OUT/f'pose_{name}_history.csv')
        shutil.copy2(net.trainer.best,OUT/f'pose_{name}_best.pt')
    save('pose_training.json',{'split_unit':'visible component in one illustration',
                             'train_objects':4,'val_objects':1,'test_objects':1,'models':metrics})
    panel('pose_predictions.png',predictions,titles,cols=4)
    import pandas as pd
    fig,ax=plt.subplots(figsize=(8,4))
    for name in ['v8s','11s']:
        frame=pd.read_csv(OUT/f'pose_{name}_history.csv');frame.columns=frame.columns.str.strip()
        ax.plot(frame['epoch'],frame['metrics/mAP50-95(P)'],label=name)
    ax.set(xlabel='Эпоха',ylabel='Validation mAP50–95(P)');ax.legend();fig.tight_layout()
    fig.savefig(OUT/'pose_learning.png',dpi=160);plt.close(fig)
    print(metrics,flush=True)

if __name__=='__main__':
    main()
