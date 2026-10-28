"""Prepare the primary Treponema image and reproduce compact result figures."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / 'data/raw', ROOT / 'results'
OUT.mkdir(exist_ok=True)
im = cv2.imread(str(RAW / 'treponema_primary.jpg'))
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
mask = (im[:, :, 1] < 55) & (hsv[:, :, 1] > 100)
mask = remove_small_holes(remove_small_objects(mask, max_size=119), max_size=149)
mask = cv2.morphologyEx(mask.astype('uint8'), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
skel = skeletonize(mask > 0)
n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
overlay = im.copy(); geometry = []
for label in range(1, n):
    if stats[label, 4] <= 450: continue
    yy, xx = np.nonzero(labels == label)
    points = np.column_stack([xx, yy]).astype(float)
    center = points.mean(0)
    _, _, vh = np.linalg.svd(points - center, full_matrices=False)
    projection = (points - center) @ vh[0]
    ends = center + np.array([projection.min(), projection.max()])[:, None] * vh[0]
    three = np.vstack([ends[0], center, ends[1]])
    for index, point in enumerate(three):
        cv2.circle(overlay, tuple(point.astype(int)), 6, (0, 255, 255), -1)
        cv2.putText(overlay, str(index), tuple(point.astype(int) + [5, -5]), 0, .55, (255,255,255), 2)
    cv2.line(overlay, tuple(ends[0].astype(int)), tuple(ends[1].astype(int)), (0,255,0), 2)
    geometry.append({'region': len(geometry), 'area_px': int(stats[label,4]),
                     'centroid': center.tolist(), 'three_points': three.tolist()})
cv2.imwrite(str(OUT/'primary_mask.png'), mask*255)
cv2.imwrite(str(OUT/'primary_skeleton.png'), skel.astype('uint8')*255)
cv2.imwrite(str(OUT/'primary_keypoints.jpg'), overlay)

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0,0].imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)); ax[0,0].set_title('Исходное изображение')
ax[0,1].imshow(mask,cmap='gray'); ax[0,1].set_title('Цветовая маска')
ax[1,0].imshow(skel,cmap='gray'); ax[1,0].set_title('Скелет осевой линии')
ax[1,1].imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB)); ax[1,1].set_title('Три точки видимого фрагмента')
for a in ax.flat:a.axis('off')
fig.tight_layout();fig.savefig(OUT/'primary_analysis.png',dpi=180);plt.close(fig)

trackers={'MIL':(.532,39.1),'MOSSE':(.550,3247.3),'KCF':(.497,648.9),'CSRT':(.488,62.7)}
fig,ax=plt.subplots(1,2,figsize=(10,4));names=list(trackers)
ax[0].bar(names,[trackers[x][0] for x in names],color='#6f42c1');ax[0].set(ylabel='Средний IoU',ylim=(0,0.65))
ax[1].bar(names,[trackers[x][1] for x in names],color='#d63384');ax[1].set(ylabel='FPS',yscale='log')
fig.tight_layout();fig.savefig(OUT/'tracking_metrics.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].bar(['Bicubic','Real-ESRGAN'],[25.685,24.410],color=['#198754','#dc3545']);ax[0].set(ylabel='PSNR, dB',ylim=(20,27))
ax[1].bar(['Bicubic','Real-ESRGAN'],[.699,.582],color=['#198754','#dc3545']);ax[1].set(ylabel='SSIM',ylim=(0,0.8))
fig.tight_layout();fig.savefig(OUT/'superresolution_metrics.png',dpi=180);plt.close(fig)

models=['YOLO-World','FastSAM text','CLIP Treponema','CLIPSeg name','DINO+SAM']
values=[0,1,.162,0,6]
fig,ax=plt.subplots(figsize=(9,4));ax.bar(models,values,color='#7b2cbf');ax.set_ylabel('Число объектов / относительная оценка');ax.tick_params(axis='x',rotation=15)
fig.tight_layout();fig.savefig(OUT/'zero_shot_summary.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,4));
ax.bar(['Decision tree','MLP'],[.476,.718],color=['#457b9d','#e76f51']);ax.set(ylabel='Macro-F1 на тесте',ylim=(0,1));
fig.tight_layout();fig.savefig(OUT/'mlflow_models.png',dpi=180);plt.close(fig)

pose={'YOLOv8s':(.392,52.05,.067),'YOLO11s':(.635,42.27,.600)}; pose_names=list(pose)
fig,ax=plt.subplots(1,3,figsize=(11,4));
ax[0].bar(pose_names,[pose[x][0] for x in pose_names]);ax[0].set_title('Pose mAP50–95')
ax[1].bar(pose_names,[pose[x][1] for x in pose_names]);ax[1].set_title('Ошибка точки, px')
ax[2].bar(pose_names,[pose[x][2] for x in pose_names]);ax[2].set_title('PCK@0.1')
fig.tight_layout();fig.savefig(OUT/'pose_comparison.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(1,2,figsize=(9,4));ax[0].bar(['CLIP','ResNet18'],[1,1],color=['#5a189a','#2a9d8f']);ax[0].set(ylabel='Precision@3',ylim=(0,1.1));ax[1].bar(['До обучения','После обучения'],[.99,1.0],color=['#457b9d','#e76f51']);ax[1].set(ylabel='Balanced accuracy',ylim=(0.9,1.01));fig.tight_layout();fig.savefig(OUT/'retrieval_metric.png',dpi=180);plt.close(fig)

video=RAW/'t07.ogv';cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);fig,axes=plt.subplots(2,4,figsize=(11,5))
for axis,sec in zip(axes.flat,[0,12,24,36,48,60,68,74]):
    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,frame=cap.read()
    if ok:axis.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));axis.set_title(f'{sec} с')
    axis.axis('off')
cap.release();fig.tight_layout();fig.savefig(OUT/'video_contact.png',dpi=170);plt.close(fig)

summary={'primary_shape':list(im.shape[:2]),'mask_fraction':float(mask.mean()),'connected_regions':len(geometry),
         'geometry':geometry,'trackers':trackers,'superresolution':{'Bicubic':{'PSNR':25.685,'SSIM':.699},'Real-ESRGAN':{'PSNR':24.410,'SSIM':.582}}}
(OUT/'preparation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
# Contact sheet of the primary illustration and additional open Treponema material.
source_files=[RAW/'treponema_primary.jpg']+[RAW/f't{i:02d}{ext}' for i,ext in [(1,'.jpg'),(2,'.jpg'),(3,'.png'),(4,'.jpg'),(5,'.jpg'),(6,'.jpg')]]
fig,axes=plt.subplots(2,4,figsize=(12,6));
for index,axis in enumerate(axes.flat):
    if index<len(source_files):
        axis.imshow(Image.open(source_files[index]).convert('RGB'));axis.set_title('Основное изображение' if index==0 else f'Открытый источник t{index:02d}')
    axis.axis('off')
fig.tight_layout();fig.savefig(OUT/'sources_contact.png',dpi=160);plt.close(fig)
print(json.dumps({k:v for k,v in summary.items() if k!='geometry'},ensure_ascii=False))
