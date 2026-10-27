from pathlib import Path
import os,json,xml.etree.ElementTree as E,zipfile,numpy as np
P=Path(__file__).resolve().parents[1];os.environ['YOLO_CONFIG_DIR']=str(P/'models/yolo_config')
from ultralytics import YOLO
import torch;torch.set_num_threads(2)
model=YOLO(str(P/'results/pose/train/weights/best.pt'));files=sorted((P/'data/pose/images/test').glob('*.jpg'));D=P/'results/cvat';D.mkdir(exist_ok=True)
root=E.Element('annotations');E.SubElement(root,'version').text='1.1';meta=E.SubElement(root,'meta');task=E.SubElement(meta,'task');E.SubElement(task,'name').text='Andreev_Lab4_pose_predictions';E.SubElement(task,'size').text=str(len(files));E.SubElement(task,'mode').text='annotation';labs=E.SubElement(task,'labels')
labels=[{'name':'bacterium','type':'rectangle','color':'#00ff00','attributes':[]}]+[{'name':f'p{i}','type':'points','color':'#ff0000','attributes':[]} for i in range(1,6)]
for lab in labels:
 node=E.SubElement(labs,'label');E.SubElement(node,'name').text=lab['name'];E.SubElement(node,'color').text=lab['color'];E.SubElement(node,'type').text=lab['type'];E.SubElement(node,'attributes')
nobj=0
for i,f in enumerate(files):
 imnode=E.SubElement(root,'image',id=str(i),name=f.name,width='256',height='256');r=model.predict(str(f),conf=.25,imgsz=256,device='cpu',verbose=False)[0]
 for j,(box,pts) in enumerate(zip(r.boxes.xyxy.cpu().numpy(),r.keypoints.xy.cpu().numpy())):
  group=str(i*100+j+1);x1,y1,x2,y2=np.clip(box,0,256);E.SubElement(imnode,'box',label='bacterium',source='auto',occluded='0',xtl=f'{x1:.3f}',ytl=f'{y1:.3f}',xbr=f'{x2:.3f}',ybr=f'{y2:.3f}',z_order='0',group_id=group)
  for k,(x,y) in enumerate(pts,1):
   x,y=np.clip([x,y],0,256);E.SubElement(imnode,'points',label=f'p{k}',source='auto',occluded='0',points=f'{x:.3f},{y:.3f}',z_order='0',group_id=group)
  nobj+=1
E.indent(root);E.ElementTree(root).write(D/'annotations.xml',encoding='utf-8',xml_declaration=True);(D/'labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2))
with zipfile.ZipFile(D/'cvat_annotations.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(D/'annotations.xml','annotations.xml')
with zipfile.ZipFile(D/'cvat_task_images.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,f.name)
# Validate parsed image names and all exported coordinate ranges.
rr=E.parse(D/'annotations.xml').getroot();assert len(rr.findall('image'))==len(files)
for node in rr.findall('image'):
 assert node.attrib['name'] in [f.name for f in files]
 for a in node.findall('points'):
  x,y=map(float,a.attrib['points'].split(','));assert 0<=x<=256 and 0<=y<=256
summary={'images':len(files),'predicted_objects':nobj,'keypoint_shapes':nobj*5,'format':'CVAT for images 1.1; grouped box and five point labels','xml_parsing_and_coordinates_validated':True,'online_import':'not executed; no authenticated CVAT session','human_review':'preannotations must be reviewed in CVAT'};(P/'results/cvat_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(summary)
