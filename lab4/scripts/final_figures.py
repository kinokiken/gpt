from pathlib import Path
import json,os,csv,numpy as np,cv2
from PIL import Image,ImageDraw
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];R=P/'results'
def panel(paths,titles,name,width=1200,cols=3,cellh=330):
 rows=(len(paths)+cols-1)//cols;out=Image.new('RGB',(width,rows*cellh),'white');draw=ImageDraw.Draw(out);cw=width//cols
 for i,(f,title) in enumerate(zip(paths,titles)):
  im=Image.open(f).convert('RGB');im.thumbnail((cw-15,cellh-40));x=i%cols*cw+(cw-im.width)//2;y=i//cols*cellh+30;out.paste(im,(x,y));draw.text((i%cols*cw+8,i//cols*cellh+8),title,fill='black')
 out.save(R/name)
panel([R/'sr_reference.png',R/'sr_lowres.png',R/'sr_neural.png'],['Reference 256x256','Input 64x64','Real-ESRGAN x4'],'sr_comparison.png')
panel([R/f'track_{s}_74.jpg' for s in ['MIL','KCF','CSRT']],['MIL / frame 74','KCF / frame 74','CSRT / frame 74'],'track_comparison.png',cellh=280)
panel([R/'calibration_before.jpg',R/'calibration_after.jpg'],['Before','Undistorted'],'calibration_comparison.png',cols=2,cellh=470)
panel([R/f'mog2_{i}.png' for i in [0,25,74]],['Frame 0','Frame 25','Frame 74'],'mog2_comparison.png',cellh=280)
panel([R/'control_pose_bus.jpg.jpg',R/'mediapipe_pose.jpg',R/'control_pose_s06_640.png.jpg'],['YOLOv8n pose / humans','MediaPipe / human','Human model / bacteria'],'human_pose_controls.png',cellh=530)
files=sorted((R/'pose/predictions').glob('*.jpg'));panel(files,[f.stem for f in files],'pose_predictions.png',cols=4,cellh=300)
files=sorted((P/'data/pose/images/train').glob('*.jpg'))[::4];panel(files,[f.stem for f in files],'pose_dataset.png',cols=3,cellh=300)
# Video with actual tracker outputs, identical colour code across all frames.
rows=list(csv.DictReader(open(R/'tracking.csv')));index={(r['tracker'],int(r['frame'])):r for r in rows};cap=cv2.VideoCapture(str(P/'data/raw/bacteria_12s.mp4'));writer=cv2.VideoWriter(str(R/'tracking_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(960,540));colors={'MIL':(0,255,255),'KCF':(255,100,0),'CSRT':(0,255,0)}
for fi in range(75):
 ok,frame=cap.read()
 if not ok:break
 for name,col in colors.items():
  r=index.get((name,fi));b=(785,342,23,45) if fi==0 else tuple(float(r[k]) for k in ['x','y','w','h']) if r and int(r['ok']) else None
  if b:x,y,w,h=map(round,b);cv2.rectangle(frame,(x,y),(x+w,y+h),col,2)
  cv2.putText(frame,name+(' LOST' if b is None else ''),(15,30+list(colors).index(name)*25),cv2.FONT_HERSHEY_SIMPLEX,.6,col,2)
 cv2.putText(frame,f't = {fi/25:.2f}s',(15,112),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);writer.write(frame)
cap.release();writer.release()
# Plain table exported from MLflow data, explicitly not a UI screenshot.
m=json.loads((R/'mlflow_summary.json').read_text());fig,ax=plt.subplots(figsize=(10,3.6));ax.axis('off');rows=[[r['name'],r['run_id'][:12],str(r.get('val_accuracy','-'))] for r in m['runs']];tab=ax.table(cellText=rows,colLabels=['Run','Run ID prefix','Validation accuracy'],cellLoc='left',loc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.6);fig.tight_layout();fig.savefig(R/'mlflow_runs.png',dpi=150);plt.close(fig)
