from pathlib import Path
import os,json,time,subprocess,csv
import cv2,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
from scipy.signal import butter,sosfilt
from scipy.io import wavfile
P=Path(__file__).resolve().parents[1];R=P/'results';cv2.setNumThreads(2)
metrics={};im=cv2.imread(str(P/'data/raw/s06_640.png'));gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
variants={'Original':im,'Gaussian 5x5':cv2.GaussianBlur(im,(5,5),0),'Median 5x5':cv2.medianBlur(im,5),'Canny':cv2.Canny(gray,60,150)}
fig,ax=plt.subplots(2,2,figsize=(10,7))
for a,(n,v) in zip(ax.ravel(),variants.items()):a.imshow(cv2.cvtColor(v,cv2.COLOR_BGR2RGB) if v.ndim==3 else v,cmap='gray');a.set_title(n);a.axis('off')
fig.tight_layout();fig.savefig(R/'filters.png',dpi=150);plt.close(fig)
hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,(100,70,30),(155,255,255));mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));cv2.imwrite(str(R/'color_mask.png'),mask)
n,lab,stats,centroids=cv2.connectedComponentsWithStats(mask)
metrics['color_segmentation']={'foreground_fraction':float((mask>0).mean()),'components_over_100px':int((stats[1:,4]>100).sum()),'note':'Connected regions merge touching cells; not a bacteria count.'}
# Downsample then reconstruct to evaluate information loss against a genuine reference.
ref=im[64:320,128:384];small=cv2.resize(ref,(64,64),interpolation=cv2.INTER_AREA);scores={}
fig,ax=plt.subplots(1,4,figsize=(12,3));ax[0].imshow(cv2.cvtColor(ref,cv2.COLOR_BGR2RGB));ax[0].set_title('Reference')
for a,(name,method) in zip(ax[1:],[('Nearest',cv2.INTER_NEAREST),('Bilinear',cv2.INTER_LINEAR),('Bicubic',cv2.INTER_CUBIC)]):
 out=cv2.resize(small,(256,256),interpolation=method);scores[name]={'PSNR':float(peak_signal_noise_ratio(ref,out,data_range=255)),'SSIM':float(structural_similarity(ref,out,channel_axis=2,data_range=255))};a.imshow(cv2.cvtColor(out,cv2.COLOR_BGR2RGB));a.set_title(f'{name}\nPSNR={scores[name]["PSNR"]:.2f}')
for a in ax:a.axis('off')
fig.tight_layout();fig.savefig(R/'resolution.png',dpi=150);plt.close(fig);metrics['resolution']=scores
cv2.imwrite(str(R/'sr_reference.png'),ref);cv2.imwrite(str(R/'sr_lowres.png'),small)
cap=cv2.VideoCapture(str(P/'data/raw/bacteria_12s.mp4'));frames=[]
while len(frames)<75:
 ok,f=cap.read()
 if not ok:break
 frames.append(f)
cap.release();h,w=frames[0].shape[:2]
# Sparse manual boxes around one visually persistent curved bacterium.
gt={0:(785,342,23,45),15:(787,341,25,44),30:(784,340,25,44),45:(789,343,24,43),60:(787,345,26,45),74:(788,347,26,45)}
def iou(a,b):
 x,y,ww,hh=a;xx,yy,www,hhh=b;inter=max(0,min(x+ww,xx+www)-max(x,xx))*max(0,min(y+hh,yy+hhh)-max(y,yy));u=ww*hh+www*hhh-inter;return inter/u if u>0 else 0
track={};allrows=[]
for name,ctor in [('MIL',cv2.TrackerMIL_create),('KCF',cv2.TrackerKCF_create),('CSRT',cv2.TrackerCSRT_create)]:
 tracker=ctor();tracker.init(frames[0],gt[0]);boxes=[gt[0]];oklist=[];times=[]
 for fi,frame in enumerate(frames[1:],1):
  t=time.perf_counter();ok,b=tracker.update(frame);times.append(time.perf_counter()-t);oklist.append(bool(ok));boxes.append(tuple(b) if ok else None)
  allrows.append({'tracker':name,'frame':fi,'seconds':fi/25,'ok':int(ok),'x':float(b[0]),'y':float(b[1]),'w':float(b[2]),'h':float(b[3])})
 vals=[iou(boxes[f],b) if boxes[f] else 0 for f,b in gt.items() if f>0]
 track[name]={'mean_IoU_sparse':float(np.mean(vals)),'success_IoU_05_sparse':float(np.mean(np.array(vals)>.5)),'valid_fraction':float(np.mean(oklist)),'FPS':1/float(np.mean(times)),'reference_frames_excluding_init':5,'ious':vals}
 print(name,track[name],flush=True)
 for fi in [0,30,74]:
  f=frames[fi].copy();b=boxes[fi]
  if b:
   x,y,bw,bh=map(int,b);cv2.rectangle(f,(x,y),(x+bw,y+bh),(0,255,0),2)
  cv2.imwrite(str(R/f'track_{name}_{fi}.jpg'),f)
metrics['tracking']=track
with (R/'tracking.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=allrows[0].keys());wr.writeheader();wr.writerows(allrows)
(R/'tracking_reference.json').write_text(json.dumps(gt,indent=2))
p=cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY);n=cv2.cvtColor(frames[1],cv2.COLOR_BGR2GRAY)
flow=cv2.calcOpticalFlowFarneback(p,n,None,.5,3,15,3,5,1.2,0)
roi=np.zeros_like(p);roi[342:387,785:808]=255
p0=cv2.goodFeaturesToTrack(p,120,.01,8,mask=roi);sparse={}
if p0 is not None:
 p1,st,_=cv2.calcOpticalFlowPyrLK(p,n,p0,None);pback,stback,_=cv2.calcOpticalFlowPyrLK(n,p,p1,None);ok=(st.ravel()==1)&(stback.ravel()==1)&(np.linalg.norm((pback-p0).reshape(-1,2),axis=1)<1);d=(p1-p0).reshape(-1,2)[ok]
 sparse={'points_detected':len(p0),'points_accepted':int(ok.sum()),'mean_vector_px':d.mean(0).tolist() if len(d) else None}
mean=flow[roi>0].mean(0);metrics['flow']={'Farneback_mean_vector_px':mean.tolist(),'ROI_speed_px_s':float(np.linalg.norm(mean)*25),'Lucas_Kanade':sparse,'note':'No dense ground-truth flow; ROI includes local background.'}
mag,ang=cv2.cartToPolar(flow[:,:,0],flow[:,:,1]);vis=np.zeros_like(frames[0]);vis[:,:,0]=ang*180/np.pi/2;vis[:,:,1]=255;vis[:,:,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX);cv2.imwrite(str(R/'flow.png'),cv2.cvtColor(vis,cv2.COLOR_HSV2BGR))
f=frames[0].copy()
for yy in range(20,h-30,20):
 for xx in range(20,w-20,20):
  dx,dy=flow[yy,xx];cv2.arrowedLine(f,(xx,yy),(round(xx+dx*12),round(yy+dy*12)),(0,255,0),1,tipLength=.3)
cv2.imwrite(str(R/'flow_arrows.jpg'),f)
# Real text from original video scale bar; no invented OCR observation.
ocr=frames[0][492:536,10:115];ocr=cv2.resize(ocr,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC);cv2.imwrite(str(R/'ocr_input.png'),ocr)
ocr_rows=[]
for variant in ['color','threshold']:
 a=ocr if variant=='color' else cv2.threshold(cv2.cvtColor(ocr,cv2.COLOR_BGR2GRAY),170,255,cv2.THRESH_BINARY)[1];f=R/f'ocr_{variant}.png';cv2.imwrite(str(f),a)
 r=subprocess.run(['tesseract',str(f),'stdout','--psm','7','-l','eng'],capture_output=True,text=True);ocr_rows.append({'variant':variant,'returncode':r.returncode,'text':r.stdout.strip(),'reference':'30 μm'})
metrics['OCR']=ocr_rows
back=cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=16,detectShadows=False);fractions=[]
for i,f in enumerate(frames):
 m=back.apply(f,learningRate=.02);m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));fractions.append(float((m>0).mean()))
 if i in [0,25,74]:cv2.imwrite(str(R/f'mog2_{i}.png'),m)
metrics['MOG2']={'foreground_fraction_frame0':fractions[0],'foreground_fraction_frame25':fractions[25],'foreground_fraction_frame74':fractions[74],'note':'Motion mask, not semantic segmentation.'}
# Explicitly synthetic audio control: the microscopy video has no audio track.
sr=16000;t=np.arange(sr)/sr;clean=np.sin(2*np.pi*440*t);noise=.4*np.sin(2*np.pi*4000*t);sig=clean+noise;out=sosfilt(butter(4,1200,fs=sr,output='sos'),sig)
wavfile.write(str(R/'audio_synthetic_before.wav'),sr,(sig/1.4*.9*32767).astype('int16'));wavfile.write(str(R/'audio_synthetic_after.wav'),sr,(out/1.4*.9*32767).astype('int16'))
freq=np.fft.rfftfreq(sr,1/sr);fig,ax=plt.subplots(figsize=(8,3));ax.plot(freq,20*np.log10(np.maximum(abs(np.fft.rfft(sig)),1e-8)),label='Before');ax.plot(freq,20*np.log10(np.maximum(abs(np.fft.rfft(out)),1e-8)),label='Low-pass 1200 Hz');ax.set(xlim=(0,5000),ylim=(-30,90),xlabel='Hz',ylabel='Magnitude, dB');ax.legend();fig.tight_layout();fig.savefig(R/'audio_filter.png',dpi=150);plt.close(fig)
metrics['audio']={'source':'synthetic two-tone control, not bacteria video audio','sample_rate':sr,'signal_Hz':440,'noise_Hz':4000,'cutoff_Hz':1200}
(R/'baselines.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2));print(json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
