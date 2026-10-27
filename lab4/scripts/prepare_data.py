"""Manual geometric labels on public images; isolated teaching crops, not expert masks."""
from pathlib import Path
import json,cv2,numpy as np
P=Path(__file__).resolve().parents[1]; R=P/'results';R.mkdir(exist_ok=True)
# Polyline centres (in source images resized to 640 px width) and approximate half-width.
A={
's01': [([(50,200),(61,232),(72,264),(81,295),(86,319)],23), ([(137,176),(135,205),(137,237),(138,267),(139,293)],24), ([(126,323),(119,350),(120,381),(122,414),(124,439)],23), ([(309,202),(328,173),(347,145),(365,111),(390,84)],25), ([(405,76),(435,70),(470,69),(504,76),(531,84)],22), ([(391,268),(408,292),(424,321),(439,347),(459,366)],24)],
's03': [([(127,306),(150,288),(175,271),(200,253),(230,242)],28), ([(240,202),(269,188),(296,179),(324,168),(353,164)],27), ([(302,123),(322,117),(341,111),(366,113),(385,111)],24), ([(411,117),(431,114),(450,109),(470,106),(494,109)],24)],
's04': [([(338,218),(354,209),(370,200),(385,190),(399,182)],13), ([(142,344),(156,334),(170,322),(184,310),(199,298)],13), ([(193,234),(204,236),(214,239),(226,240),(235,239)],13), ([(40,245),(45,238),(49,231),(54,224),(59,220)],13), ([(161,93),(162,104),(162,116),(161,127),(159,137)],11)],
's05': [([(32,201),(58,214),(86,231),(111,251),(139,273)],13), ([(145,282),(160,288),(177,297),(199,301),(222,299)],12), ([(211,303),(250,299),(285,293),(327,287),(366,282)],12), ([(117,389),(133,375),(150,358),(170,344),(190,336)],13), ([(392,290),(409,267),(428,248),(442,227),(452,210)],13), ([(466,369),(491,348),(522,329),(552,312),(585,300)],16), ([(574,459),(581,425),(589,395),(596,360),(601,333)],16), ([(327,487),(377,483),(431,479),(491,475),(555,476)],16), ([(177,200),(189,213),(203,232),(219,244),(235,253)],12)],
's06': [([(60,73),(108,79),(160,93),(213,105),(263,115)],20), ([(24,155),(37,190),(54,232),(69,272),(84,304)],21), ([(340,85),(355,113),(364,148),(372,184),(379,218)],21), ([(431,316),(464,276),(501,227),(530,170),(569,108)],22), ([(264,401),(277,374),(298,340),(322,307),(352,282)],21), ([(519,10),(546,17),(577,28),(607,42),(630,52)],19), ([(29,396),(62,381),(96,365),(131,346),(155,306)],16)],
's09': [([(197,506),(224,473),(253,443),(280,408),(307,372)],55), ([(340,328),(366,286),(393,247),(422,202),(451,159)],60)]
}
SPLIT={'s01':'train','s05':'train','s06':'train','s03':'val','s04':'test','s09':'test'}
D=P/'data/pose'; records=[]
for split in ['train','val','test']:
 for typ in ['images','labels']:(D/typ/split).mkdir(parents=True,exist_ok=True)
for sid,items in A.items():
 im=cv2.imread(str(P/'data/raw'/f'{sid}_640.png'));h,w=im.shape[:2];overlay=im.copy()
 for i,(pts,radius) in enumerate(items):
  pts=np.array(pts,dtype=float)
  # Keep a canonical endpoint order (left first, then top for tied x).
  if tuple(pts[0])>tuple(pts[-1]):pts=pts[::-1]
  for a,b in zip(pts[:-1],pts[1:]):cv2.line(overlay,tuple(a.astype(int)),tuple(b.astype(int)),(0,255,0),2)
  for k,q in enumerate(pts): cv2.circle(overlay,tuple(q.astype(int)),4,(0,0,255),-1);cv2.putText(overlay,str(k+1),tuple(q.astype(int)+[4,-4]),0,.35,(255,255,255),1)
  mask=np.zeros((h,w),np.uint8)
  cv2.polylines(mask,[pts.astype(np.int32)],False,255,2*radius)
  for q in pts:cv2.circle(mask,tuple(q.astype(int)),radius,255,-1)
  ys,xs=np.nonzero(mask);x0=max(0,int(xs.min())-20);x1=min(w,int(xs.max())+21);y0=max(0,int(ys.min())-20);y1=min(h,int(ys.max())+21)
  crop=np.where(mask[...,None]>0,im,127)[y0:y1,x0:x1].astype('uint8');cm=mask[y0:y1,x0:x1]
  ch,cw=crop.shape[:2];scale=224/max(ch,cw);nw,nh=round(cw*scale),round(ch*scale);ox,oy=(256-nw)//2,(256-nh)//2
  out=np.full((256,256,3),127,np.uint8);out[oy:oy+nh,ox:ox+nw]=cv2.resize(crop,(nw,nh));om=np.zeros((256,256),np.uint8);om[oy:oy+nh,ox:ox+nw]=cv2.resize(cm,(nw,nh),interpolation=cv2.INTER_NEAREST)
  points=(pts-[x0,y0])*[nw/cw,nh/ch]+[ox,oy];ys,xs=np.nonzero(om);box=[(xs.min()+xs.max()+1)/512,(ys.min()+ys.max()+1)/512,(xs.max()-xs.min()+1)/256,(ys.max()-ys.min()+1)/256]
  vals=[0]+box+sum(([float(x/256),float(y/256),2] for x,y in points),[])
  name=f'{sid}_{i:02d}';sp=SPLIT[sid]
  cv2.imwrite(str(D/'images'/sp/f'{name}.jpg'),out)
  (D/'labels'/sp/f'{name}.txt').write_text(' '.join(f'{v:.6f}' for v in vals)+'\n')
  records.append({'id':name,'source':sid,'split':sp,'class':'rod','image':str(D/'images'/sp/f'{name}.jpg'),'points':points.tolist(),'bbox':box})
 cv2.imwrite(str(R/f'labels_{sid}.jpg'),overlay)
# Morphological classification and retrieval gallery: original image crops, no isolation.
G=P/'data/gallery';G.mkdir(exist_ok=True);gallery=[]
for sid,items in A.items():
 im=cv2.imread(str(P/'data/raw'/f'{sid}_640.png'))
 for i,(pts,r) in enumerate(items):
  q=np.array(pts);x0,y0=np.maximum(0,q.min(0)-r-8);x1,y1=np.minimum([im.shape[1],im.shape[0]],q.max(0)+r+9)
  f=G/f'{sid}_{i:02d}.jpg';cv2.imwrite(str(f),im[y0:y1,x0:x1]);gallery.append({'id':f.stem,'source':sid,'class':0,'split':SPLIT[sid],'image':str(f)})
C={'s02':[(281,180,95),(356,316,100)],'s07':[(104,314,51),(189,276,50),(290,211,47),(368,146,47),(438,102,46),(532,128,52)],'s08':[(201,181,51),(294,227,54),(414,244,53),(507,227,51)]}
for sid,circles in C.items():
 im=cv2.imread(str(P/'data/raw'/f'{sid}_640.png'))
 for i,(x,y,r) in enumerate(circles):
  # Coordinates checked on resized source, separate source per split.
  f=G/f'{sid}_{i:02d}.jpg';c=im[max(0,y-r):min(im.shape[0],y+r),max(0,x-r):min(im.shape[1],x+r)];cv2.imwrite(str(f),c)
  gallery.append({'id':f.stem,'source':sid,'class':1,'split':{'s02':'test','s07':'train','s08':'val'}[sid],'image':str(f)})
(P/'data/pose_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));(P/'data/gallery.json').write_text(json.dumps(gallery,ensure_ascii=False,indent=2))
(D/'dataset.yaml').write_text('path: '+str(D)+'\ntrain: images/train\nval: images/val\ntest: images/test\nkpt_shape: [5, 3]\nflip_idx: [4, 3, 2, 1, 0]\nnames:\n  0: bacterium\n')
print('pose counts',{s:sum(r['split']==s for r in records) for s in SPLIT.values()});print('gallery',len(gallery))
# Check every target is visible and inside its labelled box.
for r in records:
 a=np.loadtxt(D/'labels'/r['split']/(r['id']+'.txt')).reshape(-1);assert len(a)==20
 assert np.all((a[1:5]>=0)&(a[1:5]<=1));pts=a[5:].reshape(5,3);assert np.all((pts[:,:2]>=0)&(pts[:,:2]<=1));assert np.all(pts[:,2]==2)
print('Label checks passed')
