from pathlib import Path
import urllib.request,concurrent.futures,cv2,numpy as np,json
P=Path(__file__).resolve().parents[1];D=P/'data/calibration';D.mkdir(exist_ok=True);R=P/'results'
def get(i):
 f=D/f'left{i:02d}.jpg';url='https://raw.githubusercontent.com/opencv/opencv/master/samples/data/'+f.name
 try:
  if not f.exists():
   with urllib.request.urlopen(url,timeout=20) as r:f.write_bytes(r.read())
  return f
 except Exception:return None
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:files=[f for f in ex.map(get,range(1,15)) if f]
obj=np.zeros((54,3),np.float32);obj[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2);ops=[];ips=[];good=[]
for f in files:
 im=cv2.imread(str(f));g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);ok,pts=cv2.findChessboardCorners(g,(9,6))
 if ok:ops.append(obj);ips.append(cv2.cornerSubPix(g,pts,(11,11),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,30,.001)));good.append(f)
assert len(good)>=6
ntrain=len(good)-3;rms,K,d,rvecs,tvecs=cv2.calibrateCamera(ops[:ntrain],ips[:ntrain],g.shape[::-1],None,None);errors=[]
for o,ip in zip(ops[ntrain:],ips[ntrain:]):
 ok,rv,tv=cv2.solvePnP(o,ip,K,d);pred,_=cv2.projectPoints(o,rv,tv,K,d);errors.append(float(np.sqrt(np.mean(np.sum((pred-ip)**2,axis=2)))))
im=cv2.imread(str(good[-1]));newK,_=cv2.getOptimalNewCameraMatrix(K,d,(im.shape[1],im.shape[0]),1);out=cv2.undistort(im,K,d,None,newK);cv2.imwrite(str(R/'calibration_before.jpg'),im);cv2.imwrite(str(R/'calibration_after.jpg'),out)
result={'source':'OpenCV samples/data/left*.jpg','images_downloaded':len(files),'images_with_corners':len(good),'train_images':ntrain,'heldout_images':3,'pattern':[9,6],'calibration_RMS_px':float(rms),'heldout_reprojection_RMSE_px':errors,'K':K.tolist(),'distortion':d.tolist(),'not_applicable_to_bacteria_camera':True};(R/'calibration.json').write_text(json.dumps(result,indent=2));print(result)
