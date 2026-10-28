"""Distortion, tracking, optical flow and image filtering on local Treponema data."""
from experiment_common import *
import time
from scipy.optimize import minimize
from skimage.metrics import peak_signal_noise_ratio as psnr

def main():
    source = cv2.resize(rgb(ROOT/'data/raw/treponema_primary.jpg'),(435,270))
    h,w=source.shape[:2]
    K=np.array([[.9*w,0,w/2],[0,.9*w,h/2],[0,0,1]],np.float64)
    D=np.array([-.25,.05,0,0,0],np.float64)
    xx,yy=np.meshgrid(np.arange(w),np.arange(h))
    pts=np.stack([xx,yy],-1).reshape(-1,1,2).astype(np.float32)
    coords=cv2.undistortPointsIter(pts,K,D,None,K,
              (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-9)).reshape(h,w,2)
    distorted=cv2.remap(source,coords[:,:,0],coords[:,:,1],cv2.INTER_LINEAR)
    def correct(values):
        return cv2.undistort(distorted,K,np.array([*values,0,0,0],np.float64))
    optimum=minimize(lambda v:-psnr(source,correct(v),data_range=255),[0.,0.],
                     method='Nelder-Mead',options={'initial_simplex':np.array([[0,0],[-.3,0],[0,.1]]),
                     'maxiter':300,'xatol':1e-5,'fatol':1e-6})
    panel('distortion.png',[source,distorted,correct(optimum.x)],
          ['Исходное изображение','Искусственная дисторсия','Подобранная коррекция'])
    save('distortion.json',{'true':[-.25,.05],'estimated':optimum.x.tolist(),
         'PSNR_distorted':float(psnr(source,distorted,data_range=255)),
         'PSNR_true':float(psnr(source,correct([-.25,.05]),data_range=255)),
         'PSNR_optimized':float(-optimum.fun),'iterations':int(optimum.nit)})
    panel('filters.png',[source,cv2.GaussianBlur(source,(5,5),0),cv2.medianBlur(source,5),
          cv2.Canny(cv2.cvtColor(source,cv2.COLOR_RGB2GRAY),60,140)],
          ['Исходник','Gaussian 5×5','Median 5×5','Canny'],cols=2)
    cap=cv2.VideoCapture(str(ROOT/'data/raw/t07.ogv'))
    cap.set(cv2.CAP_PROP_POS_MSEC,36000)
    frames=[]
    for _ in range(36):
        ok,frame=cap.read()
        if ok:frames.append(frame)
    cap.release()
    box=(113,88,44,170)  # Visible vertical spiral at t=36 s, coordinates in the 360×288 frame.
    first=frames[0]
    factories={'MIL':cv2.legacy.TrackerMIL_create,'MOSSE':cv2.legacy.TrackerMOSSE_create,
               'KCF':cv2.legacy.TrackerKCF_create,'CSRT':cv2.legacy.TrackerCSRT_create}
    metrics={}; visuals=[]; titles=[]; observations=[]
    for name,factory in factories.items():
        tracker=factory();tracker.init(first,box)
        times=[]; found=[]; boxes=[]
        for i,frame in enumerate(frames[1:],1):
            begin=time.perf_counter();ok,bb=tracker.update(frame);times.append(time.perf_counter()-begin)
            found.append(bool(ok));boxes.append(list(map(float,bb)))
            if i in [12,24,35]:
                shown=frame.copy()
                if ok:
                    x,y,w,h=map(int,bb);cv2.rectangle(shown,(x,y),(x+w,y+h),(0,255,0),1)
                visuals.append(cv2.cvtColor(shown,cv2.COLOR_BGR2RGB));titles.append(f'{name}: t={36+i/12:.2f}, ok={ok}')
        metrics[name]={'FPS':len(times)/sum(times),'valid_fraction':float(np.mean(found)),
                       'updates':len(times)}
        observations.append({'tracker':name,'boxes':boxes,'valid':found})
    panel('tracking_actual.png',visuals,titles,cols=3)
    save('tracking_actual.json',{'video':'t07.ogv','start_s':36,'initial_box':box,
          'metrics':metrics,'observations':observations})
    a,b=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames[:2]]
    flow=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.2,0)
    mag,angle=cv2.cartToPolar(flow[:,:,0],flow[:,:,1]);hsv=np.zeros_like(first)
    hsv[:,:,0]=angle*90/np.pi;hsv[:,:,1]=255;hsv[:,:,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    dense=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    points=cv2.goodFeaturesToTrack(a,100,.01,5)
    next_points,status,_=cv2.calcOpticalFlowPyrLK(a,b,points,None)
    drawn=cv2.cvtColor(first,cv2.COLOR_BGR2RGB)
    for p,q in zip(points[status.ravel()==1],next_points[status.ravel()==1]):
        cv2.arrowedLine(drawn,tuple(p[0].astype(int)),tuple((p[0]+5*(q[0]-p[0])).astype(int)),(255,40,0),1)
    panel('flow_actual.png',[cv2.cvtColor(first,cv2.COLOR_BGR2RGB),dense,drawn],
          ['t=36 с','Farneback: поле движения','Lucas–Kanade: стрелки ×5'])
    x,y,w,h=box;v=flow[y:y+h,x:x+w].mean(axis=(0,1))
    save('flow_actual.json',{'mean_vector_px':v.tolist(),'speed_px_s':float(np.linalg.norm(v)*12),
                            'LK_detected':len(points),'LK_accepted':int(status.sum())})
    mog=cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=16,detectShadows=False)
    images=[];titles=[];fractions={}
    for i,frame in enumerate(frames):
        mask=mog.apply(frame,learningRate=.02)
        if i in [0,12,35]:
            images.append(mask);titles.append(f'Кадр {i}');fractions[str(i)]=float((mask>0).mean())
    panel('foreground.png',images,titles)
    save('foreground.json',fractions)
    print('Completed geometry and video experiments',flush=True)

if __name__ == '__main__':
    main()
