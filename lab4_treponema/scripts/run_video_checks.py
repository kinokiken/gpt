"""Check local video search and pose tracking with a hysteresis bend counter."""
from experiment_common import *
from video_index import model, search
from ultralytics import YOLO
from PIL import Image
from collections import deque
import time

def main():
    directory=ROOT/'data/video_index'
    net,processor=model()
    query='Treponema pallidum spirochetes moving in a microscope field'
    text_hits=search(directory,query,net,processor)
    image_hits=search(directory,'',net,processor,image=Image.open(ROOT/'data/raw/treponema_primary.jpg'))
    panel('video_search_actual.png',[rgb(directory/r['path']) for r in text_hits+image_hits],
          [f"{kind}: {r['seconds']:.0f} с, {r['cosine']:.3f}"
           for kind,hits in [('Text',text_hits),('Image',image_hits)] for r in hits])
    save('video_search_actual.json',{'query':query,'text':text_hits,'image':image_hits})
    net=YOLO(OUT/'pose_v8s_best.pt')
    cap=cv2.VideoCapture(str(ROOT/'data/raw/t07.ogv'));cap.set(cv2.CAP_PROP_POS_MSEC,36000)
    state={};naive=0;flagged=0;fps=deque(maxlen=15);observations=[];images=[];titles=[]
    for number in range(36):
        ok,frame=cap.read()
        if not ok:break
        begin=time.perf_counter()
        result=net.track(frame,persist=True,tracker='bytetrack.yaml',conf=.1,imgsz=192,verbose=False)[0]
        if result.boxes.id is not None:
            for track_id,points in zip(result.boxes.id.int().tolist(),result.keypoints.xy.cpu().numpy()):
                a,b,c=points;axis=c-a;length=np.linalg.norm(axis)
                bend=float(abs(axis[0]*(b-a)[1]-axis[1]*(b-a)[0])/max(length*length,1e-6))
                before=state.get(track_id,False);now=bend>(.07 if before else .10)
                naive+=int(now);flagged+=int(now and not before);state[track_id]=now
                observations.append({'frame':number,'id':track_id,'bend':bend})
        fps.append(1/max(time.perf_counter()-begin,1e-9))
        if number in [0,12,24,35]:
            images.append(cv2.cvtColor(result.plot(),cv2.COLOR_BGR2RGB));titles.append(f't={36+number/12:.2f} с')
    cap.release()
    panel('pose_video_tracking.png',images,titles,cols=2)
    save('pose_video_tracking.json',{'tracks':len(state),'naive':naive,'flagged':flagged,
                'window_fps':float(np.mean(fps)),'observations':observations,'frames':36})
    print('Video text/image search and counters completed',flush=True)

if __name__=='__main__':
    main()
