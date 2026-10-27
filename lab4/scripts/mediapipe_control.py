from pathlib import Path
import json,cv2,mediapipe as mp
P=Path(__file__).resolve().parents[1];R=P/'results'
opt=mp.tasks.vision.PoseLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(P/'models/pose_landmarker_lite.task'),delegate=mp.tasks.BaseOptions.Delegate.CPU),num_poses=4)
with mp.tasks.vision.PoseLandmarker.create_from_options(opt) as detector:
 img=mp.Image.create_from_file(str(P/'data/raw/bus.jpg'));res=detector.detect(img);im=cv2.imread(str(P/'data/raw/bus.jpg'));h,w=im.shape[:2]
 for person in res.pose_landmarks:
  for p in person:cv2.circle(im,(round(p.x*w),round(p.y*h)),4,(0,255,255),-1)
 cv2.imwrite(str(R/'mediapipe_pose.jpg'),im)
p=R/'zero_shot_controls.json';d=json.loads(p.read_text());d['MediaPipe']={'people':len(res.pose_landmarks),'landmarks_per_person':33,'delegate':'CPU'};p.write_text(json.dumps(d,ensure_ascii=False,indent=2));print(d['MediaPipe'])
