from pathlib import Path
import os,json,time
P=Path(__file__).resolve().parents[1];R=P/'results';os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING']='false';os.environ['MLFLOW_DISABLE_AGENT_HINT']='1'
import numpy as np,torch,mlflow,mlflow.sklearn,mlflow.pytorch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
uri='sqlite:///'+str(P/'mlflow/mlflow.db');(P/'mlflow').mkdir(exist_ok=True);mlflow.set_tracking_uri(uri);mlflow.set_experiment('Andreev_Lab4_bacteria')
data=json.loads((P/'data/gallery.json').read_text());X=np.load(R/'gallery_embeddings.npy');y=np.array([r['class'] for r in data]);ix={s:np.array([i for i,r in enumerate(data) if r['split']==s]) for s in ['train','val','test']};runs=[];candidates=[]
for C in [.1,1.0]:
 m=LogisticRegression(C=C,class_weight='balanced',random_state=42,max_iter=1000);m.fit(X[ix['train']],y[ix['train']]);score=accuracy_score(y[ix['val']],m.predict(X[ix['val']]))
 with mlflow.start_run(run_name=f'logreg_C_{C}') as run:
  mlflow.log_params({'C':C,'features':'CLIP frozen 512','split':'by original source image','train_samples':len(ix['train'])});mlflow.log_metric('val_accuracy',score)
  info=mlflow.sklearn.log_model(m,name='model',signature=infer_signature(X,m.predict(X)),input_example=X[:2],registered_model_name='Andreev_Bacteria_LogReg')
  loaded=mlflow.sklearn.load_model(info.model_uri);assert np.array_equal(loaded.predict(X),m.predict(X))
  runs.append({'name':f'logreg_C_{C}','run_id':run.info.run_id,'val_accuracy':float(score),'model_uri':info.model_uri});candidates.append((score,C,m,run.info.run_id))
best=max(candidates,key=lambda t:(t[0],-t[1]));test=accuracy_score(y[ix['test']],best[2].predict(X[ix['test']]))
with mlflow.start_run(run_id=best[3]):mlflow.log_metric('test_accuracy_selected_model',test)
state=torch.load(P/'models/metric_head.pt',map_location='cpu',weights_only=True);head=torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.ReLU(),torch.nn.Linear(128,32));head.load_state_dict(state['state_dict']);head.eval()
with mlflow.start_run(run_name='triplet_projection') as run:
 sem=json.loads((R/'semantic_metrics.json').read_text());tr=sem['metric_learning'];mlflow.log_params({'epochs':tr['epochs'],'selected_epoch':tr['best_epoch'],'margin':tr['margin'],'head':'512-128-32'})
 mlflow.log_metrics({k:v for k,v in tr['results']['Triplet projection'].items() if isinstance(v,(int,float))});mlflow.log_artifact(str(R/'triplet_loss.png'))
 with torch.no_grad():pred=head(torch.from_numpy(X)).numpy()
 info=mlflow.pytorch.log_model(head,name='model',signature=infer_signature(X,pred),input_example=X[:2],registered_model_name='Andreev_Triplet_Projection')
 loaded=mlflow.pytorch.load_model(info.model_uri)
 with torch.no_grad():assert np.allclose(loaded(torch.from_numpy(X)).numpy(),pred,atol=1e-6)
 runs.append({'name':'triplet_projection','run_id':run.info.run_id,'model_uri':info.model_uri,'reload_check':True})
with mlflow.start_run(run_name='YOLO_pose') as run:
 mlflow.log_params({'epochs':30,'imgsz':256,'train_images':22,'val_images':4,'test_images':7,'input':'isolated geometric teaching crops'})
 mets=json.loads((R/'pose_metrics.json').read_text());mlflow.log_metrics({k.replace('(','_').replace(')',''):v for k,v in mets.items()});mlflow.log_artifact(str(R/'pose/train/weights/best.pt'));mlflow.log_artifact(str(R/'pose/train/results.csv'));runs.append({'name':'YOLO_pose','run_id':run.info.run_id})
for name,vals in json.loads((R/'baselines.json').read_text())['tracking'].items():
 with mlflow.start_run(run_name=f'Tracker_{name}') as run:
  mlflow.log_params({'tracker':name,'video_frames':75,'reference_frames_excluding_init':5});mlflow.log_metrics({k:v for k,v in vals.items() if isinstance(v,(float,int))});runs.append({'name':'Tracker_'+name,'run_id':run.info.run_id})
from mlflow.tracking import MlflowClient
client=MlflowClient();versions=[{'name':v.name,'version':v.version,'status':v.status} for v in client.search_model_versions()]
summary={'tracking_uri':uri,'runs':runs,'registered_versions':versions,'selected_C':best[1],'selected_logreg_test_accuracy':float(test),'sklearn_reload_verified':True,'pytorch_reload_verified':True}
(R/'mlflow_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False,indent=2))
