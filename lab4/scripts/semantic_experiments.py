from pathlib import Path
import os,json,copy,time
P=Path(__file__).resolve().parents[1];os.environ['HF_HOME']=str(P/'models/hf');os.environ['HF_HUB_DISABLE_XET']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import torch,numpy as np
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import CLIPModel,CLIPProcessor
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score
import torch.nn.functional as F
torch.set_num_threads(2);torch.manual_seed(42);rng=np.random.default_rng(42);R=P/'results';data=json.loads((P/'data/gallery.json').read_text())
model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True).eval();processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32',local_files_only=True)
def images_embed(paths):
 out=[]
 for i in range(0,len(paths),8):
  inputs=processor(images=[Image.open(f).convert('RGB') for f in paths[i:i+8]],return_tensors='pt')
  with torch.no_grad():z=model.get_image_features(**inputs);z=F.normalize(z,dim=1)
  out.append(z.cpu())
 return torch.cat(out)
def texts_embed(texts):
 with torch.no_grad():return F.normalize(model.get_text_features(**processor(text=texts,return_tensors='pt',padding=True)),dim=1)
z=images_embed([d['image'] for d in data]);np.save(R/'gallery_embeddings.npy',z.numpy());print('Embeddings',list(z.shape),flush=True)
ids={s:np.array([i for i,r in enumerate(data) if r['split']==s]) for s in ['train','val','test']};y=np.array([d['class'] for d in data]);prompts=['a microscope image of elongated rod-shaped bacteria','a microscope image of round spherical cocci bacteria']
textz=texts_embed(prompts);sim=z@textz.T;pred=sim.argmax(1).numpy();out={'n_gallery':len(data),'class_names':['rod','coccus'],'prompts':prompts,'zero_shot_test_accuracy':float(accuracy_score(y[ids['test']],pred[ids['test']])),'split_counts':{s:len(a) for s,a in ids.items()},'split_sources':{s:sorted(set(data[i]['source'] for i in a)) for s,a in ids.items()}}
# Text retrieval over all gallery; scores are similarities, not calibrated probabilities.
fig,axes=plt.subplots(2,4,figsize=(10,6));rankings=[]
for j,query in enumerate(prompts):
 order=torch.argsort(sim[:,j],descending=True)[:4].numpy();rankings.append({'query':query,'matches':[{'id':data[i]['id'],'cosine':float(sim[i,j])} for i in order]})
 for a,i in zip(axes[j],order):a.imshow(Image.open(data[i]['image']));a.set_title(f'{data[i]["id"]}: {sim[i,j]:.3f}');a.axis('off')
fig.tight_layout();fig.savefig(R/'text_retrieval.png',dpi=150);plt.close(fig);out['text_retrieval']=rankings
q=ids['test'][0];rank=ids['train'][np.argsort(-(z[ids['train']]@z[q]).numpy())[:4]];fig,ax=plt.subplots(1,5,figsize=(12,3))
for a,i,title in zip(ax,[q]+rank.tolist(),['Query']+['Match']*4):a.imshow(Image.open(data[i]['image']));a.set_title(title+' '+data[i]['id']);a.axis('off')
fig.tight_layout();fig.savefig(R/'image_retrieval.png',dpi=150);plt.close(fig)
# Frozen CLIP backbone with a trainable projection and triplet margin loss.
head=torch.nn.Sequential(torch.nn.Linear(512,128),torch.nn.ReLU(),torch.nn.Linear(128,32));opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
def pairs(ix,emb):
 d=[];labels=[]
 for k,a in enumerate(ix):
  for b in ix[k+1:]:d.append(float(torch.linalg.vector_norm(emb[a]-emb[b])));labels.append(int(y[a]==y[b]))
 return np.array(d),np.array(labels)
def threshold(emb):
 d,t=pairs(ids['val'],emb);c=np.unique(np.r_[0,(np.sort(d)[:-1]+np.sort(d)[1:])/2,2.01]);scores=[balanced_accuracy_score(t,d<=k) for k in c];i=int(np.argmax(scores));return float(c[i]),float(scores[i])
losses=[];history=[];best_score=-1;best_state=None;best_epoch=None
for epoch in range(1,151):
 aa=rng.choice(ids['train'],64);pp=np.array([rng.choice([j for j in ids['train'] if y[j]==y[i] and j!=i]) for i in aa]);nn=np.array([rng.choice(ids['train'][y[ids['train']]!=y[i]]) for i in aa]);head.train();a,p,n=[F.normalize(head(z[t]),dim=1) for t in [aa,pp,nn]];loss=F.triplet_margin_loss(a,p,n,margin=.5);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
 if epoch%10==0:
  head.eval()
  with torch.no_grad():e=F.normalize(head(z),dim=1)
  th,score=threshold(e);history.append({'epoch':epoch,'validation_balanced_accuracy':score,'threshold':th})
  if score>best_score:best_score=score;best_state=copy.deepcopy(head.state_dict());best_epoch=epoch
head.load_state_dict(best_state);head.eval()
with torch.no_grad():learned=F.normalize(head(z),dim=1)
torch.save({'state_dict':head.state_dict(),'input_dim':512,'hidden_dim':128,'output_dim':32,'best_epoch':best_epoch},P/'models/metric_head.pt');np.save(R/'learned_embeddings.npy',learned.numpy())
res={}
for name,e in [('CLIP',z),('Triplet projection',learned)]:
 th,val=threshold(e);dist,targ=pairs(ids['test'],e);pred=dist<=th;p3=[]
 for i in ids['test']:
  top=ids['train'][torch.argsort(e[ids['train']]@e[i],descending=True)[:3].numpy()];p3.append(float(np.mean(y[top]==y[i])))
 res[name]={'threshold_from_validation':th,'val_balanced_accuracy':val,'test_pair_accuracy':float(accuracy_score(targ,pred)),'test_pair_balanced_accuracy':float(balanced_accuracy_score(targ,pred)),'test_ROC_AUC':float(roc_auc_score(targ,-dist)),'test_pairs':len(targ),'positive_pairs':int(targ.sum()),'negative_pairs':int((1-targ).sum()),'test_mean_precision_at_3':float(np.mean(p3))}
out['metric_learning']={'epochs':150,'best_epoch':best_epoch,'margin':.5,'train_batch_triplets':64,'loss_first':losses[0],'loss_last':losses[-1],'results':res,'validation_history':history}
fig,ax=plt.subplots(figsize=(8,3));ax.plot(np.arange(1,151),losses);ax.set(xlabel='Epoch',ylabel='Triplet margin loss');ax.grid(alpha=.3);fig.tight_layout();fig.savefig(R/'triplet_loss.png',dpi=150);plt.close(fig)
# Index real microscopy footage, independently of unavailable RuTube transport.
import cv2
c=cv2.VideoCapture(str(P/'data/raw/bacteria.webm'));fps=c.get(cv2.CAP_PROP_FPS);count=int(c.get(cv2.CAP_PROP_FRAME_COUNT));idx=[];paths=[];D=P/'data/frames';D.mkdir(exist_ok=True)
for fi in range(0,count,round(2*fps)):
 c.set(cv2.CAP_PROP_POS_FRAMES,fi);ok,f=c.read()
 if not ok:continue
 f=cv2.resize(f,(640,360));dest=D/f'frame_{fi:06d}.jpg';cv2.imwrite(str(dest),f);paths.append(str(dest));idx.append({'frame':fi,'seconds':fi/fps,'path':str(dest),'source_url':json.loads((P/'data/raw/video_metadata.json').read_text())['descriptionurl']})
c.release();vz=images_embed(paths);np.save(R/'video_embeddings.npy',vz.numpy());(R/'video_index.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2))
queries=['a microscope image of many thin curved bacteria','a microscopic field with a few isolated bacteria'];vtext=texts_embed(queries);vsim=vz@vtext.T
fig,axes=plt.subplots(2,3,figsize=(12,5));matches=[]
for j,q in enumerate(queries):
 order=torch.argsort(vsim[:,j],descending=True).tolist();picked=[]
 for i in order:
  if all(abs(idx[i]['seconds']-idx[k]['seconds'])>=6 for k in picked):picked.append(i)
  if len(picked)==3:break
 matches.append({'query':q,'results':[{'seconds':idx[i]['seconds'],'cosine':float(vsim[i,j])} for i in picked]})
 for a,i in zip(axes[j],picked):a.imshow(Image.open(paths[i]));a.set_title(f'{idx[i]["seconds"]:.1f}s, score {vsim[i,j]:.3f}');a.axis('off')
fig.tight_layout();fig.savefig(R/'video_search.png',dpi=150);plt.close(fig)
# Known-frame control excludes identical frame; nearest temporal neighbor is only a control.
qi=5;scores=(vz@vz[qi]).numpy();scores[qi]=-2;nearest=int(np.argmax(scores));out['video_search']={'video_frames':count,'fps':fps,'indexed_frames':len(idx),'duration_s':count/fps,'text_queries':matches,'known_frame_query_s':idx[qi]['seconds'],'nearest_nonidentical_s':idx[nearest]['seconds'],'time_difference_s':abs(idx[nearest]['seconds']-idx[qi]['seconds']),'rutube_status':'HTTP 403; transport not validated'}
(R/'semantic_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)
