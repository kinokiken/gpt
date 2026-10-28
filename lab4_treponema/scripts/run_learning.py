"""CLIP/ResNet retrieval, triplet projection, 8 trees and 5 MLP runs in MLflow."""
from experiment_common import *
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import itertools

def main():
    # Near-duplicate colour variants t04/t05 are kept together in train.
    assignments=[('t01.jpg',0,'train'),('t02.jpg',0,'train'),('t04.jpg',0,'train'),
                 ('t05.jpg',0,'train'),('t03.png',0,'val'),('t06.jpg',0,'test'),
                 ('treponema_primary.jpg',0,'test'),
                 ('s01.jpg',1,'train'),('s05.jpg',1,'train'),('s06.jpg',1,'train'),
                 ('s03.jpg',1,'val'),('s04.jpg',1,'test'),('s09.jpg',1,'test'),
                 ('s07.jpg',2,'train'),('s08.jpg',2,'val'),('s02.jpg',2,'test')]
    gallery=ROOT/'data/final_gallery';gallery.mkdir(parents=True,exist_ok=True)
    images=[];rows=[]
    for filename,label,split in assignments:
        path=(LEGACY if filename.startswith('s') else ROOT)/'data/raw'/filename
        image=Image.open(path).convert('RGB');w,h=image.size
        for j,box in enumerate([(0,0,w,h),(0,0,w//2,h//2),(w//2,0,w,h//2),
                                 (0,h//2,w//2,h),(w//2,h//2,w,h)]):
            patch=image.crop(box).resize((224,224));name=Path(filename).stem+f'_{j}.jpg'
            patch.save(gallery/name);images.append(patch)
            rows.append({'file':name,'source':filename,'label':label,'split':split})
    save('gallery.json',rows)
    clip=CLIPModel.from_pretrained('openai/clip-vit-base-patch32',use_safetensors=True).eval()
    processor=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    clip_vectors=[]
    with torch.no_grad():
        for i in range(0,len(images),8):
            clip_vectors.append(F.normalize(clip.get_image_features(**processor(images=images[i:i+8],
                                    return_tensors='pt')),dim=1).numpy())
    clip_vectors=np.concatenate(clip_vectors)
    weights=ResNet18_Weights.DEFAULT;resnet=resnet18(weights=weights).eval();resnet.fc=nn.Identity()
    resnet_vectors=[]
    with torch.no_grad():
        for i in range(0,len(images),8):
            resnet_vectors.append(F.normalize(resnet(torch.stack([weights.transforms()(im)
                                 for im in images[i:i+8]])),dim=1).numpy())
    resnet_vectors=np.concatenate(resnet_vectors)
    np.save(OUT/'clip_vectors.npy',clip_vectors);np.save(OUT/'resnet_vectors.npy',resnet_vectors)
    labels=np.array([r['label'] for r in rows]);splits=np.array([r['split'] for r in rows])
    train=np.flatnonzero(splits=='train');val=np.flatnonzero(splits=='val');test=np.flatnonzero(splits=='test')
    prompts=['Treponema pallidum spiral spirochetes','rod shaped bacteria','round cocci bacteria']
    with torch.no_grad():q=F.normalize(clip.get_text_features(**processor(text=prompts,padding=True,
                                                 return_tensors='pt')),dim=1).numpy()
    hits=[];panels=[];titles=[]
    for prompt,vector in zip(prompts,q):
        scores=clip_vectors@vector;ids=np.argsort(-scores)[:3]
        hits.append({'query':prompt,'hits':[{'file':rows[i]['file'],'cosine':float(scores[i])} for i in ids]})
        for i in ids:panels.append(np.array(images[i]));titles.append(f'{prompt[:25]}\n{rows[i]["file"]}: {scores[i]:.3f}')
    panel('retrieval_text.png',panels,titles)
    def precision(vectors):
        values=[]
        for i in test:
            candidates=[j for j in test if rows[i]['source']!=rows[j]['source']]
            order=sorted(candidates,key=lambda j:float(vectors[i]@vectors[j]),reverse=True)[:3]
            values.append(float(np.mean(labels[order]==labels[i])))
        return float(np.mean(values))
    query_index=next(i for i,r in enumerate(rows) if r['file']=='treponema_primary_0.jpg')
    candidates=[i for i,r in enumerate(rows) if r['source']!='treponema_primary.jpg']
    idx=sorted(candidates,key=lambda i:float(resnet_vectors[query_index]@resnet_vectors[i]),reverse=True)[:3]
    panel('retrieval_image.png',[np.array(images[query_index])]+[np.array(images[i]) for i in idx],
          ['Запрос']+[rows[i]['file'] for i in idx],cols=4)
    save('retrieval_actual.json',{'samples':len(rows),'train':len(train),'val':len(val),'test':len(test),
       'classes':['Treponema','rods','cocci'],'text_queries':hits,
       'CLIP_precision_at_3_cross_source':precision(clip_vectors),
       'ResNet_precision_at_3_cross_source':precision(resnet_vectors)})

    x=torch.tensor(resnet_vectors,dtype=torch.float32)
    projection=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,64))
    optimizer=torch.optim.Adam(projection.parameters(),lr=.001)
    rng=np.random.default_rng(7);history=[];best=None;best_score=-1
    def pairs(ids):
        # Pairs share patches and may share a source; they are not independent samples.
        return list(itertools.combinations(ids,2))
    def evaluate(z,ids,threshold=None):
        pp=pairs(ids);truth=np.array([labels[i]==labels[j] for i,j in pp]);scores=np.array([z[i]@z[j] for i,j in pp])
        if threshold is None:
            choices=np.linspace(-1,1,401)
            predictions=scores[:,None]>=choices
            balanced=(predictions[truth].mean(0)+(~predictions[~truth]).mean(0))/2
            threshold=choices[int(np.argmax(balanced))]
        return {'threshold':float(threshold),'balanced_accuracy':float(balanced_accuracy_score(truth,scores>=threshold)),
                'AUC':float(roc_auc_score(truth,scores)),'pairs':len(pp)}
    for epoch in range(100):
        triples=[]
        for i in train:
            positives=[j for j in train if labels[j]==labels[i] and j!=i]
            negatives=[j for j in train if labels[j]!=labels[i]]
            triples.append([i,rng.choice(positives),rng.choice(negatives)])
        t=np.array(triples);z=F.normalize(projection(x),dim=1)
        loss=F.triplet_margin_loss(z[t[:,0]],z[t[:,1]],z[t[:,2]],margin=.5)
        optimizer.zero_grad();loss.backward();optimizer.step()
        with torch.no_grad():vectors=F.normalize(projection(x),dim=1).numpy()
        validation=evaluate(vectors,val)
        if validation['balanced_accuracy']>best_score:
            best_score=validation['balanced_accuracy'];best={k:v.detach().clone() for k,v in projection.state_dict().items()}
        history.append(float(loss.detach()))
    projection.load_state_dict(best)
    with torch.no_grad():learned=F.normalize(projection(x),dim=1).numpy()
    baseline=evaluate(resnet_vectors,val);after=evaluate(learned,val)
    metric={'base':evaluate(resnet_vectors,test,baseline['threshold']),
            'triplet':evaluate(learned,test,after['threshold']), 'train_loss':history}
    torch.save(projection.state_dict(),OUT/'triplet_head.pt');save('metric_actual.json',metric)
    fig,ax=plt.subplots(figsize=(8,3));ax.plot(range(1,101),history);ax.set(xlabel='Эпоха',ylabel='Triplet loss')
    fig.tight_layout();fig.savefig(OUT/'triplet_learning.png',dpi=160);plt.close(fig)

    mlflow.set_tracking_uri('sqlite:///'+str(CACHE/'mlflow.db'))
    mlflow.set_experiment('Treponema_LR4_measured')
    mlflow_runs=[];tree_models=[]
    for depth in [2,4,8,16]:
        for minimum in [2,10]:
            estimator=DecisionTreeClassifier(max_depth=depth,min_samples_split=minimum,random_state=7).fit(resnet_vectors[train],labels[train])
            pred=estimator.predict(resnet_vectors[test]);vpred=estimator.predict(resnet_vectors[val])
            with mlflow.start_run(run_name=f'tree_{depth}_{minimum}') as run:
                mlflow.log_params({'depth':depth,'min_samples_split':minimum,'backbone':'ResNet18 frozen'})
                scores={'val_f1':float(f1_score(labels[val],vpred,average='macro')),
                        'test_f1':float(f1_score(labels[test],pred,average='macro'))}
                mlflow.log_metrics(scores)
                info=mlflow.sklearn.log_model(estimator,name='model',
                         skops_trusted_types=['sklearn.tree._tree.Tree'])
                mlflow_runs.append({'name':f'tree_{depth}_{minimum}','run_id':run.info.run_id,**scores})
                tree_models.append((scores['val_f1'],info.model_uri,estimator))
    network_runs=[]
    for hidden,dropout in [(32,0),(64,0),(64,.2),(128,.2),(128,.5)]:
        torch.manual_seed(7)
        net=nn.Sequential(nn.Linear(512,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,3))
        optim=torch.optim.Adam(net.parameters(),lr=.005);ys=torch.tensor(labels,dtype=torch.long)
        values=[]
        with mlflow.start_run(run_name=f'mlp_{hidden}_{dropout}') as run:
            mlflow.log_params({'hidden':hidden,'dropout':dropout,'epochs':60,'backbone':'ResNet18 frozen'})
            for epoch in range(60):
                net.train();loss=F.cross_entropy(net(x[train]),ys[train]);optim.zero_grad();loss.backward();optim.step()
                net.eval()
                with torch.no_grad():v=net(x[val]).argmax(1).numpy()
                score=float(accuracy_score(labels[val],v));values.append(score)
                mlflow.log_metric('val_accuracy',score,step=epoch)
            with torch.no_grad():pred=net(x[test]).argmax(1).numpy()
            score=float(f1_score(labels[test],pred,average='macro'));mlflow.log_metric('test_f1',score)
            info=mlflow.pytorch.log_model(net,name='model',input_example=resnet_vectors[test])
            loaded=mlflow.pyfunc.load_model(info.model_uri)
            restored_prediction=loaded.predict(resnet_vectors[test])
            same=bool(np.array_equal(np.asarray(restored_prediction).argmax(1),pred))
            network_runs.append({'name':f'mlp_{hidden}_{dropout}','run_id':run.info.run_id,
                                 'test_f1':score,'val_history':values,'reload_identical':same})
    registered=[]
    for _,uri,estimator in sorted(tree_models,key=lambda t:t[0],reverse=True)[:2]:
        version=mlflow.register_model(uri,'Treponema_ResNet_Tree')
        restored=mlflow.sklearn.load_model(f'models:/Treponema_ResNet_Tree/{version.version}')
        identical=bool(np.array_equal(restored.predict(resnet_vectors[test]),estimator.predict(resnet_vectors[test])))
        registered.append({'name':version.name,'version':version.version,'reload_identical':identical})
    save('mlflow_actual.json',{'trees':mlflow_runs,'networks':network_runs,'registered':registered})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    runs=mlflow_runs+network_runs;axes[0].bar(range(len(runs)),[r['test_f1'] for r in runs])
    axes[0].set_xticks(range(len(runs)),[r['name'] for r in runs],rotation=70,fontsize=7);axes[0].set_ylabel('Test macro-F1')
    for r in network_runs:axes[1].plot(r['val_history'],label=r['name'])
    axes[1].set(xlabel='Эпоха',ylabel='Validation accuracy');axes[1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'mlflow_actual.png',dpi=160);plt.close(fig)
    print('Completed retrieval, metric learning and 13 MLflow runs',flush=True)

if __name__=='__main__':
    main()
