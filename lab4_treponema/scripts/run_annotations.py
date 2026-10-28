"""Produce measured skeleton landmarks and COCO/CVAT polygon exports."""
from experiment_common import *
import zipfile
import shutil
import xml.etree.ElementTree as ET

def main():
    source=rgb(ROOT/'data/raw/treponema_primary.jpg')
    mask=cv2.imread(str(ROOT/'results/primary_mask.png'),0)
    count,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype('uint8'))
    shown=source.copy();records=[];annotations=[]
    xml=ET.Element('annotations');ET.SubElement(xml,'version').text='1.1'
    node=ET.SubElement(xml,'image',id='0',name='treponema_primary.jpg',width=str(source.shape[1]),height=str(source.shape[0]))
    for label in range(1,count):
        x,y,w,h,area=map(int,stats[label])
        if area<=450:continue
        points,path=skeleton_points(labels==label)
        for a,b in zip(path[:-1],path[1:]):cv2.line(shown,tuple(a.astype(int)),tuple(b.astype(int)),(30,255,30),1)
        for i,point in enumerate(points):
            cv2.circle(shown,tuple(point.astype(int)),5,(255,220,0),-1)
            cv2.putText(shown,str(i),tuple(point.astype(int)+[4,-4]),0,.6,(0,0,0),2)
        contour=max(cv2.findContours((labels==label).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0],key=cv2.contourArea)
        simplified=cv2.approxPolyDP(contour,2,True)
        records.append({'id':label,'area_px':area,'keypoints':points.tolist(),
                        'arc_length_px':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
                        'contour_points':len(contour),'simplified_points':len(simplified)})
        annotations.append({'id':label,'image_id':1,'category_id':1,'bbox':[x,y,w,h],
                            'area':area,'iscrowd':0,'segmentation':[contour.reshape(-1).tolist()]})
        ET.SubElement(node,'polygon',label='treponema',occluded='0',source='auto',
                      points=';'.join(f'{a},{b}' for a,b in simplified[:,0,:]))
        ET.SubElement(node,'skeleton',label='treponema_pose',source='auto')
        skeleton=node[-1]
        for name,p in zip(['start','middle','end'],points):
            ET.SubElement(skeleton,'points',label=name,points=f'{p[0]},{p[1]}',occluded='0',outside='0')
    panel('skeleton_landmarks.png',[source,mask,shown],['Исходник','Цветовая маска','Точки вдоль осевой линии'])
    save('skeleton_landmarks.json',records)
    out=OUT/'cvat';out.mkdir(exist_ok=True)
    ET.indent(xml);ET.ElementTree(xml).write(out/'annotations.xml',encoding='utf-8',xml_declaration=True)
    totals={}
    for variant in ['full','simplified']:
        payload={'images':[{'id':1,'file_name':'treponema_primary.jpg','width':source.shape[1],'height':source.shape[0]}],
                 'categories':[{'id':1,'name':'treponema'}],'annotations':json.loads(json.dumps(annotations))}
        if variant=='simplified':
            for ann in payload['annotations']:
                contour=np.array(ann['segmentation'][0],np.int32).reshape(-1,1,2)
                ann['segmentation']=[cv2.approxPolyDP(contour,2,True).reshape(-1).tolist()]
        points=sum(len(a['segmentation'][0])//2 for a in payload['annotations'])
        totals[variant]={'objects':len(payload['annotations']),'polygon_points':points}
        with zipfile.ZipFile(out/f'{variant}_coco.zip','w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('annotations/instances_default.json',json.dumps(payload))
            archive.write(ROOT/'data/raw/treponema_primary.jpg','images/treponema_primary.jpg')
    save('cvat_actual.json',totals)
    print(totals,flush=True)

if __name__=='__main__':
    main()
