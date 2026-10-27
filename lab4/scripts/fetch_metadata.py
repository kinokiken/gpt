import requests,json,pathlib,time
P=pathlib.Path(__file__).resolve().parents[1];raw=P/'data/raw'
s=requests.Session();s.headers['User-Agent']='Lab4ComputerVision/1.0 (educational image analysis; https://github.com/kinokiken/gpt)'
rows=json.loads((raw/'sources.json').read_text())
for row in rows:
 if row.get('error'):continue
 try:
  d=s.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':'File:'+row['name'],'prop':'imageinfo','iiprop':'url|extmetadata'},timeout=25).json()
  page=next(iter(d['query']['pages'].values()));row['metadata']=page.get('imageinfo',[{}])[0]
  print(row['name'],'metadata',bool(row['metadata']),flush=True)
 except Exception as e: print('metadata error',e,flush=True)
(raw/'sources.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
q=s.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','list':'search','srsearch':'Коломенское москва река бактерии','srnamespace':6},timeout=25)
print('VIDEO SEARCH',q.status_code,q.text[:7000],flush=True)
(raw/'video_search.json').write_text(q.text)
for id in ['882902b3c856e37b6d974d2bc10eb25d','2fea643e38e6bd35093ce04025f6dfaf']:
 r=s.get('https://rutube.ru/api/play/options/'+id+'/',timeout=25);print('RUTUBE',id,r.status_code,r.text[:300],flush=True)
 (raw/f'rutube_{id}.json').write_text(r.text)
