from pathlib import Path
import requests,json
P=Path(__file__).resolve().parents[1];d=P/'data/raw'
s=requests.Session();s.headers['User-Agent']='Lab4ComputerVision/1.0 (educational image analysis; https://github.com/kinokiken/gpt)'
title=json.loads((d/'video_search.json').read_text())['query']['search'][0]['title']
r=s.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','titles':title,'prop':'imageinfo','iiprop':'url|extmetadata'},timeout=30);r.raise_for_status()
info=next(iter(r.json()['query']['pages'].values()))['imageinfo'][0]
(d/'video_metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
print(info['url'],flush=True)
r=s.get(info['url'],timeout=90);r.raise_for_status();(d/'bacteria.webm').write_bytes(r.content);print('downloaded bytes',len(r.content),flush=True)
