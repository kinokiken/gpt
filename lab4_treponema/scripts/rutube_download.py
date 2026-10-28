"""User-run RuTube downloader. Importing this file performs no request."""
from pathlib import Path
from urllib.parse import urlsplit
import argparse, json, re

def validate_url(url):
    part=urlsplit(url)
    if part.scheme!='https' or (part.hostname or '').lower() not in {'rutube.ru','www.rutube.ru'}:
        raise ValueError('Нужна HTTPS-ссылка RuTube')
    match=re.fullmatch(r'/video/([0-9a-fA-F]{32})/?',part.path)
    if not match:raise ValueError('Нужна ссылка на отдельный публичный ролик')
    return f'https://rutube.ru/video/{match.group(1).lower()}/'
def download(url,destination):
    from yt_dlp import YoutubeDL
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    options={'format':'bestvideo[height<=720]+bestaudio/best[height<=720]/best','outtmpl':str(destination/'%(id)s.%(ext)s'),'merge_output_format':'mp4','noplaylist':True,'socket_timeout':30,'retries':2,'overwrites':False}
    with YoutubeDL(options) as loader:
        info=loader.extract_info(validate_url(url),download=True);path=Path(loader.prepare_filename(info)).with_suffix('.mp4')
        if not path.exists():path=Path(loader.prepare_filename(info))
        (destination/'source.json').write_text(json.dumps({'url':url,'title':info.get('title'),'duration':info.get('duration'),'video_path':str(path)},ensure_ascii=False,indent=2))
        return path
def main():
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--out',default='data/rutube');p.add_argument('--step',type=float,default=2);p.add_argument('--local-video');a=p.parse_args();url=validate_url(a.url);out=Path(a.out)
    video=Path(a.local_video) if a.local_video else download(url,out/'download')
    from video_index import index_video
    rows=index_video(video,out/'index',a.step,url);print('Индекс:',out/'index','Кадров:',len(rows))
if __name__=='__main__':main()
