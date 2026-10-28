"""Download open Treponema material from Wikimedia Commons. No RuTube access."""
from pathlib import Path
import json
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
RAW.mkdir(parents=True, exist_ok=True)
session = requests.Session()
session.headers['User-Agent'] = 'Lab4ComputerVision/2.0 (educational; https://github.com/kinokiken/gpt)'
titles = [
    'TreponemaPallidum.jpg',
    'Treponema pallidum.jpg',
    'Treponema pallidum 01.png',
    'Treponemapallidum.jpg',
    'Treponema pallidum Bacteria (Syphilis) - NIAID.jpg',
    'Syphilis Bacteria (16842837731).jpg',
    'Ultramicroscope time-lapse of syphilis parasite-Comandon-1910.ogv',
]
rows = []
for number, title in enumerate(titles, 1):
    response = session.get('https://commons.wikimedia.org/w/api.php', params={
        'action': 'query', 'format': 'json', 'titles': 'File:' + title,
        'prop': 'imageinfo', 'iiprop': 'url|extmetadata|size'}, timeout=60)
    response.raise_for_status()
    info = next(iter(response.json()['query']['pages'].values()))['imageinfo'][0]
    suffix = Path(title).suffix.lower()
    destination = RAW / f't{number:02d}{suffix}'
    if not destination.exists():
        data = session.get(info['url'], timeout=180)
        data.raise_for_status()
        destination.write_bytes(data.content)
    rows.append({'id': f't{number:02d}', 'title': title, 'file': destination.name,
                 'width': info.get('width'), 'height': info.get('height'),
                 'source_page': 'https://commons.wikimedia.org/wiki/File:' + title.replace(' ', '_'),
                 'artist': info.get('extmetadata', {}).get('Artist', {}).get('value', ''),
                 'license': info.get('extmetadata', {}).get('LicenseShortName', {}).get('value', '')})
(RAW / 'sources.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print('Downloaded/checked', len(rows), 'open files')
