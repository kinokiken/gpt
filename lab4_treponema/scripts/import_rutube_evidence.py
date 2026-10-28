"""Extract the author's screenshots from the submitted DOCX, without any web request."""
import argparse
import json
from pathlib import Path
from docx import Document

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('document')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = root / 'results/final'
    out.mkdir(parents=True, exist_ok=True)
    doc = Document(args.document)
    selected = False
    text, images = [], []
    for paragraph in doc.paragraphs:
        if paragraph.style.name == 'Heading 1':
            selected = paragraph.text.startswith('7.')
        if not selected:
            continue
        if paragraph.text:
            text.append(paragraph.text)
        for node in paragraph._p.xpath('.//a:blip'):
            rid = node.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            part = doc.part.related_parts[rid]
            suffix = Path(str(part.partname)).suffix
            name = f'rutube_author_{len(images)+1}{suffix}'
            (out / name).write_bytes(part.blob)
            images.append(name)
    metadata = {'origin': 'author submitted DOCX', 'screenshots': images,
                'paragraphs': text, 'indexed_frames_from_author_terminal': 81,
                'source_url': 'https://rutube.ru/video/0b337cb693454b8b734f4d4b87f856a8/',
                'title': 'Наглядная микробиология #5 Бактерии (Bacteria)',
                'query': 'Treponema pallidum spirochetes moving in a microscope field',
                'hits_transcribed_from_screenshot': [
                    {'seconds':144, 'cosine':.341}, {'seconds':2, 'cosine':.339},
                    {'seconds':148, 'cosine':.337}]}
    (out / 'rutube_author.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('Saved', len(images), 'original screenshots')

if __name__ == '__main__':
    main()
