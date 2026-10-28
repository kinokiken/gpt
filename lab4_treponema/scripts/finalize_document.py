"""Paginate the saved DOCX locally and fill its contents page. No experiments."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
from docx import Document
from docx.enum.text import WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'deliverables'
NAME = 'Andreev_Lab4_Treponema_Report'
PREVIEW = ROOT/'.cache/report-preview'

def export():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    profile = (ROOT/'.cache/lo-submission').as_uri()
    subprocess.run(['soffice', '--headless', '-env:UserInstallation='+profile,
                    '--convert-to', 'pdf', '--outdir', str(PREVIEW),
                    str(DEST/(NAME+'.docx'))], check=True)
    return PREVIEW/(NAME+'.pdf')

def contents_pages(pdf, headings):
    result = subprocess.run(['pdftotext', '-layout', str(pdf), '-'],
                            check=True, capture_output=True, text=True)
    pages = result.stdout.split('\f')
    clean = lambda s: re.sub(r'\s+', ' ', s).strip()
    found = {}
    for heading in headings:
        matches = [i+1 for i,page in enumerate(pages) if i>=2 and clean(heading) in clean(page)]
        if not matches:
            raise ValueError('Heading not found: '+heading)
        found[heading] = matches[0]
    return found

def main():
    blocks = json.loads((ROOT/'report_blocks.json').read_text())
    headings = [b['text'] for b in blocks if b['kind']=='h' and b['level']==1]
    pages = contents_pages(export(), headings)
    doc = Document(DEST/(NAME+'.docx'))
    for heading in headings:
        paragraph = next(p for p in doc.paragraphs if p.text==heading and p.style.name=='Normal')
        paragraph.text = heading+'\t'+str(pages[heading])
        paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(16.3), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
        for run in paragraph.runs:
            run.font.name='Times New Roman'
            run.font.size=Pt(12)
    doc.save(DEST/(NAME+'.docx'))
    pdf = export()
    final_pages = contents_pages(pdf, headings)
    if final_pages != pages:
        raise RuntimeError('Contents changed pagination; update required')
    shutil.copy2(pdf, DEST/(NAME+'.pdf'))
    (DEST/'contents_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved DOCX and matching PDF with page numbers in contents')

if __name__=='__main__':
    main()
