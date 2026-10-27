from pathlib import Path
import json,hashlib,zipfile,shutil
P=Path(__file__).resolve().parents[1];D=P/'deliverables';D.mkdir(exist_ok=True)
# Keep only raw open inputs and prepared data; the 19 MB full video is retrieved by download_assets.py.
files=[]
for root in ['scripts','results','data']:
 for f in (P/root).rglob('*'):
  if not f.is_file():continue
  rel=f.relative_to(P)
  if '__pycache__' in f.parts or f.suffix=='.pyc':continue
  if f.name in ['last.pt','report_preview.png','report_pdf_text.txt']:continue
  if root=='data' and f.name in ['bacteria.webm','basketball1.png']:continue
  if 'rutube_' in f.name:continue
  files.append(f)
files += [P/'README.md',P/'requirements.txt',P/'report_blocks.json',P/'models/metric_head.pt',P/'models/RRDBNet_arch.py']
manifest=[]
for f in sorted(files):
 manifest.append({'path':str(f.relative_to(P)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(P/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));files.append(P/'MANIFEST.json')
for name,selection in [('Andreev_Lab4_Code_Results.zip',[f for f in files if f.relative_to(P).parts[0]!='data']),('Andreev_Lab4_Input_Data.zip',[f for f in files if f.relative_to(P).parts[0]=='data'])]:
 with zipfile.ZipFile(D/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for f in selection:z.write(f,'Lab4/'+str(f.relative_to(P)))
 with zipfile.ZipFile(D/name) as z:
  bad=z.testzip();assert bad is None,(name,bad)
 print(name,(D/name).stat().st_size, len(selection),flush=True)
shutil.copy2(P/'results/tracking_comparison.mp4',D/'Andreev_Lab4_Tracking.mp4')
shutil.copy2(P/'results/cvat/cvat_annotations.zip',D/'Andreev_Lab4_CVAT_Annotations.zip')
shutil.copy2(P/'results/cvat/cvat_task_images.zip',D/'Andreev_Lab4_CVAT_Images.zip')
