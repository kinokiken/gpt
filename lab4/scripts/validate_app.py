from pathlib import Path
import json
from streamlit.testing.v1 import AppTest
P=Path(__file__).resolve().parents[1]
a=AppTest.from_file(str(P/'scripts/app.py')).run(timeout=60)
assert not a.exception,[e.message for e in a.exception]
a.button[0].click().run(timeout=90)
assert not a.exception,[e.message for e in a.exception]
texts=[m.value for m in a.markdown];hits=[s for s in texts if 'Время:' in s];assert len(hits)==3,hits
(P/'results/app_check.json').write_text(json.dumps({'initial_render':True,'search_button':True,'results_rendered':len(hits),'result_texts':hits},ensure_ascii=False,indent=2));print(hits)
