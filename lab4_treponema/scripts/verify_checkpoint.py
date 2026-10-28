"""Offline validation for the partial checkpoint."""
from pathlib import Path
import importlib.util
import json
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rutube_download', ROOT/'scripts/rutube_download.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
valid = 'https://rutube.ru/video/0123456789abcdef0123456789abcdef/'
assert module.validate_url(valid) == valid
for bad in ['http://rutube.ru/video/0123456789abcdef0123456789abcdef/',
            'https://example.com/video/0123456789abcdef0123456789abcdef/']:
    try:
        module.validate_url(bad)
        raise AssertionError('Invalid URL accepted')
    except ValueError:
        pass
assert Image.open(ROOT/'data/raw/treponema_primary.jpg').size == (870, 540)
sources = json.loads((ROOT/'data/raw/sources.json').read_text(encoding='utf-8'))
assert len(sources) == 7 and (ROOT/'data/raw/t07.ogv').stat().st_size > 1_000_000
result = json.loads((ROOT/'results/preparation.json').read_text(encoding='utf-8'))
assert result['connected_regions'] == 6
assert abs(result['mask_fraction'] - .2254810557684121) < 1e-9
for name in ['primary_analysis.png','sources_contact.png','video_contact.png','tracking_metrics.png',
             'superresolution_metrics.png','retrieval_metric.png','zero_shot_summary.png',
             'mlflow_models.png','pose_comparison.png']:
    with Image.open(ROOT/'results'/name) as image:
        image.verify()
print('PASS: data, figures and offline RuTube URL validation')
