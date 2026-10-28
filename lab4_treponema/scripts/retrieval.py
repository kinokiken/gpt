"""Index a folder of images and search it with CLIP."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from video_index import model

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--images', default='data/raw')
    parser.add_argument('--out', default='results/retrieval')
    parser.add_argument('--query', default='Treponema pallidum spirochetes')
    args = parser.parse_args()
    files = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in {'.jpg', '.png', '.jpeg'})
    if not files:
        raise ValueError('No images in the input directory')
    net, processor = model()
    vectors = []
    with torch.no_grad():
        for path in files:
            with Image.open(path) as image:
                inputs = processor(images=image.convert('RGB'), return_tensors='pt')
            vectors.append(F.normalize(net.get_image_features(**inputs), dim=1)[0].numpy())
        inputs = processor(text=[args.query], padding=True, return_tensors='pt')
        query = F.normalize(net.get_text_features(**inputs), dim=1)[0].numpy()
    matrix = np.stack(vectors)
    scores = matrix @ query
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'vectors.npy', matrix)
    rows = [{'file': str(files[i]), 'cosine': float(scores[i])}
            for i in np.argsort(-scores)[:3]]
    result = {'query': args.query, 'files': list(map(str, files)), 'hits': rows}
    (out / 'search.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
