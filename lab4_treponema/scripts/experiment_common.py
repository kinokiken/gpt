"""Shared paths and reproducible CPU configuration for the laboratory experiments."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT.parent / 'lab4'
CACHE = ROOT / '.cache'
OUT = ROOT / 'results/final'
OUT.mkdir(parents=True, exist_ok=True)
for name, value in {'HF_HOME': CACHE/'hf', 'TORCH_HOME': CACHE/'torch',
                    'YOLO_CONFIG_DIR': CACHE/'yolo', 'MPLCONFIGDIR': CACHE/'mpl',
                    'XDG_CACHE_HOME': CACHE}.items():
    os.environ[name] = str(value)
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'
import json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
torch.set_num_threads(2)
cv2.setNumThreads(2)
torch.manual_seed(7)
np.random.seed(7)

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def panel(name, images, titles, cols=3):
    rows = (len(images)+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.4*rows), squeeze=False)
    for i, axis in enumerate(axes.flat):
        if i < len(images):
            im = images[i]
            axis.imshow(im, cmap='gray' if im.ndim == 2 else None)
            axis.set_title(titles[i], fontsize=10)
        axis.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)

def rgb(path):
    from PIL import Image
    return np.array(Image.open(path).convert('RGB'))

def skeleton_points(mask):
    """Return start/midpoint/end along the longest reachable skeleton path."""
    from skimage.morphology import skeletonize
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    yy,xx=np.nonzero(skeletonize(mask>0))
    coordinates=np.column_stack([xx,yy])
    lookup={(int(x),int(y)):i for i,(x,y) in enumerate(coordinates)}
    rows=[];columns=[];weights=[]
    for (x,y),i in lookup.items():
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            j=lookup.get((x+dx,y+dy))
            if j is not None:rows.append(i);columns.append(j);weights.append(float(np.hypot(dx,dy)))
    graph=coo_matrix((weights,(rows,columns)),shape=(len(xx),len(xx))).tocsr()
    distance=dijkstra(graph,indices=0)
    start=int(np.argmax(np.where(np.isfinite(distance),distance,-1)))
    distance,previous=dijkstra(graph,indices=start,return_predecessors=True)
    end=int(np.argmax(np.where(np.isfinite(distance),distance,-1)))
    indices=[end]
    while indices[-1]!=start:
        indices.append(int(previous[indices[-1]]))
    path=coordinates[indices[::-1]].astype(float)
    if tuple(path[0])>tuple(path[-1]):path=path[::-1]
    length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    midpoint=path[int(np.searchsorted(length,length[-1]/2))]
    return np.stack([path[0],midpoint,path[-1]]),path
