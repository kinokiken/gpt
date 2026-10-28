"""Track an explicit ROI and measure local optical flow on a local video."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('video')
    parser.add_argument('--box', type=int, nargs=4, required=True, metavar=('X','Y','W','H'))
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--out', default='results/video_analysis')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    ok, first = cap.read()
    if not ok:
        raise ValueError('Cannot decode the first video frame')
    x, y, w, h = args.box
    if min(x,y) < 0 or min(w,h) <= 0 or x+w > first.shape[1] or y+h > first.shape[0]:
        raise ValueError('Initial box must be inside the frame')
    factories = {'MIL': cv2.legacy.TrackerMIL_create,
                 'KCF': cv2.legacy.TrackerKCF_create,
                 'CSRT': cv2.legacy.TrackerCSRT_create}
    trackers = {name: factory() for name, factory in factories.items()}
    for tracker in trackers.values():
        tracker.init(first, tuple(args.box))
    previous = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    rows = []
    for number in range(1, args.frames):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous, gray, None, .5, 3, 15, 3, 5, 1.2, 0)
        vector = flow[y:y+h, x:x+w].mean(axis=(0, 1)).tolist()
        row = {'frame': number, 'flow_in_initial_roi_px': vector, 'trackers': {}}
        for name, tracker in trackers.items():
            start = time.perf_counter()
            found, box = tracker.update(frame)
            elapsed = time.perf_counter() - start
            row['trackers'][name] = {'found': bool(found), 'box': list(box), 'seconds': elapsed}
            if number % 12 == 0:
                annotated = frame.copy()
                if found:
                    bx, by, bw, bh = map(int, box)
                    cv2.rectangle(annotated, (bx, by), (bx+bw, by+bh), (0,255,0), 1)
                cv2.imwrite(str(out / f'{name}_{number:04d}.jpg'), annotated)
        rows.append(row)
        previous = gray
    cap.release()
    (out / 'observations.json').write_text(json.dumps(rows, indent=2))
    print('Saved', len(rows), 'frames; IoU requires independent reference boxes.')

if __name__ == '__main__':
    main()
