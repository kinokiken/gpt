"""Train two pose architectures on an explicitly supplied annotated dataset."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('data', help='Prepared YOLO dataset YAML with train/val paths')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--out', default='results/pose')
    args = parser.parse_args()
    if not Path(args.data).is_file():
        raise FileNotFoundError('Prepare and check the annotated dataset YAML first')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for weights, name in [('yolov8s-pose.pt', 'v8s'), ('yolo11s-pose.pt', '11s')]:
        net = YOLO(weights)
        net.train(data=str(Path(args.data).resolve()), epochs=args.epochs,
                  imgsz=args.size, batch=args.batch, device=args.device,
                  project=str(out), name=name, seed=0, workers=0)
        checkpoint = net.trainer.best
        metrics = YOLO(checkpoint).val(data=str(Path(args.data).resolve()),
                                       device=args.device, split='val')
        rows.append({'model': name, 'checkpoint': str(checkpoint),
                     'validation_pose_map': float(metrics.pose.map),
                     'validation_box_map': float(metrics.box.map)})
    (out / 'comparison.json').write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))

if __name__ == '__main__':
    main()
