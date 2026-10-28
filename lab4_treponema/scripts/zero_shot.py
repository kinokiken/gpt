"""Run YOLO-World on a local image and record actual detections."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--image', default='data/raw/treponema_primary.jpg')
    parser.add_argument('--weights', default='yolov8s-world.pt')
    parser.add_argument('--out', default='results/world')
    parser.add_argument('--conf', type=float, default=0.10)
    args = parser.parse_args()
    net = YOLO(args.weights)
    prompts = ['Treponema pallidum', 'spirochete', 'bacterium']
    net.set_classes(prompts)
    result = net(args.image, conf=args.conf, verbose=False)[0]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.save(filename=str(out / 'prediction.jpg'))
    rows = [{'xyxy': box.xyxy[0].tolist(), 'confidence': float(box.conf[0]),
             'class': result.names[int(box.cls[0])]} for box in result.boxes]
    payload = {'image': args.image, 'prompts': prompts, 'threshold': args.conf,
               'detections': rows}
    (out / 'detections.json').write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

if __name__ == '__main__':
    main()
