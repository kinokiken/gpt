"""Download official model artifacts; never access RuTube."""
from experiment_common import CACHE
import argparse
from huggingface_hub import hf_hub_download
import requests

def main():
    p = argparse.ArgumentParser()
    p.add_argument('group', choices=['clip','vision'])
    args = p.parse_args()
    if args.group == 'clip':
        from transformers import CLIPModel, CLIPProcessor
        CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True)
        CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    else:
        from ultralytics import YOLO
        for name in ['yolov8s-pose.pt', 'yolo11s-pose.pt', 'yolov8s-worldv2.pt', 'FastSAM-s.pt']:
            YOLO(str(CACHE/name))
        path = CACHE/'pose_landmarker_lite.task'
        if not path.exists():
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        hf_hub_download('ai-forever/Real-ESRGAN', 'RealESRGAN_x4.pth')
    print('Prepared', args.group, flush=True)

if __name__ == '__main__':
    main()
