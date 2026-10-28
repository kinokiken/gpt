"""Compare interpolation against a known reference and optionally run Tesseract."""
import argparse
import json
import subprocess
from pathlib import Path
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('image')
    parser.add_argument('--out', default='results/image_controls')
    parser.add_argument('--ocr', action='store_true')
    args = parser.parse_args()
    reference = cv2.imread(args.image)
    if reference is None or min(reference.shape[:2]) < 16:
        raise ValueError('Readable image at least 16 pixels wide and high required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    height, width = reference.shape[:2]
    small = cv2.resize(reference, (width//4, height//4), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / 'low_resolution.png'), small)
    metrics = {}
    for name, mode in [('nearest', cv2.INTER_NEAREST),
                       ('bilinear', cv2.INTER_LINEAR), ('bicubic', cv2.INTER_CUBIC)]:
        restored = cv2.resize(small, (width, height), interpolation=mode)
        cv2.imwrite(str(out / f'{name}.png'), restored)
        metrics[name] = {'PSNR': float(peak_signal_noise_ratio(reference, restored, data_range=255)),
                         'SSIM': float(structural_similarity(reference, restored,
                                                            channel_axis=2, data_range=255))}
    if args.ocr:
        result = subprocess.run(['tesseract', args.image, 'stdout', '-l', 'eng', '--psm', '6'],
                                capture_output=True, text=True, check=True)
        metrics['ocr_text'] = result.stdout.strip()
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
