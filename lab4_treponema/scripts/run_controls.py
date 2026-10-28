"""Human pose controls, RRDB superresolution and OCR on Treponema material."""
from experiment_common import *
import sys
import subprocess
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def main():
    from ultralytics import YOLO
    pose = YOLO(CACHE/'yolov8s-pose.pt')
    images, titles, observations = [], [], {}
    for name, path in [('people', LEGACY/'data/raw/bus.jpg'),
                       ('treponema', ROOT/'data/raw/treponema_primary.jpg')]:
        for threshold in ([.5] if name == 'people' else [.5, .25, .05]):
            result = pose(path, conf=threshold, imgsz=640, device='cpu', verbose=False)[0]
            observations[f'{name}_{threshold}'] = {'boxes':len(result.boxes), 'k':17}
            images.append(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
            titles.append(f'YOLOv8s-pose: {name}, conf={threshold}, N={len(result.boxes)}')
    panel('human_pose.png', images, titles, cols=2)
    import mediapipe as mp
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(CACHE/'pose_landmarker_lite.task')),
        num_poses=4)
    images, titles = [], []
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        for name, path in [('people',LEGACY/'data/raw/bus.jpg'),
                           ('treponema',ROOT/'data/raw/treponema_primary.jpg')]:
            im = rgb(path)
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
            for person in result.pose_landmarks:
                for point in person:
                    cv2.circle(im, (round(point.x*im.shape[1]), round(point.y*im.shape[0])),
                               4, (255,100,0), -1)
            observations['mediapipe_'+name] = {'people':len(result.pose_landmarks), 'k':33}
            images.append(im)
            titles.append(f'MediaPipe: {name}; N={len(result.pose_landmarks)}')
    panel('mediapipe.png', images, titles, cols=2)
    save('human_controls.json', observations)

    sys.path.insert(0, str(LEGACY/'models'))
    from RRDBNet_arch import RRDBNet
    model = RRDBNet(3,3,64,23,gc=32)
    path = hf_hub_download('ai-forever/Real-ESRGAN','RealESRGAN_x4.pth')
    raw = torch.load(path, map_location='cpu', weights_only=True)
    mapping = {'body':'RRDB_trunk','conv_body':'trunk_conv',
               'conv_up1':'upconv1','conv_up2':'upconv2','conv_hr':'HRconv'}
    state = {'.'.join([mapping.get(k.split('.')[0], k.split('.')[0])] +
                      [p.upper() if p.startswith('rdb') else p for p in k.split('.')[1:]]):v
             for k,v in raw.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    reference = cv2.resize(rgb(ROOT/'data/raw/treponema_primary.jpg'), (384,240),
                           interpolation=cv2.INTER_AREA)
    small = cv2.resize(reference, (96,60), interpolation=cv2.INTER_AREA)
    tensor = torch.tensor(small.transpose(2,0,1).copy()).float()[None]/255
    with torch.no_grad():
        neural = np.round(model(tensor).clamp(0,1)[0].permute(1,2,0).numpy()*255).astype('uint8')
    cubic = cv2.resize(small, (384,240), interpolation=cv2.INTER_CUBIC)
    metrics = {}
    for name, im in [('Bicubic',cubic),('Real-ESRGAN',neural)]:
        metrics[name] = {'PSNR':float(peak_signal_noise_ratio(reference, im, data_range=255)),
                         'SSIM':float(structural_similarity(reference,im,channel_axis=2,data_range=255))}
    panel('superresolution_actual.png', [reference,cubic,neural], ['Эталон','Bicubic ×4','Real-ESRGAN ×4'])
    save('superresolution_actual.json', metrics)

    # A known text label is intentionally drawn onto a microscopy illustration for the OCR control.
    label = Image.new('RGB',(600,100),'white')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',34)
    reference_text = 'Treponema pallidum 30 um'
    ImageDraw.Draw(label).text((12,25),reference_text,font=font,fill='black')
    label.save(OUT/'ocr_reference.png')
    small = cv2.resize(np.array(label), (150,25), interpolation=cv2.INTER_AREA)
    cubic = cv2.resize(small,(600,100),interpolation=cv2.INTER_CUBIC)
    tensor = torch.tensor(small.transpose(2,0,1).copy()).float()[None]/255
    with torch.no_grad():
        neural_text = np.round(model(tensor).clamp(0,1)[0].permute(1,2,0).numpy()*255).astype('uint8')
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=str(CACHE/'easyocr'),
                            user_network_directory=str(CACHE/'easyocr/user'))
    def distance(a,b):
        previous = list(range(len(b)+1))
        for i,ca in enumerate(a,1):
            current = [i]
            for j,cb in enumerate(b,1):
                current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
            previous=current
        return previous[-1]
    rows=[]
    for name, im in [('original',np.array(label)),('bicubic',cubic),('Real-ESRGAN',neural_text)]:
        file=OUT/f'ocr_{name}.png'
        Image.fromarray(im).save(file)
        tess=subprocess.run(['tesseract',str(file),'stdout','-l','eng','--psm','7'],
                            capture_output=True,text=True,check=True).stdout.strip()
        easy=' '.join(reader.readtext(im,detail=0))
        rows.append({'variant':name,'reference':reference_text,'Tesseract':tess,'EasyOCR':easy,
                     'CER_tesseract':distance(reference_text,tess)/len(reference_text),
                     'CER_easyocr':distance(reference_text,easy)/len(reference_text)})
    panel('ocr_comparison.png',[np.array(label),cubic,neural_text],
          ['Контрольная надпись','Уменьшение ×4 → Bicubic','Уменьшение ×4 → Real-ESRGAN'],cols=1)
    save('ocr_actual.json',rows)
    print('Completed human, SR and OCR controls', flush=True)

if __name__ == '__main__':
    main()
