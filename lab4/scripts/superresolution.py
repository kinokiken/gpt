from pathlib import Path
import sys,json,torch,cv2,numpy as np
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'models'));from RRDBNet_arch import RRDBNet
torch.set_num_threads(2)
net=RRDBNet(3,3,64,23,gc=32);raw=torch.load(P/'models/RRDB_ESRGAN_x4.pth',map_location='cpu',weights_only=True)
# Equivalent RRDB implementation uses different module names; preserve every tensor.
state={k.replace('body.','RRDB_trunk.').replace('.rdb','.RDB').replace('conv_body.','trunk_conv.').replace('conv_up1.','upconv1.').replace('conv_up2.','upconv2.').replace('conv_hr.','HRconv.'):v for k,v in raw.items()}
# Repair conv_body name without partial string substitution.
state={k:v for k,v in state.items() if not k.startswith('conv_RRDB_trunk.')}
for key in ['weight','bias']:state['trunk_conv.'+key]=raw['conv_body.'+key]
net.load_state_dict(state,strict=True);net.eval()
im=cv2.imread(str(P/'results/sr_lowres.png'));inp=torch.from_numpy(cv2.cvtColor(im,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()[None]/255
with torch.no_grad():out=net(inp).clamp(0,1)[0].permute(1,2,0).numpy()
out=np.round(out*255).astype('uint8');cv2.imwrite(str(P/'results/sr_neural.png'),cv2.cvtColor(out,cv2.COLOR_RGB2BGR));ref=cv2.cvtColor(cv2.imread(str(P/'results/sr_reference.png')),cv2.COLOR_BGR2RGB)
metrics={'PSNR':float(peak_signal_noise_ratio(ref,out,data_range=255)),'SSIM':float(structural_similarity(ref,out,channel_axis=2,data_range=255)),'model_source':'https://huggingface.co/ai-forever/Real-ESRGAN','architecture':'RRDB 23 blocks, x4; parameter-name mapping with strict loading'}
p=P/'results/zero_shot_controls.json';d=json.loads(p.read_text());d['Real_ESRGAN']=metrics;p.write_text(json.dumps(d,ensure_ascii=False,indent=2));print(metrics)
