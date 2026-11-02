import numpy as np
import torch
import torch.nn.functional as F
import os, argparse
import cv2
from data.dataloader import My_test_dataset
from models.detectors.net_pvt import FocusFormer

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='./checkpoints/best/best.pth')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = FocusFormer().to(device).eval()
ckpt = torch.load(opt.pth_path, map_location='cpu')
if isinstance(ckpt, dict) and 'model' in ckpt:
    ckpt = ckpt['model']
if any(k.startswith('module.') for k in ckpt.keys()):
    ckpt = {k.replace('module.', '', 1): v for k, v in ckpt.items()}
model.load_state_dict(ckpt, strict=False)

for _data_name in ['CHAMELEON','CAMO', 'COD10K','NC4K']:
    data_path = '/home/XY/VSSNet-main/COD-TestDataset/{}/'.format(_data_name)
    save_path = './res/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    os.makedirs(save_path, exist_ok=True)

    image_root = '{}/Image/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    print('root', image_root, gt_root)

    test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
    print('****', test_loader.size)

    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            print('***name', name)
            gt = np.asarray(gt, np.float32)
            image = image.to(device, non_blocking=True)

            outs = model(image)
            if isinstance(outs, (tuple, list)):
                if len(outs) == 2:
                    stage_list, pred = outs
                elif len(outs) >= 3:
                    stage_list, pred = outs[0], outs[1]
                else:
                    raise RuntimeError(f"Unexpected model outputs: len={len(outs)}")
            else:
                stage_list, pred = [], outs

            pred_full = stage_list[-1] + pred if len(stage_list) > 0 else pred
            res = F.interpolate(pred_full, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            print('> {} - {}'.format(_data_name, name))
            cv2.imwrite(os.path.join(save_path, name), (res * 255).astype(np.uint8))
