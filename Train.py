import os
import math
import argparse
import logging
from datetime import datetime
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from data.dataloader import get_loader, test_dataset
from models.detectors.net_pvt import FocusFormer
from utils.utils import build_scheduler, step_scheduler, get_current_lr
import eval.metrics as metrics

try:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_AVAILABLE = True
except Exception:
    _AMP_AVAILABLE = False
    autocast = None
    GradScaler = None


def load_model_state_dict(model, state_dict, print_matched=True):
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    num_matched, num_total = 0, 0
    curr = model.state_dict()
    for key in curr.keys():
        num_total += 1
        if key in state_dict and curr[key].shape == state_dict[key].shape:
            curr[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr)
    if print_matched:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')

def state_dict_for_saving(model: nn.Module):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

def weighted_bce_and_iou_loss(pred, mask):
    mask = F.interpolate(mask, size=(pred.size(2), pred.size(3)), mode="bilinear", align_corners=False)
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, weight=weit, reduction='none')
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou.unsqueeze(-1).unsqueeze(-1)).mean()

@torch.no_grad()
def val(test_loader, model, epoch, save_path, device, is_main=True):
    if not is_main:
        return
    global best_metric_dict, best_score, best_epoch
    SM = metrics.Smeasure(); WFM = metrics.WeightedFmeasure(); mae = metrics.MAE(); EM = metrics.Emeasure()
    metrics_dict = dict()

    model.eval()
    for i in range(test_loader.size):
        image, gt, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        image = image.to(device, non_blocking=True)

        outs = model(image)
        stage_list, pred = outs[0], outs[1]

        res = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=False).sigmoid()
        res = (res - res.amin(dim=(2, 3), keepdim=True)) / (
            res.amax(dim=(2, 3), keepdim=True) - res.amin(dim=(2, 3), keepdim=True) + 1e-8
        )
        pred_uint8 = (res * 255.0).clamp(0, 255).to(torch.uint8).squeeze(0).squeeze(0).cpu().numpy()
        SM.step(pred=pred_uint8, gt=gt); WFM.step(pred=pred_uint8, gt=gt); mae.step(pred=pred_uint8, gt=gt); EM.step(pred=pred_uint8, gt=gt)

    metrics_dict.update(Sm=SM.get_results()['sm'].round(3))
    metrics_dict.update(weightFm=WFM.get_results()['wfm'].round(3))
    metrics_dict.update(MAE=mae.get_results()['mae'].round(5))
    metrics_dict.update(meanEm=EM.get_results()['em']['curve'].mean().round(3))

    cur_score = (metrics_dict['Sm'] + metrics_dict['weightFm'] + metrics_dict['meanEm'])
    save_best = False

    if cur_score > best_score:
        best_metric_dict = metrics_dict; best_score = cur_score; best_epoch = epoch; save_best = True

    if save_best:
        torch.save(state_dict_for_saving(model), os.path.join(save_path, 'best.pth'))
        print(f'>>> Saved BEST(state_dict by metrics) at epoch {best_epoch} (score={best_score:.3f}).')

    torch.save(state_dict_for_saving(model), os.path.join(save_path, 'last.pth'))

    msg = ('[Cur Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})\n'
           '[Best Epoch: {}] Metrics (Sm={}, weightFm={}, MAE={}, meanEm={})').format(
           epoch, metrics_dict['Sm'], metrics_dict['weightFm'], metrics_dict['MAE'], metrics_dict['meanEm'],
           best_epoch, best_metric_dict['Sm'], best_metric_dict['weightFm'], best_metric_dict['MAE'], best_metric_dict['meanEm'])
    print(msg); logging.info(msg)



def train(train_loader, model, optimizer, epoch, device, total_step, use_amp=False, is_main=True, grad_accum=1):
    model.train()

    if use_amp and _AMP_AVAILABLE and device.type == 'cuda':
        amp_ctx = autocast; scaler = GradScaler()
    else:
        from contextlib import contextmanager
        @contextmanager
        def _null_ctx(): yield
        amp_ctx = _null_ctx; scaler = None

    running_total = 0.0
    optimizer.zero_grad(set_to_none=True)

    bce = torch.nn.BCEWithLogitsLoss()  

    for i, pack in enumerate(train_loader, start=1):
        images, gts = pack
        images = images.to(device, non_blocking=True)
        gts = gts.to(device, non_blocking=True).float()
        if gts.max() > 1.0: gts = gts / 255.0

        with amp_ctx():
            outs = model(images)

            stage_list, final_pred = outs[0], outs[1]
            p_fg5 = p_bg5 = None
            if len(outs) >= 3:
                sid = outs[2]
                if isinstance(sid, (tuple, list)) and len(sid) >= 2:
                    p_fg5, p_bg5 = sid[0], sid[1]
                elif isinstance(sid, dict):
                    p_fg5, p_bg5 = sid.get('p_fg5', None), sid.get('p_bg5', None)

            aux_preds = stage_list[:-1] if len(stage_list) > 0 else []
            losses_aux = [weighted_bce_and_iou_loss(out, gts) for out in aux_preds]
            gamma = 0.2
            weights = [gamma * (j + 1) for j in range(len(losses_aux))]
            loss_result1 = sum(w * l for w, l in zip(weights, losses_aux)) if losses_aux else 0.0

            loss_result2 = weighted_bce_and_iou_loss(final_pred, gts)

            loss_sid = 0.0
            if (p_fg5 is not None) and (p_bg5 is not None):
                gt5 = F.interpolate(gts, size=p_fg5.shape[-2:], mode='nearest')
                loss_fg5 = weighted_bce_and_iou_loss(p_fg5, gt5)
                loss_bg5 = weighted_bce_and_iou_loss(p_bg5, 1.0 - gt5)
                loss_sid = loss_fg5 + loss_bg5

            loss = loss_result1 + loss_result2 + opt.lambda_sid * loss_sid

        loss_to_backward = loss / grad_accum
        if scaler is not None:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if i % grad_accum == 0:
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_total += loss.detach().item()

        if is_main and (i % 80 == 0 or i == total_step):
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_aux: {:0.4f}, loss_main: {:0.4f}, loss_sid: {:0.4f}, loss_total: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         float(loss_result1), float(loss_result2), float(loss_sid), float(loss)))
            logging.info('Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss_aux: {:0.4f}, loss_main: {:0.4f}, loss_sid: {:0.4f}, loss_total: {:0.4f}'.
                         format(epoch, opt.epoch, i, total_step,
                                float(loss_result1), float(loss_result2), float(loss_sid), float(loss)))

    if is_main and epoch % opt.epoch_save == 0:
        torch.save(state_dict_for_saving(model), os.path.join(opt.save_path, f'{epoch}.pth'))

    return running_total / total_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--augmentation', action='store_true')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--trainsize', type=int, default=384)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--train_path', type=str, default='/home/XY/VSSNet-text/COD-Traindataset')
    parser.add_argument('--test_path', type=str, default='/home/XY/VSSNet-text/COD-TestDataset')
    parser.add_argument('--save_path', type=str, default='./checkpoints/smt_s-2/')
    parser.add_argument('--epoch_save', type=int, default=10)
    parser.add_argument('--lambda_sid', type=float, default=0.2, help='f5 粗分支 (fg/bg) 监督损失权重')
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_tf32', action='store_true')
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--channels_last', action='store_true')
    parser.add_argument('--ema_alpha', type=float, default=0.9, help='EMA 平滑系数')
    parser.add_argument('--trainloss_min_delta', type=float, default=1e-4, help='最小改进阈值')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='开始比较最佳前的预热轮数')
    parser.add_argument('--use_ddp', action='store_true')
    parser.add_argument('--grad_accum', type=int, default=1, help='梯度累积步数（每 rank）')

    global opt
    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(opt.save_path, 'log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # --------- DDP 初始化 ---------
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    use_ddp = opt.use_ddp and torch.cuda.is_available() and world_size > 1

    if use_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=1800))
        device = torch.device('cuda', local_rank)
    else:
        # 单卡/CPU
        # if torch.cuda.is_available():
        #     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu  # 仅单卡时使用
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    is_main = (not use_ddp) or (dist.get_rank() == 0)

    if is_main:
        print(">>>:", opt); logging.info(">>>:%s", opt)
        print(f"[CUDA] available={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}, "
              f"visible_gpus={os.environ.get('CUDA_VISIBLE_DEVICES','ALL')}")

    torch.backends.cudnn.benchmark = True

    if opt.use_tf32 and device.type == 'cuda':
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as e:
            if is_main:
                print("[TF32] enabling failed:", e)


    model = FocusFormer().to(device)
    if opt.channels_last and device.type == 'cuda':
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as e:
            if is_main: print("[channels_last] model convert failed:", e)

    if use_ddp:

        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, static_graph=True, find_unused_parameters=False)
    else:

        pass

    if opt.load is not None:
        map_loc = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if use_ddp else device
        model_dict = torch.load(opt.load, map_location=map_loc)
        if is_main: print("start loading model:", opt.load)
        load_model_state_dict(model.module if hasattr(model, 'module') else model, model_dict)
        if is_main: print('Successfully loaded.')

    if is_main:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('model parameters', num_params)
        logging.info("model parameters: %d", num_params)


    params = model.parameters()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, lr=opt.lr, weight_decay=2e-4, momentum=0.9)
    if is_main: print(optimizer); logging.info("optimizer: %s", optimizer)
 
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=opt.warmup_epochs,      
        warmup_start_factor=0.1,
        num_epochs=opt.epoch,          
        eta_min=1e-6,                  
        use_warm_restarts=False,       
        T_mult=1,
    )

    image_root = f'{opt.train_path}/Image/'; gt_root = f'{opt.train_path}/GT/'
    base_train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                                   augmentation=opt.augmentation)
    train_set = base_train_loader.dataset
    num_workers = getattr(base_train_loader, 'num_workers', 8)

    if use_ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=False)
        train_loader = DataLoader(train_set, batch_size=opt.batchsize, sampler=train_sampler,
                                  num_workers=num_workers if num_workers is not None else 8,
                                  pin_memory=True, persistent_workers=(num_workers if num_workers else 0) > 0,
                                  prefetch_factor=2 if (num_workers if num_workers else 0) > 0 else None)
    else:
        train_loader = base_train_loader
        train_sampler = None

    total_step = len(train_loader)


    val_loader = None
    if is_main and opt.eval_every > 0:
        val_loader = test_dataset(
            image_root=os.path.join(opt.test_path, 'CHAMELEON/Image/'),
            gt_root=os.path.join(opt.test_path, 'CHAMELEON/GT/'),
            testsize=opt.trainsize
        )

    global best_epoch, best_score, best_metric_dict
    best_epoch = 0
    best_score = float('-inf')         
    best_metric_dict = {}

    trainloss_ema = None
    best_train_ema_val = float('inf')

    for epoch in range(1, opt.epoch + 1):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        avg_train_loss = train(train_loader, model, optimizer, epoch, device, total_step,
                               use_amp=opt.use_amp, is_main=is_main, grad_accum=opt.grad_accum)
        
        step_scheduler(scheduler)
        if is_main:
            curr_lr = get_current_lr(optimizer)
            print(f"[Epoch {epoch:03d}] lr={curr_lr:.6g}, avg_train_loss={avg_train_loss:.6f}")
            logging.info(f"[Epoch {epoch:03d}] lr={curr_lr:.6g}, avg_train_loss={avg_train_loss:.6f}")

        if is_main:
            if trainloss_ema is None:
                trainloss_ema = avg_train_loss
            else:
                trainloss_ema = opt.ema_alpha * trainloss_ema + (1 - opt.ema_alpha) * avg_train_loss

            print(f'[Epoch {epoch:03d}] avg_train_loss={avg_train_loss:.6f}, ema_train_loss={trainloss_ema:.6f}')
            logging.info(f'[Epoch {epoch:03d}] avg_train_loss={avg_train_loss:.6f}, ema_train_loss={trainloss_ema:.6f}')

            if epoch >= opt.warmup_epochs and (trainloss_ema + opt.trainloss_min_delta) < best_train_ema_val:
                best_train_ema_val = trainloss_ema
                torch.save(state_dict_for_saving(model), os.path.join(opt.save_path, 'best_by_trainloss.pth'))
                print(f'>>> Saved BEST(by train-EMA) at epoch {epoch}: ema_loss={best_train_ema_val:.6f}')
                logging.info(f'>>> Saved BEST(by train-EMA) at epoch {epoch}: ema_loss={best_train_ema_val:.6f}')

            if opt.eval_every > 0 and epoch % opt.eval_every == 0:
                val_loader.index = 0
                val(val_loader, model, epoch, opt.save_path, device, is_main=is_main)

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == '__main__':
    opt = argparse.Namespace(ema_alpha=0.9, trainloss_min_delta=1e-4, warmup_epochs=3)
    main()


