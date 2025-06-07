import os
import sys
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from models.PIMNet import PIMNet # Assuming PIMNet is in the models directory
from tensorboardX import SummaryWriter
from dataload.data_util import Config, Data # Assuming Config and Data are in dataload.data_util
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from tqdm import tqdm # You've already imported this!
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def loss_fn(pred, mask): # Renamed from 'loss' to avoid conflict with total_loss variable
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred_sigmoid = torch.sigmoid(pred) # Use a different variable name for sigmoid output
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


if __name__ == '__main__':
    nets_name = 'polyp'
    cfg = Config(datapath='/mnt/d/BaiduDownload/data/UltraEdit', # Make sure this path is correct
                 savepath='./pths/', mode='train',
                 batch=16, lr=1e-4, momen=0.9, decay=5e-4, epoch=30, lr_decay_gamma=0.1)

    data = Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=0)

    net = PIMNet(cfg) # Ensure PIMNet is correctly defined and imported
    save_tensorboard_dir = './tensorboard_server/' + nets_name + '/'
    if not os.path.exists(save_tensorboard_dir):
        os.makedirs(save_tensorboard_dir)
    save_pth_dir = './pths/' + nets_name + '/'
    if not os.path.exists(save_pth_dir):
        os.makedirs(save_pth_dir)

    net.train(True)
    net.cuda()

    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)

    optimizer = torch.optim.Adam([{'params': base}, {'params': head}], lr=cfg.lr, betas=(0.9, 0.999),
                                 weight_decay=cfg.decay)

    sw = SummaryWriter(
        save_tensorboard_dir)

    global_step = 0
    scaler = GradScaler() # Initialize GradScaler if using autocast

    for epoch in range(cfg.epoch):
        net.train(True)
        # --- Progress bar for the inner loop ---
        # Wrap the loader with tqdm
        # Add a description to the progress bar that includes the current epoch
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epoch}", unit="batch")

        current_step_in_epoch = 0 # Renamed from current_step to avoid confusion with global_step
        max_iter = cfg.epoch * len(loader)

        # Iterate over the progress_bar instead of loader directly
        for step, (image, mask) in enumerate(progress_bar):
            image, mask = image.cuda().float(), mask.cuda().float()

            # If you plan to use autocast, uncomment the following line
            # with autocast():
            pred, PRM1_out, PRM2_out, PRM3_out, PRM4_out = net(image)

            loss_pred_val = loss_fn(pred, mask) # Use the renamed loss function
            loss_PRM1_out_val = loss_fn(PRM1_out, mask)
            loss_PRM2_out_val = loss_fn(PRM2_out, mask)
            loss_PRM3_out_val = loss_fn(PRM3_out, mask)
            loss_PRM4_out_val = loss_fn(PRM4_out, mask)

            current_total_loss = loss_pred_val + loss_PRM1_out_val / 2 + loss_PRM2_out_val / 4 + loss_PRM3_out_val / 8 + loss_PRM4_out_val / 16

            optimizer.zero_grad()
            # If using GradScaler with autocast, uncomment these lines:
            # scaler.scale(current_total_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # If not using GradScaler:
            current_total_loss.backward()
            optimizer.step()

            global_step += 1
            current_step_in_epoch += 1

            current_iter = epoch * len(loader) + step + 1
            current_lr = poly_learning_rate(cfg.lr, current_iter, max_iter, power=0.9)
            optimizer.param_groups[0]['lr'] = current_lr
            optimizer.param_groups[1]['lr'] = current_lr

            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'total_loss': current_total_loss.item(),
                                     'loss_pred': loss_pred_val.item(),
                                     'loss_PRM1_out': loss_PRM1_out_val.item(),
                                     'loss_PRM2_out': loss_PRM2_out_val.item(),
                                     'loss_PRM3_out': loss_PRM3_out_val.item(),
                                     'loss_PRM4_out': loss_PRM4_out_val.item()}, global_step=global_step)

            # Update progress bar with current loss
            progress_bar.set_postfix(loss=f"{current_total_loss.item():.4f}", lr=f"{current_lr:.6f}")

            if step % 10 == 0:
                # The tqdm progress bar already shows progress,
                # so this print statement might be redundant unless you want specific formatting.
                # You can choose to keep it or remove it.
                print(f'{datetime.datetime.now()} | global_step:{global_step} | epoch_step:{current_step_in_epoch}/{len(loader)} | epoch:{epoch + 1}/{cfg.epoch} | lr={current_lr:.6f} | loss={current_total_loss.item():.6f}')

            del current_total_loss, loss_pred_val, loss_PRM1_out_val, loss_PRM2_out_val, loss_PRM3_out_val, loss_PRM4_out_val, pred, PRM1_out, PRM2_out, PRM3_out, PRM4_out

        if epoch >= 10: # Or some other condition like (epoch + 1) % save_interval == 0
            torch.save(net.state_dict(), os.path.join(save_pth_dir, f'epoch_{epoch + 1}.pth'))
    sw.close()
    print("Training finished.")