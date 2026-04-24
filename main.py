import os

os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from time import time,sleep
import argparse
import logging
import os

import scipy.io as sio
import random
import matplotlib.pyplot as plt
from accelerate import Accelerator
from ema_pytorch import EMA
from model import ModelT
from dataset import EITdataset

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def lossloc():
    x = np.linspace(-1, 1, 128)
    y = np.linspace(-1, 1, 128)
    X, Y = np.meshgrid(x, y)
    x_shape = X.shape
    X = X.reshape([-1, 1])
    Y = Y.reshape([-1, 1])
    Res = []
    for i, theta in enumerate(reversed(range(0, 16))):
        theta = np.deg2rad((theta + 4) * 22.5 + 22.5 / 2)
        m = [np.cos(theta) * 0.8, np.sin(theta) * 0.8]
        res = (X - m[0]) ** 2 + (Y - m[1]) ** 2
        res = np.exp(-res/0.5)
        res = res.reshape(x_shape)
        Res.append(res)
    return np.stack(Res)


def lossweight(v1):
    # vi [b,1,16,16]
    v1 = v1[:, 0]
    v1 = torch.mean(v1, 2).abs()  # [b,16]
    v1 = v1.unsqueeze(0).unsqueeze(0)
    return v1


def init_seed(seed=2019, reproducibility=True) -> None:
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    accelerator = Accelerator(mixed_precision='no')
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    seed = args.global_seed
    init_seed(seed)
    device = accelerator.device
    gpus = torch.cuda.device_count()
    if accelerator.is_local_main_process:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        checkpoint_dir = args.results_dir

        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{checkpoint_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Experiment directory created at {checkpoint_dir}")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    model = ModelT(modelname=args.results_dir[0:3])
    model = model.to(device)
    



    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"batch size one gpu: {args.global_batch_size}")
    logger.info(f"gpus: {gpus}")
    lr = 1e-4
    logger.info(f"lr: {lr}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gpuname = torch.cuda.get_device_name(0)
    modelname = 'DEIT'
    
    datapath = args.data_path


    path = datapath + '/train40k/'
    dataset = EITdataset(path, modelname)

    path = datapath + '/valid40k/'   
    dataVal = EITdataset(path, modelname)
    #375_000
    args.epochs =  int(np.ceil(50000/ (len(dataset) / args.global_batch_size / 1)))  #  
    batch_size = args.global_batch_size

    loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
    )
    Y=[]
    Y_ST=[]
    X=[]
    for y, y_st, x in loader:
       Y.append(y.to(device))
       Y_ST.append(y_st.to(device))
       X.append(x.to(device))
    Y=torch.cat(Y,dim=0)
    Y_ST=torch.cat(Y_ST,dim=0)
    X=torch.cat(X,dim=0)   
    cached_dataset = TensorDataset(Y, Y_ST, X)   
    loader = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )

   
    loaderVal = DataLoader(
    dataVal,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
    )
    Y=[]
    Y_ST=[]
    X=[]
    for y, y_st, x in loaderVal:
       Y.append(y.to(device))
       Y_ST.append(y_st.to(device))
       X.append(x.to(device))
    Y=torch.cat(Y,dim=0)
    Y_ST=torch.cat(Y_ST,dim=0)
    X=torch.cat(X,dim=0)
    cached_dataset = TensorDataset(Y, Y_ST, X)   
    loaderVal = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    


    if accelerator.is_main_process:
        ema = EMA(model, beta=0.995, update_every=10)
        ema.to(device)
        ema.ema_model.eval()

    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    load_weight = False  # True  #
    if load_weight == True:

        checkpoint_dir = args.results_dir
        checkpoint = torch.load(checkpoint_dir + '/rre.pt', map_location='cpu')

        current_epoch = checkpoint["epoch"] + 1

        accelerator.print('load weight')

        model.load_state_dict(checkpoint['model'])
        if accelerator.is_main_process:
            ema = EMA(model, beta=0.995, update_every=10)
            ema.to(device)
            ema.ema_model.load_state_dict(checkpoint['model'])
            ema.ema_model.eval()
        model, opt, loader, loaderVal = accelerator.prepare(model, opt, loader, loaderVal)
    else:
        current_epoch = 0
        model, opt, loader, loaderVal = accelerator.prepare(model, opt, loader, loaderVal)

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    best_loss = 1000000
    Loss_tr = []
    Loss_val = []
    model.train()
    crterion = torch.nn.MSELoss(reduction='none')
   
    loc_loss = lossloc()

    loc_loss = torch.from_numpy(loc_loss).unsqueeze(1).permute(2, 3, 0, 1)
    loc_loss = loc_loss.float().to(device)


    for epoch in range(current_epoch, current_epoch + args.epochs):
        for y, y_st, x in loader:
            x = x.to(device)  # image
            y = y.to(device)  # voltage
            y_st = y_st.to(device)

            weight_n = lossweight(y)

            weight_n = weight_n @ loc_loss
            weight_n = weight_n.permute(2, 3, 0, 1)

 



            out = model(y, y_st)
            x=model.c2p(x)
            weight_n=model.c2p(weight_n)
            
            x_min = weight_n.amin(dim=(-2, -1), keepdim=True)  # [B,1,1,1]
            x_max = weight_n.amax(dim=(-2, -1), keepdim=True)  # [B,1,1,1]
            
       
            
            
            loss_mse=crterion(out, x)
            loss = loss_mse * weight_n
            loss = loss.mean()

            accelerator.backward(loss)


            accelerator.wait_for_everyone()
            accelerator.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            opt.zero_grad()

            if accelerator.is_local_main_process:
                ema.update()

            running_loss += loss_mse.mean().item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                logger.info('*' * 40)
                

                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)

                # avg_loss = avg_loss.item()
                # if accelerator.is_local_main_process:
                avg_loss = accelerator.gather(avg_loss)
                avg_loss = avg_loss.mean().item()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {loss.item():.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:

                running_loss = 0
                log_steps = 0
                start_time = time()
                Loss_tr.append(avg_loss)


            if train_steps % args.ckpt_every == 0:

                model.eval()
                val_loss_v = 0
                log_steps_v = 0
                with torch.no_grad():
                    for y, y_st, x in loaderVal:
                        x = x.to(device)
                        y = y.to(device)
                        y_st = y_st.to(device)
                        
                        
                        x=model.c2p(x)
                        out = model(y, y_st)
                        loss = crterion(out, x) #* weight_n
                        loss = loss.mean()
                        val_loss_v += loss.item()
                        log_steps_v += 1

                    val_loss_v = torch.tensor(val_loss_v / log_steps_v, device=device)



                    val_loss_v = accelerator.gather(val_loss_v)
                    val_loss_v = val_loss_v.mean().item()
                    logger.info(
                        f"(step={train_steps:07d}) Valid Loss: {val_loss_v:.4f}")
                    Loss_val.append(val_loss_v)
                    if val_loss_v < best_loss:
                        best_loss = val_loss_v
                        if accelerator.is_local_main_process:
                            checkpoint = {
                                "model": ema.ema_model.state_dict(),
                                "epoch": epoch
                            }
                            checkpoint_path = f"{checkpoint_dir}/rre.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f"Saved checkpoint to {checkpoint_path}")
                model.train()

    if accelerator.is_local_main_process:
        sio.savemat(checkpoint_dir + '/loss1.mat',
                    {'loss_stage1Tr': np.stack(Loss_tr),
                     'loss_stage1Val': np.stack(Loss_val)})


def test(args):
    accelerator = Accelerator(mixed_precision='no')
    device = accelerator.device
    gpus = torch.cuda.device_count()
    seed = args.global_seed
    init_seed(seed)

    checkpoint_dir = args.results_dir
    if accelerator.is_local_main_process:
        os.makedirs(args.results_dir, exist_ok=True) 
        checkpoint_dir = args.results_dir

        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{checkpoint_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Experiment directory created at {checkpoint_dir}")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    model = ModelT(modelname=args.results_dir[0:3])
    model = model.to(device)
    state_dict = torch.load(checkpoint_dir + '/rre.pt', map_location='cpu', weights_only = True)

    model.load_state_dict(state_dict['model'])
    gpuname = torch.cuda.get_device_name(0)

    modelname = 'DEIT'

    datapath = args.data_path

    path = datapath + '/test40k/'

    dataTe = EITdataset(path, modelname, dataset='simulate')  #
 
    loaderTe = DataLoader(
        dataTe,
        batch_size=args.global_batch_size * 1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    model, loaderTe = accelerator.prepare(model, loaderTe)
    model = accelerator.unwrap_model(model)
    model.eval()
    loc_loss = lossloc()

    loc_loss = torch.from_numpy(loc_loss).unsqueeze(1).permute(2, 3, 0, 1)
    loc_loss = loc_loss.float().to(device)
    with torch.no_grad():
        pred = []
        gt1 = []
        RMSE = []
        yy=[]
        for i, (y, y_st, x) in enumerate(loaderTe):
            x = x.to(device)
            y = y.to(device)
            y_st = y_st.to(device)
            
            weight_n = lossweight(y)
            weight_n = weight_n @ loc_loss
            weight_n = weight_n.permute(2, 3, 0, 1)
            

            out = model(y, y_st)


            out=model.p2c(out)
            out = accelerator.gather(out)
            x = accelerator.gather(x)
            rmse = (x - out).abs().mean()
            accelerator.print('out', i, out.shape, 'rmse: ', rmse)

            RMSE.append(rmse)
            out = out.squeeze()
            x = x.squeeze()
            yy.append(weight_n)
            gt1.append(x)
            pred.append(out)

        pred = torch.cat(pred, dim=0)
        gt1 = torch.cat(gt1, dim=0)
        yy = torch.cat(yy, dim=0)


        accelerator.print('out', pred.shape)
        RMSE = torch.stack(RMSE, dim=0)

        rmse = (gt1 - pred).square().mean().sqrt()
        pred1 = pred / 2 + 0.5
        gt2 = gt1 / 2 + 0.5
        max1, _ = torch.max(gt2, 1)
        max1, _ = torch.max(max1, 1)

        psnr = 10 * torch.log10(max1.square() / ((gt2 - pred1).square().mean([1, 2]) + 1e-12))
        accelerator.print('PSNR ', psnr.mean())
        accelerator.print('RMSE whole ', rmse)
        PSNR = psnr.mean()
        logger.info(f"(PSNR={PSNR})  ")
        torch.save(psnr.mean(), checkpoint_dir + '/' + 'psnr.pt')

        if accelerator.is_main_process:
            sio.savemat(checkpoint_dir + '/' + modelname + '.mat',
                        {'pred': pred.cpu() * dataTe.current / dataTe.voltage})
            # sio.savemat(checkpoint_dir + '/' +     'GT.mat',
            # {'GT': gt1.cpu()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,default="../../data/dataNew")
    parser.add_argument("--results-dir", type=str, default="rre")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=500)
    args = parser.parse_args()

    if args.mode == 'test':
        test(args)
    else:
 
        main(args)
 
 
     
