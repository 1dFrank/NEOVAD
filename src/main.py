from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import numpy as np
import random


from configs_base2novel import build_config

#from configs_fully import build_config
from utils import *
from log import get_logger
from model import Model, AdaptedModel
from dataset import *
from loss import *

from train import train_func
from test import test_func
from infer import infer_func

import argparse
import copy
import wandb
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_checkpoint(model, ckpt_path, logger):
    if os.path.isfile(ckpt_path):
        logger.info('loading pretrained checkpoint from {}.'.format(ckpt_path))
        weight_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    logger.info('{} size mismatch: load {} given {}'.format(
                        name, param.size(), model_dict[name].size()))
            else:
                logger.info('{} not found in model dict.'.format(name))
        return True
    else:
        logger.info('Not found pretrained checkpoint file.')
        return False


def train(model, train_loader, test_loader, gt, logger,clslist):
    save_dir = cfg.save_dir + args.test + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = torch.nn.BCELoss()
    xentropy = torch.nn.CrossEntropyLoss()
    # pseloss = PseLoss(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr) 
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10],gamma=0.1)

    #seed2024 for ub weightdecaty5e-4
    #optimizer = optim.Adam(model.parameters(), lr=cfg.lr,betas = (0.9, 0.999), weight_decay = 5e-4)
    
    # logger.info('Model:{}\n'.format(model))
    logger.info('Optimizer:{}\n'.format(optimizer))
    if cfg.eval_on_cpu:
        torch.save(model.state_dict(), 'tmp.ckpt')
        # model_eval = Model(cfg,device='cpu')
        model_eval = AdaptedModel(cfg,device='cpu')
        load_checkpoint(model_eval,'tmp.ckpt',logger)
        initial_auc,top1acc,top5acc,mauc,mauc_WOnorm = test_func(test_loader, model_eval, gt, cfg.dataset,clslist,'cpu',cfg)
    else:
        initial_auc,top1acc,top5acc,mauc,mauc_WOnorm = test_func(test_loader, model, gt, cfg.dataset,clslist,cfg.device,cfg)
    logger.info('Random initialize {}:{:.4f} top1ACC:{:.4f} top5ACC:{:.4f} mauc:{:.4f} mauc_ab:{:.4f}'.format(cfg.metrics, initial_auc,top1acc,top5acc,mauc,mauc_WOnorm))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0


    st = time.time()
    for epoch in range(cfg.max_epoch):
        # loss1, loss2 = train_func(train_loader, model, optimizer, criterion, xentropy, clslist,cfg.device,cfg.lamda)
        loss1, loss2, orthogonal_loss, contrast_loss= train_func(train_loader, model, optimizer, criterion, xentropy,  clslist,cfg)
        # scheduler.step() #for ucf
        if cfg.eval_on_cpu:
            torch.save(model.state_dict(), 'tmp.ckpt')
            # model_eval = Model(cfg,device='cpu')
            model_eval = AdaptedModel(cfg,device='cpu')
            load_checkpoint(model_eval,'tmp.ckpt',logger)
            auc,top1acc,top5acc,mauc,mauc_WOnorm = test_func(test_loader, model_eval, gt, cfg.dataset,clslist,'cpu',cfg)
        else:
          
            auc,top1acc,top5acc,mauc,mauc_WOnorm = test_func(test_loader, model, gt, cfg.dataset,clslist,cfg.device,cfg)


        if auc >= best_auc:
            best_auc = auc
           
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_dir + cfg.model_name + '_ckpt_best_tmp.pkl')

            # logger.info('save model at epoch{}, {}:{:.4f}\n'.format(epoch+1, cfg.metrics,best_auc))
                

        if WANDB:
            wandb.log({'auc':auc,'epoch':epoch,'loss_bin':loss1,'loss_Mul':loss2,'best_auc':best_auc,'top1acc':top1acc,'top5acc':top5acc,'mauc':mauc,'mauc_WOnorm':mauc_WOnorm})

        logger.info('[Epoch:{}/{}]: loss1:{:.4f} loss2:{:.4f} orthogonal_loss:{:.4f} contrast_loss:{:.4f}| {}:{:.4f} \n top1ACC:{:.4f} top5ACC:{:.4f} mauc:{:.4f} mauc_ab:{:.4f}'.format(
        epoch + 1, cfg.max_epoch, loss1, loss2, orthogonal_loss, contrast_loss, cfg.metrics,auc,top1acc,top5acc,mauc,mauc_WOnorm))
        # write a txt, show best auc
        auc_str = str(round(best_auc, 4))
        update_auc(auc_str,cfg)
        txt_name = auc_str + ".txt"
        txt_path = os.path.join(save_dir, txt_name)

        for file in os.listdir(save_dir):
            if file.endswith(".txt"):
                os.remove(os.path.join(save_dir, file))

        with open(txt_path, "w") as f: pass


    time_elapsed = time.time() - st
    torch.save(model.state_dict(), save_dir + cfg.model_name + '_final.pkl')
    # logger.info('save model {}'.format(save_dir + cfg.model_name + '_final.pkl'))
    model.load_state_dict(best_model_wts)
    # torch.save(model.state_dict(), save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl')
    # logger.info('save model {}'.format(save_dir + cfg.model_name + '_' + str(round(best_auc, 4)).split('.')[1] + '.pkl'))
    logger.info('Training completes in {:.0f}m {:.0f}s | best {}:{:.4f}\n'.
                format(time_elapsed // 60, time_elapsed % 60, cfg.metrics, best_auc))
    if WANDB:
        wandb.finish()

def main(cfg):
    save_dir = cfg.save_dir + args.test + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # torch.backends.cuda.flash_sdp_enabled()
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    logger = get_logger(save_dir + cfg.logs_dir)
    setup_seed(cfg.seed)
    logger.info('Config:{}'.format(cfg.__dict__))

    clslist,cls_dict = text_prompting(dataset=args.dataset, cls=cfg.clslist,clipbackbone=cfg.backbone, device=device)
    print("loading cls info and encoder the text of cls...")
    print(cls_dict)
    if cfg.dataset == 'ucf-crime':
        train_data = UCFDataset(cfg, cls_dict,test_mode=False)
        test_data = UCFDataset(cfg, cls_dict,test_mode=True)
    elif cfg.dataset == 'shanghaitech':
        train_data = SHDataset(cfg,cls_dict, test_mode=False)
        test_data = SHDataset(cfg,cls_dict, test_mode=True)
    elif cfg.dataset == 'xd-violence':
        train_data = XDDataset(cfg,cls_dict, test_mode=False)
        test_data = XDDataset(cfg,cls_dict, test_mode=True)
    elif cfg.dataset == 'ubnormal':
        train_data = UBDataset(cfg,cls_dict, test_mode=False)
        test_data = UBDataset(cfg,cls_dict, test_mode=True)
    else:
        raise RuntimeError("Do not support this dataset!")

    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True)

    test_loader = DataLoader(test_data, batch_size=cfg.test_bs, shuffle=False,
                             num_workers=cfg.workers, pin_memory=True)

    
    if cfg.preprompt:
        # model = Model(cfg,device=device)
        model = AdaptedModel(cfg,device=device)
    else:
        # model = Model(cfg,clslist,device=device)
        model = AdaptedModel(cfg,clslist,device=device)

    gt = np.load(cfg.gt)
   
    model = model.to(device)
    if WANDB:
        wandb.watch(model,log='all')

    param = sum(p.numel() for p in model.parameters() if p.requires_grad)

    
    logger.info('total requires_grad params:{:.4f}M'.format(param / (1000 ** 2)))


    if args.mode == 'train':
        logger.info('Training Mode')
        if cfg.load_ckpt :
            is_load = load_checkpoint(model, './ckpt/base2novel/ucf__final.pkl', logger)
        else:
            logger.info('train from scratch')
        train(model, train_loader, test_loader, gt, logger,clslist)

    elif args.mode == 'infer':
        logger.info('Test Mode')
        if cfg.ckpt_path is not None:
            is_load = load_checkpoint(model, save_dir + cfg.ckpt_path, logger)
            if is_load:
                infer_func(model.to(device), test_loader, gt, logger, cfg,clslist)
            else:
                logger.info('infer model not exist')
        else:
            logger.info('infer model not exist')
        

    else:
        raise RuntimeError('Invalid status!')


if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ('true', '1', 'yes', 'y')
        
    parser = argparse.ArgumentParser(description='VAD')

    parser.add_argument('--dataset', default='ucf', help='anomaly video dataset')
    parser.add_argument('--mode', default='train', help='model status: (train or infer)')
    parser.add_argument('--test', default='test', help='for file path')


    parser.add_argument('--lamda2', type=float, default=1, help='second lambda weight')
    parser.add_argument('--lamda3', type=float, default=2, help='third lambda weight')
    parser.add_argument('--device', type=str, default="cuda:0", help='device to use, e.g. cuda:0 or cpu')
    parser.add_argument('--seed', type=int, default=2, help='seed')
    parser.add_argument('--text_adapt_weight', type=float, default=0.1, help='adapter weight')

    parser.add_argument('--temporal', type=str2bool, default=True, help='temporal modeling')
    parser.add_argument('--adapter', type=str2bool, default=True, help='adapter modeling')
    parser.add_argument('--gen_npy', type=str2bool, default=False, help='generate numpy files')
    args = parser.parse_args()


    cfg = build_config(args.dataset)
    # print("args.adapter:",args.adapter)

    cfg.gen_npy = args.gen_npy
    cfg.test = args.test
    cfg.mode = args.mode
    cfg.lamda2 = args.lamda2
    cfg.lamda3 = args.lamda3
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.text_adapt_weight = args.text_adapt_weight
    cfg.temporal = args.temporal
    cfg.adapter = args.adapter
    # print("cfg.temporal:",cfg.temporal)
    # print("cfg.adapter:",cfg.adapter)

    config = cfg.__dict__

    WANDB = cfg.WANDB
    if WANDB:
        wandb.init(project='OVVAD', config=config)

    main(cfg)
    