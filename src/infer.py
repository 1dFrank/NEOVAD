import time
from test import *
from utils import *


def infer_func(model, dataloader, gt, logger, cfg,clslist):
    st = time.time()

    vis_v_input = []
    vis_v_feat = []
    vis_labels = []      # 0: Normal, 1: Abnormal
    vis_multilabel = []  # 0–13

    mauc_metric = MulticlassAUROC(num_classes=len(clslist), average=None, thresholds=None)
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(cfg.device)
        abnormal_preds = torch.zeros(0).to(cfg.device)
        abnormal_labels = torch.zeros(0).to(cfg.device)
        normal_preds = torch.zeros(0).to(cfg.device)
        normal_labels = torch.zeros(0).to(cfg.device)
        # gt_tmp = torch.tensor(gt.copy()).to(cfg.device)
        similarity, targets = torch.zeros(0).to(cfg.device), torch.zeros(0).to(cfg.device)
        for i, (v_input, label,multi_label) in enumerate(dataloader):
            v_input = v_input.float().to(cfg.device)
            seq_len = torch.sum(torch.max(torch.abs(v_input), dim=2)[0] > 0, 1)   
            logits,v_feat,t_feat_pre,t_feat_le,pair_features= model(v_input, seq_len,clslist)
            logit_scale = model.logit_scale.exp()
            #align and get the simlarity,multilabels of each batch
            #v2t_logits = MILAlign(v_feat,t_feat,logit_scale,label,seq_len,cfg.device)
            v2t_logits_pre = MILAlign(v_feat,t_feat_pre,logit_scale,label,seq_len,cfg.device)
            v2t_logits_le = MILAlign(v_feat,t_feat_le,logit_scale,label,seq_len,cfg.device)
            
                        # video-level features
            v_input_vid = v_input.mean(dim=1)   # [B, D]
            v_feat_vid  = v_feat.mean(dim=1)    # [B, D]

            vis_v_input.append(v_input_vid.cpu())
            vis_v_feat.append(v_feat_vid.cpu())

            # binary label: Normal vs Abnormal
            binary_label = (multi_label != 0).long()
            vis_labels.append(binary_label.cpu())

            # multi-class label (0–13)
            vis_multilabel.append(multi_label.cpu())


            # v2t_logits = torch.where(v2t_logits_le>v2t_logits_pre,v2t_logits_le,v2t_logits_pre)
            v2t_logits = v2t_logits_le

            sim_batch = v2t_logits.softmax(dim=-1)
            target_batch = multi_label.to(cfg.device)
                #for multicrop
            sim_batch = torch.mean(sim_batch,dim=0).unsqueeze(0)
            target_batch = target_batch[0].unsqueeze(0)

            similarity = torch.cat([similarity, sim_batch], dim=0)
            targets = torch.cat([targets, target_batch], dim=0)

            batch_mcauc = mauc_metric.update(sim_batch, target_batch)
            # binary logits 
            logits = torch.mean(logits, 0)

            pred = torch.cat((pred, logits))
                #gt(binary),and repeat16
           
            
            # labels = gt_tmp[: seq_len[0] * 16]
         
            
            # if torch.sum(labels) == 0:
            #     normal_labels = torch.cat((normal_labels, labels))
            #     normal_preds = torch.cat((normal_preds, logits))
            # else:
            #     abnormal_labels = torch.cat((abnormal_labels, labels))
            #     abnormal_preds = torch.cat((abnormal_preds, logits))
            # gt_tmp = gt_tmp[seq_len[0] * 16:]
 
        values,indices = similarity.topk(5)
        
        vis_v_input = torch.cat(vis_v_input, dim=0).numpy()
        vis_v_feat = torch.cat(vis_v_feat, dim=0).numpy()
        vis_labels = torch.cat(vis_labels, dim=0).numpy()
        vis_multilabel = torch.cat(vis_multilabel, dim=0).numpy()

        
        if cfg.gen_npy: 
            np.save("tsne/tsne_v_input.npy", vis_v_input)
            np.save("tsne/tsne_v_feat.npy", vis_v_feat)
            np.save("tsne/tsne_binary_label.npy", vis_labels)
            np.save("tsne/tsne_multilabel.npy", vis_multilabel)


        # np.save("result/sim.npy",similarity.cpu().detach().numpy())
        # np.save("result/cls.npy",clslist)
        # np.save("result/target.npy",targets.cpu().detach().numpy())
        # np.save("result/pred.npy",indices[:,0].cpu().detach().numpy())
        top1 = (indices[:, 0] == targets).tolist()
        top5 = ((indices == einops.repeat(targets, 'b -> b k', k=5)).sum(-1)).tolist()
        top1ACC = np.array(top1).sum() / len(top1)
        top5ACC = np.array(top5).sum() / len(top5)
        mc_auc = mauc_metric.compute()
        # print("mauc")
        # print(mc_auc)
        mauc = torch.nanmean(mc_auc[mc_auc!=0])

        mc_auc_ab = mc_auc[1:]
        mauc_WOnorm = torch.nanmean(mc_auc_ab[mc_auc_ab!=0]) #delete cls normal
        mauc_metric.reset()

        pred = list(pred.cpu().detach().numpy())
        # np.save("result/base_pred.npy",np.repeat(pred,16))
        # np.save("result/base_gt.npy",gt)
        # np.save("result/novel_pred.npy",pred)
        # np.save("result/novel_gt.npy",gt)
        # n_far = cal_false_alarm(normal_labels, normal_preds)
        if cfg.dataset == 'ucf-crime':
            fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
            roc_auc = cal_auc(fpr, tpr, cfg)
            pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
            pr_auc = cal_ap(rec, pre, cfg)
        elif cfg.dataset == 'shanghaitech':
            fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
            roc_auc = cal_auc(fpr, tpr, cfg)
            pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
            pr_auc = cal_ap(rec, pre, cfg)
        elif cfg.dataset == 'ubnormal':
            fpr, tpr, _ = roc_curve(list(gt), pred)
            roc_auc = cal_auc(fpr, tpr, cfg)
            pre, rec, _ = precision_recall_curve(list(gt), pred)
            pr_auc = cal_ap(rec, pre, cfg)
            top1ACC = cal_top1_ACC(indices, targets, cfg)
            top5ACC = cal_top5_ACC(indices, targets, cfg)
        elif cfg.dataset == 'xd-violence':
            fpr, tpr, _ = roc_curve(list(gt), np.repeat(pred, 16))
            roc_auc = cal_auc(fpr, tpr, cfg)
            pre, rec, _ = precision_recall_curve(list(gt), np.repeat(pred, 16))
            pr_auc = cal_ap(rec, pre, cfg)
        else:
            raise RuntimeError('Invalid dataset.')

    time_elapsed = time.time() - st
    if cfg.dataset == 'xd-violence':
        logger.info('AP:{:.4f} top1ACC:{:.4f} top5ACC:{:.4f} mauc:{:.4f} mauc_ab:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
            pr_auc, top1ACC, top5ACC, mauc, mauc_WOnorm, time_elapsed // 60, time_elapsed % 60))
    else:
        logger.info('AUC:{:.4f} top1ACC:{:.4f} top5ACC:{:.4f} mauc:{:.4f} mauc_ab:{:.4f} | Complete in {:.0f}m {:.0f}s\n'.format(
            roc_auc, top1ACC, top5ACC, mauc, mauc_WOnorm, time_elapsed // 60, time_elapsed % 60))
