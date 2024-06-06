from tensorboardX import SummaryWriter
from torch import nn

from dataset.semi import SemiDataset
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map

import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import ramps

def fast_hist(a, b, n):
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

MODE = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, default='dataset/splits/pascal/1_2/labeled.txt')
    parser.add_argument('--unlabeled-id-path', type=str, default='dataset/splits/pascal/1_2/unlabeled.txt')
    parser.add_argument('--pseudo-mask-path', type=str, default='outdir/pseudo_masks/pascal/1_2')

    parser.add_argument('--save-path', type=str, default='outdir/models/pascal/1_2')

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str, default='outdir/reliable_id_path')
    parser.add_argument('--plus', dest='plus', default=True, action='store_true',
                        help='whether to use ST++')

    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=80.0, help='consistency_rampup')

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if not args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    global MODE
    MODE = 'train'
    labeled_trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    unlabeled_trainset = SemiDataset(args.dataset, args.data_root, 'first_stage',args.crop_size, args.unlabeled_id_path)
    scale_nums = len(unlabeled_trainset.ids) // len(labeled_trainset.ids)
    print("Number of labeled data = ",len(labeled_trainset.ids))
    scale_nums = scale_nums // 2
    print("scale_nums = ",scale_nums)
    # labeled_trainset.ids = 2 * labeled_trainset.ids if len(labeled_trainset.ids) < 200 else labeled_trainset.ids
    labeled_trainset.ids = scale_nums * labeled_trainset.ids if scale_nums > 1 else labeled_trainset.ids * 2
    print("Augmented labeled data = ",len(labeled_trainset.ids))
    train_loader = DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=4, drop_last=True)
    unsupervised_train_loader = DataLoader(unlabeled_trainset, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=4, drop_last=True)

    model1, optimizer1 = init_basic_elems(args)

    model2, optimizer2 = init_basic_elems(args)

    print('\nParams: %.1fM' % count_params(model1))
    print('\nParams: %.1fM' % count_params(model2))
    best_model, checkpoints = train1(model1=model1,model2=model2, train_loader=train_loader,unsupervised_train_loader=unsupervised_train_loader, valloader=valloader, criterion=criterion, optimizer1=optimizer1, optimizer2=optimizer2,args=args)
    print("finish SupOnly......")
    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')

        dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        label(best_model, dataloader, args)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                               args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=4, drop_last=True)

        model, optimizer = init_basic_elems(args)

        train(model, trainloader, valloader, criterion, optimizer, args)

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')

    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    labeled_trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    unlabeled_trainset = SemiDataset(args.dataset, args.data_root, 'train', args.crop_size, cur_unlabeled_id_path)
    train_loader = DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True,pin_memory=True, num_workers=4, drop_last=True)
    unsupervised_train_loader = DataLoader(unlabeled_trainset, batch_size=args.batch_size, shuffle=True,pin_memory=True,num_workers=4, drop_last=True)

    model1, optimizer1 = init_basic_elems(args)

    model2, optimizer2 = init_basic_elems(args)




    best_model = train1(model1=model1, model2=model2, train_loader=train_loader,
                                     unsupervised_train_loader=unsupervised_train_loader, valloader=valloader,
                                     criterion=criterion, optimizer1=optimizer1, optimizer2=optimizer2, args=args)

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=4, drop_last=True)

    # model, optimizer = init_basic_elems(args)

    # train2(model, trainloader, valloader, criterion, optimizer, args)
    model1, optimizer1 = init_basic_elems(args)

    model2, optimizer2 = init_basic_elems(args)

    best_model = train2(model1=model1, model2=model2, train_loader=trainloader,
                                     unsupervised_train_loader=trainloader, valloader=valloader,
                                     criterion=criterion, optimizer1=optimizer1, optimizer2=optimizer2, args=args)

def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, 4 if args.dataset == 'pascal' else 19)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    return model, optimizer

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

# train1(model1,model2, train_loader,unsupervised_train_loader, valloader, criterion, optimizer1, optimizer2,args)
def train1(model1=None,model2=None, train_loader=None,unsupervised_train_loader=None, valloader=None, criterion=None, optimizer1=None,optimizer2=None, args=None):
    iters = 0
    total_iters = len(train_loader) * args.epochs

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []
    iter_num = 0
    writer = SummaryWriter('./log')

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer1.param_groups[0]["lr"], previous_best))
        model1.train()
        model2.train()
        model1 = model1.to(device)
        model2 = model2.to(device)

        total_loss = 0.0
        niters_per_epoch = min(len(train_loader),len(unsupervised_train_loader))
        tbar = tqdm(range(niters_per_epoch))

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)
        iter_num = iter_num + 1
        num_classes = 5
        hist2 = np.zeros((num_classes, num_classes))
        hist3 = np.zeros((num_classes, num_classes))
        for i in tbar:
            # minibatch = dataloader.next()
            minibatch = next(dataloader)
            # unsup_minibatch = unsupervised_dataloader.next()
            unsup_minibatch = next(unsupervised_dataloader)
            imgs = minibatch[0]
            gts = minibatch[1]
            unsup_imgs = unsup_minibatch[0]
            img = imgs.to(device)
            unsup_imgs = unsup_imgs.to(device)
            mask = gts.to(device)

            # img, mask = imgs.cuda(), gts.cuda()

            pred_sup_l = model1(img)
            one = torch.ones((pred_sup_l.shape[0], 1, pred_sup_l.shape[2], pred_sup_l.shape[3])).cuda()
            pred_sup_l = torch.cat([pred_sup_l, (100 * one * (mask.cuda() == 4).unsqueeze(dim=1))], dim=1)
            pred_unsup_l = model1(unsup_imgs)

            pred_sup_r = model2(img)
            one = torch.ones((pred_sup_r.shape[0], 1, pred_sup_r.shape[2], pred_sup_r.shape[3])).cuda()
            pred_sup_r = torch.cat([pred_sup_r, (100 * one * (mask.cuda() == 4).unsqueeze(dim=1))], dim=1)
            pred_unsup_r = model2(unsup_imgs)

            # pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
            # pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)


            uncentainty_l = abs(torch.max(torch.softmax(pred_sup_l,dim=1), dim=1)[0] - torch.max(torch.softmax(pred_sup_r,dim=1), dim=1)[0])
            scale_uncentainty_l = torch.rand_like(uncentainty_l)
            for bix in range(uncentainty_l.shape[0]):
                max_v = torch.max(uncentainty_l[bix])
                min_v = torch.min(uncentainty_l[bix])
                scale_uncentainty_l[bix] = (uncentainty_l[bix] - min_v) / (max_v - min_v)

            # scale_weights_l = 1 - scale_uncentainty_l
            # scale_weights_l = scale_weights_l ** 4 + 1e-2
            scale_weights_l = 1 - uncentainty_l
            scale_weights_l = scale_weights_l ** 4

            uncentainty_un = abs(torch.max(torch.softmax(pred_unsup_l,dim=1), dim=1)[0] - torch.max(torch.softmax(pred_unsup_r,dim=1), dim=1)[0])
            scale_uncentainty_un = torch.rand_like(uncentainty_un)
            for bix in range(uncentainty_un.shape[0]):
                max_v = torch.max(uncentainty_un[bix])
                min_v = torch.min(uncentainty_un[bix])
                scale_uncentainty_un[bix] = (uncentainty_un[bix] - min_v) / (max_v - min_v)


            scale_weights_un = 1 - uncentainty_un
            scale_weights_un = scale_weights_un ** 4

            _, max_sup_l = torch.max(pred_sup_l, dim=1)
            _, max_unsup_l = torch.max(pred_unsup_l, dim=1)
            _, max_sup_r = torch.max(pred_sup_r, dim=1)
            _, max_unsup_r = torch.max(pred_unsup_r, dim=1)

            max_sup_l = max_sup_l.long()
            max_unsup_l = max_unsup_l.long()
            max_sup_r = max_sup_r.long()
            max_unsup_r = max_unsup_r.long()

            ### cps loss ###
            # cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
            criterion2 = nn.CrossEntropyLoss(reduction="none")
            cps_loss_l = ((criterion2(pred_sup_l, max_sup_r) * scale_weights_l)/ (max_sup_r.shape[0] * max_sup_r.shape[1] * max_sup_r.shape[2])).sum()  + ((criterion2(pred_sup_r, max_sup_l) * scale_weights_l)/ (max_sup_l.shape[0] * max_sup_l.shape[1] * max_sup_l.shape[2])).sum()

            cps_loss_un = ((criterion2(pred_unsup_l, max_unsup_r) * scale_weights_un)/ (max_unsup_r.shape[0] * max_unsup_r.shape[1] * max_unsup_r.shape[2])).sum()  + ((criterion2(pred_unsup_r, max_unsup_l) * scale_weights_un)/ (max_unsup_l.shape[0] * max_unsup_l.shape[1] * max_unsup_l.shape[2])).sum()

            ### standard cross entropy loss ###
            loss_sup = criterion(pred_sup_l, mask)
            loss_sup_r = criterion(pred_sup_r, mask)

            consistency_weight = get_current_consistency_weight(iter_num)
            
            hist2 += fast_hist(mask.cpu().numpy().flatten(),torch.max(pred_sup_l, dim=1)[1].cpu().numpy().flatten(), num_classes)
            hist3 += fast_hist(mask.cpu().numpy().flatten(),torch.max(pred_sup_r, dim=1)[1].cpu().numpy().flatten(), num_classes)

            loss = loss_sup + loss_sup_r + consistency_weight * (cps_loss_l + cps_loss_un)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer1.param_groups[0]["lr"] = lr
            optimizer1.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            optimizer2.param_groups[0]["lr"] = lr
            optimizer2.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        epoch_class_iou2 = per_class_iu(hist2)
        epoch_class_iou3 = per_class_iu(hist3)
        mean_iou2 = np.mean(epoch_class_iou2)
        mean_iou3 = np.mean(epoch_class_iou3)

        print("========================================Training Set===================================================>")
        print("========================================model1===================================================>")
        print("tumor_iou : ", epoch_class_iou2[0], "stroma_iou : ",
              epoch_class_iou2[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou2[2],
              "necrosis_or_debris_iou : ", epoch_class_iou2[3],"other_iou : ", epoch_class_iou2[4])
        print("mean_iou = ", mean_iou2)
        writer.add_scalar('stage1_train/model1_tumor_iou', epoch_class_iou2[0], iter_num)
        writer.add_scalar('stage1_train/model1_stroma_iou', epoch_class_iou2[1], iter_num)
        writer.add_scalar('stage1_train/model1_lymphocytic_infiltrate_iou', epoch_class_iou2[2], iter_num)
        writer.add_scalar('stage1_train/model1_necrosis_or_debris_iou', epoch_class_iou2[3], iter_num)
        writer.add_scalar('stage1_train/model1_other_iou', epoch_class_iou2[4], iter_num)
        writer.add_scalar('stage1_train/model1_mean_iou', mean_iou2, iter_num)

        print("========================================model2===================================================>")
        print("tumor_iou : ", epoch_class_iou3[0], "stroma_iou : ",
              epoch_class_iou3[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou3[2],
              "necrosis_or_debris_iou : ", epoch_class_iou3[3],"other_iou : ", epoch_class_iou3[4])
        print("mean_iou = ", mean_iou3)
        writer.add_scalar('stage1_train/model2_tumor_iou', epoch_class_iou3[0], iter_num)
        writer.add_scalar('stage1_train/model2_stroma_iou', epoch_class_iou3[1], iter_num)
        writer.add_scalar('stage1_train/model2_lymphocytic_infiltrate_iou', epoch_class_iou3[2], iter_num)
        writer.add_scalar('stage1_train/model2_necrosis_or_debris_iou', epoch_class_iou3[3], iter_num)
        writer.add_scalar('stage1_train/model2_other_iou', epoch_class_iou3[4], iter_num)
        writer.add_scalar('stage1_train/model2_mean_iou', mean_iou3, iter_num)

        metric = meanIOU(num_classes=5 if args.dataset == 'pascal' else 19)
        print("consistency_weight = ",consistency_weight)
        model1.eval()
        model2.eval()

        tbar = tqdm(valloader)

        num_classes = 5
        hist4 = np.zeros((num_classes, num_classes))
        hist5 = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model1(img)
                one = torch.ones((pred.shape[0], 1, pred.shape[2], pred.shape[3])).cuda()
                pred = torch.cat([pred, (100 * one * (mask.cuda() == 4).unsqueeze(dim=1))], dim=1)
                pred = torch.argmax(pred, dim=1)

                pred2 = model2(img)
                one = torch.ones((pred2.shape[0], 1, pred2.shape[2], pred2.shape[3])).cuda()
                pred2 = torch.cat([pred2, (100 * one * (mask.cuda() == 4).unsqueeze(dim=1))], dim=1)
                pred2 = torch.argmax(pred2, dim=1)

                # hist4 += fast_hist(pred.cpu().numpy().flatten(),
                #                    mask.cpu().numpy().flatten(), num_classes)
                # hist5 += fast_hist(pred2.cpu().numpy().flatten(),
                #                    mask.cpu().numpy().flatten(), num_classes)
                hist4 += fast_hist(mask.cpu().numpy().flatten(), pred.cpu().numpy().flatten(),num_classes)
                hist5 += fast_hist(mask.cpu().numpy().flatten(),pred2.cpu().numpy().flatten(), num_classes)
                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
        epoch_class_iou2 = per_class_iu(hist4)
        epoch_class_iou3 = per_class_iu(hist5)
        mean_iou2 = np.mean(epoch_class_iou2)
        mean_iou3 = np.mean(epoch_class_iou3)

        print("========================================Validation Set===================================================>")
        print("========================================model1===================================================>")
        print("tumor_iou : ", epoch_class_iou2[0], "stroma_iou : ",
              epoch_class_iou2[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou2[2],
              "necrosis_or_debris_iou : ", epoch_class_iou2[3], "other_iou : ", epoch_class_iou2[4])
        print("mean_iou = ", mean_iou2)
        writer.add_scalar('stage1_val/model1_tumor_iou', epoch_class_iou2[0], iter_num)
        writer.add_scalar('stage1_val/model1_stroma_iou', epoch_class_iou2[1], iter_num)
        writer.add_scalar('stage1_val/model1_lymphocytic_infiltrate_iou', epoch_class_iou2[2], iter_num)
        writer.add_scalar('stage1_val/model1_necrosis_or_debris_iou', epoch_class_iou2[3], iter_num)
        writer.add_scalar('stage1_val/model1_other_iou', epoch_class_iou2[4], iter_num)
        writer.add_scalar('stage1_val/model1_mean_iou', mean_iou2, iter_num)

        print("========================================model2===================================================>")
        print("tumor_iou : ", epoch_class_iou3[0], "stroma_iou : ",
              epoch_class_iou3[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou3[2],
              "necrosis_or_debris_iou : ", epoch_class_iou3[3], "other_iou : ", epoch_class_iou3[4])
        print("mean_iou = ", mean_iou3)
        writer.add_scalar('stage1_val/model2_tumor_iou', epoch_class_iou3[0], iter_num)
        writer.add_scalar('stage1_val/model2_stroma_iou', epoch_class_iou3[1], iter_num)
        writer.add_scalar('stage1_val/model2_lymphocytic_infiltrate_iou', epoch_class_iou3[2], iter_num)
        writer.add_scalar('stage1_val/model2_necrosis_or_debris_iou', epoch_class_iou3[3], iter_num)
        writer.add_scalar('stage1_val/model2_other_iou', epoch_class_iou3[4], iter_num)
        writer.add_scalar('stage1_val/model2_mean_iou', mean_iou3, iter_num)

        metric = meanIOU(num_classes=5 if args.dataset == 'pascal' else 19)
        print("consistency_weight = ", consistency_weight)

        mIOU *= 100.0
        if (mean_iou2 * 100 > previous_best) or (mean_iou3 * 100 > previous_best):
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))


            if mean_iou2 > mean_iou3:# model1 > model2
                previous_best = mean_iou2*100.0
                torch.save(model1.module.state_dict(),os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mean_iou2*100.0)))
                best_model = deepcopy(model1)
            else:
                previous_best = mean_iou3*100.0
                torch.save(model2.module.state_dict(),os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mean_iou3*100.0)))
                best_model = deepcopy(model2)


            # best_model = deepcopy(model1)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model1))
            print("checkpoints.length = ", len(checkpoints))


    if MODE == 'train':
        return best_model, checkpoints

    return best_model
def train2(model1=None,model2=None, train_loader=None,unsupervised_train_loader=None, valloader=None, criterion=None, optimizer1=None,optimizer2=None, args=None):
    iters = 0
    total_iters = len(train_loader) * args.epochs

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []
    iter_num = 0
    writer = SummaryWriter('./log')

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer1.param_groups[0]["lr"], previous_best))
        model1.train()
        model2.train()
        model1 = model1.to(device)
        model2 = model2.to(device)


        total_loss = 0.0
        niters_per_epoch = min(len(train_loader),len(unsupervised_train_loader))
        tbar = tqdm(range(niters_per_epoch))

        dataloader = iter(train_loader)
        unsupervised_dataloader = iter(unsupervised_train_loader)
        iter_num = iter_num + 1

        num_classes = 5
        hist2 = np.zeros((num_classes, num_classes))
        hist3 = np.zeros((num_classes, num_classes))

        for i in tbar:
            # minibatch = dataloader.next()
            minibatch = next(dataloader)
            # unsup_minibatch = unsupervised_dataloader.next()
            unsup_minibatch = next(unsupervised_dataloader)
            imgs = minibatch[0]
            gts = minibatch[1]
            unsup_imgs = unsup_minibatch[0]
            img = imgs.to(device)
            unsup_imgs = unsup_imgs.to(device)
            mask = gts.to(device)

            # img, mask = imgs.cuda(), gts.cuda()

            pred_sup_l = model1(img)
            one = torch.ones((pred_sup_l.shape[0], 1, pred_sup_l.shape[2], pred_sup_l.shape[3])).cuda()
            pred_sup_l = torch.cat([pred_sup_l, (100 * one * (mask.cuda() == 4).unsqueeze(dim=1))], dim=1)
            pred_unsup_l = model1(unsup_imgs)

            pred_sup_r = model2(img)
            one = torch.ones((pred_sup_r.shape[0], 1, pred_sup_r.shape[2], pred_sup_r.shape[3])).cuda()
            pred_sup_r = torch.cat([pred_sup_r, (100 * one * (mask.cuda() == 4).unsqueeze(dim=1))], dim=1)
            pred_unsup_r = model2(unsup_imgs)

            # pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
            # pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)

            uncentainty_l = abs(torch.max(torch.softmax(pred_sup_l, dim=1), dim=1)[0] -
                                torch.max(torch.softmax(pred_sup_r, dim=1), dim=1)[0])
            scale_uncentainty_l = torch.rand_like(uncentainty_l)
            for bix in range(uncentainty_l.shape[0]):
                max_v = torch.max(uncentainty_l[bix])
                min_v = torch.min(uncentainty_l[bix])
                scale_uncentainty_l[bix] = (uncentainty_l[bix] - min_v) / (max_v - min_v)

            # scale_weights_l = 1 - scale_uncentainty_l
            # scale_weights_l = scale_weights_l ** 4 + 1e-2
            scale_weights_l = 1 - uncentainty_l
            scale_weights_l = scale_weights_l ** 4

            uncentainty_un = abs(torch.max(torch.softmax(pred_unsup_l, dim=1), dim=1)[0] -
                                 torch.max(torch.softmax(pred_unsup_r, dim=1), dim=1)[0])
            scale_uncentainty_un = torch.rand_like(uncentainty_un)
            for bix in range(uncentainty_un.shape[0]):
                max_v = torch.max(uncentainty_un[bix])
                min_v = torch.min(uncentainty_un[bix])
                scale_uncentainty_un[bix] = (uncentainty_un[bix] - min_v) / (max_v - min_v)

            scale_weights_un = 1 - uncentainty_un
            scale_weights_un = scale_weights_un ** 4

            _, max_sup_l = torch.max(pred_sup_l, dim=1)
            _, max_unsup_l = torch.max(pred_unsup_l, dim=1)
            _, max_sup_r = torch.max(pred_sup_r, dim=1)
            _, max_unsup_r = torch.max(pred_unsup_r, dim=1)

            max_sup_l = max_sup_l.long()
            max_unsup_l = max_unsup_l.long()
            max_sup_r = max_sup_r.long()
            max_unsup_r = max_unsup_r.long()

            ### cps loss ###
            # cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
            criterion2 = nn.CrossEntropyLoss(reduction="none")
            cps_loss_l = ((criterion2(pred_sup_l, max_sup_r) * scale_weights_l) / (
                        max_sup_r.shape[0] * max_sup_r.shape[1] * max_sup_r.shape[2])).sum() + (
                                     (criterion2(pred_sup_r, max_sup_l) * scale_weights_l) / (
                                         max_sup_l.shape[0] * max_sup_l.shape[1] * max_sup_l.shape[2])).sum()

            cps_loss_un = ((criterion2(pred_unsup_l, max_unsup_r) * scale_weights_un) / (
                        max_unsup_r.shape[0] * max_unsup_r.shape[1] * max_unsup_r.shape[2])).sum() + (
                                      (criterion2(pred_unsup_r, max_unsup_l) * scale_weights_un) / (
                                          max_unsup_l.shape[0] * max_unsup_l.shape[1] * max_unsup_l.shape[2])).sum()

            ### standard cross entropy loss ###
            loss_sup = criterion(pred_sup_l, mask)
            loss_sup_r = criterion(pred_sup_r, mask)

            consistency_weight = get_current_consistency_weight(iter_num)

            hist2 += fast_hist(mask.cpu().numpy().flatten(), torch.max(pred_sup_l, dim=1)[1].cpu().numpy().flatten(),
                               num_classes)
            hist3 += fast_hist(mask.cpu().numpy().flatten(), torch.max(pred_sup_r, dim=1)[1].cpu().numpy().flatten(),
                               num_classes)

            loss = loss_sup + loss_sup_r + consistency_weight * (cps_loss_l + cps_loss_un)
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()
            optimizer1.step()
            optimizer2.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer1.param_groups[0]["lr"] = lr
            optimizer1.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            optimizer2.param_groups[0]["lr"] = lr
            optimizer2.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        epoch_class_iou2 = per_class_iu(hist2)
        epoch_class_iou3 = per_class_iu(hist3)
        mean_iou2 = np.mean(epoch_class_iou2)
        mean_iou3 = np.mean(epoch_class_iou3)

        print("========================================Training Set===================================================>")
        print("========================================model1===================================================>")
        print("tumor_iou : ", epoch_class_iou2[0], "stroma_iou : ",
              epoch_class_iou2[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou2[2],
              "necrosis_or_debris_iou : ", epoch_class_iou2[3], "other_iou : ", epoch_class_iou2[4])
        print("mean_iou = ", mean_iou2)
        writer.add_scalar('stage2_train/model1_tumor_iou', epoch_class_iou2[0], iter_num)
        writer.add_scalar('stage2_train/model1_stroma_iou', epoch_class_iou2[1], iter_num)
        writer.add_scalar('stage2_train/model1_lymphocytic_infiltrate_iou', epoch_class_iou2[2], iter_num)
        writer.add_scalar('stage2_train/model1_necrosis_or_debris_iou', epoch_class_iou2[3], iter_num)
        writer.add_scalar('stage2_train/model1_other_iou', epoch_class_iou2[4], iter_num)
        writer.add_scalar('stage2_train/model1_mean_iou', mean_iou2, iter_num)

        print("========================================model2===================================================>")
        print("tumor_iou : ", epoch_class_iou3[0], "stroma_iou : ",
              epoch_class_iou3[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou3[2],
              "necrosis_or_debris_iou : ", epoch_class_iou3[3], "other_iou : ", epoch_class_iou3[4])
        print("mean_iou = ", mean_iou3)
        writer.add_scalar('stage2_train/model2_tumor_iou', epoch_class_iou3[0], iter_num)
        writer.add_scalar('stage2_train/model2_stroma_iou', epoch_class_iou3[1], iter_num)
        writer.add_scalar('stage2_train/model2_lymphocytic_infiltrate_iou', epoch_class_iou3[2], iter_num)
        writer.add_scalar('stage2_train/model2_necrosis_or_debris_iou', epoch_class_iou3[3], iter_num)
        writer.add_scalar('stage2_train/model2_other_iou', epoch_class_iou3[4], iter_num)
        writer.add_scalar('stage2_train/model2_mean_iou', mean_iou3, iter_num)

        metric = meanIOU(num_classes=5 if args.dataset == 'pascal' else 19)
        print("consistency_weight = ",consistency_weight)
        model1.eval()
        model2.eval()

        tbar = tqdm(valloader)

        num_classes = 5
        hist4 = np.zeros((num_classes, num_classes))
        hist5 = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model1(img)
                one = torch.ones((pred.shape[0], 1, pred.shape[2], pred.shape[3])).cuda()
                pred = torch.cat([pred, (100 * one * (mask.cuda() == 4).unsqueeze(dim=1))], dim=1)
                pred = torch.argmax(pred, dim=1)

                pred2 = model2(img)
                one = torch.ones((pred2.shape[0], 1, pred2.shape[2], pred2.shape[3])).cuda()
                pred2 = torch.cat([pred2, (100 * one * (mask.cuda() == 4).unsqueeze(dim=1))], dim=1)
                pred2 = torch.argmax(pred2, dim=1)

                # hist4 += fast_hist(pred.cpu().numpy().flatten(),
                #                    mask.cpu().numpy().flatten(), num_classes)
                # hist5 += fast_hist(pred2.cpu().numpy().flatten(),
                #                    mask.cpu().numpy().flatten(), num_classes)
                hist4 += fast_hist(mask.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), num_classes)
                hist5 += fast_hist(mask.cpu().numpy().flatten(), pred2.cpu().numpy().flatten(), num_classes)
                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))
        epoch_class_iou2 = per_class_iu(hist4)
        epoch_class_iou3 = per_class_iu(hist5)
        mean_iou2 = np.mean(epoch_class_iou2)
        mean_iou3 = np.mean(epoch_class_iou3)

        print("========================================Validation Set===================================================>")
        print("========================================model1===================================================>")
        print("tumor_iou : ", epoch_class_iou2[0], "stroma_iou : ",
              epoch_class_iou2[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou2[2],
              "necrosis_or_debris_iou : ", epoch_class_iou2[3], "other_iou : ", epoch_class_iou2[4])
        print("mean_iou = ", mean_iou2)
        writer.add_scalar('stage2_val/model1_tumor_iou', epoch_class_iou2[0], iter_num)
        writer.add_scalar('stage2_val/model1_stroma_iou', epoch_class_iou2[1], iter_num)
        writer.add_scalar('stage2_val/model1_lymphocytic_infiltrate_iou', epoch_class_iou2[2], iter_num)
        writer.add_scalar('stage2_val/model1_necrosis_or_debris_iou', epoch_class_iou2[3], iter_num)
        writer.add_scalar('stage2_val/model1_other_iou', epoch_class_iou2[4], iter_num)
        writer.add_scalar('stage2_val/model1_mean_iou', mean_iou2, iter_num)

        print("========================================model2===================================================>")
        print("tumor_iou : ", epoch_class_iou3[0], "stroma_iou : ",
              epoch_class_iou3[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou3[2],
              "necrosis_or_debris_iou : ", epoch_class_iou3[3], "other_iou : ", epoch_class_iou3[4])
        print("mean_iou = ", mean_iou3)
        writer.add_scalar('stage2_val/model2_tumor_iou', epoch_class_iou3[0], iter_num)
        writer.add_scalar('stage2_val/model2_stroma_iou', epoch_class_iou3[1], iter_num)
        writer.add_scalar('stage2_val/model2_lymphocytic_infiltrate_iou', epoch_class_iou3[2], iter_num)
        writer.add_scalar('stage2_val/model2_necrosis_or_debris_iou', epoch_class_iou3[3], iter_num)
        writer.add_scalar('stage2_val/model2_other_iou', epoch_class_iou3[4], iter_num)
        writer.add_scalar('stage2_val/model2_mean_iou', mean_iou3, iter_num)

        mIOU *= 100.0

        if (mean_iou2 * 100 > previous_best) or (mean_iou3 * 100 > previous_best):
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))


            if mean_iou2 > mean_iou3:# model1 > model2
                previous_best = mean_iou2*100.0
                torch.save(model1.module.state_dict(),os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mean_iou2*100.0)))
                best_model = deepcopy(model1)
            else:
                previous_best = mean_iou3*100.0
                torch.save(model2.module.state_dict(),os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mean_iou3*100.0)))
                best_model = deepcopy(model2)


            # best_model = deepcopy(model1)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model1))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model

def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []
    iter_num = 0
    writer = SummaryWriter('./log')

    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        tbar = tqdm(trainloader)

        iter_num = iter_num + 1
        num_classes = 5
        hist = np.zeros((num_classes, num_classes))
        for i, (img, mask) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            # hist += fast_hist(pred.cpu().numpy().flatten(), mask.cpu().numpy().flatten(), num_classes)
            hist += fast_hist(mask.cpu().numpy().flatten(),pred.cpu().numpy().flatten(), num_classes)

            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        epoch_class_iou = per_class_iu(hist)
        mean_iou = np.mean(epoch_class_iou)
        # name_classes    = ["other","tumor","stroma","lymphocytic_infiltrate","necrosis_or_debris"]
        print("========================================Training Set===================================================>")
        print("tumor_iou : ", epoch_class_iou[0], "stroma_iou : ",
              epoch_class_iou[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou[2], "necrosis_or_debris_iou : ",
              epoch_class_iou[3], "other_iou : ", epoch_class_iou[4])
        print("mean_iou = ", mean_iou)
        writer.add_scalar('stage3_train/model_other_iou', epoch_class_iou[0], iter_num)
        writer.add_scalar('stage3_train/model__tumor_iou', epoch_class_iou[1], iter_num)
        writer.add_scalar('stage3_train/model__stroma_iou', epoch_class_iou[2], iter_num)
        writer.add_scalar('stage3_train/model_lymphocytic_infiltrate_iou', epoch_class_iou[3], iter_num)
        writer.add_scalar('stage3_train/model_necrosis_or_debris_iou', epoch_class_iou[4], iter_num)
        writer.add_scalar('stage3_train/model_mean_iou', mean_iou, iter_num)
        metric = meanIOU(num_classes=5 if args.dataset == 'pascal' else 19)

        model.eval()
        tbar = tqdm(valloader)
        hist2 = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                # hist2 += fast_hist(pred.cpu().numpy().flatten(), mask.cpu().numpy().flatten(), num_classes)
                hist2 += fast_hist(mask.cpu().numpy().flatten(),pred.cpu().numpy().flatten(), num_classes)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        epoch_class_iou2 = per_class_iu(hist2)
        mean_iou2 = np.mean(epoch_class_iou2)
        # name_classes    = ["other","tumor","stroma","lymphocytic_infiltrate","necrosis_or_debris"]
        print("========================================Training Set===================================================>")
        print("tumor_iou : ", epoch_class_iou2[0], "stroma_iou : ",
              epoch_class_iou2[1], "lymphocytic_infiltrate_iou : ", epoch_class_iou2[2], "necrosis_or_debris_iou : ",
              epoch_class_iou2[3], "other_iou : ", epoch_class_iou2[4])
        print("mean_iou = ", mean_iou2)
        writer.add_scalar('stage3_val/model_other_iou', epoch_class_iou2[0], iter_num)
        writer.add_scalar('stage3_val/model__tumor_iou', epoch_class_iou2[1], iter_num)
        writer.add_scalar('stage3_val/model__stroma_iou', epoch_class_iou2[2], iter_num)
        writer.add_scalar('stage3_val/model_lymphocytic_infiltrate_iou', epoch_class_iou2[3], iter_num)
        writer.add_scalar('stage3_val/model_necrosis_or_debris_iou', epoch_class_iou2[4], iter_num)
        writer.add_scalar('stage3_val/model_mean_iou', mean_iou2, iter_num)
        metric = meanIOU(num_classes=5 if args.dataset == 'pascal' else 19)


        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model


def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(num_classes=5 if args.dataset == 'pascal' else 19)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')



def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    metric = meanIOU(num_classes=5 if args.dataset == 'pascal' else 19)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)

            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.01, 'cityscapes': 0.004}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'pascal': 224, 'cityscapes': 721}[args.dataset]

    print()
    print(args)

    main(args)
