import argparse
import logging
import os
import random
import shutil
import sys
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
# import wandb
# from dataloaders import utils
import sys
import torch.nn.functional as F
sys.path.append("..")
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
# from networks.net_factory import net_factory
from networks.unet import UNet
from utils import losses, ramps
from utils.val_2D import test_single_volume

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/EVIL', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=887, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=21,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "MMWHS" in dataset:
        ref_dict = {"1": 96, "2": 192, "3": 288, "4": 384, "5": 480, "6": 576, "7":672, "8":768}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    def create_model(ema=False):
        # Network definition
        model = UNet(in_chns=1, class_num=num_classes).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model()
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=1, pin_memory=True)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            label0 = (label_batch == 0).unsqueeze(1)
            label1 = (label_batch == 1).unsqueeze(1)
            label2 = (label_batch == 2).unsqueeze(1)
            label3 = (label_batch == 3).unsqueeze(1)
            onehot = torch.cat([label0, label1, label2, label3], dim=1).int()

            outputs1 = model1(volume_batch)
            evidence1 = torch.exp(torch.tanh(outputs1) / 0.25)

            alpha1 = evidence1 + 1
            S = torch.sum(alpha1, dim=1, keepdim=True)
            belief1 = evidence1 / S
            # preds1 = alpha1 / S
            preds1 = F.softmax(belief1, dim=1)
            uncertainty = num_classes / S

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            classic_dice = dice_loss(preds1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            evidence_ce = losses.edl_digamma_loss(
                evidence1[:args.labeled_bs].float(), onehot[:args.labeled_bs].float(), epoch_num,
                num_classes, (max_epoch // 2)
            )

            loss1 = 0.5 * (evidence_ce + classic_dice)

            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(preds1[args.labeled_bs:].detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)

            un_onehot = torch.zeros_like(preds1[args.labeled_bs:])
            for i in range(args.num_classes):
                un_onehot[:, i, :, :] = (pseudo_outputs2 == i).unsqueeze(0)

            pseudo_supervision1 = losses.edl_digamma_loss(
                evidence1[args.labeled_bs:].float(), un_onehot.float(), epoch_num,
                num_classes, (max_epoch // 2))


            # threshold = 0.25 - 0.15 * ramps.sigmoid_rampup(epoch_num, max_epoch)
            threshold = 0.2
            mask = (uncertainty[args.labeled_bs:] < threshold).float()
            pseudo_supervision2 = ce_loss((mask * outputs2[args.labeled_bs:]), (mask.squeeze(1) * pseudo_outputs1).long())

            model1_loss = loss1 + consistency_weight * pseudo_supervision1
            model2_loss = loss2 + consistency_weight * pseudo_supervision2

            loss = model1_loss + model2_loss


            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            # lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            lr2 = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            # if iter_num % 4000 == 0:
            #     learning_rate1 = learning_rate1 * 0.5
            lr1 = base_lr * (1.0 - iter_num / max_iterations) ** 0.99
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr1
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr2

            writer.add_scalar('lr1', lr1, iter_num)
            writer.add_scalar('lr2', lr2, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f model2 loss : %f' % (iter_num, model1_loss.item(), model2_loss.item()))

            if iter_num > 0 and iter_num % 1000 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_{}/{}".format(
        args.exp, args.labeled_num, args.seed, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
