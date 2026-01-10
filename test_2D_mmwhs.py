import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/MMWHS_ROI', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='MMWHS_ROI', help='Dataset')
parser.add_argument('--exp', type=str,
                    default='EVIL', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=8,
                    help='output channel of network')
parser.add_argument('--seed', type=int,  default=1337,
                    help='random seed')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--labeled_num', type=int, default=21,
                    help='labeled data')


def calculate_metric_percase(pred, gt):

    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume3d(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/val/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    input = np.zeros((image.shape[0], FLAGS.patch_size[0], FLAGS.patch_size[1]))
    x, y = image.shape[1], image.shape[2]
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        slice = zoom(slice, (FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
        input[ind,:] = slice

    input_tensor = torch.from_numpy(input).unsqueeze(1).float().cuda()
    with torch.no_grad():
        if FLAGS.model == "unet_urpc" or FLAGS.model == "unet_cct":
            out_main, _, _, _ = net(input_tensor)
        elif 'fcn' in FLAGS.model:
            out_main = net(input_tensor.repeat_interleave(3, dim=1))
        else:
            out_main = net(input_tensor)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(1)
        out = out.cpu().detach().numpy()
        for ind in range(out.shape[0]):
            slice = out[ind, :, :]
            pred = zoom(slice, (x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
            prediction[ind] = pred

    LV_metric = calculate_metric_percase(prediction == 1, label == 1)
    RV_metric = calculate_metric_percase(prediction == 2, label == 2)
    LA_metric = calculate_metric_percase(prediction == 3, label == 3)
    RA_metric = calculate_metric_percase(prediction == 4, label == 4)
    MYO_metric = calculate_metric_percase(prediction == 5, label == 5)
    ASA_metric = calculate_metric_percase(prediction == 6, label == 6)
    PUA_metric = calculate_metric_percase(prediction == 7, label == 7)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii")
    return LV_metric, RV_metric, LA_metric, RA_metric, MYO_metric, ASA_metric, PUA_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = "./model/{}/{}_{}_{}/{}".format(FLAGS.dataset,
        FLAGS.exp, FLAGS.labeled_num, FLAGS.seed, FLAGS.model)
    test_save_path = "./model/{}/{}_{}_{}/{}_predictions/".format(FLAGS.dataset,
        FLAGS.exp, FLAGS.labeled_num,  FLAGS.seed, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes).cuda()
    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
    # save_mode_path = os.path.join(
    #     snapshot_path, 'iter_12000.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    LV_total = 0.0
    RV_total = 0.0
    LA_total = 0.0
    RA_total = 0.0
    MYO_total = 0.0
    ASA_total = 0.0
    PUA_total = 0.0

    for case in tqdm(image_list):
        LV_metric, RV_metric, LA_metric, RA_metric, MYO_metric, ASA_metric, PUA_metric = test_single_volume3d(
            case, net, test_save_path, FLAGS)
        LV_total += np.asarray(LV_metric)
        RV_total += np.asarray(RV_metric)
        LA_total += np.asarray(LA_metric)
        RA_total += np.asarray(RA_metric)
        MYO_total += np.asarray(MYO_metric)
        ASA_total += np.asarray(ASA_metric)
        PUA_total += np.asarray(PUA_metric)

    avg_metric = [LV_total / len(image_list), RV_total / len(image_list),
                  LA_total / len(image_list), RA_total / len(image_list),
                  MYO_total / len(image_list), ASA_total / len(image_list), PUA_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print('LV: ', metric[0])
    print('RV: ', metric[1])
    print('LA: ', metric[2])
    print('RA: ', metric[3])
    print('MYO: ', metric[4])
    print('ASA: ', metric[5])
    print('PUA: ', metric[6])
    # print(metric)
    print('Mean: ', (metric[0]+metric[1]+metric[2]+metric[3]+metric[4]+metric[5]+metric[6])/7)
