import argparse
import os
import shutil
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
# from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
from networks.unet import UNet
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='Dataset')
parser.add_argument('--exp', type=str,
                    default='EVIL', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--seed', type=int,  default=1337,
                    help='random seed')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
parser.add_argument('--patch_size', type=int, default=[256, 256],
                    help='labeled data')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    dice = metric.binary.dc(pred, gt) * 100.0
    asd = metric.binary.asd(pred, gt) if np.sum(pred) != 0 else 100
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume3d(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
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
        else:
            out_main = net(input_tensor)
        out = torch.argmax(torch.softmax(
            out_main, dim=1), dim=1).squeeze(1)
        out = out.cpu().detach().numpy()
        for ind in range(out.shape[0]):
            slice = out[ind, :, :]
            pred = zoom(slice, (x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
            prediction[ind] = pred

    RV_metric = calculate_metric_percase(prediction == 1, label == 1)
    LV_metric = calculate_metric_percase(prediction == 2, label == 2)
    Myo_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii")
    return RV_metric, LV_metric, Myo_metric


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
    net = UNet(in_chns=1, class_num=FLAGS.num_classes).cuda()
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model1.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    LV_total = 0.0
    RV_total = 0.0
    Myo_total = 0.0


    for case in tqdm(image_list):
        RV_metric, LV_metric, Myo_metric = test_single_volume3d(
            case, net, test_save_path, FLAGS)
        LV_total += np.asarray(LV_metric)
        RV_total += np.asarray(RV_metric)
        Myo_total += np.asarray(Myo_metric)

    avg_metric = [RV_total / len(image_list), LV_total / len(image_list),
                  Myo_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metrics = Inference(FLAGS)
    print('RV: ', metrics[0])
    print('LV: ', metrics[1])
    print('Myo: ', metrics[2])
    # print(metric)
    print('Mean: ', (metrics[0]+metrics[1]+metrics[2])/3)
