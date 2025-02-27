import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import scipy.io as io
import os
import re

class VGG(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        self.root_path = data_root
        self.train_lists = os.listdir(data_root+'VGG/train/images/')
        self.eval_list = os.listdir(data_root+'VGG/test/images/')
        if train:
            self.img_list = [os.path.join(data_root, 'VGG/train/images/', f) for f in self.train_lists]
        else:
            self.img_list = [os.path.join(data_root, 'VGG/test/images/', f) for f in self.eval_list]
        self.img_map = {}
        # 生成对应的txt标注路径
        for img_path in self.img_list:
            # 替换目录为annotations，并修改扩展名为.txt
            gt_path = img_path.replace('images', 'labels').replace('cell', 'dots')
            gt_path = os.path.splitext(gt_path)[0] + '.txt'
            self.img_map[img_path] = gt_path
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point = load_data((img_path, gt_path), self.train)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point = point.astype(np.float64)
                point *= float(scale)
        # random crop augumentaiton
        if self.train and self.patch:
            img, point = random_crop(img, point)
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(re.search(r'\d+', img_path.split('/')[-1].split('.')[0]).group())
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []
    # 从txt文件读取坐标
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    x, y = line.split()
                    points.append([float(x), float(y)])
    points = np.array(points, dtype=np.float64)
    return img, points

    # random crop augumentation


def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        if den.shape[0]:
            idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
            # shift the corrdinates
            record_den = den[idx]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
            result_den.append(record_den)
        else:
            result_den.append(np.empty((0, 2), dtype='float64'))

    return result_img, result_den
