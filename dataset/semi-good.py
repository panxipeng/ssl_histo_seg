from dataset.transform import crop, hflip, normalize, resize, blur, cutout,rand_bbox

import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from albumentations import Compose
from  albumentations  import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,VerticalFlip,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose,Resize,Sharpen,PiecewiseAffine,Emboss,RandomBrightnessContrast
) # 图像变换函数
import cv2

class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None, pseudo_mask_path=None):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        :param labeled_id_path: path of labeled image ids, needed in train or semi_train mode.
        :param unlabeled_id_path: path of unlabeled image ids, needed in semi_train or label mode.
        :param pseudo_mask_path: path of generated pseudo masks, needed in semi_train mode.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids = \
                self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids

        else:
            if mode == 'val':
                id_path = 'dataset/splits/%s/val.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path
            elif mode == 'first_stage':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))

        # if self.mode == 'val' or self.mode == 'label':
        if self.mode == 'val' :
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            img, mask = normalize(img, mask)
            return img, mask, id
        if self.mode == 'label':
            fake_mask = np.random.randn(img.size[0], img.size[1])
            mask = Image.fromarray(fake_mask)
            img, mask = normalize(img, mask)
            return img, mask, id
        # 如果训练模式是train的话
        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            if os.path.exists(os.path.join(self.root, id.split(' ')[1])):
                mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
            else:
                fake_mask = np.random.randn(img.size[0], img.size[1])
                mask = Image.fromarray(fake_mask)
        elif self.mode == 'first_stage':
            fake_mask = np.random.randn(img.size[0],img.size[1])
            mask = Image.fromarray(fake_mask)
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # basic augmentation on all training images
        base_size = 400 if self.name == 'pascal' else 2048
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, p=0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            img, mask = cutout(img, mask, p=0.5)
            image = np.array(img)
            masks = np.array(mask)
            
            # ------------------------------  CutMix  ------------------------------------------
            if random.random() < 0.5:
                # rand_index:[0,]
                # 这里改一下，我们另一个图片只选择有标签的图像
                rand_index = random.randint(0, len(self.labeled_ids) - 1)
                id = self.ids[rand_index]
                rand_image = Image.open(os.path.join(self.root, id.split(' ')[0]))
                rand_masks = Image.open(os.path.join(self.root, id.split(' ')[1]))
                rand_image, rand_masks = resize(rand_image, rand_masks, base_size, (0.5, 2.0))
                rand_image, rand_masks = crop(rand_image, rand_masks, self.size)
                rand_image = np.array(rand_image)
                rand_masks = np.array(rand_masks)
                lam = np.random.beta(1, 1)
                # 原图转为np
                
                bbx1, bby1, bbx2, bby2 = rand_bbox(image.shape, lam)

                image[bbx1:bbx2, bby1:bby2, :] = rand_image[bbx1:bbx2, bby1:bby2, :]
                masks[bbx1:bbx2, bby1:bby2] = rand_masks[bbx1:bbx2, bby1:bby2]


            # ---------------------------------  CutMix  ---------------------------------------

            # 病理数据增强
            p = 0.3
            combine_transform = Compose([
                RandomRotate90(p=0.3),  # 随机旋转
                Flip(p=0.2),  # 水平翻转或垂直翻转
                Transpose(p=0.1),  # 行列转置
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5,
                                 border_mode=cv2.BORDER_REFLECT),  # 仿射变换：线性变换+平移
                OneOf([
                    MedianBlur(blur_limit=3, p=0.3),  # 中心模糊
                    Blur(blur_limit=3, p=0.3),  # 模糊图像
                ], p=0.3),
                OneOf([
                    OpticalDistortion(p=0.3),  # 光学畸变
                    IAAPiecewiseAffine(p=0.3),  # 形态畸变
                ], p=0.3),
                OneOf([
                    IAASharpen(),  # 锐化
                    IAAEmboss(),  # 类似于锐化
                    RandomContrast(limit=0.5),  # 对比度变化
                ], p=0.3),
                OneOf([  # HSV变换
                    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10),
                    # CLAHE(clip_limit=2),
                    RandomBrightnessContrast(),  # 随机亮度和对比度变化
                ], p=0.8),
            ], p=p)

            augmented = combine_transform(image=image, mask=masks)
            image = augmented['image']
            masks = augmented['mask']

        # 将image和mask转换为PIL
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            img = Image.fromarray(image)
            mask = Image.fromarray(masks)


        img, mask = normalize(img, mask)

        mask_false = (mask >= 4)
        mask[mask_false] = 4

        return img, mask

    def __len__(self):
        return len(self.ids)
