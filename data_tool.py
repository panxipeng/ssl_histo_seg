import glob
import os

import numpy as np
import cv2
from PIL import Image


def npz():
    #图像路径
    path = r'D:\papers\Transformer\TransUnet\TransUNet-main\valid_data\images\*.png'
    #项目中存放训练所用的npz文件路径
    path2 = r'D:\GitCode\Swin-Unet\data\Synapse\test_vol_h5\\'
    for i,img_path in enumerate(glob.glob(path)):
    	#读入图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #读入标签
        label_path = img_path.replace('images','labels')
        # label = cv2.imread(label_path,flags=0)
        label = np.array(Image.open(label_path))
		#保存npz
        name = img_path.split('\\')[-1]
        pid = name[:-4]
        np.savez(path2 + pid,image=image,label=label)
        print('------------',i)

    # 加载npz文件
    # data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
    # image, label = data['image'], data['label']

    print('ok')

# 制作数据集
# npz()


# 将文件名称写入train.txt和test_vol.txt
# filenames = os.listdir(r'C:\Users\86159\Desktop\ST-PlusPlus\data\1_100\img')
# print("filenames = ",filenames)
# labeled_txt = open(r'C:\Users\86159\Desktop\ST-PlusPlus\data\1_100\labeled.txt', 'w')
#
# for filename in filenames:
#     if "png" in filename:
#         labeled_txt.write("JPEGImages/" + filename+ " " +"SegmentationClass/" +  filename + "\n")
# labeled_txt.close()


# filenames = os.listdir(r'C:\Users\86159\Desktop\ST-PlusPlus\data\1_100\img_rest')
# # print("filenames = ",filenames)
# print("len(filenames) = ",len(filenames))
# filenames2 = os.listdir(r'C:\Users\86159\Desktop\ST-PlusPlus\data\training')
# # print("filenames2 = ",filenames2)
# print("len(filenames2) = ",len(filenames2))
#
# filenames3 = filenames + filenames2
# print("len(filenames3) = ",len(filenames3))
#
# unlabeled_txt = open(r'C:\Users\86159\Desktop\ST-PlusPlus\data\1_100\unlabeled.txt', 'w')
#
# for filename in filenames3:
#     if "png" in filename:
#         unlabeled_txt.write("JPEGImages/" + filename+ " " +"SegmentationClass/" +  filename + "\n")
# unlabeled_txt.close()



# 将文件名称写入train.txt和test_vol.txt
filenames = os.listdir(r'D:\BaiduNetdiskDownload\tissue_seg\1_1_2418\unlabeled_1_2_11711')
print("filenames = ",filenames)
labeled_txt = open(r'D:\BaiduNetdiskDownload\tissue_seg\1_1_2418\unlabeled.txt', 'w')

for filename in filenames:
    if "png" in filename:
        labeled_txt.write("JPEGImages/" + filename+ " " +"SegmentationClass/" +  filename + "\n")
labeled_txt.close()
















# stage1
# ==> Epoch 79, learning rate = 0.0002					 previous best = 63.13
# Loss: 0.867: 100%|██████████| 253/253 [02:18<00:00,  1.83it/s]
#   0%|          | 0/50 [00:00<?, ?it/s]consistency_weight =  0.3
# mIOU: 59.57: 100%|██████████| 50/50 [00:01<00:00, 26.88it/s]
# finish SupOnly......

# stage2
# ==> Epoch 79, learning rate = 0.0056					 previous best = 55.36
# mIOU: 52.16: 100%|██████████| 50/50 [00:01<00:00, 28.56it/s]
# Loss: 1.333: 100%|█████████▉| 259/260 [02:29<00:00,  1.64it/s]consistency_weight =  0.3
# Loss: 1.335: 100%|██████████| 260/260 [02:29<00:00,  1.74it/s]
# mIOU: 49.57: 100%|██████████| 50/50 [00:01<00:00, 27.24it/s]

# stage3
# mIOU: 60.00:  98%|█████████▊| 49/50 [00:01<00:00, 28.66it/s]
# ==> Epoch 79, learning rate = 0.0002					 previous best = 62.68
# mIOU: 60.27: 100%|██████████| 50/50 [00:01<00:00, 28.49it/s]
# Loss: 1.047: 100%|█████████▉| 1058/1059 [10:00<00:00,  1.75it/s]consistency_weight =  0.3
# Loss: 1.047: 100%|██████████| 1059/1059 [10:00<00:00,  1.76it/s]
# mIOU: 60.75: 100%|██████████| 50/50 [00:01<00:00, 28.99it/s]
#

