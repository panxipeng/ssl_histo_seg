import os
import numpy as np
from PIL import Image

filename_path = r'D:\BaiduNetdiskDownload\semi-data\LUAD-HistoSeg\semi\valid\ori_mask'

file_path = os.path.join(filename_path,"387709-19860-36232.png")
mask_pil = Image.open(file_path)
mask_np = np.array(mask_pil)
mask_unique = np.unique(mask_np)
print("mask_unique = ",mask_unique)


# 0 肿瘤
# 1 间质
# 2 淋巴
# 3 坏死
# 4 背景

# 它
# mask_unique =  [0 1 4]
# #
# wo
# 0,1,2