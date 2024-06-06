import os
import numpy as np
from PIL import Image


palette = [0] * 768
palette[0:3] = [255,255,255] # 0
palette[3:6] = [205,51,51] # 1
palette[6:9] = [255,165,0] # 2
palette[9:12] = [65,105,225] # 3
palette[12:15] = [0,255,0] # 4

root_path = r"D:\BaiduNetdiskDownload\semi-data\LUAD-HistoSeg\semi\luad_expenriments\expenriment1\test\mask"
mask_filenames = os.listdir(root_path)
new_root_path = r"D:\BaiduNetdiskDownload\semi-data\LUAD-HistoSeg\semi\luad_expenriments\expenriment1\test\remask"

for mask_filename in mask_filenames:
    mask_path = os.path.join(root_path, mask_filename)
    new_mask_path = os.path.join(new_root_path, mask_filename)

    mask_pil = Image.open(mask_path)
    mask_np = np.array(mask_pil)

    bool_0 = [mask_np == 0] # 肿瘤 0
    bool_1 = [mask_np == 1] # 间质 1
    bool_2 = [mask_np == 2] # 淋巴 2
    bool_3 = [mask_np == 3] # 坏死 3
    bool_4 = [mask_np == 4] # 背景

    mask_np[bool_0] = 5 # 肿瘤
    mask_np[bool_1] = 6 # 间质
    mask_np[bool_2] = 7 # 淋巴
    mask_np[bool_3] = 8 # 坏死
    mask_np[bool_4] = 9 # 背景

    # 转回来
    bool_5 = [mask_np == 5]  # 肿瘤 0
    bool_6 = [mask_np == 6]  # 间质 1
    bool_7 = [mask_np == 7]  # 淋巴 2
    bool_8 = [mask_np == 8]  # 坏死 3
    bool_9 = [mask_np == 9]  # 背景

    mask_np[bool_5] = 1  # 肿瘤
    mask_np[bool_6] = 2  # 间质
    mask_np[bool_7] = 3  # 淋巴
    mask_np[bool_8] = 4  # 坏死
    mask_np[bool_9] = 0  # 背景

    # 重新染色
    # 染色
    # 将预测pred label从numpy转为uint8的P模式的PIL Image方便上色
    visualimg = Image.fromarray(mask_np.astype(np.uint8), "P")
    # 通过调色板上色
    visualimg.putpalette(palette)
    # P模式转RGB的PIL
    # visualimg = visualimg.convert("RGB")

    visualimg.save(new_mask_path)




