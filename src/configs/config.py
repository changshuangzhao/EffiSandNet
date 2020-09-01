from easydict import EasyDict

cfg = EasyDict()

cfg.InputSize_w = 896
cfg.InputSize_h = 896

cfg.DataRoot = '/Users/yanyan/data'
cfg.TrainData = '../generators/convert_data/val.csv'
cfg.ValData = '../generators/convert_data/val.csv'
cfg.Cls = '../generators/convert_data/cls.csv'
cfg.Pro = '../generators/convert_data/pro.csv'

cfg.w_bifpn = 64  # [64, 88, 112, 160, 224, 288, 384]
cfg.d_bifpn = 3  # [3, 4, 5, 6, 7, 7, 8]
cfg.d_head = 3  # [3, 3, 3, 4, 4, 4, 5]


