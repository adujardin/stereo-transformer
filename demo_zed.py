from PIL import Image
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./') # add relative path
import skimage.io

from module.sttr import STTR
from dataset.preprocess import normalization, compute_left_occ_region
from utilities.misc import NestedTensor

# Default parameters
args = type('', (), {})() # create empty args
args.channel_dim = 128
args.position_encoding='sine1d_rel'
args.num_attn_layers=6
args.nheads=8
args.regression_head='ot'
args.context_adjustment_layer='cal'
args.cal_num_blocks=8
args.cal_feat_dim=16
args.cal_expansion_ratio=4

model = STTR(args).cuda().eval()

# Load the pretrained model
model_file_name = "./sceneflow_pretrained_model.pth.tar"#"./kitti_finetuned_model.pth.tar"
checkpoint = torch.load(model_file_name)
pretrained_dict = checkpoint['state_dict']
model.load_state_dict(pretrained_dict)
print("Pre-trained model successfully loaded.")

left = np.array(Image.open('./sample_data/ZED/left.png'))[:,:,0:3]
right = np.array(Image.open('./sample_data/ZED/right.png'))[:,:,0:3]

# normalize
input_data = {'left': left, 'right':right}
input_data = normalization(**input_data)

# donwsample attention by stride of 3
h, w, _ = left.shape
bs = 1

downsample = 3
col_offset = int(downsample / 2)
row_offset = int(downsample / 2)
sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()

# build NestedTensor
input_data = NestedTensor(input_data['left'].cuda()[None,],input_data['right'].cuda()[None,], sampled_cols=sampled_cols, sampled_rows=sampled_rows)

output = model(input_data)

# set disparity of occ area to 0
disp_pred = output['disp_pred'].data.cpu().numpy()[0]
occ_pred = output['occ_pred'].data.cpu().numpy()[0] > 0.5
disp_pred[occ_pred] = 0.0

# Saving disp
skimage.io.imsave("disp_pred.png", (disp_pred * 256.).astype(np.uint16))

# Ptcloud generation from predicted disp, save into ply (require plyfile package)
if True:
    # Load it
    import json
    from plyfile import PlyData, PlyElement
    save_name_cloud = "cloud.ply"
    with open("./sample_data/ZED/calibration.json") as f:
        data = json.load(f)
        fx = data["fx"]
        fy = data["fy"]
        cx = data["cx"]
        cy = data["cy"]
        baseline = data["baseline"]
        H = disp_pred.shape[0]
        W = disp_pred.shape[1]
        depth = (baseline * fx) / disp_pred
        pts_color_full = []

        clamping_depth=40*1000
        print("clamping at " + str(clamping_depth) + "mm")

        for m in range(0, H):
            for n in range(0, W):
                z = depth[m, n]
                if z >= 0.: #z < 0.:
                    color = left[m, n]
                    x = ((n - cx) * z) / fx
                    y = ((m - cy) * z) / fy

                    if not np.isnan(x*z*y) \
                            and np.isfinite(x*z*y) \
                            and abs(z) < clamping_depth \
                            and abs(y) < clamping_depth \
                            and abs(x) < clamping_depth:
                        pts_color_full.append((x, y, z, int(color[0]), int(color[1]), int(color[2])))

        vertex = np.array(pts_color_full,
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                                    ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], text=False).write(save_name_cloud)