#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score

from current_map_module import get_current_map
from pdn_map_module import get_pdn_density_map
from voltage_distance_map_module import get_voltage_distance_map
from ir_map_module import get_ir_drop_map

def pad_to_common(cur, pdn, vd, ir):
    H = max(cur.shape[0], pdn.shape[0], vd.shape[0], ir.shape[0])
    W = max(cur.shape[1], pdn.shape[1], vd.shape[1], ir.shape[1])
    def pad(a):
        th = (H - a.shape[0]) // 2
        bh = H - a.shape[0] - th
        tw = (W - a.shape[1]) // 2
        bw = W - a.shape[1] - tw
        return np.pad(a, ((th, bh), (tw, bw)), mode='constant')
    return pad(cur), pad(pdn), pad(vd), pad(ir)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
    def forward(self, x):
        return self.net(x)

def inference(spice_file, voltage_file, model_path, output_csv, prefix):
    start_time = time.time()

    # 1) Load maps
    cm  = get_current_map(spice_file)
    pdn = get_pdn_density_map(spice_file)
    vd  = get_voltage_distance_map(spice_file)
    gt  = get_ir_drop_map(spice_file, voltage_file)

    # 2) Pad to common size
    cm, pdn, vd, gt = pad_to_common(cm, pdn, vd, gt)

    # 3) Stack features into tensor
    x = np.stack([cm, pdn, vd], axis=0)[None,...].astype(np.float32)
    x_t = torch.from_numpy(x).cuda()

    # 4) Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleCNN().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 5) Predict
    with torch.no_grad():
        pred = model(x_t).cpu().numpy()[0,0]  # shape [H,W]

    elapsed = time.time() - start_time  # 🕒 measure elapsed time

    # 6) Save raw prediction
    pd.DataFrame(pred).to_csv(output_csv, header=False, index=False)
    print(f"✅ Predicted CSV saved to {output_csv}")

    # 7) Metrics
    mae = np.mean(np.abs(pred - gt))

    gt_thr = 0.9 * gt.max()
    gbin = (gt >= gt_thr).astype(int).ravel()

    pthr = np.percentile(pred, 90)
    pbin = (pred >= pthr).astype(int).ravel()

    prec = precision_score(gbin, pbin, zero_division=0)
    rec  = recall_score(gbin, pbin, zero_division=0)
    f1   = f1_score(gbin, pbin, zero_division=0)

    # 8) Print metrics
    print(f"🔎 Metrics:")
    print(f"MAE       = {mae:.6e}")
    print(f"Precision = {prec:.10f}")
    print(f"Recall    = {rec:.10f}")
    print(f"F1-score  = {f1:.10f}")
    print(f"Runtime   = {elapsed:.4f} seconds")

    # 9) Save images (with flip + transpose)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)

    pred_img = np.flipud(pred.T)
    gt_img = np.flipud(gt.T)

    plt.imsave(f"{prefix}_predicted.png", pred_img, cmap='jet',
               vmin=pred_img.min(), vmax=pred_img.max(), origin='upper')

    plt.imsave(f"{prefix}_groundtruth.png", gt_img, cmap='jet',
               vmin=gt_img.min(), vmax=gt_img.max(), origin='upper')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-spice_file',   required=True)
    parser.add_argument('-voltage_file', required=True)
    parser.add_argument('-ml_model',     required=True)
    parser.add_argument('-output',       required=True)
    parser.add_argument('-prefix',       required=True)
    args = parser.parse_args()

    inference(
        args.spice_file,
        args.voltage_file,
        args.ml_model,
        args.output,
        args.prefix
    )
