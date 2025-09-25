# training.py

import os, re, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

def pad_to_common(cur, pdn, vdd, ir):
    H = max(cur.shape[0], pdn.shape[0], vdd.shape[0], ir.shape[0])
    W = max(cur.shape[1], pdn.shape[1], vdd.shape[1], ir.shape[1])
    def pad(a):
        th, bh = (H - a.shape[0])//2, (H - a.shape[0]) - (H - a.shape[0])//2
        tw, bw = (W - a.shape[1])//2, (W - a.shape[1]) - (W - a.shape[1])//2
        return np.pad(a, ((th,bh),(tw,bw)), mode='constant')
    return pad(cur), pad(pdn), pad(vdd), pad(ir)

class IRDataset(Dataset):
    def __init__(self, csv_dir):
        self.dir = csv_dir
        pat = re.compile(r"current_map_(data_point\d+)\.csv")
        self.names = sorted(m.group(1)
            for f in os.listdir(csv_dir)
            if (m := pat.match(f))
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        n = self.names[idx]
        cur = pd.read_csv(f"{self.dir}/current_map_{n}.csv",              header=None).values
        pdn = pd.read_csv(f"{self.dir}/pdn_density_map_{n}.csv",         header=None).values
        vdd = pd.read_csv(f"{self.dir}/voltage_source_map_{n}.csv",      header=None).values
        ir  = pd.read_csv(f"{self.dir}/ir_drop_map_{n}.csv",             header=None).values

        cur, pdn, vdd, ir = pad_to_common(cur, pdn, vdd, ir)
        x = np.stack([cur, pdn, vdd], axis=0).astype(np.float32)   # [3,H,W]
        y = ir.astype(np.float32)[None,...]                        # [1,H,W]
        return torch.from_numpy(x), torch.from_numpy(y)

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

def train(csv_dir, out_model, epochs=6000, lr=1e-3):
    ds = IRDataset(csv_dir)
    dl = DataLoader(ds, batch_size=1, shuffle=True)  # variable H×W
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    best_mae = float('inf')
    for ep in range(1, epochs+1):
        total = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        mae = total / len(ds)
        print(f"Epoch {ep}/{epochs} — MAE={mae:.6e}")
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), out_model)

    print(f"\nTraining complete! Best MAE: {best_mae:.6e}")
    print(f"Model saved to: {out_model}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-input',  required=True, help="CSV maps directory")
    p.add_argument('-output', required=True, help="Where to save model (.pth)")
    args = p.parse_args()
    train(args.input, args.output)
