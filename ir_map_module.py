import numpy as np
import pandas as pd

def get_ir_drop_map(spice_path, voltage_path, VDD=1.1):
    coords = {}
    with open(spice_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) >= 4:
                for tok in tokens[1:3]:
                    if tok.startswith('n1_m1_') and tok not in coords:
                        parts = tok.split('_')
                        x, y = int(int(parts[2]) / 2000), int(int(parts[3]) / 2000)
                        coords[tok] = (y, x)
    rows = []
    with open(voltage_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2 and parts[0] in coords:
                r, c = coords[parts[0]]
                val = VDD - float(parts[1])
                rows.append((r, c, val))
    if not rows:
        return np.zeros((1,1))
    max_r = max(r for r,c,v in rows) + 1
    max_c = max(c for r,c,v in rows) + 1
    grid = np.zeros((max_r, max_c), dtype=float)
    for r, c, v in rows:
        grid[r, c] = v
    return grid
