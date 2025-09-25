#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse

from current_map_module import get_current_map
from pdn_map_module import get_pdn_density_map
from voltage_distance_map_module import get_voltage_distance_map
from ir_map_module import get_ir_drop_map

def _save_heatmap(arr, fname, cmap, vmin=None, vmax=None, title="", origin='lower'):
    """
    Save a filled 2D heatmap of `arr` to `fname`.
    `origin` can be 'lower' (default) or 'upper'.
    """
    plt.figure(figsize=(6,6))
    norm = mcolors.Normalize(
        vmin=(vmin if vmin is not None else np.nanmin(arr)),
        vmax=(vmax if vmax is not None else np.nanmax(arr))
    )
    plt.imshow(arr,
               origin=origin,
               interpolation='none',
               cmap=cmap,
               norm=norm,
               aspect='equal')
    plt.colorbar(label=title)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

def generate_all_maps(spice_path, voltage_path, output_dir):
    # derive a label from the spice filename
    label = os.path.splitext(os.path.basename(spice_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    # 1) Current map
    cm = get_current_map(spice_path)
    _save_heatmap(cm,
                  os.path.join(output_dir, f"current_map_{label}.png"),
                  cmap='viridis',
                  title="Current Map",
                  origin='lower')

    # 2) PDN density map (3×3 tile normalization)
    raw = get_pdn_density_map(spice_path)
    tiles_y, tiles_x = 3, 3
    h, w = raw.shape
    tile_h, tile_w = h // tiles_y, w // tiles_x

    block = np.zeros((tiles_y, tiles_x), dtype=float)
    for i in range(tiles_y):
        for j in range(tiles_x):
            sub = raw[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            block[i,j] = np.sum(sub)

    mn, mx = block.min(), block.max()
    norm = (3 * (block - mn) / (mx - mn)) if mx != mn else np.zeros_like(block)
    up = np.kron(np.round(norm,1), np.ones((tile_h, tile_w)))
    pdn_map = up[:h, :w]

    _save_heatmap(pdn_map,
                  os.path.join(output_dir, f"pdn_density_map_{label}.png"),
                  cmap='jet',
                  vmin=0, vmax=3,
                  title="PDN Density Map",
                  origin='lower')

    # 3) Voltage-distance map
    vd = get_voltage_distance_map(spice_path)
    _save_heatmap(vd,
                  os.path.join(output_dir, f"voltage_distance_map_{label}.png"),
                  cmap='plasma',
                  title="Voltage Distance Map",
                  origin='lower')

    # 4) IR-drop map — post-processing for display
    ir = get_ir_drop_map(spice_path, voltage_path)    # raw [Y, X]
    ir_oriented = np.flipud(ir.T)                     # swap X/Y & flip Y
    _save_heatmap(ir_oriented,
                  os.path.join(output_dir, f"ir_drop_map_{label}.png"),
                  cmap='jet',
                  vmin=ir_oriented.min(),
                  vmax=ir_oriented.max(),
                  title="IR Drop Map",
                  origin='upper')

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate current, PDN-density, voltage-distance and IR-drop maps"
    )
    p.add_argument("spice_path",   help="Path to the SPICE .sp netlist")
    p.add_argument("voltage_path", help="Path to the corresponding .voltage file")
    p.add_argument("output_dir",   help="Directory where the four PNGs will be saved")
    args = p.parse_args()

    generate_all_maps(args.spice_path, args.voltage_path, args.output_dir)
