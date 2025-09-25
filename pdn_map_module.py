# pdn_map_module.py

import numpy as np

def get_pdn_density_map(spice_path):
    """
    Returns a raw 2D grid of resistor‐endpoint counts.
    Each SPICE resistor (R*) contributes +1 to each of its two node coords.
    Coordinates are integer µm derived via int(DBU/2000).
    """
    def get_coord(n):
        parts = n.split('_')
        if len(parts) == 4:
            try:
                return int(int(parts[2]) / 2000), int(int(parts[3]) / 2000)
            except ValueError:
                return None
        return None

    coords = []
    with open(spice_path, 'r') as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) == 4 and toks[0].startswith('R'):
                c1 = get_coord(toks[1])
                c2 = get_coord(toks[2])
                if c1: coords.append(c1)
                if c2: coords.append(c2)

    if not coords:
        return np.zeros((1,1), dtype=float)

    max_x = max(x for x,y in coords)
    max_y = max(y for x,y in coords)
    grid = np.zeros((max_y+1, max_x+1), dtype=float)
    for x,y in coords:
        grid[y, x] += 1.0

    return grid
