import numpy as np

def get_coord(nodename):
    parts = nodename.split('_')
    if len(parts) == 4:
        return int(int(parts[2]) / 2000), int(int(parts[3]) / 2000)
    return None

def get_current_map(spice_path):
    coords, values = [], []
    with open(spice_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 4 and tokens[0].startswith('I'):
                n1, n2, val = tokens[1], tokens[2], float(tokens[3])
                c1, c2 = get_coord(n1), get_coord(n2)
                if c1: coords.append(c1); values.append(val)
                if c2: coords.append(c2); values.append(-val)

    if not coords:
        return np.zeros((1,1))
    max_x = max(x for x,y in coords)
    max_y = max(y for x,y in coords)
    grid = np.zeros((max_y+1, max_x+1), dtype=float)
    for (x,y), v in zip(coords, values):
        grid[y, x] += v
    return grid
