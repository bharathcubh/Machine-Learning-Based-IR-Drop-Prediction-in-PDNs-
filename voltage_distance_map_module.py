import numpy as np

def get_coord(nodename):
    parts = nodename.split('_')
    if len(parts) == 4:
        return int(int(parts[2]) / 2000), int(int(parts[3]) / 2000)
    return None

def extract_voltage_coords(spice_path):
    voltage_coords, all_coords = set(), set()
    with open(spice_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 4:
                comp, n1, n2, _ = tokens
                for n in (n1, n2):
                    c = get_coord(n)
                    if c: all_coords.add(c)
                if comp.startswith('V'):
                    c1 = get_coord(n1)
                    if c1: voltage_coords.add(c1)
    return voltage_coords, all_coords

def get_voltage_distance_map(spice_path):
    vsrc, allc = extract_voltage_coords(spice_path)
    if not allc:
        return np.zeros((1,1))
    max_x = max(x for x,y in allc) + 1
    max_y = max(y for x,y in allc) + 1
    yy, xx = np.meshgrid(np.arange(max_y), np.arange(max_x), indexing='ij')
    eff = np.zeros((max_y, max_x), dtype=float)
    for vx, vy in vsrc:
        dist = np.sqrt((xx-vx)**2 + (yy-vy)**2)
        inv = np.where(dist>0, 1/dist, 0)
        eff += inv
    mask = eff != 0
    eff[mask] = 1/eff[mask]
    return eff
