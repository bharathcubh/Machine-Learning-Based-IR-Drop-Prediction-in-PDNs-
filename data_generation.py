#!/usr/bin/env python3
import os
import argparse
import pandas as pd

from current_map_module import get_current_map
from pdn_map_module import get_pdn_density_map
from voltage_distance_map_module import get_voltage_distance_map
from ir_map_module import get_ir_drop_map

def main():
    parser = argparse.ArgumentParser(
        description="Generate feature & label CSVs from a SPICE netlist + voltage file"
    )
    parser.add_argument(
        '-spice_netlist', required=True,
        help="Path to the SPICE .sp netlist"
    )
    parser.add_argument(
        '-voltage_file', required=True,
        help="Path to the .voltage file"
    )
    parser.add_argument(
        '-output', required=True,
        help="Directory where four CSVs will be saved"
    )
    args = parser.parse_args()

    spice_path   = args.spice_netlist
    voltage_path = args.voltage_file
    out_dir      = args.output
    os.makedirs(out_dir, exist_ok=True)

    # base name for CSV files, e.g. "testcase1" from "testcase1.sp"
    base = os.path.splitext(os.path.basename(spice_path))[0]

    # (a) current_map_<base>.csv
    cm = get_current_map(spice_path)
    pd.DataFrame(cm).to_csv(
        os.path.join(out_dir, f"current_map_{base}.csv"),
        header=False, index=False
    )

    # (b) pdn_density_map_<base>.csv
    pdn = get_pdn_density_map(spice_path)
    pd.DataFrame(pdn).to_csv(
        os.path.join(out_dir, f"pdn_density_map_{base}.csv"),
        header=False, index=False
    )

    # (c) voltage_source_map_<base>.csv
    vdd = get_voltage_distance_map(spice_path)
    pd.DataFrame(vdd).to_csv(
        os.path.join(out_dir, f"voltage_source_map_{base}.csv"),
        header=False, index=False
    )

    # (d) ir_drop_map_<base>.csv
    ir = get_ir_drop_map(spice_path, voltage_path)
    pd.DataFrame(ir).to_csv(
        os.path.join(out_dir, f"ir_drop_map_{base}.csv"),
        header=False, index=False
    )

    print(f"✅ Generated CSVs for {base} in {out_dir}")

if __name__ == "__main__":
    main()
