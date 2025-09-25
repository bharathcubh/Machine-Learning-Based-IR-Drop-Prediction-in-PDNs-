#!/usr/bin/env python3
"""
process_and_generate.py

Unzips a structured dataset ZIP (with folders of .sp/.voltage pairs),
generates current, PDN density, voltage-distance, and IR-drop maps
for each testcase, and writes them as CSVs into the specified output directory.

Usage:
  python process_and_generate.py \
    --zip-file final_dataset.zip \
    --output   training_csvs
"""

import os
import zipfile
import tempfile
import shutil
import argparse
import glob
import pandas as pd

from current_map_module import get_current_map
from pdn_map_module import get_pdn_density_map
from voltage_distance_map_module import get_voltage_distance_map
from ir_map_module import get_ir_drop_map

def process_zip_and_generate(zip_path, output_dir):
    # 1) Create temporary directory and unzip
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        os.makedirs(output_dir, exist_ok=True)

        # 2) Find all .sp files recursively
        for sp_path in glob.glob(os.path.join(temp_dir, '**', '*.sp'), recursive=True):
            base = os.path.splitext(os.path.basename(sp_path))[0]
            dirpath = os.path.dirname(sp_path)

            # 3) Locate matching .voltage file
            vf_path = os.path.join(dirpath, base + '.voltage')
            if not os.path.isfile(vf_path):
                volts = glob.glob(os.path.join(dirpath, '*.voltage'))
                if len(volts) == 1:
                    vf_path = volts[0]
                else:
                    print(f"⚠️  Skipping {base}: cannot find matching .voltage")
                    continue

            # 4) Generate the four maps
            cm  = get_current_map(sp_path)
            pdn = get_pdn_density_map(sp_path)
            vdd = get_voltage_distance_map(sp_path)
            ir  = get_ir_drop_map(sp_path, vf_path)

            # 5) Save each map as CSV
            pd.DataFrame(cm).to_csv(
                os.path.join(output_dir, f"current_map_{base}.csv"),
                header=False, index=False
            )
            pd.DataFrame(pdn).to_csv(
                os.path.join(output_dir, f"pdn_density_map_{base}.csv"),
                header=False, index=False
            )
            pd.DataFrame(vdd).to_csv(
                os.path.join(output_dir, f"voltage_source_map_{base}.csv"),
                header=False, index=False
            )
            pd.DataFrame(ir).to_csv(
                os.path.join(output_dir, f"ir_drop_map_{base}.csv"),
                header=False, index=False
            )

            print(f"✅ Generated CSVs for {base}")

    finally:
        # 6) Clean up temporary directory
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unzip structured ZIP and generate CSV feature/label maps."
    )
    parser.add_argument(
        '--zip-file', '-z',
        required=True,
        help="Path to the structured dataset ZIP file"
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help="Directory where generated CSVs will be saved"
    )
    args = parser.parse_args()

    process_zip_and_generate(args.zip_file, args.output)
