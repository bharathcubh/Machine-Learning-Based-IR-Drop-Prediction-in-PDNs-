#!/usr/bin/env python3
"""
structure_and_zip.py

Given a directory containing files like
    t1.sp, t1.voltage, t2.sp, t2.voltage, ...
this script will:

  1. Create an output directory (e.g. `structured/`).
  2. For each base name (`t1`, `t2`, ...):
      - Make `structured/t1/`
      - Copy `t1.sp` and `t1.voltage` into it.
  3. Produce a ZIP archive of the entire `structured/` tree.

Usage:
  python structure_and_zip.py \
    --input-dir path/to/raw_files \
    --output-dir path/to/structured \
    --zip-file   final_dataset.zip
"""

import os
import shutil
import argparse
import zipfile

def structure_and_zip(input_dir: str, output_dir: str, zip_path: str):
    # 1) Gather all .sp and .voltage files
    files = os.listdir(input_dir)
    bases = set()
    for fn in files:
        name, ext = os.path.splitext(fn)
        if ext.lower() in ('.sp', '.voltage'):
            bases.add(name)

    # 2) Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 3) For each base, create subfolder + copy files
    for base in sorted(bases):
        subdir = os.path.join(output_dir, base)
        os.makedirs(subdir, exist_ok=True)

        sp_src  = os.path.join(input_dir, base + '.sp')
        v_src   = os.path.join(input_dir, base + '.voltage')

        # Only copy if the source file exists
        if os.path.isfile(sp_src):
            shutil.copy2(sp_src, os.path.join(subdir, base + '.sp'))
        else:
            print(f"⚠️  Warning: {sp_src} not found, skipping .sp")

        if os.path.isfile(v_src):
            shutil.copy2(v_src, os.path.join(subdir, base + '.voltage'))
        else:
            print(f"⚠️  Warning: {v_src} not found, skipping .voltage")

    # 4) Zip the entire output_dir tree
    zip_base, _ = os.path.splitext(zip_path)
    # shutil.make_archive will append .zip for us if you omit it
    archive_path = shutil.make_archive(zip_base, 'zip', root_dir=output_dir)
    print(f"\n✅ Structured files in '{output_dir}' and created archive '{archive_path}'")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Structure .sp/.voltage files into per-testcase folders and zip them."
    )
    p.add_argument(
        '--input-dir', '-i',
        required=True,
        help="Directory containing raw .sp and .voltage files"
    )
    p.add_argument(
        '--output-dir', '-o',
        required=True,
        help="Directory in which to create per-base subfolders"
    )
    p.add_argument(
        '--zip-file', '-z',
        required=True,
        help="Path (with or without .zip) for the output ZIP archive"
    )
    args = p.parse_args()

    structure_and_zip(args.input_dir, args.output_dir, args.zip_file)
