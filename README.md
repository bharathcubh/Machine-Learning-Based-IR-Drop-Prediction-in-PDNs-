# IR-Drop Prediction Pipeline

This repository converts SPICE netlists into feature maps, trains a CNN to predict IR-drop heatmaps, and provides tools for inference and visualization.

---

## Python Modules (imported by scripts, not run standalone)

- **`current_map_module.py`**  
  Parses `.sp` netlist and returns a 2D NumPy array of net currents per µm² via `get_current_map(spice_path)`.

- **`pdn_map_module.py`**  
  Parses `.sp` netlist and returns a raw 2D grid of PDN resistor-endpoint counts via `get_pdn_density_map(spice_path)`.

- **`voltage_distance_map_module.py`**  
  Parses `.sp` netlist and returns a 2D effective-distance map to all voltage sources via `get_voltage_distance_map(spice_path)`.

- **`ir_map_module.py`**  
  Parses `.sp` + `.voltage` files and returns a raw 2D IR-drop grid via `get_ir_drop_map(spice_path, voltage_path)`.

---

## Scripts

### 1. `structure_and_zip.py`  
Organizes raw `.sp` and `.voltage` files into per-testcase folders and creates a ZIP.

```bash
python structure_and_zip.py   --input-dir  raw_files/   --output-dir structured/   --zip-file   final_dataset.zip
```

### 2. `dataset_generation.py`  
(Unzip + CSV) Extracts a structured ZIP, discovers every testcase, and generates four CSVs each.

```bash
python dataset_generation.py   --zip-file final_dataset.zip   --output   training_csvs/
```

### 3. `data_generation.py`  
Generate four CSV maps from a single SPICE netlist + voltage file.

```bash
python data_generation.py   -spice_netlist path/to/testcase.sp   -voltage_file  path/to/testcase.voltage   -output        training_csvs/
```

Outputs in `training_csvs/`:
- `current_map_<base>.csv`
- `pdn_density_map_<base>.csv`
- `voltage_source_map_<base>.csv`
- `ir_drop_map_<base>.csv`

### 4. `generate_image_main.py`  
Plots and saves PNG heatmaps of current, PDN density, voltage-distance, and IR-drop. Applies X↔Y swap + Y-flip only for IR-drop display.

```bash
python generate_image_main.py path/to/testcase1.sp path/to/testcase1.voltage path/to/output_images/

```

Produces:
- `current_map_<label>.png`
- `pdn_density_map_<label>.png`
- `voltage_distance_map_<label>.png`
- `ir_drop_map_<label>.png`

### 5. `training.py`  
Train a 3-channel → 1-channel CNN on the CSV dataset.

```bash
python training.py   -input  training_csvs/   -output best_ir_model.pth
```

### 6. `inference.py`  
Predict IR-drop on a new SPICE + voltage case using a saved model. Saves CSV, metrics, and oriented PNGs.

```bash
python inference.py   -spice_file   path/to/testcase.sp   -voltage_file path/to/testcase.voltage   -ml_model     best_ir_model.pth   -output       ir_pred_testcase.csv   -prefix       results/testcase
```

Produces:
- `ir_pred_testcase.csv`
- `results/testcase_predicted.png`
- `results/testcase_groundtruth.png`

---

## Notes

- Modules in `*_map_module.py` are **imported** by the scripts; do **not** run them directly.  
- All scripts create missing output directories automatically.  
- The pipeline supports both single-case and batch (ZIP-based) workflows.  
- Ensure all dependencies from `requirements.txt` are installed before running.
