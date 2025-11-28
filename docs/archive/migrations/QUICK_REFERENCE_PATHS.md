# Quick Reference: File Path Structure
**Status**: ✅ Ready to Execute

---

## Input Files (Already in Place ✅)

```
data/inputs/
├── fault_merged_data.xlsx      ✅ 10.4 MB - Ready
└── health_merged_data.xlsx     ✅ 197 KB - Ready
```

Both input files are present and ready for pipeline execution.

---

## Pipeline Execution

```bash
cd /home/user/PoF2
python run_pipeline.py
```

---

## Generated Files (Will be Created)

### Step 2: Data Transformation
```
data/outputs/equipment_level_data.csv
```

### Step 3: Feature Engineering
```
data/outputs/features_engineered.csv
```

### Step 4: Feature Selection
```
data/outputs/features_reduced.csv
```

### Optional Step 2a: Health Equipment Loader
```
data/outputs/health_equipment_prepared.csv
```

---

## File Path Configuration

All paths defined in: **`config.py`**

**Never** use hardcoded paths in scripts - always import from config:

```python
from config import (
    INPUT_FILE,
    HEALTHY_EQUIPMENT_FILE,
    EQUIPMENT_LEVEL_FILE,
    FEATURES_ENGINEERED_FILE,
    FEATURES_REDUCED_FILE,
    DATA_OUTPUTS_DIR
)
```

---

## Diagram: Data Flow

```
data/inputs/
│
├─ fault_merged_data.xlsx
│  │
│  └─> Step 00: Input Analysis
│  │
│  └─> Step 01: Data Profiling
│  │
│  └─> Step 02: Data Transformation
│       └─> data/outputs/equipment_level_data.csv
│
├─ health_merged_data.xlsx
│  │
│  └─> Step 02a: Health Equipment Loader (Optional)
│       └─> data/outputs/health_equipment_prepared.csv
│
data/outputs/
│
├─ equipment_level_data.csv
│  │
│  └─> Step 03: Feature Engineering
│       └─> data/outputs/features_engineered.csv
│
├─ features_engineered.csv
│  │
│  └─> Step 04: Feature Selection
│       └─> data/outputs/features_reduced.csv
│
└─ features_reduced.csv
   │
   └─> Step 06+: Model Training
        ├─> models/
        ├─> outputs/
        └─> predictions/
```

---

## Key Points

1. **Input files are in `data/inputs/`** - Don't move them
2. **Intermediate files go to `data/outputs/`** - Automatically generated
3. **Final outputs go to `outputs/`, `models/`, etc.** - Created by pipeline
4. **All paths managed in `config.py`** - Single source of truth
5. **No hardcoded paths in scripts** - All use config imports

---

## Ready to Execute ✅

```
data/inputs/fault_merged_data.xlsx     ✅ Ready
data/inputs/health_merged_data.xlsx    ✅ Ready
data/outputs/                          ✅ Empty & Ready
config.py                              ✅ Updated
All scripts                            ✅ Updated
```

**Status: READY FOR PIPELINE EXECUTION**

Run: `python run_pipeline.py`
