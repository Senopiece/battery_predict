# BatteryLife Raw Dataset

## Acknowledgements

Data sourced from the **BatteryLife** dataset:

> Ruifeng Tan et al., *BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction*  
> GitHub: https://github.com/Ruifeng-Tan/BatteryLife

This project uses **only the NA-ion (sodium-ion) subset** of BatteryLife.

## Download Instructions

1. Go to https://zenodo.org/records/18646655
2. Download **NA-ion.zip**
3. Extract the archive
4. Place the extracted contents (the individual `.pkl` cell files) into `data/raw/batterylife/set/naion/`

After placing the files, run the converter from the repository root:

```bash
python data/raw/batterylife/convert.py
```

Converted tensors will be written to `data/set/` in the project tensor format.
