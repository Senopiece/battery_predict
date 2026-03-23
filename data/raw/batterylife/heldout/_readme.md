# BatteryLife Heldout Raw Inputs

Place heldout NA-ion battery files here under `naion/`.

Expected heldout source format:
- `.pkl` files

Converter:

```bash
python data/raw/batterylife/convert.py
```

Output location:
- `data/set/heldout/*.jsonl`

The converter applies the same preprocessing and deterministic naming as the main
training/validation conversion path.
