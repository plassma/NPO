# Preparing Datasets

`prepare_datasets.py` generates and serializes dataset instances for the supported problems (currently HCP).

## Create Datasets

e.g.
```setup
python prepare_datasets.py --datasets HCP_dummy --problems HCP --seed 123 --save True --modes test train val
```

All possible datasets and CO problems are listed within `prepare_datasets.py`.

### Parameter details

`--time_limits` is kept for compatibility and stored in the per-instance metadata.

## Dataset Path
All data will be saved in the folder
```
DatasetCreator/loadGraphDatasets/DatasetSolutions/
```

