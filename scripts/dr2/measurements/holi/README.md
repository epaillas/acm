# Holi clustering measurements

This folder contains `holi.py`, a driver to measure clustering statistics from DR2 Holi **altmtl** mocks (other versions will be incorporated soon).

The script reads catalogs from:

- `/global/cfs/cdirs/desicollab/mocks/cai/LSS/DA2/mocks/holi_v1/altmtl{PHASE}/loa-v1/mock{PHASE}/LSScats/`

with files:

- data: `{TRACER}_{REGION}_clustering.dat.h5`
- randoms: `{TRACER}_{REGION}_{I}_clustering.ran.h5` (typically `I=0..18`)

## Quick run

From the repository root (`acm/`):

```bash
python scripts/dr2/measurements/holi/holi.py \
  --statistics spectrum \
  --start_phase 201 --n_phase 1 \
  --tracer LRG --region NGC \
  --zrange 0.4 0.6
```

## Run both statistics

```bash
python scripts/dr2/measurements/holi/holi.py \
  --statistics spectrum density_split \
  --start_phase 201 --n_phase 5 \
  --tracer LRG --region NGC \
  --zrange 0.4 0.6 \
  --n_randoms 18
```

## Useful options

- `--base_dir`: input Holi root directory.
- `--save_dir`: output directory for measurements.
- `--statistics`: one or more of `spectrum`, `density_split`.
- `--start_phase`, `--n_phase`: realization range to process.
- `--n_randoms`: number of random files per phase.

## Notes

- Missing phases are skipped automatically if the data file does not exist.
- Outputs are written under:
  - `{save_dir}/spectrum/phXXX/`
  - `{save_dir}/density_split/phXXX/`
