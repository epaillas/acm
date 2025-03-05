# The BGS project

This project aims to measure the different Alternative Clustering Methods for the BGS sample of DESI.

## Folders

The AbacusSummit Dark Matter catalogs are precomputed in the `prepare_simulations` folder, to apply a halo filter adapted to the BGS sample.

> The LRG filter did not allow the population of galaxies to reach the expected BGS number density.

The statistics measurements scripts are in the `measurements` folder.

The models training scripts are in the `training` folder.

The inference scripts are in the `inference` folder.

The final plots are in the `plots` folder.

In each of these folders, you will find a `control_plots.ipynb` notebook to check the results of the scripts.


### Data

The data is stored in `/pscratch/sd/s/sbouchar/acm/bgs/` on NERSC. The folder follows the `acm` storage structure recommandations.

## TODO

- [x] Finish the readme
- [ ] Recompute statistics with new boxes 
- [ ] Add plots scripts and notebooks
- [ ] Write the new statistic computation when needed again (+ control notebooks and small boxes !)