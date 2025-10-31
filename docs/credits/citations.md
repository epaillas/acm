# Citations

If you use the `acm` package in your research, please cite the relevant papers:

## ACM Pipeline

```{note}
The ACM pipeline paper is in preparation. For now, please cite the individual statistics and methods you use.
```

## Statistics and Methods

If you use specific statistics from this package, please cite the relevant papers:

### Density-Split Statistics

If you use the density-split clustering statistics:

```bibtex
@article{paillas2023,
    title={Cosmological constraints from the density-split clustering of BOSS galaxies},
    author={Paillas, Enrique and others},
    journal={Monthly Notices of the Royal Astronomical Society},
    year={2023}
}
```

### Simulation-Based Inference

If you use the neural network emulators and inference methods:

```bibtex
@article{cuesta-lazaro2023,
    title={Simulation-based inference of cosmology and galaxy formation from DESI},
    author={Cuesta-Lazaro, Carolina and others},
    journal={arXiv preprint},
    year={2023}
}
```

### AbacusSummit Simulations

If you use galaxy catalogs generated with AbacusHOD on AbacusSummit:

```bibtex
@article{maksimova2021,
    title={AbacusHOD: a highly efficient method for mock galaxy generation},
    author={Maksimova, Natalie A and others},
    journal={Monthly Notices of the Royal Astronomical Society},
    volume={508},
    pages={4017--4037},
    year={2021}
}

@article{garrison2021,
    title={The Abacus Cosmos: A Suite of Cosmological N-body Simulations},
    author={Garrison, Lehman H and others},
    journal={The Astrophysical Journal Supplement Series},
    volume={258},
    pages={11},
    year={2021}
}
```

## Dependencies

Please also consider citing the key dependencies used by this package:

- **sunbird**: Neural network emulation and inference ([GitHub](https://github.com/florpi/sunbird))
- **pycorr**: Two-point correlation function estimation ([GitHub](https://github.com/cosmodesi/pycorr))
- **pypower**: Power spectrum estimation ([GitHub](https://github.com/cosmodesi/pypower))
- **PolyBin3D**: Bispectrum estimation ([GitHub](https://github.com/oliverphilcox/PolyBin3D))

See the [Dependencies](dependencies) page for a complete list of packages used by `acm`.

## How to Cite

We recommend citing this package as:

```
ACM Topical Group (2025). Alternative Clustering Methods (ACM) Pipeline. 
GitHub repository: https://github.com/epaillas/acm
```

```{seealso}
For information on contributors to this project, see the [Contributors](contributors) page.
```