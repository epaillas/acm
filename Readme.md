### Alternative Clustering Methods -- a common analysis pipeline for DESI

To install in Perlmutter in developer mode, within the cosmodesi environment:

    source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
    python setup.py develop --user

Note: the `--user` flag is important -- it ensures that installation takes place in your own directories.

### Examples

Examples are provided under the `nb` folder.

### Credits

 - Mesh routines for particle assignment and FFT calculations are based on pyrecon (https://github.com/cosmodesi/pyrecon).

 - Correlation function measurements are based on pycorr (https://github.com/cosmodesi/pycorr/).

 - Power spectrum measurements are based on pypower (https://github.com/cosmodesi/pypower/).

 - The bispectrum estimator uses a wrapper around the PolyBin3D code developed by Oliver Philcox & Thomas Flöss (https://github.com/oliverphilcox/PolyBin3D).
 
 - The voxel voids estimator uses a wrapper around the Revolver code developed by Seshadri Nadathur (https://github.com/seshnadathur/Revolver).

 - Georgios Valogiannis for providing his wavelet scattering transform code, which is based around the Kymatio package (https://www.kymat.io/).

 - Carolina Cuesta Lazaro for inspiration from her ili-summarizer code (https://github.com/florpi/ili-summarizer).