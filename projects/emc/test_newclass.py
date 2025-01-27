from acm.observables.emc import GalaxyCorrelationFunctionMultipoles
import numpy as np


select_filters = {'cosmo_idx': [0]}
slice_filters = {'s': [0, 150]}

tpcf = GalaxyCorrelationFunctionMultipoles(
    select_filters=select_filters, slice_filters=slice_filters
)

# load Latin hypercubes of input and output features
lhc_x = tpcf.lhc_x
lhc_y = tpcf.lhc_y

# load features from small AbacusSummit box for covariance estimation
small_box_y = tpcf.small_box_y

# load model
model = tpcf.model

# Separation vector
sep = tpcf.separation

print(tpcf.coords_model)
print(tpcf.lhc_x)