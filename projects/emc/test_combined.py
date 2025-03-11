from acm.observables.emc import (GalaxyCorrelationFunctionMultipoles,
    GalaxyPowerSpectrumMultipoles, CombinedObservable)

select_filters = {'cosmo_idx': [0, 1], 'hod_idx': [30]}

# you can pass filters to each statistic separately
observable = CombinedObservable([
    GalaxyCorrelationFunctionMultipoles(select_filters=select_filters),
    GalaxyPowerSpectrumMultipoles(select_filters=select_filters),
])

# output features
lhc_y = observable.lhc_y

# covariance
cov = observable.covariance_matrix(increase_factor=8)

# theory model
model = observable.model