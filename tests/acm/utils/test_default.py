"""
Small tests on the default values of acm.utils.defaults.

We mainly test that those values match the expected types.
"""
from acm.utils.default import cosmo_list, is_nersc

def test_cosmo_list():
    assert isinstance(cosmo_list, list)
    assert all(type(l) is int for l in cosmo_list)
    
def test_is_nersc():
    assert type(is_nersc) is bool