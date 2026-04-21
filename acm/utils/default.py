import os

# This file contains default values used in the ACM package

# List of cosmologies in AbacusSummit
cosmo_list = (list(range(5)) + list(range(13, 14)) + list(range(100, 127)) + list(range(130, 182)))

# Flag to indicate if running on NERSC
is_nersc = (os.environ.get("NERSC_HOST") == "perlmutter")
