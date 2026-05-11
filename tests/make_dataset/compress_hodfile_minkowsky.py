import numpy as np
from astropy.io import fits
import sys


print("Reading HOD file:", sys.argv[1])

hdul = fits.open(sys.argv[1])

data = hdul[1].data

cxp =  fits.Column(name='X_PERP',format='E', array=data["X_PERP"].astype(np.float32))
cyp =  fits.Column(name='Y_PERP',format='E', array=data["Y_PERP"].astype(np.float32))
czr =  fits.Column(name='Z_RSD',format='E', array=data["Z_RSD"].astype(np.float32))

new_cols = fits.ColDefs([cxp,cyp,czr])
new_hdu = fits.BinTableHDU.from_columns(new_cols)

new_hdu.header['Q_PAR'] = 1.0
new_hdu.header['Q_PERP'] = 1.0

new_hdu.writeto("hod_f32.fits.gz", overwrite=True)
hdul.close()