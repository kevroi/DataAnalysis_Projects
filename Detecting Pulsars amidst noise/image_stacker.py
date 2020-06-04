import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def mean_fits(files):
    n = len(files)
    result = np.zeros(())
    for i in files:
        hdulist = fits.open(i)
        imgdata = hdulist[0].data
        result = result + imgdata
    return result / n

meanstack = mean_fits(['image0.fits', 'image1.fits', 'image2.fits', 'image3.fits', 'image4.fits'])
plt.imshow(meanstack.T, cmap=plt.cm.viridis)
plt.colorbar().set_label('Spectral Flux Density', rotation=270)
plt.show()