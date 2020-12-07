from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
print('Starting convertfluxtoadu.py')
image_file = get_pkg_data_filename('output_file.fits')
hdul = fits.open(image_file)
hdul.info()
data = hdul['PRIMARY'].data
print(data[5,6]) #testing
fluxconv = 0.1088
exptime = 1161.6
print(data)
print(fluxconv)
print(exptime)
print('Converting now')
data = data * exptime / fluxconv
hdul['PRIMARY'].data = data
#hdul.close()
print(data)
hdul.writeto('newimage.fits')
#hdul.close()
print('Ending converfluxtoadu.py')

