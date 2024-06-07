from astropy.io import fits
from astropy.wcs import WCS

import jax; jax.config.update('jax_enable_x64',True)
import jax.numpy as jp
import jax_finufft

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore',category=AstropyWarning)

import numpy as np

# Load npz visibility data
# ----------------------------------------------------------

def loadvis(visname,visindx=np.arange(5)):
    visload = np.load('{0}.data.npz'.format(visname),fix_imports=True,encoding='bytes')
    visdata = np.array([np.copy(visload[visload.files[0]][idx].flatten()) for idx in visindx])
    visload.close(); del visload, visindx
    return vis(visdata)

# Basic visibility class
# ----------------------------------------------------------
class vis:
    def __init__(self,data):
        self.u  = data[0].copy()
        self.v  = data[1].copy()
        self.re = data[2].copy()
        self.im = data[3].copy()
        self.wt = data[4].copy()
        self.hermitian = False

    def getxy(self,cdelt):
        rdelt = np.deg2rad(cdelt.to('deg').value)
        x = -2.00*np.pi*self.v*np.abs(rdelt)
        y =  2.00*np.pi*self.u*np.abs(rdelt)
        return x, y
    
    def doherm(self):
        if self.hermitian == False:
            self.u  = jp.append(self.u, -self.u)
            self.v  = jp.append(self.v, -self.v)
            self.re = jp.append(self.re, self.re)
            self.im = jp.append(self.im,-self.im)
            self.wt = jp.append(self.wt, self.wt)
            self.hermitian = True
    
# Load primary beam fits and remove frequency+stokes axes
# ----------------------------------------------------------

def loadpb(visname):
    inphdu = fits.open('{0}.pbeam.fits'.format(visname))
    inpwcs = WCS(inphdu[0].header)
    
    outhdu = fits.PrimaryHDU(data=inphdu[0].data[0,0],header=inpwcs.celestial.to_header())
    outhdu.data[np.isnan(outhdu.data)] = 0.00
    outhdu.header['FREQ'] = inphdu[0].header['CRVAL3']
    outhdu.header['BAND'] = inphdu[0].header['CDELT3']

    inphdu.close()
    return outhdu

# ----------------------------------------------------------

def backward(caxis,x,y,re,im,wt):
    return jax_finufft.nufft1((caxis,caxis),wt*(re+1j*im)/jp.sum(wt),x,y).real
