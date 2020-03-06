import numpy
import re
import util_efs
import pdb
from astropy.io import fits

def getpars(dir=None, space_for_fluxes=False):
    """Return parameters of spectra in MARCS grid.

    Args:
        dir: Directory where the spectra reside, default to COELHO_DIR
        space_for_fluxes (bool): include empty space for SEDs in output array

    Returns:
        table of synthetic spectra parameters
    """
    if dir is None:
        dir = os.getenv('COELHO_DIR', None)
    if dir is None:
        raise ValueError('Need to set COELHO_DIR')

    filelist = list(util_efs.locate('*.fits', root=dir))
    if len(filelist) == 0:
        raise ValueError('No files found; bad MARCS_DIR?')
    # eg
    # p3500_g+4.0_m0.0_t02_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx.gz
    # 1 geo 2 temp 3 grav 4 mass 5 microturbulence
    # 6 Z 7 alpha 8 C 9 N 10 O 11 r 12 s
    fs = '([+-]?[0-9]*\.?[0-9]+)'  # this matches any number
    fs2 = '([pm][0-9]*\.?[0-9]+)'  # this matches any number
    prog = re.compile('.*/t'+fs+'_g'+fs+'_'+fs2+fs2+'_sed.fits')
    res = [prog.match(f) for f in filelist]
    filelist = [f for f, m in zip(filelist, res) if m is not None]
    res = [m.groups() for m in res if m is not None]
    param1dict = {'teff': 0., 'logz': 0., 'logg': 0., 'alpha': 0.,
                  'fname': ' '*200}
    if space_for_fluxes:
        param1dict['flux'] = numpy.zeros(2250, dtype='f4')-1
    param1 = util_efs.dict_to_struct(param1dict)
    param = numpy.zeros(len(res), param1.dtype)
    param['teff'] = [r[0] for r in res]
    param['logg'] = [r[1] for r in res]
    param['logz'] = [r[2][1:] for r in res]
    param['alpha'] = [r[3][1:] for r in res]
    param['logz'] *= [1-2*(r[2][0] == 'm') for r in res]
    param['alpha'] *= [1-2*(r[3][0] == 'm') for r in res]
    param['logz'] /= 10
    param['alpha'] /= 10
    param['fname'] = filelist
    return param


def getspec(fname):
    ss = fits.getdata(fname)
    crval1 = 3.1139433523068
    cdelt1 = 0.00083862011902764
    ww = 10.**(crval1 + numpy.arange(2250)*cdelt1)
    return ww, ss


def intgrid(grid, filt):
    import filters
    nspec = len(grid)
    nfilt = len(filt)
    out = numpy.zeros((nspec, nfilt), dtype='f8')
    wave = getspec(grid['fname'][0])[0]
    specregions = []
    from copy import deepcopy
    filt = deepcopy(filt)
    import marcs
    filt = marcs.minimize_filters(filt)
    for i in xrange(nfilt):
        wf, sf = filt[i]
        pix = numpy.arange(len(wave), dtype='i4')
        minwind = numpy.clip(numpy.min(pix[wave >= wf[0]])-1, 0, len(wave)-1)
        maxwind = numpy.clip(numpy.max(pix[wave <= wf[-1]])+1, 0, len(wave)-1)
        specregions.append([minwind, maxwind])
        # clip down the filters and spectra to just the relevant bits
    for i in xrange(nspec):
        for j in xrange(nfilt):
            wf, sf = filt[j]
            minwind, maxwind = specregions[j]
            out[i, j] = filters.intfilter(wave[minwind:maxwind],
                                          grid['flux'][i, minwind:maxwind],
                                          wf, sf)
    return out
