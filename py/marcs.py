import os
import re
import gzip
import pdb
import numpy
from astropy.io import fits
import util_efs


def getpars(dir=None, clip=False, space_for_fluxes=False):
    """Return parameters of spectra in MARCS grid.

    Args:
        dir: Directory where the spectra reside, default to MARCS_DIR
        clip: clip just to unique subset
              (plane parallel for log g >= 3, spherical for log g < 3)

    Returns:
        table of synthetic spectra parameters
    """
    if dir is None:
        dir = os.getenv('MARCS_DIR', None)
    if dir is None:
        raise ValueError('Need to set MARCS_DIR')

    filelist = list(util_efs.locate('*.flx.gz', root=dir))
    if len(filelist) == 0:
        raise ValueError('No files found; bad MARCS_DIR?')
    # eg
    # p3500_g+4.0_m0.0_t02_st_z+0.00_a+0.00_c+0.00_n+0.00_o+0.00_r+0.00_s+0.00.flx.gz
    # 1 geo 2 temp 3 grav 4 mass 5 microturbulence
    # 6 Z 7 alpha 8 C 9 N 10 O 11 r 12 s
    fs = '([+-]?[0-9]*\.?[0-9]+)'  # this matches any number
    prog = re.compile('.*/(.)'+fs+'_g'+fs+'_m'+fs+'_t'+fs+'_st_z'+fs+
                      '_a'+fs+'_c'+fs+'_n'+fs+'_o'+fs+'_r'+fs+'_s'+fs+
                      '.flx.gz')
    res = [prog.match(f) for f in filelist]
    filelist = [f for f, m in zip(filelist, res) if m is not None]
    res = [m.groups() for m in res if m is not None]
    param1dict = {'teff': 0., 'logz': 0., 'logg': 0., 'alpha': 0., 'mass': 0.,
                  'c': 0., 'n': 0., 'o': 0.,
                  'r': 0., 's': 0., 'geometry': ' ', 'fname': ' '*200,
                  'vturb': -1.}
    if space_for_fluxes:
        param1dict['flux'] = numpy.zeros(100724, dtype='f4')-1
    param1 = util_efs.dict_to_struct(param1dict)
    param = numpy.zeros(len(res), param1.dtype)
    param['geometry'] = [r[0] for r in res]
    param['teff'] = [r[1] for r in res]
    param['logg'] = [r[2] for r in res]
    param['mass'] = [r[3] for r in res]
    param['vturb'] = [r[4] for r in res]
    param['logz'] = [r[5] for r in res]
    param['alpha'] = [r[6] for r in res]
    param['c'] = [r[7] for r in res]
    param['n'] = [r[8] for r in res]
    param['o'] = [r[9] for r in res]
    param['r'] = [r[10] for r in res]
    param['s'] = [r[11] for r in res]
    param['fname'] = filelist
    if clip:
        m = (((param['logg'] >= 3) & (param['geometry'] == 'p')) |
             ((param['logg'] < 3) & (param['geometry'] == 's')))
        param = param[m]
    return param


def readspecfile(fname):
    if fname[-3:] == '.gz':
        openfn = gzip.open
    else:
        openfn = open
    dat = openfn(fname, 'r').readlines()
    return numpy.array([float(d.split()[0]) for d in dat], dtype='f4')


def asciitofits(griddir, filename):
    pars = getpars(griddir, space_for_fluxes=True)
    for i in xrange(len(pars)):
        pars['flux'][i, :] = readspecfile(pars['fname'][i])
    fits.writeto(os.path.join(griddir, filename), pars, clobber=True)
    return pars


def getspec(fname):
    """Return spectrum corresponding to fname.

    Args:
        fname: filename of MARCS spectrum to return

    Returns:
        wave, spectrum
        wave: wavelengths (A)
        spectrum: spectrum (erg/cm^2/s/A)
    """
    self = getspec
    wave = getattr(self, 'wave', None)
    if wave is None:
        # wave = self.wave = numpy.loadtxt(os.path.join(os.path.dirname(fname),
        #                                 'flx_wavelengths.vac.gz'))
        wave = self.wave = readspecfile(os.path.join(os.path.dirname(fname),
                                                     'flx_wavelengths.vac.gz'))
    return wave, readspecfile(fname)


def minimize_filters(filt):
    nfilt = len(filt)
    out = []
    for i in xrange(nfilt):
        wf, sf = filt[i]
        minwind = numpy.argmin(wf+(sf <= 0)*1e99)-5
        maxwind = numpy.argmax(wf-(sf <= 0)*1e99)+5
        minwind, maxwind = [numpy.clip(x, 0, len(wf))
                            for x in [minwind, maxwind]]
        wf = wf[minwind:maxwind]
        sf = sf[minwind:maxwind]
        out.append([wf, sf])
    return out


def intgrid(grid, filt):
    import filters
    nspec = len(grid)
    nfilt = len(filt)
    out = numpy.zeros((nspec, nfilt), dtype='f8')
    wave = readspecfile(os.path.join(os.environ['MARCS_DIR'],
                                     'flx_wavelengths.vac.gz'))
    specregions = []
    from copy import deepcopy
    filt = deepcopy(filt)
    filt = minimize_filters(filt)
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
