import os
import numpy
from astropy.io import fits
import util_efs


def readspec(dir):
    if isinstance(dir, str):
        fnall = list(util_efs.locate('*_stisnic_*.fits', root=dir))
        fnall += list(util_efs.locate('*_stis_*.fits', root=dir))
        fnall += list(util_efs.locate('*_nic_*.fits', root=dir))
    else:
        fnall = dir
    spec = []
    for fn in fnall:
        # hdr = fits.getheader(fn)
        # this works for _model_ spectra, but I don't think that's what
        # we want.
        # for the moment, make: teff, logg, logz, ebv, wavelength, flux
        # if '/' in hdr['teffgrav']:
        #     parsplit = hdr['teffgrav'].split('/')
        #     par = [float(parsplit[0].split('K')[0]),
        #            float(parsplit[1]), 0., 0.]
        # else:
        #     parsplit = hdr['teffgrav'].split(',')
        #     par = [float(parsplit[0].split('K')[0]),
        #            float(parsplit[1]), float(parsplit[2]), float(parsplit[3])]
        tspec = fits.getdata(fn, 1)
        spec.append([os.path.basename(fn)] +
                    # par +
                    [tspec['wavelength'].copy(), tspec['flux'].copy()])
    return spec


def intspec(spec, filt):
    import filters
    nspec = len(spec)
    nfilt = len(filt)
    out = numpy.zeros((nspec, nfilt), dtype='f8')
    for i in xrange(nspec):
        for j in xrange(nfilt):
            wf, sf = filt[j]
            scopy = spec[i][2].copy()
            scopy[0] = 0
            scopy[-1] = 0
            minwave = numpy.min(wf[sf > 0])
            maxwave = numpy.max(wf[sf > 0])
            # make sure spectrum covers filter
            if (numpy.any((scopy == 0) & (spec[i][1] >= minwave) &
                          (spec[i][1] <= maxwave)) |
                (spec[i][1][0] > minwave) | (spec[i][1][-1] < maxwave)):
                out[i, j] = numpy.nan
            else:
                out[i, j] = filters.intfilter(spec[i][1],
                                              scopy, wf, sf)
    return out
