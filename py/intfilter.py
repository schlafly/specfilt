import numpy
import scipy.integrate


def intfilter(ws, ss, wf, sf, wa=None, sa=None, atm=1., plot=False):
    """Return AB magnitude corresponding to filter and atmosphere (photon
    counting).

    Args:
        ws, ss: wavelength (A) and spectrum (erg/cm^2/s/A) of the source
        wf, sf: wavelength (A) and QE (0-1) of the filter
        wa, sa: wavelength (A) and transparency (0-1) of the atmosphere
                (optional)
        atm: airmass to use

    Returns:
        the AB magnitude of the source in the filter
    """

    if wa is None:
        wa = ws
        sa = numpy.ones(len(ws))

    s = numpy.argsort(ws)
    ws = ws[s]
    ss = ss[s]
    s = numpy.argsort(wf)
    wf = wf[s]
    sf = sf[s]
    s = numpy.argsort(wa)
    wa = wa[s]
    sa = sa[s]
    wall = numpy.unique(numpy.concatenate([ws, wf, wa]))
    ssall = numpy.interp(wall, ws, ss)
    sfall = numpy.interp(wall, wf, sf)
    saall = numpy.interp(wall, ws, sa)
    sfall *= saall

    h = 6.62606957e-27  # erg * s
    c = 2.99792458e10  # cm/s
    flux_to_number = h*c/(wall*1e-8)  # 1e8 Angstrom / cm
    flux = scipy.integrate.trapz(sfall*ssall*flux_to_number, wall)

    szall = 3631e-23*c/wall/(wall*1e-8)
    zflux = scipy.integrate.trapz(sfall*szall*flux_to_number, wall)

    if plot:
        import matplotlib.pyplot as p
        p.plot(wall, ssall)
        p.plot(wall, ssall*sfall)
        p.twinx()
        p.plot(wf, sf)
        p.plot(wa, sa)
        p.plot(wall, sfall)

    if (flux <= 0) or (zflux <= 0):
        raise ValueError('flux and zero flux must be > 0')

    return -2.5*numpy.log10(flux/zflux)
