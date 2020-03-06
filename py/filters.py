import os
import numpy
import scipy.integrate


def iter_loadtxt(filename, delimiter=None, skiprows=0, dtype=float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                if line[0] == '#':
                    continue
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield float(item)
        iter_loadtxt.rowlength = len(line)

    data = numpy.fromiter(iter_func(), dtype='f4')
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data


def read_filter(fname, multiply_wave=1):
    """Read a filter.

    Args:
        fname: the filter file name to read in

    Returns:
        wave, throughput
        wave: wavelengths at which the throughput is evaluated (A)
        throughput: the filter throughput (0-1)
    """

    # res = numpy.loadtxt(fname, dtype=[('wavelength', 'f4'),
    #                                   ('throughput', 'f4')])
    res = iter_loadtxt(fname).astype('f4')
    return res[:, 0]*multiply_wave, res[:, 1]


def intfilter(ws, ss, wf, sf, wa=None, sa=None, atm=1., plot=False):
    """Return AB magnitude corresponding to filter and atmosphere (photon
    counting).

    Args:
        ws, ss: wavelength (A) and spectrum (erg/cm^2/s/A) of the source
        wf, sf: wavelength (A) and QE (0-1) of the filter
        wa, sa: wavelength (A) and transparency (0-1) of the atmosphere
                (optional)
        atm: airmass to use
        ws, wf, wa must be sorted by wavelengeth

    Returns:
        the AB magnitude of the source in the filter
    """

    if wa is None:
        wa = ws
        sa = numpy.ones(len(ws))

    if atm != 1:
        sa = sa ** atm

#     demand presorting
#     s = numpy.argsort(ws)
#     ws = ws[s]
#     ss = ss[s]
#     s = numpy.argsort(wf)
#     wf = wf[s]
#     sf = sf[s]
#     s = numpy.argsort(wa)
#     wa = wa[s]
#     sa = sa[s]

    wall = numpy.unique(numpy.concatenate([ws, wf, wa]))
    ssall = numpy.interp(wall, ws, ss)
    sfall = numpy.interp(wall, wf, sf, left=0., right=0.)
    saall = numpy.interp(wall, ws, sa)
    sfall *= saall

    h = 6.62606957e-27  # erg * s
    c = 2.99792458e10  # cm/s
    flux_to_number = h*c/(wall*1e-8)  # 1e8 Angstrom / cm

    def integrator(x, y):
        return scipy.integrate.trapz(x, y)
    # integrator = lambda x, y: scipy.integrate.trapz(x, y)
    # investigate why Simpson vs. trapz can make small differences here?

    flux = integrator(sfall*ssall/flux_to_number, wall)

    szall = 3631e-23*c/wall/(wall*1e-8)
    zflux = integrator(sfall*szall/flux_to_number, wall)

    if (flux <= 0) or (zflux <= 0):
        raise ValueError('flux and zero flux must be > 0')

    if plot:
        import matplotlib.pyplot as p
        p.figure()
        p.plot(wall, ssall)
        p.plot(wall, ssall*sfall)
        p.twinx()
        p.plot(wf, sf)
        p.plot(wa, sa)
        p.plot(wall, sfall)
        effwavelength = integrator(sfall*ssall*wall/flux_to_number, wall)/flux
        p.axvline(effwavelength)

    return -2.5*numpy.log10(flux/zflux)


def ls_filters(fdir=None, oldbass=True):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    if oldbass:
        filters = (['decam.%s.am1p4.dat' % s for s in 'ugrizY'] +
                   ['bass-g.am1p1.conv.dat', 'bass-r.am1p1.conv.dat',
                    'kpzdccdcorr3.am1p1.conv.dat'])
    else:
        filters = (['decam.%s.am1p4.dat' % s for s in 'ugrizY'] +
                   ['BASS_g_corr.bp', 'BASS_r_corr.bp',
                    'kpzdccdcorr3.am1p1.conv.dat'])
    return [read_filter(os.path.join(fdir, f)) for f in filters]


def ps_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    filters = ['ps1.g.dat', 'ps1.r.dat', 'ps1.i.dat', 'ps1.z.dat', 'ps1.y.dat']
    return [read_filter(os.path.join(fdir, f)) for f in filters]


def sdss_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    filters = ['sdss_jun2001_'+f+'_atm.dat' for f in 'ugriz']
    return [read_filter(os.path.join(fdir, f)) for f in filters]


def tmass_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    filters = ['jrsr.tbl.ang', 'hrsr.tbl.ang', 'krsr.tbl.ang']
    return [read_filter(os.path.join(fdir, f)) for f in filters]


def wise_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    filters = ['RSR-'+s+'.txt.conv.dat' for s in ['W1', 'W2', 'W3', 'W4']]
    return [read_filter(os.path.join(fdir, f)) for f in filters]


def landolt_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    filters = ['landolt_'+s+'.dat' for s in ['U', 'B', 'V', 'R', 'I']]
    print 'this is not the whole filter curve!'
    return [read_filter(os.path.join(fdir, f)) for f in filters]

def decam_filters(fdir=None, am=1.3):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    from astropy.io import ascii
    dat = ascii.read(os.path.join(fdir, 'DECam_filters.csv'))
    return [(dat['wavelength']*10., dat[f]*dat['atm1p3']**(am/1.3-1))
            for f in 'ugrizY']

def tonry2012tofilterdat(tonryfilename):
    dat = numpy.genfromtxt(tonryfilename, skip_header=26,
                           dtype=[('wave', 'f4'), ('o', 'f4'), ('g', 'f4'),
                                  ('r', 'f4'), ('i', 'f4'), ('z', 'f4'),
                                  ('y', 'f4'), ('w', 'f4'), ('aoeu', 'f4'),
                                  ('ray', 'f4'), ('mol', 'f4')])
    dat['wave'] *= 10  # nm -> angstroms
    import matplotlib
    for f in 'grizy':
        fname = 'ps1.%s.dat' % f
        dat0 = matplotlib.mlab.rec_keep_fields(dat, ['wave', f])
        numpy.savetxt(fname, dat0, fmt='%10.2f   %15.9f')
        str = file(fname).read()
        header = ('# PS1 %s total system throughput from Tonry+2012 ApJ '
                  'published version\n' % f)
        open(fname, 'w').write(header+str)


def tmass2filterdat(dir):
    import util_efs
    filenames = util_efs.locate('*.tbl', root=dir)
    for f in filenames:
        dat = numpy.genfromtxt(f, comments='#',
                               dtype=[('wave', 'f4'), ('throughput', 'f4')])
        dat['wave'] *= 10*1000  # micron -> angstroms
        fnameout = f+'.dat'
        numpy.savetxt(fnameout, dat, fmt='%10.2f   %15.9f')
        str = file(fnameout).read()
        header = '# 2MASS system throughput from %s\n' % f
        open(fnameout, 'w').write(header+str)


def tmass_filters_cohen(fdir=None, dividebywavelength=False):
    # my reading of the paper is that these are energy,
    # not photon sensitivity, so these files are almost certainly
    # wrong.
    print 'probably wrong'
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    filters = [f+'rsrcohen.tbl.dat' for f in 'jhk']
    res = [list(read_filter(os.path.join(fdir, f))) for f in filters]
    if dividebywavelength:
        for i in xrange(len(res)):
            res[i][1] = res[i][1] / (res[i][0]+(res[i][0] == 0))
            res[i][1] /= numpy.max(res[i][1])
    return res


def tmass_filters_photon(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    filters = [f+'rsrphoton.tbl.dat' for f in 'jhk']
    return [read_filter(os.path.join(fdir, f)) for f in filters]


def central_wavelengths(filts):
    cw = []
    for ww, ff in filts:
        cw.append(numpy.trapz(ww*ff, x=ww)/numpy.trapz(ff, x=ww))
    return numpy.array(cw)


def landolt_full_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']
    wq, sq = read_filter(os.path.join(fdir, 'landolt_qe2.dat'))
    wa, sa = read_filter(os.path.join(fdir, 'ctio_atmos.dat'))

    filters = ['landolt_'+s+'.dat' for s in ['U', 'B', 'V', 'R', 'I']]
    out = []
    for filtfile in filters:
        wf, sf = read_filter(os.path.join(fdir, filtfile))
        wi = numpy.sort(numpy.unique(numpy.concatenate([wq, wa, wf])))
        sqi = numpy.interp(wi, wq, sq, left=0., right=0.)
        sai = numpy.interp(wi, wa, sa)
        sfi = numpy.interp(wi, wf, sf)
        out.append([wi, sfi*sqi*10.**(-sai/2.5)])
    return out


def landolt_brad_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']+'/landolt_brad'

    filters = ['Landolt'+s for s in ['U', 'B', 'V', 'R', 'I']]
    return [read_filter(os.path.join(fdir, f)) for f in filters]


def landolt_adps_filters(fdir=None, divide_by_wavelength=0):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']+'/landolt_other'

    filters = ['adps_landolt_1983_'+s+'.dat' for s in 'ubvri']
    out = []
    for f in filters:
        ww, ff = read_filter(os.path.join(fdir, f))
        out.append([ww, ff/((ww/numpy.median(ww))**divide_by_wavelength)])
    return out


def vista_filters_partial(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']+'/vista/Filters_QE_Atm_curves'

    filters = ['VISTA_Filters_at80K_forETC_'+s+'.dat'
               for s in ['J', 'H', 'Ks']]
    return [read_filter(os.path.join(fdir, f), multiply_wave=10)
            for f in filters]


def ukirt_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']+'/ukirt'
    filters = ['wfcam'+x+'.dat' for x in 'zyjhk']
    return [read_filter(os.path.join(fdir, f), multiply_wave=10000)
            for f in filters]


def gaia_dr2_revised_filters(fdir=None):
    if fdir is None:
        fdir = os.environ['FILTER_DIR']+'/gaia/'
    from astropy.io import ascii
    dat = ascii.read(os.path.join(fdir, 'GaiaDR2_RevisedPassbands.dat'),
                     names=['wavelength', 'g', 'dg', 'bp', 'dbp', 'rp', 'drp'])
    wave = dat['wavelength']*10
    return [[wave, dat['g']*(dat['dg'] != 99.99)],
            [wave, dat['bp']*(dat['dbp'] != 99.99)],
            [wave, dat['rp']*(dat['drp'] != 99.99)]]
    
