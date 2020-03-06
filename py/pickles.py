import os, pdb, re
import numpy
import util_efs

def read_file(file):
    rgx = re.compile('.*/(r|w)?(o|b|a|f|g|k|m)([0-9])(i|ii|iii|iv|v).dat')
    match = rgx.match(file)
    if match is None:
        print 'bad file name %s' % file
        return None
    fnparts = match.groups()
    dat = numpy.genfromtxt(file, dtype=[('lamaa', 'f4'), ('flux', 'f4')]) #readcol, files[i], aa, ff, /silent
    if fnparts[0] == 'w':
        logz = -0.5
    elif fnparts[0] == 'r':
        logz = 0.5
    else:
        logz = 0.
    lc0 = fnparts[3]
    if lc0 == 'i':
        lumclass = 1.
    elif lc0 == 'ii':
        lumclass = 2.
    elif lc0 == 'iii':
        lumclass = 3.
    elif lc0 == 'iv':
        lumclass = 4.
    elif lc0 == 'v':
        lumclass = 5.
    else:
        pdb.set_trace()
    specclass = fnparts[1]+fnparts[2]
    return dat['lamaa'], dat['flux'], logz, lumclass, specclass
    

def read_grid(dir=None, uvk=False):
    if dir is None:
        dir = os.path.join(os.environ['PICKLES_DIR'],
                           'dat'+('k' if uvk else ''))
    files = list(util_efs.locate('*.dat', root=dir))
    nfiles = len(files)
    npts = 1895 if not uvk else 4771
    wavea = numpy.zeros(npts, dtype='f4')
    flux = numpy.zeros((npts, nfiles), dtype='f4')
    logz = numpy.zeros(nfiles, dtype='f4')
    lumclass = numpy.zeros(nfiles, dtype='f4')
    specclass = numpy.zeros(nfiles, dtype='a8')
    j = 0
    for i, f in enumerate(files):
        res0 = read_file(f)
        if res0 is None:
            continue
        lam0, flux0, logz0, lc0, sc0 = res0
        wavea[:] = lam0
        flux[:,j] = flux0
        logz[j] = logz0
        lumclass[j] = lc0
        specclass[j] = sc0
        j += 1
    flux = flux[:,0:j]
    specclass = specclass[0:j]
    lumclass = lumclass[0:j]
    logz = logz[0:j]
    return util_efs.dict_to_struct(
        {'wave':wavea, 'flux':flux, 'specclass':specclass,
         'lumclass':lumclass, 'logz':logz })
