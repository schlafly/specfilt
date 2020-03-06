import numpy
from astropy.io import ascii
import util_efs
from matplotlib.mlab import rec_append_fields

def read(system='WFC3', dir='/n/fink1/schlafly/mist/bcs'):
    files = list(util_efs.locate('*%s' % system, root=dir))
    res = []
    for file in files:
        lastline = ''
        fehline = -1
        if file[-3:] == 'iso':
            fehline = 4
        with open(file, 'r') as fp:
            for i, line in enumerate(fp):
                if line[0] != '#':
                    break
                if i == fehline:
                    isoline = line
                lastline = line
        if fehline > 0:
            feh = float(isoline.split()[3])
        names = lastline.split()[1:]
        grid = ascii.read(file, comment='#', names=names).as_array()
        if fehline > 0:
            grid = rec_append_fields(grid, '[Fe/H]', 
                                     feh*numpy.ones(len(grid), dtype='f4'))
        if 'Av' in grid.dtype.names:
            m = grid['Av'] == 0
            res.append(grid[m])
        else:
            res.append(grid)
    return numpy.concatenate(res)


# for eventual extinction curve sensitivity predictions, need to:
# * compute something simple color proxy for R(V) uncertainty, but
#   using the PHAT bands.  Something like E(F275W-F160W)/E(F275W-F336W)
