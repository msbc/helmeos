import numpy as np
import sys

default_names = ['p/e(e/rho,rho)', 'e/p(p/rho,rho)', 'asq*rho/p(p/rho,rho)',
                 'T*rho/p(p/rho,rho)']


def write_athena_tables(dlim, elim, varlist, fn=None, out_type=None, eOp=1.0,
                        ratios=None, ftype=None, sdim=0, opt=None, var_names=None):
    """
    Writes an EOS table for Athena++

    :param dlim: (float, float) log_10 density limits
    :param elim: (float, float) log_10 specific internal energy limits
    :param varlist: (n_var x array(n_eng, n_dim)) data arrays to be written
    :param fn: (str) filename for output table; defaults to eos_table.<extension>
    :param out_type: (str) choose from ['ascii', 'hdf5', (raw) 'binary']
    :param eOp: (float) sets ratio for energy limits / pressure limits
    :param ratios: (float x n_var) defaults to [1, eOp, eOp], ratios to multiply
                   independent variables by
    :param ftype: float type, defaults to 'float64'
    :param sdim: stacking dimension, defaults to 0; should not change
    :param opt: (dict) options to pass to np.ndarray.tofile
    :param var_names: (nvar * str) list of names in varlist, defaults to default_names
    :return: output filename
    """
    nvar = len(varlist)
    if var_names is None:
        var_names = default_names[:nvar]
    if fn is None:
        fn = 'eos_tables.data'
    if out_type is None:
        ext = fn.split('.')[-1]
        if ext in ['tab', 'txt', 'ascii']:
            out_type = 'ascii'
        elif ext in ['hdf5']:
            out_type = 'hdf5'
    if ftype is None:
        ftype = 'float64'
        if out_type == 'hdf5':
            ftype = 'float32'
    if out_type == 'ascii':
        if opt is None:
            opt = {'sep': ' ', 'format': '%.4e'}
    if opt is None:
        opt = {'sep': ''}
    iopt = {'sep': opt.get('sep', '')}
    dlim = np.atleast_1d(dlim).astype(ftype)
    elim = np.atleast_1d(elim).astype(ftype)
    nd = np.array(varlist[0].shape[1], 'int32')
    ne = np.array(varlist[0].shape[0], 'int32')
    eOp = np.array(eOp)  # , ftype)
    if ratios is None:
        ratios = [1., eOp, eOp]
        if nvar > 3:
            ratios += [1] * (nvar - 3)
        ratios = np.array(ratios, dtype=ftype)
    if out_type == 'hdf5':
        import h5py
        with h5py.File(fn, 'w') as f:
            f.create_dataset('LogDensLim', data=dlim.astype(ftype))
            f.create_dataset('LogEspecLim', data=elim.astype(ftype))
            f.create_dataset('ratios', data=ratios.astype(ftype))
            for i, d in enumerate(varlist):
                f.create_dataset(var_names[i], data=np.log10(d).astype(ftype))
    elif out_type == 'ascii':
        with open(fn, 'w') as f:
            f.write('# Entries must be space separated.\n')
            f.write('# n_var, n_espec, n_rho\n')
            np.array([nvar, ne, nd]).tofile(f, **iopt)
            f.write('\n# Log espec lim\n')
            elim.tofile(f, **opt)
            f.write('\n# Log rho lim\n')
            dlim.tofile(f, **opt)
            f.write('\n# 1, eint/pres, eint/pres, eint/h\n')
            ratios.tofile(f, **opt)
            f.write('\n')
            for i, d in enumerate(varlist):
                f.write('# ' + var_names[i] + '\n')
                np.savetxt(f, np.log10(d), opt['format'], delimiter=opt['sep'])
    else:  # if binary:
        with open(fn, 'wb') as f:
            np.array([nvar, ne, nd], dtype='int32').tofile(f, **iopt)
            elim.tofile(f, **opt)
            dlim.tofile(f, **opt)
            ratios.tofile(f, **opt)
            out = np.stack(varlist, axis=sdim).astype(ftype)
            np.log10(out).tofile(f, **opt)
    return fn


def update_progress(progress):
    """Usage : update_progress(progress)
    An ASCII progression bar. Values of input should range from 0 to 1."""
    bar_length = 50  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length * progress))
    text = "\rPercent: [{0}] {1}% {2}".format("#" * block + "-" * (bar_length - block),
                                              "%.3f" % (progress * 100), status)
    sys.stdout.write(text)
    sys.stdout.flush()
