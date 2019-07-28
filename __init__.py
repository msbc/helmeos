from .helm_table import HelmTable, _DelayedTable

# create the default helmholtz table
default = _DelayedTable()


def eos_DT(den, temp, abar, zbar, outvar=None):
    return default.eos_DT(den, temp, abar, zbar, outvar=outvar)


def eos_invert(dens, abar, zbar, var, var_name, t0=None):
    return default.eos_invert(dens, abar, zbar, var, var_name, t0=t0)
