from .helm_table import HelmTable, _DelayedTable

# create the default helmholtz table
default = _DelayedTable()


def eos_call(den, temp, abar, zbar, outvar=None):
    return default.eos_call(den, temp, abar, zbar, outvar=outvar)
