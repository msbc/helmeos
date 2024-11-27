#! /usr/bin/env python

import numpy as np
import os
try:
    import helmeos
except ImportError:
    try:
        from .. import helmeos
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        import helmeos

vars_to_print = ['etot', 'ptot', 'cs', 'sele']

# defining inputs
density = np.array([1e-7, 1e2])  # g/cc
temperature = np.array([1e4, 1e7])  # K
abar = 1.0
zbar = 1.0

# format string
fmt = ("{:s}: " + "{:g}, " * len(density))[:-1]

# easiest way to call EOS
print("~ Easy call ~")
out = helmeos.eos_DT(density, temperature, abar, zbar)
for i in vars_to_print:
    print(fmt.format(i, *out[i]))
    inv = helmeos.eos_invert(density, abar, zbar, out[i], i)
    print(fmt.format("Inverted temp", *inv))
print()

# load default table
print("~ Load default table ~")
default = helmeos.HelmTable()
out = default.eos_DT(density, temperature, abar, zbar)
for i in vars_to_print:
    print(fmt.format(i, *out[i]))
    inv = default.eos_invert(density, abar, zbar, out[i], i)
    print(fmt.format("Inverted temp", *inv))
print()

# load table by filename
print("~ Load table by filename ~")
fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "helm_table.dat")
by_filename = helmeos.HelmTable(fn=fn, temp_n=201, dens_n=541)
out = by_filename.eos_DT(density, temperature, abar, zbar)
for i in vars_to_print:
    print(fmt.format(i, *out[i]))
    inv = by_filename.eos_invert(density, abar, zbar, out[i], i)
    print(fmt.format("Inverted temp", *inv))
print()
