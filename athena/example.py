#! /usr/bin/env python
try:
    from helmeos.athena import helm2athena
except ImportError:
    try:
        from . import helm2athena
    except ImportError:
        import os
        import sys
        path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        sys.path.append(path)
        from helmeos.athena import helm2athena

savefig = __name__ == '__main__'
heos = helm2athena(nd=32, ne=64, return_class=True, ldmin=-1, ldmax=11)
heos.quad_plot(dpi=200, save=savefig)
