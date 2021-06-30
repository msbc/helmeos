from . import helm2athena

heos = helm2athena(nd=32, ne=64, return_class=True, ldmin=-1, ldmax=11)
heos.quad_plot(dpi=200)
