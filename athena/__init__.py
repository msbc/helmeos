import numpy as np
from .util import write_athena_tables, update_progress
from .. import phys_const


def helm2athena(abar=1.0, zbar=1.0, helm_eos=None, ldmin=None, ldmax=None, lemin=None,
                lemax=None, nd=None, ne=None, ltmin=None, ltmax=None, force_safe=True,
                return_class=False, loop_invert=True, **kwargs):
    """
    Converts the Helmholtz EOS into an Athena++ EOS Table
    Unless otherwise stated, units are assumed to be cgs

    :param abar: mean atomic/ion weight in amu
    :param zbar: mean ion charge
    :param helm_eos: instance of HelmholtzEOS
    :param ldmin: log_10 density minimum, defaults to that of helm_eos
    :param ldmax: log_10 density maximum, defaults to that of helm_eos
    :param lemin: log_10 specific energy density minimum, default inferred from helm_eos
    :param lemax: log_10 specific energy density maximum, default inferred from helm_eos
    :param nd: number of density points, default inferred from helm_eos
    :param ne: number of specific energy density points, default inferred from helm_eos
    :param ltmin: set to use a log_10 temperature min to infer lemin
    :param ltmax: set to use a log_10 temperature max to infer lemax
    :param force_safe: ensure lemin/lemax stay within Helmholtz table
    :param return_class: set True to return an instance of AthenaTable class
    :param loop_invert: Use explicit loops to do the EOS inversion
    :param kwargs: extra parameters/options to pass to write_athena_tables
    :return: output filename or AthenaTable if return_class
    """
    if helm_eos is None:
        from .. import default as helm_eos
    if nd is None and ldmin is None and ldmax is None:
        nd = helm_eos.dens_n
    ldmin = ldmin if ldmin is not None else helm_eos.dens_log_min
    ldmax = ldmax if ldmax is not None else helm_eos.dens_log_max
    if nd is None:
        nd = (ldmax - ldmin) / (helm_eos.dens_log_max - helm_eos.dens_log_min)
        nd = int(np.ceil(nd * (helm_eos.dens_n - 1.0)) + 1)
    dens = 10**np.linspace(ldmin, ldmax, nd)

    def get_le(lt):
        return np.log10(helm_eos.eos_DT(dens, 10**lt, abar, zbar, 'etot')['etot'])

    def get_lp(lt):
        return np.log10(helm_eos.eos_DT(dens, 10**lt, abar, zbar, 'ptot')['ptot'] / dens)

    if lemin is None:
        ltmin = ltmin if ltmin is not None else helm_eos.temp_log_min
        lemin = get_le(ltmin).min()
    if lemax is None:
        ltmax = ltmax if ltmax is not None else helm_eos.temp_log_max
        lemax = get_le(ltmax).max()
    if force_safe:
        lemin = max(lemin, get_le(helm_eos.temp_log_min).max(),
                    get_lp(helm_eos.temp_log_min).max())
        lemax = min(lemax, get_le(helm_eos.temp_log_max).min(),
                    get_lp(helm_eos.temp_log_max).min())

    ne = ne if ne is not None else helm_eos.temp_n
    etot = 10**np.linspace(lemin, lemax, ne)
    one = np.ones((ne, nd))
    # specific internal energy density
    energy = etot[:, None] * one
    density = dens[None, :] * one
    print("Inverting energy into temperature (phase 1 of 2)")
    if loop_invert:
        temp = np.empty_like(energy)
        for di in range(nd):
            # print('di', di)
            update_progress(float(di) / nd)
            Trad = (dens[di] * etot[-1] / phys_const.asol) ** 0.25
            Tgas = 2. / 3. * (abar / (1.0 + zbar)) * etot[-1] / phys_const.kergavo
            t0 = min(Trad, Tgas)
            for ei in reversed(range(ne)):
                # print('\tei', ei, t0)
                t0 = helm_eos.eos_invert(dens[di], abar, zbar, etot[ei], 'etot', t0=t0)
                temp[ei, di] = t0
        update_progress(1)
    else:
        temp = helm_eos.eos_invert(density, abar, zbar, energy, 'etot')
    pspec = helm_eos.eos_DT(density, temp, abar, zbar, 'ptot')['ptot'] / density
    print("Inverting pressure into temperature (phase 2 of 2)")
    # now the "energy/etot" arrays are actually specific pressure (P/rho)
    if loop_invert:
        for di in range(nd):
            rho = dens[di]
            update_progress(float(di) / nd)
            Trad = (rho * etot[-1] / phys_const.asoli3) ** 0.25
            Tgas = (abar / (1.0 + zbar)) * etot[-1] / phys_const.kergavo
            t0 = min(Trad, Tgas)
            for ei in reversed(range(ne)):
                args = rho, abar, zbar, etot[ei] * rho, 'ptot'
                try:
                    t0 = helm_eos.eos_invert(*args, t0=t0)
                except ValueError:
                    # if we fail then try the temperature from energy inversion
                    t0 = helm_eos.eos_invert(*args, t0=temp[ei, di])
                temp[ei, di] = t0
        update_progress(1)
    else:
        temp = helm_eos.eos_invert(density, abar, zbar, energy * density, 'ptot')
    data = helm_eos.eos_DT(density, temp, abar, zbar, ['etot', 'asq'])
    # p/e(e/rho,rho), e/p(p/rho,rho), asq*rho/p(p/rho,rho), T(p/rho,rho)
    varlist = [pspec / energy, data['etot'] / energy, data['asq'] / energy, temp]
    fn = write_athena_tables((ldmin, ldmax), (lemin, lemax), varlist, **kwargs)
    if return_class:
        from .eos import AthenaTable
        return AthenaTable(varlist, np.log10(dens), np.log10(etot))
    return fn
