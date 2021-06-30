import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS


class AthenaEOS(object):
    """Parent class to implement equation of state functions"""
    def __init__(self):
        """Initialize EOS class"""
        self.ideal = False  # Whether this EOS is ideal
        self.indep = None   # independent variables not including density

    def valid(self):
        """Determine if this EOS is valid"""
        try:
            self.asq_of_rho_p(1, 1)
            self.ei_of_rho_p(1, 1)
        except NotImplementedError:
            return False
        return True

    def asq_of_rho_p(self, rho, p):
        """Adiabatic sound speed^2 as a function of density (rho) and pressure (p)"""
        raise NotImplementedError

    def ei_of_rho_p(self, rho, p):
        """Internal energy density as a function of density (rho) and pressure (p)"""
        raise NotImplementedError

    def es_of_rho_p(self, rho, p):
        """Specific internal energy as a function of density (rho) and pressure (p)"""
        return self.ei_of_rho_p(rho, p) / rho

    def p_of_rho_es(self, rho, es):
        """Pressure as a function of density (rho) and specific internal energy (es)"""
        raise NotImplementedError


class Helmholtz(AthenaEOS):
    def __init__(self, abar=1, zbar=1, helmtable=None, dunit=1, eunit=1, tunit=1):
        """Initialize EOS class"""
        super(Helmholtz, self).__init__()
        self.abar = abar
        self.zbar = zbar
        if helmtable is None:
            from .. import default
            self._ht = default
        else:
            self._ht = helmtable
        self.indep = "T"
        self.dunit = dunit
        if eunit is None:
            if tunit is not None:
                eunit = self._ht.eos_DT(dunit, tunit, self.abar, self.zbar)['etot'] / dunit
            else:
                raise ValueError("eunit and tunit can't both be None.")
        if tunit is None:
            self._ht.load()
            tunit = float(self._ht.eos_invert(dunit, abar, zbar, eunit / dunit, 'etot'))
        self.tunit = tunit
        self.eunit = eunit
        self.esunit = eunit / dunit
        self.asqunit = eunit / dunit
        self._tlast = tunit

    def _DT(self, rho, T, outvar=None):
        """Shorthand density, temperature call"""
        return self._ht.eos_DT(rho * self.dunit, T * self.tunit, self.abar, self.zbar,
                               outvar=outvar)

    def p_of_rho_T(self, rho, T):
        """Pressure as a function of density (rho) and temperature (T)"""
        return self._DT(rho, T)['ptot'] / self.eunit

    def es_of_rho_T(self, rho, T):
        """Specific energy density as a function of density (rho) and temperature (T)"""
        return self._DT(rho, T)['etot'] / self.esunit

    def ei_of_rho_T(self, rho, T):
        """Internal energy density as a function of density (rho) and temperature (T)"""
        return self.es_of_rho_T(rho, T) * rho

    def asq_of_rho_T(self, rho, T):
        """Adiabatic sound speed^2 as a function of density (rho) and temperature (T)"""
        return self._DT(rho, T)['asq'] / self.asqunit

    def T_of_rho_p(self, rho, p):
        out = self._ht.eos_invert(rho * self.dunit, self.abar, self.zbar, p * self.eunit,
                                  'ptot', t0=self._tlast)
        try:
            out = float(out)
            self._tlast = out
        except TypeError:
            self._tlast = float(out[0])
        return out / self.tunit

    def T_of_rho_es(self, rho, es):
        out = self._ht.eos_invert(rho * self.dunit, self.abar, self.zbar,
                                  es * self.esunit, 'etot', t0=self._tlast)
        try:
            out = float(out)
            self._tlast = out
        except TypeError:
            self._tlast = float(out[0])
        return out / self.tunit

    def ei_of_rho_p(self, rho, p):
        """Internal energy density as a function of density (rho) and pressure (p)"""
        return self.ei_of_rho_T(rho, self.T_of_rho_p(rho, p))

    def asq_of_rho_p(self, rho, p):
        """Adiabatic sound speed^2 as a function of density (rho) and pressure (p)"""
        out = self.asq_of_rho_T(rho, self.T_of_rho_p(rho, p))
        try:
            return float(out)
        except TypeError:
            return out

    def p_of_rho_es(self, rho, es):
        """Pressure as a function of density (rho) and specific internal energy (es)"""
        return self.p_of_rho_T(rho, self.T_of_rho_es(rho, es))

    def gamma1(self, rho, T):
        """First adiabatic index as a function of density (rho) and temperature (T)"""
        return self._DT(rho, T)['gam1']


class AthenaTable(AthenaEOS):
    def __init__(self, data, lrho, le, ratios=None, indep=None, dens_pow=-1, fn=None,
                 add_var=None):
        super(AthenaTable, self).__init__()
        self.ideal = False
        self.fn = fn
        if ratios is None:
            ratios = np.ones(len(data))
        lr = np.log(ratios)
        self._lr = lr
        if indep is None:
            indep = 'es'
        self.indep = indep
        self.data = data
        self.lrho = lrho
        self.le = le
        self.dens_pow = dens_pow
        var = ['p', 'e', 'asq_p']
        if add_var is not None:
            var.extend(add_var)
        d = {var[i]: RBS(lrho, le + lr[i], np.log10(data[i].T), kx=1, ky=1).ev
             for i in range(len(var))}
        self._interp_dict = d
        axes = [r'\epsilon'] + [r'P/\rho'] * 4
        self._axes = [r'$\log_{10}' + i + '$' for i in axes]
        lbls = ['P/e', 'e/P', r'\Gamma_1', r'\log_{\rm 10}T']
        self._lbls = ['${0:}$'.format(i) for i in lbls]

    def _interp(self, rho, e, var):
        ld = np.log10(rho)
        return 10**self._interp_dict[var](ld, np.log10(e) + self.dens_pow * ld)

    def asq_of_rho_p(self, rho, p):
        """Adiabatic sound speed^2 as a function of density (rho) and pressure (p)"""
        return self._interp(rho, p, 'asq_p') * p / rho

    def ei_of_rho_p(self, rho, p):
        """Internal energy density as a function of density (rho) and pressure (p)"""
        return self._interp(rho, p, 'e') * p

    def es_of_rho_p(self, rho, p):
        """Specific internal energy as a function of density (rho) and pressure (p)"""
        return self._interp(rho, p, 'e') * p / rho

    def p_of_rho_ei(self, rho, ei):
        """Pressure as a function of density (rho) and internal energy density (ei)"""
        return self._interp(rho, ei, 'p') * ei

    def p_of_rho_es(self, rho, es):
        """Pressure as a function of density (rho) and specific internal energy (es)"""
        return self.p_of_rho_ei(rho, es / rho)

    def imshow(self, data, save=False, fn=None, fig=None, ax=None, fig_opt=None, dpi=None,
               figsize=None, log=False, popt=None, cb=True, cbl=None, aspect='auto',
               title=None, lbls=True, vmin=None, vmax=None, ext='png', lr=0, cbax=None,
               interpolation='nearest', origin='lower'):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        var = None
        ylbl = None
        var_index = None
        try:
            if int(data) == data:
                var_index = data
                var = self._lbls[data]
                ylbl = self._axes[data]
                lr = self._lr[data]
                data = self.data[data]
                if 'log' in var:
                    data = np.log10(data)
        except TypeError:
            pass
        if popt is None:
            popt = {}
        norm = None
        if log:
            norm = LogNorm()
        extent = [self.lrho[0], self.lrho[-1], self.le[0] + lr, self.le[-1] + lr]
        _popt = dict(norm=norm, extent=extent, aspect=aspect, vmin=vmin, vmax=vmax,
                     interpolation=interpolation, origin=origin)
        _popt.update(popt)

        if fig is None and ax is None:
            _fopt = {'dpi': dpi, 'figsize': figsize}
            if fig_opt is None:
                fig_opt = {}
            _fopt.update(fig_opt)
            fig = plt.figure(**_fopt)
        if ax is not None:
            plt.sca(ax)
        im = plt.imshow(data, **_popt)
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        if cb:
            if cbax:
                cb = plt.colorbar(im, cax=cbax)
            else:
                cb = plt.colorbar(im)
            cb.ax.tick_params(axis='both', direction='in')
            if cbl is None:
                try:
                    if str(var) == var:
                        cbl = var
                except TypeError:
                    pass
            if cbl:
                cb.set_label(cbl)
        if title:
            plt.title(title)
        if lbls:
            plt.xlabel(r'$\log_{10} \rho$')
            if ylbl:
                plt.ylabel(ylbl)

        if save or fn:
            if not fn:
                fn = 'eos'
                try:
                    if str(var) == var:
                        fn += '_' + var
                except TypeError:
                    pass
                fn += '_plot.' + ext
            plt.savefig(fn)
            plt.close()

    def quad_plot(self, save=False, fn=None, fig=None, fig_opt=None, dpi=None,
                  figsize=(7, 5), popt=None, ext='png', **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        if popt is None:
            popt = dict()
        popt.update(kwargs)
        if save and dpi is None:
            dpi = 150
        if fig is None:
            _fopt = {'dpi': dpi, 'figsize': figsize}
            if fig_opt is None:
                fig_opt = {}
            _fopt.update(fig_opt)
            fig = plt.figure(**_fopt)
        gs = GridSpec(2, 5, width_ratios=[1, .05, .4, 1, .05], wspace=.1, hspace=.15,
                      left=.1, top=.98, right=.9, bottom=.1)
        axs = [plt.subplot(gs[0, 0])]
        tmp = [(0, 3), (1, 0), (1, 3)]
        axs += [plt.subplot(gs[i], sharex=axs[0]) for i in tmp]
        tmp = [(0, 1), (0, 4), (1, 1), (1, 4)]
        caxs = [plt.subplot(gs[i]) for i in tmp]
        for i in range(4):
            self.imshow(i, ax=axs[i], cbax=caxs[i], **popt)
            if i < 2:
                plt.xlabel('')
        if save or fn:
            if not fn:
                fn = 'eos_plot.' + ext
            plt.savefig(fn)
            plt.close()
