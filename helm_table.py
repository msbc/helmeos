import numpy as np
import os
from scipy.optimize import fsolve
from . import table_param as tab
from .phys_const import *

_default_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "helm_table.dat")
_maxiter = None


def OldStyleInputs(fn=None, jmax=None, tlo=None, thi=None, imax=None, dlo=None, dhi=None):
    """Use Frank's variable names to call HelmTable"""
    return HelmTable(fn=fn, temp_n=jmax, temp_log_min=tlo, temp_log_max=thi, dens_n=imax,
                     dens_log_min=dlo, dens_log_max=dhi)


class HelmTable(object):
    """Class to store and interpolate Helmholtz table."""

    def __init__(self, fn=None, temp_n=None, temp_log_min=None, temp_log_max=None,
                 dens_n=None, dens_log_min=None, dens_log_max=None):
        # set defaults
        if fn is None:
            fn = _default_fn
        if temp_n is None:
            temp_n = tab.temp_n
        if temp_log_min is None:
            temp_log_min = tab.temp_log_min
        if temp_log_max is None:
            temp_log_max = tab.temp_log_max
        if dens_n is None:
            dens_n = tab.dens_n
        if dens_log_min is None:
            dens_log_min = tab.dens_log_min
        if dens_log_max is None:
            dens_log_max = tab.dens_log_max
        # store parameters
        self.fn = fn
        self.temp_n = temp_n
        self.temp_log_min = temp_log_min
        self.temp_log_max = temp_log_max
        self.dens_n = dens_n
        self.dens_log_min = dens_log_min
        self.dens_log_max = dens_log_max
        self._full_table = None
        jmax = temp_n
        tlo = temp_log_min
        thi = temp_log_max
        imax = dens_n
        dlo = dens_log_min
        dhi = dens_log_max

        # load data from file
        with open(fn) as f:
            # read the helmholtz free energy and its derivatives
            col9 = np.genfromtxt(f, delimiter='  ', max_rows=jmax * imax)
            col9 = np.swapaxes(col9.reshape((jmax, imax, 9)), 0, 1)
            self._f = col9[:, :, 0]
            self._fd = col9[:, :, 1]
            self._ft = col9[:, :, 2]
            self._fdd = col9[:, :, 3]
            self._ftt = col9[:, :, 4]
            self._fdt = col9[:, :, 5]
            self._fddt = col9[:, :, 6]
            self._fdtt = col9[:, :, 7]
            self._fddtt = col9[:, :, 8]

            # read the pressure derivative with density table
            col4 = np.genfromtxt(f, delimiter='  ', max_rows=jmax * imax)
            col4 = np.swapaxes(col4.reshape((jmax, imax, 4)), 0, 1)
            self._dpdf = col4[:, :, 0]
            self._dpdfd = col4[:, :, 1]
            self._dpdft = col4[:, :, 2]
            self._dpdfdt = col4[:, :, 3]

            # read the electron chemical potential table
            col4 = np.genfromtxt(f, delimiter='  ', max_rows=jmax * imax)
            col4 = np.swapaxes(col4.reshape((jmax, imax, 4)), 0, 1)
            self._ef = col4[:, :, 0]
            self._efd = col4[:, :, 1]
            self._eft = col4[:, :, 2]
            self._efdt = col4[:, :, 3]

            # read the number density table
            col4 = np.genfromtxt(f, delimiter='  ', max_rows=jmax * imax)
            col4 = np.swapaxes(col4.reshape((jmax, imax, 4)), 0, 1)
            self._xf = col4[:, :, 0]
            self._xfd = col4[:, :, 1]
            self._xft = col4[:, :, 2]
            self._xfdt = col4[:, :, 3]

        # pre-compute deltas and temp/dens arrays
        self._tstp = (thi - tlo) / (jmax - 1.0)
        self._tstpi = 1.0 / self._tstp
        self._t = 10 ** (np.arange(jmax) * self._tstp + tlo)
        self._dt_sav = np.diff(self._t)
        self._dt2_sav = self._dt_sav ** 2
        self._dti_sav = self._dt_sav ** -1
        self._dt2i_sav = self._dti_sav ** 2
        self._dt3i_sav = self._dti_sav * self._dt2i_sav

        self._dstp = (dhi - dlo) / (imax - 1.0)
        self._dstpi = 1.0 / self._dstp
        self._d = 10 ** (np.arange(imax) * self._dstp + dlo)
        self._dd_sav = np.diff(self._d)
        self._dd2_sav = self._dd_sav ** 2
        self._ddi_sav = self._dd_sav ** -1
        self._dd2i_sav = self._ddi_sav ** 2
        self._dd3i_sav = self._ddi_sav * self._dd2i_sav

        # inversion function
        self._vec_inv_with_d = np.vectorize(self._scalar_eos_invert_with_d,
                                            otypes=[np.float64])
        self._vec_inv_no_d = np.vectorize(self._scalar_eos_invert_no_d,
                                          otypes=[np.float64])

    def eos_DT(self, den, temp, abar, zbar, outvar=None):
        scalar_input = False
        if ((not hasattr(den, "dtype")) and (not hasattr(temp, "dtype"))
                and (not hasattr(abar, "dtype")) and (not hasattr(zbar, "dtype"))):
            scalar_input = True
        den = np.atleast_1d(den)
        temp = np.atleast_1d(temp)
        abar = np.atleast_1d(abar)
        zbar = np.atleast_1d(zbar)
        shape = (den + temp + abar + zbar).shape
        one = np.ones(shape)
        den = den * one
        temp = temp * one
        abar = abar * one
        zbar = zbar * one

        # use Frank's name convention for porting ease
        jmax = self.temp_n
        tlo = self.temp_log_min
        thi = self.temp_log_max
        tstpi = self._tstpi
        imax = self.dens_n
        dlo = self.dens_log_min
        dhi = self.dens_log_max
        dstpi = self._dstpi

        den = np.maximum(10 ** dlo, np.minimum(10 ** dhi, den))
        temp = np.maximum(10 ** tlo, np.minimum(10 ** thi, temp))

        ytot1 = 1.0 / abar
        ye = np.maximum(1.0e-16, ytot1 * zbar)
        din = ye * den

        jat = np.floor((np.log10(temp) - tlo) * tstpi).astype(int)
        jat = np.minimum(jmax - 1, np.maximum(0, jat))
        jatp1 = np.minimum(jmax - 1, jat + 1)
        djat = np.minimum(jmax - 2, jat)
        iat = np.floor((np.log10(din) - dlo) * dstpi).astype(int)
        iat = np.minimum(imax - 1, np.maximum(0, iat))
        iatp1 = np.minimum(imax - 1, iat + 1)
        diat = np.minimum(imax - 2, iat)
        deni = 1.0 / den
        tempi = 1.0 / temp
        kt = kerg * temp
        ktinv = 1.0 / kt

        #####################
        # Radiation section #
        #####################

        prad = asoli3 * temp * temp * temp * temp
        dpraddd = 0.0
        dpraddt = 4.0 * prad * tempi
        dpradda = 0.0
        dpraddz = 0.0
        erad = 3.0 * prad * deni
        deraddd = -erad * deni
        deraddt = 3.0 * dpraddt * deni
        deradda = 0.0
        deraddz = 0.0
        srad = (prad * deni + erad) * tempi
        dsraddd = (dpraddd * deni - prad * deni * deni + deraddd) * tempi
        dsraddt = (dpraddt * deni + deraddt - srad) * tempi
        dsradda = 0.0
        dsraddz = 0.0

        ###############
        # Ion section #
        ###############

        xni = avo * ytot1 * den
        dxnidd = avo * ytot1
        dxnida = -xni * ytot1

        pion = xni * kt
        dpiondd = dxnidd * kt
        dpiondt = xni * kerg
        dpionda = dxnida * kt
        dpiondz = 0.0

        eion = 1.5 * pion * deni
        deiondd = (1.5 * dpiondd - eion) * deni
        deiondt = 1.5 * dpiondt * deni
        deionda = 1.5 * dpionda * deni
        deiondz = 0.0

        # sackur-tetrode equation for the ion entropy of
        # a single ideal gas characterized by abar

        x = abar * abar * np.sqrt(abar) * deni / avo
        s = sioncon * temp
        z = x * s * np.sqrt(s)
        y = np.log(z)
        sion = (pion * deni + eion) * tempi + kergavo * ytot1 * y
        dsiondd = (dpiondd * deni - pion * deni * deni + deiondd) * tempi \
                  - kergavo * deni * ytot1

        dsiondt = (dpiondt * deni + deiondt) * tempi - (pion * deni + eion) * tempi ** 2 \
                  + 1.5 * kergavo * tempi * ytot1

        x = avo * kerg / abar
        dsionda = (dpionda * deni + deionda) * tempi + kergavo * ytot1 ** 2 * (2.5 - y)
        dsiondz = 0.0

        #############################
        # electron-positron section #
        #############################

        # assume complete ionization
        xnem = xni * zbar

        # access the table locations only once
        fi = np.empty((36,) + shape)
        fi[0] = self._f[iat, jat]
        fi[1] = self._f[iatp1, jat]
        fi[2] = self._f[iat, jatp1]
        fi[3] = self._f[iatp1, jatp1]
        fi[4] = self._ft[iat, jat]
        fi[5] = self._ft[iatp1, jat]
        fi[6] = self._ft[iat, jatp1]
        fi[7] = self._ft[iatp1, jatp1]
        fi[8] = self._ftt[iat, jat]
        fi[9] = self._ftt[iatp1, jat]
        fi[10] = self._ftt[iat, jatp1]
        fi[11] = self._ftt[iatp1, jatp1]
        fi[12] = self._fd[iat, jat]
        fi[13] = self._fd[iatp1, jat]
        fi[14] = self._fd[iat, jatp1]
        fi[15] = self._fd[iatp1, jatp1]
        fi[16] = self._fdd[iat, jat]
        fi[17] = self._fdd[iatp1, jat]
        fi[18] = self._fdd[iat, jatp1]
        fi[19] = self._fdd[iatp1, jatp1]
        fi[20] = self._fdt[iat, jat]
        fi[21] = self._fdt[iatp1, jat]
        fi[22] = self._fdt[iat, jatp1]
        fi[23] = self._fdt[iatp1, jatp1]
        fi[24] = self._fddt[iat, jat]
        fi[25] = self._fddt[iatp1, jat]
        fi[26] = self._fddt[iat, jatp1]
        fi[27] = self._fddt[iatp1, jatp1]
        fi[28] = self._fdtt[iat, jat]
        fi[29] = self._fdtt[iatp1, jat]
        fi[30] = self._fdtt[iat, jatp1]
        fi[31] = self._fdtt[iatp1, jatp1]
        fi[32] = self._fddtt[iat, jat]
        fi[33] = self._fddtt[iatp1, jat]
        fi[34] = self._fddtt[iat, jatp1]
        fi[35] = self._fddtt[iatp1, jatp1]

        # various differences
        xt = np.maximum((temp - self._t[jat]) * self._dti_sav[djat], 0.0)
        xd = np.maximum((din - self._d[iat]) * self._ddi_sav[diat], 0.0)
        mxt = 1 - xt
        mxd = 1 - xd

        # the six density and six temperature basis functions
        si0t = psi0(xt)
        si1t = psi1(xt) * self._dt_sav[djat]
        si2t = psi2(xt) * self._dt2_sav[djat]

        si0mt = psi0(mxt)
        si1mt = -psi1(mxt) * self._dt_sav[djat]
        si2mt = psi2(mxt) * self._dt2_sav[djat]

        si0d = psi0(xd)
        si1d = psi1(xd) * self._dd_sav[diat]
        si2d = psi2(xd) * self._dd2_sav[diat]

        si0md = psi0(mxd)
        si1md = -psi1(mxd) * self._dd_sav[diat]
        si2md = psi2(mxd) * self._dd2_sav[diat]

        # derivatives of the weight functions
        dsi0t = dpsi0(xt) * self._dti_sav[djat]
        dsi1t = dpsi1(xt)
        dsi2t = dpsi2(xt) * self._dt_sav[djat]

        dsi0mt = -dpsi0(mxt) * self._dti_sav[djat]
        dsi1mt = dpsi1(mxt)
        dsi2mt = -dpsi2(mxt) * self._dt_sav[djat]

        dsi0d = dpsi0(xd) * self._ddi_sav[diat]
        dsi1d = dpsi1(xd)
        dsi2d = dpsi2(xd) * self._dd_sav[diat]

        dsi0md = -dpsi0(mxd) * self._ddi_sav[diat]
        dsi1md = dpsi1(mxd)
        dsi2md = -dpsi2(mxd) * self._dd_sav[diat]

        # second derivatives of the weight functions
        ddsi0t = ddpsi0(xt) * self._dt2i_sav[djat]
        ddsi1t = ddpsi1(xt) * self._dti_sav[djat]
        ddsi2t = ddpsi2(xt)

        ddsi0mt = ddpsi0(mxt) * self._dt2i_sav[djat]
        ddsi1mt = -ddpsi1(mxt) * self._dti_sav[djat]
        ddsi2mt = ddpsi2(mxt)

        ddsi0d = ddpsi0(xd) * self._dd2i_sav[diat]
        ddsi1d = ddpsi1(xd) * self._ddi_sav[diat]
        ddsi2d = ddpsi2(xd)

        ddsi0md = ddpsi0(mxd) * self._dd2i_sav[diat]
        ddsi1md = -ddpsi1(mxd) * self._ddi_sav[diat]
        ddsi2md = ddpsi2(mxd)

        # the free energy
        free = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, si0d, si1d, si2d, si0md,
                  si1md, si2md)

        # derivative with respect to density
        df_d = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, dsi0d, dsi1d, dsi2d,
                  dsi0md, dsi1md, dsi2md)

        # derivative with respect to temperature
        df_t = h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, si0d, si1d, si2d,
                  si0md, si1md, si2md)

        # derivative with respect to density**2
        df_dd = h5(fi, si0t, si1t, si2t, si0mt, si1mt, si2mt, ddsi0d, ddsi1d, ddsi2d,
                   ddsi0md, ddsi1md, ddsi2md)

        # derivative with respect to temperature**2
        df_tt = h5(fi, ddsi0t, ddsi1t, ddsi2t, ddsi0mt, ddsi1mt, ddsi2mt, si0d, si1d,
                   si2d, si0md, si1md, si2md)

        # derivative with respect to temperature and density
        df_dt = h5(fi, dsi0t, dsi1t, dsi2t, dsi0mt, dsi1mt, dsi2mt, dsi0d, dsi1d, dsi2d,
                   dsi0md, dsi1md, dsi2md)

        # now get the pressure derivative with density, chemical potential, and
        # electron positron number densities
        # get the interpolation weight functions
        si0t = xpsi0(xt)
        si1t = xpsi1(xt) * self._dt_sav[djat]

        si0mt = xpsi0(mxt)
        si1mt = -xpsi1(mxt) * self._dt_sav[djat]

        si0d = xpsi0(xd)
        si1d = xpsi1(xd) * self._dd_sav[diat]

        si0md = xpsi0(mxd)
        si1md = -xpsi1(mxd) * self._dd_sav[diat]

        # derivatives of weight functions
        dsi0t = xdpsi0(xt) * self._dti_sav[djat]
        dsi1t = xdpsi1(xt)

        dsi0mt = -xdpsi0(mxt) * self._dti_sav[djat]
        dsi1mt = xdpsi1(mxt)

        dsi0d = xdpsi0(xd) * self._ddi_sav[diat]
        dsi1d = xdpsi1(xd)

        dsi0md = -xdpsi0(mxd) * self._ddi_sav[diat]
        dsi1md = xdpsi1(mxd)

        # look in the pressure derivative only once
        fi[0] = self._dpdf[iat, jat]
        fi[1] = self._dpdf[iatp1, jat]
        fi[2] = self._dpdf[iat, jatp1]
        fi[3] = self._dpdf[iatp1, jatp1]
        fi[4] = self._dpdft[iat, jat]
        fi[5] = self._dpdft[iatp1, jat]
        fi[6] = self._dpdft[iat, jatp1]
        fi[7] = self._dpdft[iatp1, jatp1]
        fi[8] = self._dpdfd[iat, jat]
        fi[9] = self._dpdfd[iatp1, jat]
        fi[10] = self._dpdfd[iat, jatp1]
        fi[11] = self._dpdfd[iatp1, jatp1]
        fi[12] = self._dpdfdt[iat, jat]
        fi[13] = self._dpdfdt[iatp1, jat]
        fi[14] = self._dpdfdt[iat, jatp1]
        fi[15] = self._dpdfdt[iatp1, jatp1]

        # pressure derivative with density
        dpepdd = h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md)
        dpepdd = np.maximum(ye * dpepdd, 1.0e-30)

        # look in the electron chemical potential table only once
        fi[1] = self._ef[iat, jat]
        fi[2] = self._ef[iatp1, jat]
        fi[3] = self._ef[iat, jatp1]
        fi[4] = self._ef[iatp1, jatp1]
        fi[5] = self._eft[iat, jat]
        fi[6] = self._eft[iatp1, jat]
        fi[7] = self._eft[iat, jatp1]
        fi[8] = self._eft[iatp1, jatp1]
        fi[9] = self._efd[iat, jat]
        fi[10] = self._efd[iatp1, jat]
        fi[11] = self._efd[iat, jatp1]
        fi[12] = self._efd[iatp1, jatp1]
        fi[13] = self._efdt[iat, jat]
        fi[14] = self._efdt[iatp1, jat]
        fi[15] = self._efdt[iat, jatp1]
        fi[16] = self._efdt[iatp1, jatp1]

        # electron chemical potential etaele
        etaele = h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md)

        # derivative with respect to density
        x = h3(fi, si0t, si1t, si0mt, si1mt, dsi0d, dsi1d, dsi0md, dsi1md)
        detadd = ye * x

        # derivative with respect to temperature
        detadt = h3(fi, dsi0t, dsi1t, dsi0mt, dsi1mt, si0d, si1d, si0md, si1md)

        # derivative with respect to abar and zbar
        detada = -x * din * ytot1
        detadz = x * den * ytot1

        # look in the number density table only once
        fi[1] = self._xf[iat, jat]
        fi[2] = self._xf[iatp1, jat]
        fi[3] = self._xf[iat, jatp1]
        fi[4] = self._xf[iatp1, jatp1]
        fi[5] = self._xft[iat, jat]
        fi[6] = self._xft[iatp1, jat]
        fi[7] = self._xft[iat, jatp1]
        fi[8] = self._xft[iatp1, jatp1]
        fi[9] = self._xfd[iat, jat]
        fi[10] = self._xfd[iatp1, jat]
        fi[11] = self._xfd[iat, jatp1]
        fi[12] = self._xfd[iatp1, jatp1]
        fi[13] = self._xfdt[iat, jat]
        fi[14] = self._xfdt[iatp1, jat]
        fi[15] = self._xfdt[iat, jatp1]
        fi[16] = self._xfdt[iatp1, jatp1]

        # electron + positron number densities
        xnefer = h3(fi, si0t, si1t, si0mt, si1mt, si0d, si1d, si0md, si1md)

        # derivative with respect to density
        x = h3(fi, si0t, si1t, si0mt, si1mt, dsi0d, dsi1d, dsi0md, dsi1md)
        x = np.maximum(x, 1.0e-30)
        dxnedd = ye * x

        # derivative with respect to temperature
        dxnedt = h3(fi, dsi0t, dsi1t, dsi0mt, dsi1mt, si0d, si1d, si0md, si1md)

        # derivative with respect to abar and zbar
        dxneda = -x * din * ytot1
        dxnedz = x * den * ytot1

        # the desired electron-positron thermodynamic quantities

        # dpepdd at high temperatures and low densities is below the
        # floating point limit of the subtraction of two large terms.
        # since dpresdd doesn't enter the maxwell relations at all, use the
        # bicubic interpolation done above instead of the formally correct expression
        x = din * din
        pele = x * df_d
        dpepdt = x * df_dt
        # dpepdd = ye * (x * df_dd + 2.0 * din * df_d)
        s = dpepdd / ye - 2.0 * din * df_d
        dpepda = -ytot1 * (2.0 * pele + s * din)
        dpepdz = den * ytot1 * (2.0 * din * df_d + s)

        x = ye * ye
        sele = -df_t * ye
        dsepdt = -df_tt * ye
        dsepdd = -df_dt * x
        dsepda = ytot1 * (ye * df_dt * din - sele)
        dsepdz = -ytot1 * (ye * df_dt * den + df_t)

        eele = ye * free + temp * sele
        deepdt = temp * dsepdt
        deepdd = x * df_d + temp * dsepdd
        deepda = -ye * ytot1 * (free + df_d * din) + temp * dsepda
        deepdz = ytot1 * (free + ye * df_d * den) + temp * dsepdz

        # coulomb section:

        # uniform background corrections only
        # from yakovlev & shalybkov 1989
        # lami is the average ion seperation
        # plasg is the plasma coupling parameter

        z = forth * pi
        s = z * xni
        dsdd = z * dxnidd
        dsda = z * dxnida

        lami = s ** -third
        inv_lami = 1.0 / lami
        z = -third * lami
        lamidd = z * dsdd / s
        lamida = z * dsda / s

        plasg = zbar * zbar * esqu * ktinv * inv_lami

        z = -plasg * inv_lami
        plasgdd = z * lamidd
        plasgda = z * lamida
        plasgdt = -plasg * ktinv * kerg
        plasgdz = 2.0 * plasg / zbar

        ecoul = np.empty(shape)
        pcoul = np.empty(shape)
        scoul = np.empty(shape)
        decouldd = np.empty(shape)
        decouldt = np.empty(shape)
        decoulda = np.empty(shape)
        decouldz = np.empty(shape)
        dpcouldd = np.empty(shape)
        dpcouldt = np.empty(shape)
        dpcoulda = np.empty(shape)
        dpcouldz = np.empty(shape)
        dscouldd = np.empty(shape)
        dscouldt = np.empty(shape)
        dscoulda = np.empty(shape)
        dscouldz = np.empty(shape)

        # yakovlev & shalybkov 1989 equations 82, 85, 86, 87
        loc = np.where(plasg >= 1.0)
        if loc[0].size:
            x = plasg[loc] ** 0.25
            y = avo * ytot1[loc] * kerg
            ecoul[loc] = y * temp[loc] * (a1 * plasg[loc] + b1 * x + c1 / x + d1)
            pcoul[loc] = third * den[loc] * ecoul[loc]
            idkbro = 3.0 * b1 * x - 5.0 * c1 / x + d1 * (np.log(plasg[loc]) - 1.0) - e1
            scoul[loc] = -y * idkbro

            y = avo * ytot1[loc] * kt[loc] * (a1 + 0.25 / plasg[loc] * (b1 * x - c1 / x))
            decouldd[loc] = y * plasgdd[loc]
            decouldt[loc] = y * plasgdt[loc] + ecoul[loc] / temp[loc]
            decoulda[loc] = y * plasgda[loc] - (ecoul / abar)[loc]
            decouldz[loc] = y * plasgdz[loc]

            y = third * den[loc]
            dpcouldd[loc] = third * ecoul[loc] + y * decouldd[loc]
            dpcouldt[loc] = y * decouldt[loc]
            dpcoulda[loc] = y * decoulda[loc]
            dpcouldz[loc] = y * decouldz[loc]

            y = -avo * kerg / abar[loc] * plasg[loc] * (.75 * b1 * x + 1.25 * c1 / x + d1)
            dscouldd[loc] = y * plasgdd[loc]
            dscouldt[loc] = y * plasgdt[loc]
            dscoulda[loc] = y * plasgda[loc] - scoul[loc] / abar[loc]
            dscouldz[loc] = y * plasgdz[loc]

        # yakovlev & shalybkov 1989 equations 102, 103, 104
        loc = np.where(plasg < 1.0)
        if loc[0].size:
            x = plasg[loc] * np.sqrt(plasg[loc])
            y = plasg[loc] ** b2
            z = c2 * x - third * a2 * y
            pcoul[loc] = -pion[loc] * z
            ecoul[loc] = 3.0 * pcoul[loc] / den[loc]
            scoul[loc] = -avo / abar[loc] * kerg * (c2 * x - a2 * (b2 - 1.0) / b2 * y)

            s = 1.5 * c2 * x / plasg[loc] - third * a2 * b2 * y / plasg[loc]
            dpcouldd[loc] = -dpiondd[loc] * z - pion[loc] * s * plasgdd[loc]
            dpcouldt[loc] = -dpiondt[loc] * z - pion[loc] * s * plasgdt[loc]
            dpcoulda[loc] = -dpionda[loc] * z - pion[loc] * s * plasgda[loc]
            dpcouldz[loc] = -dpiondz * z - pion[loc] * s * plasgdz[loc]

            s = 3.0 / den[loc]
            decouldd[loc] = s * dpcouldd[loc] - ecoul[loc] / den[loc]
            decouldt[loc] = s * dpcouldt[loc]
            decoulda[loc] = s * dpcoulda[loc]
            decouldz[loc] = s * dpcouldz[loc]

            s = -avo * kerg / (abar[loc] * plasg[loc]) * (
                    1.5 * c2 * x - a2 * (b2 - 1.0) * y)
            dscouldd[loc] = s * plasgdd[loc]
            dscouldt[loc] = s * plasgdt[loc]
            dscoulda[loc] = s * plasgda[loc] - scoul[loc] / abar[loc]
            dscouldz[loc] = s * plasgdz[loc]

        x = prad + pion + pele + pcoul
        y = erad + eion + eele + ecoul
        z = srad + sion + sele + scoul

        loc = np.where(np.logical_or(x < 0.0, y < 0.0))
        if loc[0].size:
            pcoul[loc] = 0.0
            dpcouldd[loc] = 0.0
            dpcouldt[loc] = 0.0
            dpcoulda[loc] = 0.0
            dpcouldz[loc] = 0.0
            ecoul[loc] = 0.0
            decouldd[loc] = 0.0
            decouldt[loc] = 0.0
            decoulda[loc] = 0.0
            decouldz[loc] = 0.0
            scoul[loc] = 0.0
            dscouldd[loc] = 0.0
            dscouldt[loc] = 0.0
            dscoulda[loc] = 0.0
            dscouldz[loc] = 0.0

        # sum all the gas components
        pgas = pion + pele + pcoul
        egas = eion + eele + ecoul
        sgas = sion + sele + scoul

        dpgasdd = dpiondd + dpepdd + dpcouldd
        dpgasdt = dpiondt + dpepdt + dpcouldt
        dpgasda = dpionda + dpepda + dpcoulda
        dpgasdz = dpiondz + dpepdz + dpcouldz

        degasdd = deiondd + deepdd + decouldd
        degasdt = deiondt + deepdt + decouldt
        degasda = deionda + deepda + decoulda
        degasdz = deiondz + deepdz + decouldz

        dsgasdd = dsiondd + dsepdd + dscouldd
        dsgasdt = dsiondt + dsepdt + dscouldt
        dsgasda = dsionda + dsepda + dscoulda
        dsgasdz = dsiondz + dsepdz + dscouldz

        # add in radiation to get the total
        pres = prad + pgas
        ener = erad + egas
        entr = srad + sgas

        dpresdd = dpraddd + dpgasdd
        dpresdt = dpraddt + dpgasdt
        dpresda = dpradda + dpgasda
        dpresdz = dpraddz + dpgasdz

        denerdd = deraddd + degasdd
        denerdt = deraddt + degasdt
        denerda = deradda + degasda
        denerdz = deraddz + degasdz

        dentrdd = dsraddd + dsgasdd
        dentrdt = dsraddt + dsgasdt
        dentrda = dsradda + dsgasda
        dentrdz = dsraddz + dsgasdz

        # for the gas
        # the temperature and density exponents (c&g 9.81 9.82)
        # the specific heat at constant volume (c&g 9.92)
        # the third adiabatic exponent (c&g 9.93)
        # the first adiabatic exponent (c&g 9.97)
        # the second adiabatic exponent (c&g 9.105)
        # the specific heat at constant pressure (c&g 9.98)
        # and relativistic formula for the sound speed (c&g 14.29)

        zz = pgas * deni
        zzi = den / pgas
        chit_gas = temp / pgas * dpgasdt
        chid_gas = dpgasdd * zzi
        cv_gas = degasdt
        x = zz * chit_gas / (temp * cv_gas)
        gam3_gas = x + 1.0
        gam1_gas = chit_gas * x + chid_gas
        nabad_gas = x / gam1_gas
        gam2_gas = 1.0 / (1.0 - nabad_gas)
        cp_gas = cv_gas * gam1_gas / chid_gas
        z = 1.0 + (egas + light2) * zzi
        sound_gas = clight * np.sqrt(gam1_gas / z)

        # for the totals
        zz = pres * deni
        zzi = den / pres
        chit = temp / pres * dpresdt
        chid = dpresdd * zzi
        cv = denerdt
        x = zz * chit / (temp * cv)
        gam3 = x + 1.0
        gam1 = chit * x + chid
        nabad = x / gam1
        gam2 = 1.0 / (1.0 - nabad)
        cp = cv * gam1 / chid
        z = 1.0 + (ener + light2) * zzi
        sound = clight * np.sqrt(gam1 / z)
        asq = light2 * gam1 / z

        # maxwell relations; each is zero if the consistency is perfect
        x = den * den
        dse = temp * dentrdt / denerdt - 1.0
        dpe = (denerdd * x + temp * dpresdt) / pres - 1.0
        dsp = -dentrdd * x / dpresdt - 1.0

        loc = locals()
        ignore = ['loc', 'fi', 'x', 'y', 'z', 'zz', 'zzi', 'self']
        out = {_translate.get(i, i): loc[i] for i in loc
               if i[0] != '_' and i not in ignore}
        if outvar is not None:
            outvar = np.atleast_1d(outvar)
            alt = {i: _translate.get(i, i) for i in outvar}
            tmp = {i: out[i] for i in out if i in outvar}
            tmp.update({i: out[alt[i]] for i in outvar})
            out = tmp
        if scalar_input:
            tmp = dict()
            for i in out:
                try:
                    tmp[i] = float(out[i])
                except TypeError:
                    tmp[i] = out[i]
            out = tmp
        return out

    def _invert_helper_with_der(self, temp, dens, abar, zbar, var, var_name, der_name):
        data = self.eos_DT(dens, temp, abar, zbar)
        ivar = 1.0 / var
        out = float(data[var_name]) * ivar - 1
        der = float(data[der_name]) * ivar
        return out, der

    def _invert_helper_without_der(self, log_t, dens, abar, zbar, var, var_name):
        return self.eos_DT(dens, 10 ** log_t, abar, zbar)[var_name] / var - 1.0

    def _adaptive_root_find(self, x0=None, args=None, maxiter=100, bracket=None,
                            tol=1e-10, newt_err=0.1):
        func = self._invert_helper_with_der
        var = args[3]
        if bracket is None:
            bracket = 10 ** np.array([self.temp_log_min, self.temp_log_max])

        if x0 in bracket:
            raise ValueError

        tmp = func(x0, *args)
        if tmp[0] <= 0:
            if x0 > bracket[0]:
                bracket[0] = x0
        else:
            if x0 < bracket[1]:
                bracket[1] = x0
        if np.abs(tmp[0]) > newt_err:
            xlast = bracket[0]
            flast = func(xlast, *args)[0]
        # else:
        #    x0 = x0 - tmp[0] / tmp[1]
        mode = 0
        n = 0
        while np.abs(tmp[0]) > tol:
            # if error is large take secant method step
            if np.abs(tmp[0]) > newt_err:
                delta = tmp[0] * (x0 - xlast) / (tmp[0] - flast)
                xlast = x0
                x0 -= delta
                flast = tmp[0]
                mode = 2
            # if error is small take Newton-Raphson step
            else:
                delta = tmp[0] / tmp[1]
                xlast = x0
                x0 -= delta
                flast = tmp[0]
                mode = 3
            # if we are outside brackets use bisection method for a step
            if x0 < bracket[0] or x0 > bracket[1] or not np.isfinite(x0):
                x0 = np.sqrt(np.prod(bracket))
                mode = 1
            tmp = func(x0, *args)
            # update bracketing values
            if tmp[0] <= 0:
                bracket[0] = x0
            else:
                bracket[1] = x0
            n += 1
            if n > maxiter:
                raise ValueError("Did not converge.")
        return x0

    def _scalar_eos_invert_with_d(self, dens, abar, zbar, var, var_name, der_name, t0,
                                  t1):
        out = self._adaptive_root_find(x0=t0,
                                       args=(dens, abar, zbar, var, var_name, der_name))
        return out

    def _scalar_eos_invert_no_d(self, dens, abar, zbar, var, var_name, lt0, lt1):
        out = fsolve(self._invert_helper_without_der, x0=lt0,
                     args=(dens, abar, zbar, var, var_name))
        try:
            assert out.converged
        except AssertionError:
            raise ValueError("Root not converged.")
        return 10 ** out.root

    def eos_invert(self, dens, abar, zbar, var, var_name, der_name=None, t0=None,
                   t1=None):
        if t0 is None:
            t0 = 10 ** (.5 * (self.temp_log_min + self.temp_log_max))
        if t1 is None:
            t1 = 2 * t0
        if der_name is None:
            der_name = _derivatives.get(var_name, None)
        if der_name is None:
            try:
                lt0 = np.log10(t0)
            except AttributeError:
                lt0 = None
            try:
                lt1 = np.log10(t1)
            except AttributeError:
                lt1 = None
            return self._vec_inv_no_d(dens, abar, zbar, var, var_name, lt0, lt1)
        else:
            return self._vec_inv_with_d(dens, abar, zbar, var, var_name, der_name, t0, t1)

    def full_table(self, abar=1.0, zbar=1.0, overwite=False):
        if overwite:
            self._full_table = None
        if self._full_table is not None:
            return self._full_table
        d, t = np.meshgrid(self._d, self._t, indexing='ij')
        self._full_table = self.eos_DT(d, t, abar, zbar)
        return self._full_table

    def plot_var(self, var, log=False, abar=1.0, zbar=1.0, fig=None, ax=None, cb=True,
                 aspect=None, vmin=None, vmax=None, fig_opt=None, dpi=None, figsize=None,
                 popt=None, cmap=None, cbl=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.colors import LogNorm

        ft = self.full_table(abar, zbar)

        if popt is None:
            popt = dict()

        norm = None
        if log:
            norm = LogNorm()

        extent = [self.dens_log_min, self.dens_log_max,
                  self.temp_log_min, self.temp_log_max]

        _popt = {'norm': norm, 'extent': extent, 'aspect': aspect, 'vmin': vmin,
                 'vmax': vmax, 'cmap': cmap}
        _popt.update(popt)

        if fig is None and ax is None:
            _fopt = {'dpi': dpi, 'figsize': figsize}
            if fig_opt is None:
                fig_opt = {}
            _fopt.update(fig_opt)
            fig = plt.figure(**_fopt)
        if ax is not None:
            plt.sca(ax)

        im = plt.imshow(ft[var].T, **_popt)
        ax = plt.gca()

        if cb:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = plt.colorbar(im, cax=cax)
            if cbl is None:
                cbl = var
            if cbl:
                cb.set_label(cbl)


class _DelayedTable(HelmTable):
    """A child class of HelmTable which only loads the tables once needed"""

    def __init__(self, fn=None, temp_n=None, temp_log_min=None, temp_log_max=None,
                 dens_n=None, dens_log_min=None, dens_log_max=None, silent=False):
        self._silent = silent
        if fn is None:
            fn = _default_fn
        self._kwargs = dict(fn=fn, temp_n=temp_n, temp_log_min=temp_log_min,
                            temp_log_max=temp_log_max, dens_n=dens_n,
                            dens_log_min=dens_log_min, dens_log_max=dens_log_max)
        inherit = vars(HelmTable)
        self._loader = HelmTable.__init__
        self.names = [i for i in inherit.keys() if
                      (i not in vars(self)) and (i[0] != '_')]
        self.initB = False
        for attr in self.names:
            val = inherit[attr]
            if hasattr(val, '__call__') or attr in ['eos_interp']:
                setattr(self, attr, self.wrapped_func(attr))

    def load(self):
        if not self.initB:
            for i in self.names:
                delattr(self, i)
            if not self._silent:
                print('Initializing Helmholtz EOS table ' + str(self._kwargs['fn']))
            self._loader(self, **self._kwargs)
            self.initB = True

    def wrapped_func(self, name):
        def out(*args, **kw):
            self.load()
            func = getattr(self, name)
            return func(*args, **kw)

        out.__doc__ = getattr(self, name).__doc__
        return out

    def __getattr__(self, item):
        if item in self._kwargs:
            self.load()
            return self.__getattribute__(item)
        raise AttributeError("'{}' object has no attribute '{}'".format(self.__class__,
                                                                        item))


# quintic hermite polynomial statement functions
# psi0 and its derivatives
def psi0(z):
    return z * z * z * (z * (-6 * z + 15) - 10) + 1


def dpsi0(z):
    return z * z * (z * (-30 * z + 60) - 30)


def ddpsi0(z):
    return z * (z * (-120 * z + 180) - 60)


# psi1 and its derivatives
def psi1(z):
    return z * (z * z * (z * (-3 * z + 8) - 6) + 1)


def dpsi1(z):
    return z * z * (z * (-15 * z + 32) - 18) + 1


def ddpsi1(z):
    return z * (z * (-60 * z + 96) - 36)


# psi2  and its derivatives
def psi2(z):
    return 0.5 * z * z * (z * (z * (-z + 3) - 3) + 1)


def dpsi2(z):
    return 0.5 * z * (z * (z * (-5 * z + 12) - 9) + 2)


def ddpsi2(z):
    return 0.5 * (z * (z * (-20 * z + 36) - 18) + 2)


# cubic hermite polynomial statement functions
# psi0 & derivatives
def xpsi0(z):
    return z * z * (2 * z - 3) + 1.0


def xdpsi0(z):
    return z * (6 * z - 6)


# psi1 & derivatives
def xpsi1(z):
    return z * (z * (z - 2) + 1)


def xdpsi1(z):
    return z * (3 * z - 4) + 1


# biquintic hermite polynomial statement function
def h5(data, w0t, w1t, w2t, w0mt, w1mt, w2mt, w0d, w1d, w2d, w0md, w1md, w2md):
    return data[0] * w0d * w0t + data[1] * w0md * w0t \
           + data[2] * w0d * w0mt + data[3] * w0md * w0mt \
           + data[4] * w0d * w1t + data[5] * w0md * w1t \
           + data[6] * w0d * w1mt + data[7] * w0md * w1mt \
           + data[8] * w0d * w2t + data[9] * w0md * w2t \
           + data[10] * w0d * w2mt + data[11] * w0md * w2mt \
           + data[12] * w1d * w0t + data[13] * w1md * w0t \
           + data[14] * w1d * w0mt + data[15] * w1md * w0mt \
           + data[16] * w2d * w0t + data[17] * w2md * w0t \
           + data[18] * w2d * w0mt + data[19] * w2md * w0mt \
           + data[20] * w1d * w1t + data[21] * w1md * w1t \
           + data[22] * w1d * w1mt + data[23] * w1md * w1mt \
           + data[24] * w2d * w1t + data[25] * w2md * w1t \
           + data[26] * w2d * w1mt + data[27] * w2md * w1mt \
           + data[28] * w1d * w2t + data[29] * w1md * w2t \
           + data[30] * w1d * w2mt + data[31] * w1md * w2mt \
           + data[32] * w2d * w2t + data[33] * w2md * w2t \
           + data[34] * w2d * w2mt + data[35] * w2md * w2mt


# bicubic hermite polynomial statement function
def h3(data, w0t, w1t, w0mt, w1mt, w0d, w1d, w0md, w1md):
    return data[0] * w0d * w0t + data[1] * w0md * w0t \
           + data[2] * w0d * w0mt + data[3] * w0md * w0mt \
           + data[4] * w0d * w1t + data[5] * w0md * w1t \
           + data[6] * w0d * w1mt + data[7] * w0md * w1mt \
           + data[8] * w1d * w0t + data[9] * w1md * w0t \
           + data[10] * w1d * w0mt + data[11] * w1md * w0mt \
           + data[12] * w1d * w1t + data[13] * w1md * w1t \
           + data[14] * w1d * w1mt + data[15] * w1md * w1mt


_translate = {"pres": "ptot", "dpresdt": "dpt", "dpresdd": "dpd", "dpresda": "dpa",
              "dpresdz": "dpz", "ener": "etot", "denerdt": "det", "denerdd": "ded",
              "denerda": "dea", "denerdz": "dez", "entr": "stot", "dentrdt": "dst",
              "dentrdd": "dsd", "dentrda": "dsa", "dentrdz": "dsz", "pgas": "pgas",
              "dpgasdt": "dpgast", "dpgasdd": "dpgasd", "dpgasda": "dpgasa",
              "dpgasdz": "dpgasz", "egas": "egas", "degasdt": "degast",
              "degasdd": "degasd", "degasda": "degasa", "degasdz": "degasz",
              "sgas": "sgas", "dsgasdt": "dsgast", "dsgasdd": "dsgasd",
              "dsgasda": "dsgasa", "dsgasdz": "dsgasz", "prad": "prad",
              "dpraddt": "dpradt", "dpraddd": "dpradd", "dpradda": "dprada",
              "dpraddz": "dpradz", "erad": "erad", "deraddt": "deradt",
              "deraddd": "deradd", "deradda": "derada", "deraddz": "deradz",
              "srad": "srad", "dsraddt": "dsradt", "dsraddd": "dsradd",
              "dsradda": "dsrada", "dsraddz": "dsradz", "pion": "pion",
              "dpiondt": "dpiont", "dpiondd": "dpiond", "dpionda": "dpiona",
              "dpiondz": "dpionz", "eion": "eion", "deiondt": "deiont",
              "deiondd": "deiond", "deionda": "deiona", "deiondz": "deionz",
              "sion": "sion", "dsiondt": "dsiont", "dsiondd": "dsiond",
              "dsionda": "dsiona", "dsiondz": "dsionz", "xni": "xni", "pele": "pele",
              "dpepdt": "dpept", "dpepdd": "dpepd", "dpepda": "dpepa", "dpepdz": "dpepz",
              "eele": "eele", "deepdt": "deept", "deepdd": "deepd", "deepda": "deepa",
              "deepdz": "deepz", "sele": "sele", "dsepdt": "dsept", "dsepdd": "dsepd",
              "dsepda": "dsepa", "dsepdz": "dsepz", "xnem": "xnem", "xnefer": "xne",
              "dxnedt": "dxnet", "dxnedd": "dxned", "dxneda": "dxnea", "dxnedz": "dxnez",
              "zbar": "zeff", "etaele": "etaele", "detadt": "detat", "detadd": "detad",
              "detada": "detaa", "detadz": "detaz", "pcoul": "pcou", "dpcouldt": "dpcout",
              "dpcouldd": "dpcoud", "dpcoulda": "dpcoua", "dpcouldz": "dpcouz",
              "ecoul": "ecou", "decouldt": "decout", "decouldd": "decoud",
              "decoulda": "decoua", "decouldz": "decouz", "scoul": "scou",
              "dscouldt": "dscout", "dscouldd": "dscoud", "dscoulda": "dscoua",
              "dscouldz": "dscouz", "plasg": "plasg", "dse": "dse", "dpe": "dpe",
              "dsp": "dsp", "cv_gas": "cv_gas", "cp_gas": "cp_gas",
              "gam1_gas": "gam1_gas", "gam2_gas": "gam2_gas", "gam3_gas": "gam3_gas",
              "nabad_gas": "nabad_gas", "sound_gas": "cs_gas", "cv": "cv", "cp": "cp",
              "gam1": "gam1", "gam2": "gam2", "gam3": "gam3", "nabad": "nabad",
              "sound": "cs"}

_derivatives = {"etot": "det", "ptot": "dpt"}
