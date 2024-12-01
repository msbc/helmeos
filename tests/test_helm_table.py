"""Test accuracy of code"""
import numpy as np
from helmeos import HelmTable
from helmeos.helm_table import OldStyleInputs, _translate, _DelayedTable
import helmeos.table_param as tab
try:
    import helmholtz
except ImportError:
    print('Python package "helmholtz" not found.')
    print('See https://github.com/jschwab/python-helmholtz for code download.')
    raise


def test_table_values(nrand=100, vars_to_test=None, silent=False, err_dict=None, tol=1e-14):
    if err_dict is None:
        err_dict = {'etot': 4e-15, 'ptot': 6e-16, 'cs': 3e-13, 'sele': 0}
    if vars_to_test is None:
        vars_to_test = ['etot', 'ptot', 'cs', 'sele']
    var_len = max(len(i) for i in vars_to_test)
    fmt = f"{{:{var_len}s}}, {{:}}"

    ht = HelmTable()

    np.random.seed(0)  # for reproducibility
    # density
    dlo, dhi = tab.dens_log_min, tab.dens_log_max
    dens = np.arange(dlo, dhi + 1)
    dens = 10**np.concatenate((dens,  (dhi - dlo) * np.random.random(nrand) + dlo))
    # temperature
    tlo, thi = tab.temp_log_min, tab.temp_log_max
    temp = np.arange(tlo, thi + 1)
    temp = 10**np.concatenate((temp,  (thi - tlo) * np.random.random(nrand) + tlo))

    rho_in = np.zeros(dens.size * temp.size)
    temp_in = np.zeros(dens.size * temp.size)

    # make 1D array of all combinations of dens, temp
    ind = 0
    for i in dens:
        for j in temp:
            rho_in[ind] = i
            temp_in[ind] = j
            ind += 1

    print("Loading tables.")
    # our method
    ours = ht.eos_DT(rho_in, temp_in, 1.0, 1.0)
    # their method
    theirs = helmholtz.helmeos(rho_in, temp_in, 1.0, 1.0)

    if not silent:
        print("Showing max relative error for a few variables.")
        print(fmt.format("var", "rel_err"))

    for var in vars_to_test:
        a, b = ours[var], getattr(theirs, var)
        dif = np.abs((a - b) / b)
        i = dif.argmax()
        mytol = err_dict.get(var, tol)
        assert dif[i] <= mytol, f"Test failed for {var}. Max relative error: {dif[i]}"
        print(fmt.format(var, dif[i]))


def test_table_features():
    ht = HelmTable()
    ot = OldStyleInputs()
    dt = _DelayedTable()
    assert dt.fn == ht.fn
    h_data = ht.full_table(overwite=True)
    o_data = ot.full_table()
    assert h_data.keys() == o_data.keys()
    for key in h_data:
        assert np.all(h_data[key] == o_data[key])
    # test cache
    h_data = ht.full_table()
    for key in h_data:
        assert np.all(h_data[key] == o_data[key])
    data = ot.eos_DT(1e-7, 1e4, 1.0, 1.0)
    assert ht.eos_DT(1e-7, 1e4, 1.0, 1.0) == data
    assert dt.eos_DT(1e-7, 1e4, 1.0, 1.0) == data
    for key, val in _translate.items():
        assert ot.eos_DT(1e-7, 1e4, 1.0, 1.0, outvar=key)[key] == data[val]
        assert ot.eos_DT(1e-7, 1e4, 1.0, 1.0, outvar=val)[val] == data[val]


if __name__ == "__main__":
    test_table_values()
    test_table_features()
