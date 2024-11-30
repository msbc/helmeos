import numpy as np
import helmeos
import pytest


def test_inversion(tol=2e-10, vars_to_print=None):
    vars_to_print = vars_to_print or ['etot', 'ptot', 'cs', 'sele']

    # defining inputs
    density = np.array([1e-7, 1e2])  # g/cc
    temperature = np.array([1e4, 1e7])  # K
    abar = 1.0
    zbar = 1.0

    out = helmeos.eos_DT(density, temperature, abar, zbar)
    for i in vars_to_print:
        inv = helmeos.eos_invert(density, abar, zbar, out[i], i)
        diff = np.abs(inv / temperature - 1)
        assert diff.max() <= tol, f"Test failed for {i}. Relative error: {diff.max()}"
        print(f"Test passed for {i}. Max relative error: {diff.max()}")

    # test inversion error handling and bracketing
    dens = 1e-7
    var = 165585  # ptot for dens=1e-7, temp=1e4
    var_name = "ptot"
    der_name = "none"
    args = (dens, abar, zbar, var, var_name, der_name)
    with pytest.raises(ValueError, match="Did not converge."):
        helmeos.default._adaptive_root_find(1e7, args=args, tol=1e-14, maxiter=3)
    with pytest.raises(ValueError, match="Bracket must be strictly monotonically increasing."):
        helmeos.default._adaptive_root_find(1e7, args=args, bracket=(1e7, 1e7))
    # test initial bracketing correction
    with pytest.warns(RuntimeWarning, match="invalid value encountered in scalar divide"):
        temp = helmeos.default._adaptive_root_find(9e3, args=args, bracket=(1e3, 1e5))
    assert np.isclose(temp, 1e4, rtol=1e-5), f"Test failed for {var_name}. Temp: {temp}"


if __name__ == "__main__":
    test_inversion()
