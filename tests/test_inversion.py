import numpy as np
import helmeos


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


if __name__ == "__main__":
    test_inversion()
