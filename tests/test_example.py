import io
import sys

EXPECTED = """\
~ Easy call ~
etot: 2.47321e+12, 2.56199e+15,
Inverted temp: 10000, 1e+07,
ptot: 165585, 1.7131e+17,
Inverted temp: 10000, 1e+07,
cs: 1.66154e+06, 5.34186e+07,
Inverted temp: 10000, 1e+07,
sele: 1.14675e+09, 2.88628e+08,
Inverted temp: 10000, 1e+07,

~ Load default table ~
etot: 2.47321e+12, 2.56199e+15,
Inverted temp: 10000, 1e+07,
ptot: 165585, 1.7131e+17,
Inverted temp: 10000, 1e+07,
cs: 1.66154e+06, 5.34186e+07,
Inverted temp: 10000, 1e+07,
sele: 1.14675e+09, 2.88628e+08,
Inverted temp: 10000, 1e+07,

~ Load table by filename ~
etot: 2.47321e+12, 2.56199e+15,
Inverted temp: 10000, 1e+07,
ptot: 165585, 1.7131e+17,
Inverted temp: 10000, 1e+07,
cs: 1.66154e+06, 5.34186e+07,
Inverted temp: 10000, 1e+07,
sele: 1.14675e+09, 2.88628e+08,
Inverted temp: 10000, 1e+07,"""


def test_example():
    old_stdout = sys.stdout  # Save the current stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout  # Redirect stdout to the StringIO object

    try:
        import helmeos.example  # noqa: F401
    finally:
        sys.stdout = old_stdout  # Restore the original stdout

    out = new_stdout.getvalue().strip().splitlines()
    out = [line for line in out
           if not line.startswith("Initializing Helmholtz EOS table")]
    out = "\n".join(out)
    assert out == EXPECTED


if __name__ == "__main__":
    test_example()
