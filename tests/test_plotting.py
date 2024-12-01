from helmeos import default
import pytest


def test_import_error(monkeypatch):
    # Save the original `__import__` method
    original_import = __import__

    # Mock `__import__` to raise ImportError for `matplotlib.pyplot`
    def mock_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("No module named 'matplotlib.pyplot'")
        return original_import(name, *args, **kwargs)

    # Apply the monkeypatch
    monkeypatch.setattr("builtins.__import__", mock_import)

    with pytest.raises(ImportError):
        default.plot_var('cs')


def test_plotting(monkeypatch):
    import matplotlib.pyplot as plt
    default.plot_var('cs')
    plt.close()
    cs = default.full_table()['cs']
    default.plot_var(cs, log=True)
    plt.close()
    ax = plt.subplot()
    default.plot_var('cs', ax=ax)
    plt.close()


if __name__ == "__main__":
    test_import_error(pytest.MonkeyPatch)
    test_plotting()
