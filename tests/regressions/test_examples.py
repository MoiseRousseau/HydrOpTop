import subprocess
import sys, os
from pathlib import Path
import numpy as np
import pytest

# Disable interactive plotting for test
os.environ["MPLBACKEND"] = "Agg"

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "Permeability_calibration"

ENV = os.environ.copy()
ENV["PYTHONPATH"] = (
    str(Path(__file__).parent.parent.parent)
    + os.pathsep
    + ENV.get("PYTHONPATH", "")
)


def load_output(path):
    """
    Load ASCII output with a header line and numeric data.
    """
    with open(path) as f:
        header = f.readline().strip()
    data = np.loadtxt(path, skiprows=1)
    return header, data


@pytest.mark.parametrize(
    "example",
    [
        "make_calibration_permeability_zonal.py",
        "make_calibration_permeability_free.py",
    ],
)
def test_least_square_regression(example):
    """
    Numerical regression test for least square calibration scripts.
    """

    script = EXAMPLES_DIR / example
    ref_file = EXAMPLES_DIR / example.replace(".py", ".ref")

    assert script.exists()
    assert ref_file.exists()

    output_file = EXAMPLES_DIR / "out.txt"

    # Run example exactly as a user would
    result = subprocess.run(
        [sys.executable, script],
        cwd=EXAMPLES_DIR,
        capture_output=True,
        env=ENV,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_file.exists(), "Example did not create output file"

    # Load reference and new output
    ref_header, ref_data = load_output(ref_file)
    out_header, out_data = load_output(output_file)

    # Header must be identical
    assert out_header == ref_header

    # Numerical regression check
    np.testing.assert_allclose(
        out_data,
        ref_data,
        rtol=1e-10,
        atol=1e-12,
    )
