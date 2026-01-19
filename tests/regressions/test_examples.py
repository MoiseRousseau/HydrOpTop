import subprocess
import sys, os
from pathlib import Path
import numpy as np
import pytest

# Disable interactive plotting for test
os.environ["MPLBACKEND"] = "Agg"

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"

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


def run_and_compare(directory, script, output_file, ref_file, 
                    rtol=1e-10, atol=1e-12):
    assert script.exists()
    assert ref_file.exists()

    # Run example exactly as a user would
    result = subprocess.run(
        [sys.executable, script],
        cwd=directory,
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


@pytest.mark.slow
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
    directory = EXAMPLES_DIR / "Permeability_calibration"
    script = directory / example
    output_file = directory / "out.txt"
    ref_file = directory / example.replace(".py", ".ref")
    run_and_compare(directory, script, output_file, ref_file)


#@pytest.mark.slow
@pytest.mark.parametrize(
    "example",
    [
        "make_optimization_drainage.py",
        "make_maximize_area_drawdown.py"
    ],
)
def test_constrained_optimization(example):
    """
    Regression test for constrained optimization
    """
    directory = EXAMPLES_DIR / "miscellaneous"
    script = directory / example
    output_file = directory / "out.txt"
    ref_file = directory / example.replace(".py", ".ref")
    run_and_compare(directory, script, output_file, ref_file, rtol=3e-3)


#@pytest.mark.slow
@pytest.mark.parametrize(
    "example",
    [
        "make_pervious_surround_2D.py",
        #"make_pervious_surround_3D.py",
    ],
)
def test_pervious_surround(example):
    """
    Regression test for classical linear elasticity benchmark
    """
    directory = EXAMPLES_DIR / "pervious_surround_optimization"
    script = directory / example
    output_file = directory / "out.txt"
    ref_file = directory / example.replace(".py", ".ref")
    run_and_compare(directory, script, output_file, ref_file, rtol=3e-3)


#@pytest.mark.slow
@pytest.mark.parametrize(
    "example",
    [
        "make_cantilever_simple.py",
        "make_cantilever_min_volume.py",
    ],
)
def test_cantilever_benchmark(example):
    """
    Regression test for classical linear elasticity benchmark
    """
    directory = EXAMPLES_DIR / "linear_elasticity"
    script = directory / example
    output_file = directory / "out.txt"
    ref_file = directory / example.replace(".py", ".ref")
    run_and_compare(directory, script, output_file, ref_file, rtol=3e-3)