import os

import pytest
import glob

# Import the Network and instantiate
from swotann.nnetwork import NNetwork

testspath = os.path.dirname(__file__)
# Train
test_model_output_prefix = "trained_"
test_files = glob.glob(os.path.join(testspath, "test*.csv"))
output_prefix = "out"
output_arg1 = f"{output_prefix}.csv"
output_arg2 = f"{output_prefix}.html"
output_names = [
    f"{output_prefix}_worst_case_pm.csv",
    f"{output_prefix}_worst_case_am.csv",
    f"{output_prefix}_average_case_pm.csv",
    f"{output_prefix}_average_case_am.csv",
    f'{output_prefix}_Calibration_Diagnostic_Figs.png',
    f'{output_prefix}_Histograms_Fig.png',
    f'{output_prefix}_Predictions_Fig.png',
    f'{output_prefix}_Risk_Fig.png',
]

NETWORK_COUNT = 5
EPOCHS = 10
STORAGE_TARGET = 3


def test_run_harness():
    for file in test_files:
        net = NNetwork(NETWORK_COUNT, EPOCHS)
        net.run_swot(file, output_arg1, output_arg2, STORAGE_TARGET)
        for f in output_names + [output_arg2]:
            assert os.path.exists(f)
            output_stat = os.stat(f)
            assert output_stat.st_size > 0
            os.remove(f)


pytest.main()
