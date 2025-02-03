import glob
import os

import pytest

# Import the Network and instantiate
from swotann.swot_ml import SWOT_ML

testspath = os.path.dirname(__file__)
# Train
test_model_output_prefix = "trained_"
test_files = glob.glob(os.path.join(testspath, "test*.csv"))
output_prefix = "out"
output_arg1 = f"{output_prefix}.csv"
output_names = [
    f"{output_prefix}_calibration_scores.csv",
    f"{output_prefix}_full_prediction_results.csv",
    f"{output_prefix}_max_predicted_safety.csv",
    f"{output_prefix}_min_predicted_safety.csv",
]

STORAGE_TARGET = 3


def test_run_harness():
    for file in test_files:
        net = SWOT_ML()
        net.run_swot(file, output_arg1, STORAGE_TARGET)
        for f in output_names:
            assert os.path.exists(f)
            output_stat = os.stat(f)
            assert output_stat.st_size > 0
            os.remove(f)


pytest.main()
