# Add parent directory to path
import os, sys, glob, unittest, shutil, pathlib, contextlib, pytest, datetime

testspath = pathlib.Path(__file__).parent.resolve() / "tests"
sys.path.insert(0, str((testspath / "..").resolve()))

# Import the Network and instantiate
from swotann.nnetwork import NNetwork

# Train
test_model_output_prefix = "trained_"
pretrained_model_path = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "tests", "pretrained")
)
test_files = [str(x) for x in testspath.glob("test*.csv")]
TMP_OUTPUT_NAME = "out.csv"
TMP_OUTPUT_NAME_AVG = "out_average_case.csv"
TMP_OUTPUT_NAME_WORST = "out_worst_case.csv"
TMP_REPORT_NAME = "out.html"

NETWORK_COUNT = 5
EPOCHS = 10
STORAGE_TARGET = 3


def test_run_harness():
    for file in test_files:
        net = NNetwork(NETWORK_COUNT, EPOCHS)
        net.run_swot(file, TMP_OUTPUT_NAME, TMP_REPORT_NAME, STORAGE_TARGET)
        for f in [TMP_OUTPUT_NAME_AVG, TMP_OUTPUT_NAME_WORST, TMP_REPORT_NAME]:
            assert os.path.exists(f)
            output_stat = os.stat(f)
            assert output_stat.st_size > 0
            os.remove(f)


def test_creates_set(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()

    for i in range(len(test_files)):
        nn = NNetwork()
        training_file = test_files[i]
        test_model_output = str(outdir / (test_model_output_prefix + str(i + 1)))
        before = os.listdir(outdir)
        nn.import_data_from_csv(training_file)
        nn.train_SWOT_network(test_model_output)
        after = os.listdir(outdir)
        diff = set(after) - set(before)
        assert len(diff) == 1
        assert diff.pop() == os.path.basename(test_model_output)


pytest.main()