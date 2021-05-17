# Add parent directory to path
import os, sys, glob, unittest, shutil, pathlib, contextlib,pytest,datetime

testspath = pathlib.Path(__file__).parent.resolve()/'tests'
sys.path.insert(0, str((testspath / '..').resolve()))

# Import the Network and instantiate
from NNetwork import NNetwork

# Train
test_model_output_prefix = 'trained_'
pretrained_model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), 'tests','pretrained'))
test_files = [str(x) for x in testspath.glob('test*.csv')]
TMP_OUTPUT_NAME='out.csv'
TMP_OUTPUT_NAME_AVG = 'out_average_case.csv'
TMP_OUTPUT_NAME_WORST = 'out_worst_case.csv'
TMP_REPORT_NAME = 'out.html'

def test_run_harness():
    import run_swot_script

    @contextlib.contextmanager
    def run_swot(filename):
        now = datetime.datetime.now()
        sys._argv = sys.argv[:]
        sys.argv = [sys.argv[0], filename, '', TMP_OUTPUT_NAME, TMP_REPORT_NAME,15]
        yield
        sys.argv = sys._argv

    for file in test_files:
        with run_swot(file):
            run_swot_script.run_swot()
            '''for f in [TMP_OUTPUT_NAME_AVG,TMP_OUTPUT_NAME_WORST, TMP_REPORT_NAME]:
                assert os.path.exists(f)
                output_stat = os.stat(f)
                assert output_stat.st_size > 0
                os.remove(f)'''
def test_creates_set(tmp_path):
    outdir = tmp_path / 'out'
    outdir.mkdir()

    for i in range(len(test_files)):
        nn = NNetwork()
        training_file = test_files[i]
        test_model_output = str(outdir / (test_model_output_prefix + str(i+1)))
        before = os.listdir(outdir)
        nn.import_data_from_csv(training_file)
        nn.train_SWOT_network(test_model_output)
        after = os.listdir(outdir)
        diff = set(after) - set(before)
        assert len(diff) == 1
        assert diff.pop() == os.path.basename(test_model_output)

'''def test_validations(tmp_path):#This one is not needed because it uses the pre-trained models which we should not be doing
    import pandas as pd
    import numpy as np

    outdir = tmp_path / 'out'
    outdir.mkdir()

    # hard-coded
    FRC_IN = 'ts_frc'
    WATTEMP = 'ts_wattemp'
    COND = 'ts_cond'
    FRC_OUT = 'hh_frc'

    for i in range(len(test_files)):
        test_file = test_files[i]
        before = os.listdir(outdir)

        out_csv = os.path.join(outdir, 'test_output_%s.csv' % str(i+1))
        out_csv_avg=os.path.join(outdir, 'test_output_%s_average_case.csv' % str(i+1))
        out_csv_worst=os.path.join(outdir, 'test_output_%s_worst_case.csv' % str(i+1))

        out_html = os.path.join(outdir, 'test_output_report_%s.html' % str(i+1))
        out_diagram = out_html.rstrip('.html') + '.png'
        out_frc = out_html.rstrip('.html') + '-frc.jpg'

        nn = NNetwork()

        nn.import_data_from_csv(test_file)
        nn.import_pretrained_model(pretrained_model_path)

        # hard-coded
        file = pd.read_csv(test_file)
        med_temp = np.median(file[WATTEMP].dropna().to_numpy())
        med_cond = np.median(file[COND].dropna().to_numpy())
        nn.set_inputs_for_table()

        nn.predict()
        nn.export_results_to_csv(out_csv)
        nn.generate_html_report(out_html)

        after = os.listdir(outdir)

        diff = set(after) - set(before)

        assert len(diff) == 4
        for f in [out_csv_avg,out_csv_worst, out_diagram, out_html]:
            assert os.path.isfile(f)
            assert os.stat(f).st_size > 0'''
pytest.main()