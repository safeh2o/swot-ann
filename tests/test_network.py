# Add parent directory to path
import os, sys, glob, unittest, shutil, pathlib

testspath = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str((testspath / '..').resolve()))

# Import the Network and instantiate
from NNetwork import NNetwork

# Train
test_model_output_prefix = 'trained_'
pretrained_model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), 'pretrained'))
test_files = [str(x) for x in testspath.glob('test*.csv')]


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

def test_import_pretrained_model():
    import sklearn
    nn = NNetwork()
    nn.import_pretrained_model(pretrained_model_path)
    assert nn.predictors_scaler.n_samples_seen_ == 533
    assert nn.targets_scaler.n_samples_seen_ == 533

def test_validations(tmp_path):
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
        nn.set_inputs_for_table(med_temp ,med_cond)

        nn.predict()
        nn.export_results_to_csv(out_csv)
        nn.generate_html_report(out_html)

        after = os.listdir(outdir)

        diff = set(after) - set(before)

        assert len(diff) == 4
        for f in [out_csv, out_diagram, out_html, out_frc]:
            assert os.path.isfile(f)
            assert os.stat(f).st_size > 0
