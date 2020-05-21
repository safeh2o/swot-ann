# Add parent directory to path
import os, sys, glob, unittest, shutil, pathlib
sys.path.insert(0, os.path.realpath(os.path.join(__file__, '..', '..')))

# Import the Network and instantiate
from NNetwork import NNetwork

os.chdir(os.path.realpath(os.path.dirname(__file__)))

# Train
outdir = 'out'
test_model_output_prefix = os.path.join(outdir, 'trained_')
pretrained_model_path = 'pretrained'
test_files = glob.glob('test*.csv')

def test_creates_set():
    for i in range(len(test_files)):
        nn = NNetwork()
        training_file = test_files[i]
        test_model_output = test_model_output_prefix + str(i+1)
        nn = NNetwork()
        shutil.rmtree(outdir, ignore_errors=True)
        pathlib.Path(outdir).mkdir()
        before = os.listdir(outdir)
        nn.import_data_from_csv(training_file)
        nn.train_SWOT_network(test_model_output)
        after = os.listdir(outdir)
        diff = set(after) - set(before)
        assert len(diff) == 1
        assert diff.pop() == test_model_output

def test_import_pretrained_model():
    import sklearn
    nn = NNetwork()
    nn.import_pretrained_model(pretrained_model_path)
    assert len(nn.pretrained_networks) == 100
    assert isinstance(nn.predictors_scaler, sklearn.preprocessing.MinMaxScaler)
    assert isinstance(nn.outputs_scaler, sklearn.preprocessing.MinMaxScaler)
    assert nn.predictors_scaler.n_samples_seen_ == 385
    assert nn.outputs_scaler.n_samples_seen_ == 385

def test_validations():
    import pandas as pd
    import numpy as np
    import pathlib

    # hard-coded
    FRC_IN = 'ts_frc'
    WATTEMP = 'ts_wattemp'
    COND = 'ts_cond'
    FRC_OUT = 'hh_frc'

    shutil.rmtree(outdir, ignore_errors=True)
    pathlib.Path(outdir).mkdir()

    for i in range(len(test_files)):
        test_file = test_files[i]
        before = os.listdir(outdir)

        out_csv = os.path.join(outdir, 'test_output_%s.csv' % str(i+1))
        out_html = os.path.join(outdir, 'test_output_report_%s.html' % str(i+1))
        out_jpg = out_html.rstrip('.html') + '.jpg'

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


        assert len(diff) == 3
        assert os.path.isfile(out_csv)
        assert os.stat(out_csv).st_size() > 0
        assert os.path.isfile(out_jpg)
        assert os.stat(out_jpg).st_size() > 0
        assert os.path.isfile(out_html)
        assert os.stat(out_html).st_size() > 0
