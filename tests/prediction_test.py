# Add parent directory to path
import os, sys, glob, unittest, shutil
sys.path.insert(0, os.path.realpath(os.path.join(__file__, '..', '..')))

# Import the Network and instantiate
from NNetwork import NNetwork

# Train 
test_model_output = 'pretrained_1'
test_files = glob.glob('test*.csv')

def test_validations():
    import pandas as pd
    import numpy as np
    import pathlib

    # hard-coded
    FRC_IN = 'ts_frc'
    WATTEMP = 'ts_wattemp'
    COND = 'ts_cond'
    FRC_OUT = 'hh_frc'

    shutil.rmtree('out', ignore_errors=True)
    pathlib.Path('out').mkdir()
            
    for i in range(len(test_files)):
        test_file = test_files[i]
        before = os.listdir('out')

        out_csv = 'out/test_output_%s.csv' % str(i+1)
        out_html = 'out/test_output_report_%s.html' % str(i+1)
        out_jpg = out_html.rstrip('.html') + '.jpg'

        nn = NNetwork()

        nn.import_data_from_csv(test_file)
        nn.import_pretrained_model(test_model_output)

        # hard-coded
        file = pd.read_csv(test_file)
        med_temp = np.median(file[WATTEMP].dropna().to_numpy())
        med_cond = np.median(file[COND].dropna().to_numpy())
        nn.set_inputs_for_table(med_temp ,med_cond)

        nn.predict()
        nn.export_results_to_csv(out_csv)
        nn.generate_html_report(out_html)

        after = os.listdir('out')

        diff = set(after) - set(before)


        assert len(diff) == 3
        assert os.path.isfile(out_csv)
        assert os.path.isfile(out_html)
        assert os.path.isfile(out_jpg)

