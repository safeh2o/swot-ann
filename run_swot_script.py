import sys, os
from NNetwork import NNetwork
import pandas as pd
import numpy as np
import datetime

now = datetime.datetime.now()
SWOT_net = NNetwork()
input_file = sys.argv[1]
directory = "model_retraining" + os.sep + now.strftime("%m%d%Y_%H%M%S") + "_" + os.path.basename(input_file)
results_file = sys.argv[3]
report_file = sys.argv[4]

# Uncomment for Excel processing
# file = pd.read_excel(input_file)

file = pd.read_csv(input_file)

# Support from 3 different input templates se1_frc, ts_frc, and ts frc1
if 'se1_frc' in file.columns:
    FRC_IN = 'se1_frc'
    WATTEMP = 'se1_wattemp'
    COND = 'se1_cond'
    FRC_OUT = "se4_frc"
elif 'ts_frc1' in file.columns:
    FRC_IN = 'ts_frc1'
    WATTEMP = 'ts_wattemp'
    COND = 'ts_cond'
    FRC_OUT = "hh_frc1"
elif 'ts_frc' in file.columns:
    FRC_IN = 'ts_frc'
    WATTEMP = 'ts_wattemp'
    COND = 'ts_cond'
    FRC_OUT = "hh_frc"

# Calculate the median water temperature and conductivity values for the output table
med_temp = np.median(file[WATTEMP].dropna().to_numpy())
med_cond = np.median(file[COND].dropna().to_numpy())

SWOT_net.import_data_from_csv(input_file)
SWOT_net.train_SWOT_network(directory)
SWOT_net.set_inputs_for_table(med_temp ,med_cond)
SWOT_net.import_pretrained_model(directory)
SWOT_net.predict()
SWOT_net.display_results()
SWOT_net.export_results_to_csv(results_file)
SWOT_net.generate_html_report(report_file)
