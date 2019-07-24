import sys
from NNetwork import NNetwork

SWOT_net = NNetwork()
input_file = sys.argv[1]
pretrained_network_dir = sys.argv [2]
results_file = sys.argv[3]

'''The following lines train a network from 
data written a .csv format. The trained network is saved 
in a folder called 'Pretrained Network'.'''
SWOT_net.import_data_from_csv(input_file)
SWOT_net.import_pretrained_model(pretrained_network_dir)
SWOT_net.predict()
SWOT_net.display_results()
SWOT_net.export_results_to_csv(results_file)