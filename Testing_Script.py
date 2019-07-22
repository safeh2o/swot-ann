from NNetwork import NNetwork

filename = "SWOTAllData.csv"
SWOT_net = NNetwork()

'''The following lines train a network from 
data provided by a .csv file. The network is saved 
in a folder called 'Pretrained Network'.'''
SWOT_net.import_data_from_csv(filename)
SWOT_net.train_SWOT_network("Pretrained Network")

'''The following lines load a pretrained network
from the specified directory ('Network_dir_Name') and make a prediction based 
on the loaded data. It then later saves the results to a csv file'''
SWOT_net.import_data_from_csv(filename)
SWOT_net.import_pretrained_model("Network_Dir_Name")
SWOT_net.predict()
SWOT_net.export_results_to_csv("Saved Results")
