from NNetwork import NNetwork

filename = "SWOTAllData.csv"
SWOT_net = NNetwork()

'''The following lines train a network from 
data written a .csv format. The trained network is saved 
in a folder called 'Pretrained Network'.'''
SWOT_net.import_data_from_csv(filename)
SWOT_net.train_SWOT_network("Pretrained Network")