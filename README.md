# swot-webapp
Repo for the Safe Water Optimization Tool Web Application 

The implementation of the network is encapsulated in a single class called
NNetwork. This class has all the necessary methods to create a neural network
and interact with it for training on a given data-set and making predictions 
on user inputs.

#### Training Workflow

To train the swot network, instantiate a `NNetwork` object and call the 
`import_data_from_csv(filename)` method to feed the training parameters
into the network. After this the network is ready to get trained.

The NNetwork class has two different methods for training: `train_network()`
and `train_SWOT_network(directory)`. The first one is used for network performance testing,
and it trains a single network only. The SWOT model uses 100 networks to make 
reliable predictions. For that reason, the `train_network()` method should not be
used except in cases of network performance testing. Instead, use the 
`train_SWOT_network(directory)` as the training method. After the training
method is executed, the pre-trained network will be saved on the specified 
directory and it will contain 3 items:
1. An **arcitecture.json** file containing the structure of the Neural Network
2. A **scalers.save** file containing the saved scaling parameters of the input and output datasets
3. A **network_weights** directory including 100 files with the saved weights of each network after training

#### Using the Pre-trained Networks

Once a train network exists in a directory, it can be loaded through an empty instance of the NNetwork class.
To load the desired network use `import_pretrained_model(directory)` with `directory` being the
directory of the saved, pre-trained network.

After a pre-trained model is successfully imported, the inputs that will be used to make predictions,
must be specified. To import input data from a .csv file use `import_data_from_csv(filename)`, specifying
the name of the csv file containing the inputs, with the parameter `filename`.

### Scripts

#### run_swot_script

This script makes predictions on inputs from a csv file, using a pretrained SWOT network and 
outputs the results in the console and saves the results in a csv file. This script is called by
the command line. The script can be called using the following command:

`python run_swot_script arg1 arg2 arg3`

where:
1. arg1: The csv file containing the inputs (ex. SWOTAllData.csv)
2. arg2: The directory containing the pretrained models (ex. untitled_network)
3. arg3: The output csv file (ex. results.csv)

i.e. `python run_swot_script SWOTAllData.csv untitled_network results.csv`
`

#### Libraries

Install the following libraries for the Network to work:
1. keras
2. Tensorflow
3. numpy
4. matplotlib
5. pandas
6. sklearn
7. Pillow
8. xlrd

