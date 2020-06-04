[![Build Status](https://dev.azure.com/swot-dighr/SWOT/_apis/build/status/dighr.swot-python-analysis?branchName=master)](https://dev.azure.com/swot-dighr/SWOT/_build/latest?definitionId=2&branchName=master)

# swot-python-analysis
Safe Water Optimization Tool ANN Model

The implementation of the network is encapsulated in a single class called
NNetwork. This class has all the necessary methods to create a neural network
and interact with it for training on a given data-set and making predictions 
on user inputs.

#### Version History

v1.1 - Initial version

v1.2 (May 06, 2020) - Updates include:
  1. Modifying NNetwork to accept time either as a number (as formatted in Excel) or the standard Kobo output
  2. Modified the 'run_swot_script' to retrain on the new dataset as a temporary measure until a model retraining protocol is established
  
v1.3 (May 20, 2020) - Updates include:
  1. Modified the data division to split dataset used for training the model into training, validation, and testing (previous versions only split into training and validation). Training results graphs have been updated to reflect this split.
  2. Changed from using a fixed number of epochs for training to using an early stopping procedure based on validation MSE with a patience of 10 epochs. To ensure this works, training results are no longer saved to the self.history and instead the history is reset for each model in the ensemble.
  3. Forcing the model to randomize the weights and biases of each ANN in the ensemble to properly represent a multi-start ensemble.


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

`python run_swot_script.py arg1 arg2 arg3 arg4`

where:
1. arg1: The csv file containing the inputs (ex. SWOTAllData.csv)
2. arg2: The directory containing the pretrained models (ex. untitled_network)
3. arg3: The output csv file (ex. results.csv)
4. arg4: The output HTML report file (ex. results.html)

i.e. `python run_swot_script.py SWOTAllData.csv untitled_network results.csv results.html`

### Requirements

The network uses the following Python libraries, see [setup](#setup):
- keras - a more user-friendly library for building neural networks
- tensorflow - Tensorflow is the ML library
- numpy - adds necessary math functions
- matplotlib - for figure production
- pandas - for data structure management
- sklearn - data science library, contains MinMaxScaler for scaling inputs and outputs, as well as several built in performance metrics
- pillow
- xlrd - for reading excel libraries
- yattag - for HTML reporting
- pytest (optional) - for testing the model

## Setup

Clone or download this repository, then run the following to create a virtual environment:

#### On a Linux or Mac terminal, from the repository folder:
- `make`
- `make dependencies`

#### On Windows, run the following in Command Prompt from the repository folder:
- `python3 -m venv env` to create the virtual environment
- `start env\Scripts\activate.bat` to activate the environment
- `pip install -r requirements.txt` to install the required libraries

Activate the environment before running the [scripts](#scripts):
- `source env/bin/activate` on Linux/Mac
- `start env\Scripts\activate.bat` on Windows
- `python run_swot_scripts.py args`


### Testing

There are five sample datasets included in the `tests` directory. To test the model, [setup the system](#setup), then:
- Linux/Mac: `make test`
- Windows: activate the environment, then run `pytest`
