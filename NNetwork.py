import joblib
import time
import os
import base64
import io
from yattag import Doc
import datetime
from datetime import timedelta
from xlrd.xldate import xldate_as_datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from pandas import ExcelWriter

from mpl_toolkits.mplot3d import Axes3D

'''
TF_CPP_MIN_LOG_LEVEL:
Defaults to 0, so all logs are shown. Set TF_CPP_MIN_LOG_LEVEL to 1 to filter out INFO logs, 2 to additionally filter out WARNING, 3 to additionally filter out ERROR.
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow import keras

class NNetwork:

    def __init__(self):

        self.model = None
        self.pretrained_networks = []

        self.software_version = "1.3"
        self.input_filename = None
        self.today = str(datetime.date.today())
        self.avg_time_elapsed = timedelta(seconds=0)

        self.predictors_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.targets_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.history = None
        self.file = None

        self.layer1_neurons = 5
        self.epochs =1000

        self.predictors = None

        self.targets = None
        self.predictions = None
        self.results = None

        self.total_mse = []
        self.total_rsquared = []
        self.total_mse_val = []
        self.total_rsquared_val = []

        self.avg_mse_train = 0
        self.avg_rsq_train = 0
        self.avg_mse_val = 0
        self.avg_rsq_val = 0
        self.avg_mse_test = 0
        self.avg_rsq_test = 0

        self.optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999)
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.layer1_neurons, input_dim=3, activation="tanh"))
        self.model.add(keras.layers.Dense(1, activation="linear"))
        self.model.compile(loss='mse', optimizer=self.optimizer, metrics=['mse'])

    def import_data_from_excel(self, filename, sheet=0):
        """
        Imports data to the network by an excel file.

        Load data to a netwokr that are stored in a Microsoft Excel file format.
        The data loaded from this method can be used both for training reasons as
        well as to make predictions.

        :param filename: String containing the filename of the excel file containing the input data (e.g "input_data.xlsx")
        :param sheet: The number of the sheet in the .xlsx file that the data are stored in.
        """

        df = pd.read_excel(filename, sheet_name=sheet)

        # Uncomment the following line to filter out samples that are in shade
        # Change 1 to 0 to filter out samples that are exposed to sunlight
        # df = df.drop(df[df['sb_sunshade'] == 1].index)

        # Uncomment the next line to filter samples with input frc > 2 mg/L
        # df = df.drop(df[df["se1_frc"] > 2].index)

        self.file = df

        global FRC_IN
        global FRC_OUT
        global WATTEMP
        global COND

        # Locate the fields used as inputs/predictors and outputs in the loaded file
        # and split them

        if 'se1_frc' in self.file.columns:
            FRC_IN = 'se1_frc'
            WATTEMP = 'se1_wattemp'
            COND = 'se1_cond'
            FRC_OUT = "se4_frc"
        elif 'ts_frc1' in self.file.columns:
            FRC_IN = 'ts_frc1'
            WATTEMP = 'ts_wattemp'
            COND = 'ts_cond'
            FRC_OUT = "hh_frc1"
        elif 'ts_frc' in self.file.columns:
            FRC_IN = 'ts_frc'
            WATTEMP = 'ts_wattemp'
            COND = 'ts_cond'
            FRC_OUT = "hh_frc"

        self.file.dropna(subset=[FRC_IN], how='all', inplace=True)
        self.file.dropna(subset=[FRC_OUT], how='all', inplace=True)
        self.file.dropna(subset=['ts_datetime'], how='all', inplace=True)
        self.file.dropna(subset=['hh_datetime'], how='all', inplace=True)
        self.file.reset_index(drop=True, inplace=True)  # fix dropped indices in pandas

        self.median_wattemp = np.median(self.file[WATTEMP].dropna().to_numpy())
        self.median_cond = np.median(self.file[COND].dropna().to_numpy())

        start_date = self.file["ts_datetime"]
        end_date = self.file["hh_datetime"]

        durations = []
        if start_date.dtype == 'float64':  # Excel type
            for i in range(len(start_date)):
                start = xldate_as_datetime(start_date[i], datemode=0)
                end = xldate_as_datetime(end_date[i], datemode=0)
                durations.append(end - start)

        else:  # kobo type
            for i in range(len(start_date)):
                temp_sta = start_date[i][:16]
                temp_end = end_date[i][:16]
                start = datetime.datetime.strptime(temp_sta, "%Y-%m-%dT%H:%M")
                end = datetime.datetime.strptime(temp_end, "%Y-%m-%dT%H:%M")
                durations.append(end - start)

        sumdeltas = timedelta(seconds=0)
        i = 1
        while i < len(durations):
            sumdeltas += abs(durations[i] - durations[i - 1])
            i = i + 1

        self.avg_time_elapsed = sumdeltas / (len(durations) - 1)

        # Extract the column of dates for all data and put them in YYYY-MM-DD format
        all_dates = []
        for row in self.file.index:
            all_dates.append(str(self.file.loc[row, 'ts_datetime'])[0:10])
        self.file['formatted_date'] = all_dates

        # Locate the rows of the missing data
        nan_rows_watt = self.file.loc[self.file[WATTEMP].isnull()]
        nan_rows_cond = self.file.loc[self.file[COND].isnull()]

        # For every row of the missing data find the rows on the same day
        for i in nan_rows_watt.index:
            today = self.file.loc[i, 'formatted_date']
            same_days = self.file[self.file['formatted_date'] == today]
            temps = same_days[WATTEMP].dropna().to_numpy()
        avg_daily_temp = np.mean(temps)

        for i in nan_rows_cond.index:
            today = self.file.loc[i, 'formatted_date']
            same_days = self.file[self.file['formatted_date'] == today]
            conds = same_days[COND].dropna().to_numpy()
        avg_daily_cond = np.mean(conds)

        self.file[WATTEMP] = self.file[WATTEMP].fillna(value=avg_daily_temp)
        self.file[COND] = self.file[COND].fillna(value=avg_daily_cond)

        # From these rows get the temperatures and avg them

        # self.file.dropna(subset=[WATTEMP], how='all', inplace=True)
        self.file.dropna(subset=[FRC_OUT], how='all', inplace=True)
        self.predictors = self.file.loc[:, [FRC_IN, WATTEMP, COND]]
        self.datainputs = self.predictors
        self.targets = self.file.loc[:, FRC_OUT].values.reshape(-1, 1)

        self.input_filename = filename

    def import_data_from_csv(self, filename):
        """
                Imports data to the network by a comma-separated values (CSV) file.

                Load data to a network that are stored in .csv file format.
                The data loaded from this method can be used both for training reasons as
                well as to make predictions.

                :param filename: String containing the filename of the .csv file containing the input data (e.g "input_data.xlsx")
        """

        df = pd.read_csv(filename)
        self.file = df

        global FRC_IN
        global FRC_OUT
        global WATTEMP
        global COND

        # Locate the fields used as inputs/predictors and outputs in the loaded file
        # and split them

        if 'se1_frc' in self.file.columns:
            FRC_IN = 'se1_frc'
            WATTEMP = 'se1_wattemp'
            COND = 'se1_cond'
            FRC_OUT = "se4_frc"
        elif 'ts_frc1' in self.file.columns:
            FRC_IN = 'ts_frc1'
            WATTEMP = 'ts_wattemp'
            COND = 'ts_cond'
            FRC_OUT = "hh_frc1"
        elif 'ts_frc' in self.file.columns:
            FRC_IN = 'ts_frc'
            WATTEMP = 'ts_wattemp'
            COND = 'ts_cond'
            FRC_OUT = "hh_frc"

        self.file.dropna(subset=[FRC_IN], how='all', inplace=True)
        self.file.dropna(subset=[FRC_OUT], how='all', inplace=True)
        self.file.dropna(subset=['ts_datetime'], how='all', inplace=True)
        self.file.dropna(subset=['hh_datetime'], how='all', inplace=True)
        self.file.reset_index(drop=True, inplace=True)  # fix dropped indices in pandas
        self.median_wattemp = np.median(self.file[WATTEMP].dropna().to_numpy())
        self.median_cond = np.median(self.file[COND].dropna().to_numpy())

        start_date = self.file["ts_datetime"]
        end_date = self.file["hh_datetime"]

        durations = []
        dateformat = "%Y-%m-%dT%H:%M"

        for i in range(len(start_date)):
            try:
                # kobo type
                start = float(start_date[i])
                end = float(end_date[i])
                start = xldate_as_datetime(start, datemode=0)
                end = xldate_as_datetime(end, datemode=0)

            except ValueError:
                # excel type
                start = start_date[i][:16]
                end = end_date[i][:16]
                start = datetime.datetime.strptime(start, dateformat)
                end = datetime.datetime.strptime(end, dateformat)

            durations.append(end-start)

        sumdeltas = timedelta(seconds=0)

        for i in range (0,len(durations)):
            sumdeltas += abs(durations[i])

        self.avg_time_elapsed = sumdeltas / (len(durations) - 1)

        # Extract the column of dates for all data and put them in YYYY-MM-DD format
        all_dates = []
        for row in self.file.index:
            all_dates.append(str(self.file.loc[row, 'ts_datetime'])[0:10])
        self.file['formatted_date'] = all_dates

        # Locate the rows of the missing data
        nan_rows_watt = self.file.loc[self.file[WATTEMP].isnull()]
        nan_rows_cond = self.file.loc[self.file[COND].isnull()]

        temps = []
        # For every row of the missing data find the rows on the same day
        for i in nan_rows_watt.index:
            today = self.file.loc[i, 'formatted_date']
            same_days = self.file[self.file['formatted_date'] == today]
            temps = same_days[WATTEMP].dropna().to_numpy()
        avg_daily_temp = np.mean(temps)
        conds = []
        for i in nan_rows_cond.index:
            today = self.file.loc[i, 'formatted_date']
            same_days = self.file[self.file['formatted_date'] == today]
            conds = same_days[COND].dropna().to_numpy()
        avg_daily_cond = np.mean(conds)

        self.file[WATTEMP] = self.file[WATTEMP].fillna(value=avg_daily_temp)
        self.file[COND] = self.file[COND].fillna(value=avg_daily_cond)

        # From these rows get the temperatures and avg them

        # self.file.dropna(subset=[WATTEMP], how='all', inplace=True)

        self.predictors = self.file.loc[:, [FRC_IN, WATTEMP, COND]]
        self.datainputs = self.predictors
        self.targets = self.file.loc[:, FRC_OUT].values.reshape(-1, 1)

        self.input_filename = filename
    def train_SWOT_network(self, directory):
        """Train the set of 100 neural networks on SWOT data

        Trains an ensemble of 100 neural networks on se1_frc, water temperature,
        water conductivity."""

        self.predictors_scaler = self.predictors_scaler.fit(self.predictors)
        self.targets_scaler = self.targets_scaler.fit(self.targets)

        self.total_mse_train = []
        self.total_rsquared_train = []
        self.total_mse_val = []
        self.total_rsquared_val = []
        self.total_mse_test = []
        self.total_rsquared_test = []

        x=self.predictors
        t=self.targets

        if not os.path.exists(directory):
            os.mkdir(directory)

        if not os.path.exists(directory + os.sep + 'network_weights'):
            os.mkdir(directory + os.sep + 'network_weights')

        model_json = self.model.to_json()
        with open(directory + os.sep + "architecture.json", 'w') as json_file:
            json_file.write(model_json)

        json_file.close()

        for i in range(0, 100):
            print('Training Network ' + str(i))
            self.train_network(x, t)
            self.model.save_weights(directory + os.sep + "network_weights" + os.sep + "network" + str(i) + ".h5")

        self.avg_mse_train = np.median(np.array(self.total_mse_train))
        self.avg_rsq_train = np.median(np.array(self.total_rsquared_train))
        self.avg_mse_val = np.median(np.array(self.total_mse_val))
        self.avg_rsq_val = np.median(np.array(self.total_rsquared_val))
        self.avg_mse_test = np.median(np.array(self.total_mse_test))
        self.avg_rsq_test = np.median(np.array(self.total_rsquared_test))

        scaler_filename = "scaler.save"
        scalers = {"input": self.predictors_scaler, "output": self.targets_scaler}
        joblib.dump(scalers, directory + os.sep + scaler_filename)
        print("Model Saved!")

    def train_network(self, x, t):
        """
        Trains a single Neural Network on imported data.

        This method trains Neural Network on data that have previously been imported
        to the network using the import_data_from_csv() or import_data_from_excel() methods.
        The network used is a Multilayer Perceptron (MLP). Input and Output data are
        normalized using MinMax Normalization.

        The input dataset is split in training and validation datasets, where 80% of the inputs
        are the training dataset and 20% is the validation dataset.

        The training history is stored in a variable called self.history (see keras documentation:
        keras.model.history object)

        Performance metrics are calculated and stored for evaluating the network performance.
        """
        early_stopping_monitor=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10,
                                      restore_best_weights=True)

        x_norm=self.predictors_scaler.transform(x)
        t_norm=self.targets_scaler.transform(t)

        x_norm_train, xx, t_norm_train, tt = train_test_split(x_norm, t_norm, test_size=0.5, shuffle=True)
        x_norm_val, x_norm_test, t_norm_val, t_norm_test, = train_test_split(xx, tt, test_size=0.5, shuffle=True)
        new_weights = [np.random.uniform(-0.05, 0.05, w.shape) for w in self.model.get_weights()]
        self.model.set_weights(new_weights)
        self.model.fit(x_norm_train, t_norm_train, epochs=1000, validation_data=(x_norm_val, t_norm_val),
                  callbacks=[early_stopping_monitor], verbose=0)

        # Get real data for training
        y_norm_train = self.model.predict(x_norm_train)

        # Get real data for validation
        y_norm_val = self.model.predict(x_norm_val)

        y_norm_test = self.model.predict(x_norm_test)

        rsquared_train = r2_score(self.targets_scaler.inverse_transform(t_norm_train), self.targets_scaler.inverse_transform(y_norm_train))
        rsquared_val=r2_score(self.targets_scaler.inverse_transform(t_norm_val), self.targets_scaler.inverse_transform(y_norm_val))
        rsquared_test = r2_score(self.targets_scaler.inverse_transform(t_norm_test), self.targets_scaler.inverse_transform(y_norm_test))

        mse_train = mean_squared_error(self.targets_scaler.inverse_transform(t_norm_train), self.targets_scaler.inverse_transform(y_norm_train))
        mse_val = mean_squared_error(self.targets_scaler.inverse_transform(t_norm_val), self.targets_scaler.inverse_transform(y_norm_val))
        mse_test = mean_squared_error(self.targets_scaler.inverse_transform(t_norm_test), self.targets_scaler.inverse_transform(y_norm_test))

        self.total_mse_train.append(mse_train)
        self.total_mse_val.append(mse_val)
        self.total_mse_test.append(mse_test)
        self.total_rsquared_train.append(rsquared_train)
        self.total_rsquared_val.append(rsquared_val)
        self.total_rsquared_test.append(rsquared_test)


    def import_pretrained_model(self, directory):
        """Loads a pretrained SWOT neural Network.

        Loads a neural network that has previously been trained and saved by
        the SWOT NN. The saved networks are stored as directories.

        Args:
            directory: The name of the directory where the pretrained network exists.
                The directory name can be specified as a relative or an absolute path.
        """

        # Look in the pretrained-net directory for the JSON file that contains
        # the NN architecture and load it.
        json_architecture = open(directory + os.sep + 'architecture.json', 'r')
        network_architecture = json_architecture.read()
        json_architecture.close()

        # Load all the pretrained networks and store them in an array
        # called self.pretrained_networks (see __init__).
        pretrained_networks = os.listdir(directory + os.sep + "network_weights")
        for network in pretrained_networks:
            temp = keras.models.model_from_json(network_architecture)
            temp.load_weights(directory + os.sep + 'network_weights' + os.sep + network)
            self.pretrained_networks.append(temp)
            print(network + "loaded")
            del temp

        # Load the scalers used for normalizing the data before training
        # the NN (see train_SWOT_network()).
        scalers = joblib.load(directory + os.sep + "scaler.save")
        self.predictors_scaler = scalers["input"]
        self.targets_scaler = scalers["output"]

    def predict(self):
        """
        Make predictions on loaded data.

        This method makes predictions on data loaded to the network by the import_data_from_excel/csv() methods.
        To make the predictions, a pretrained model must be loaded using the import_pretrained_model() method.
        The SWOT ANN uses an ensemble of 100 ANNs. All of the 100 ANNs make a prediction on the inputs and the results are
        stored. The median of all the 100 predictions is calculated and stored here.

        The method also calculates the probabilities of the target FRC levels to be less than 0.2, 0.25 and 0.3 mg/L respectively.

        The predictions are target FRC values in  mg/L, and the probability values range from 0 to 1.

        All of the above results are saved in the self.results class field.
        """

        # Initialize empty arrays for the probabilities to be appended in.
        prob_02 = []
        prob_025 = []
        prob_03 = []

        results = {}

        # Normalize the inputs using the input scaler loaded
        input_scaler = self.predictors_scaler
        inputs_norm = input_scaler.transform(self.predictors)

        # Iterate through all 100 pretrained networks, make predictions based on the inputs,
        # calculate the median of the predictions and store everything to self.results
        for j, network in enumerate(self.pretrained_networks):
            key = "se4_frc_net-" + str(j)
            y_norm = network.predict(inputs_norm)
            predictions = self.targets_scaler.inverse_transform(y_norm).tolist()
            temp = sum(predictions, [])
            results.update({key: temp})
        self.results = pd.DataFrame(results)
        self.results["median"] = self.results.median(axis=1)

        # Include the inputs/predictors in the self.results variable
        for i in self.predictors.keys():
            self.results.update({i: self.predictors[i].tolist()})
            self.results[i] = self.predictors[i].tolist()

        # Calculate all the probability fields and store them to self.results
        for k in range(len(self.results['median'])):
            row = self.results.iloc[k, 0:100].to_numpy()
            count_02 = 0
            count_025 = 0
            count_03 = 0
            for j in row:
                if j <= 0.2:
                    count_02 += 1
                if j <= 0.25:
                    count_025 += 1
                if j <= 0.3:
                    count_03 +=1

            p02 = count_02/len(row)
            p025 = count_025/len(row)
            p03 = count_03/len(row)
            prob_02.append(p02)
            prob_025.append(p025)
            prob_03.append(p03)
        self.results["probability<=0.20"] = prob_02
        self.results["probability<=0.25"] = prob_025
        self.results["probability<=0.30"] = prob_03

    def display_results(self):
        """
        Display the results of the predictions as a console output.

        Display and return all the contents of the self.results variable which is a pandas Dataframe object
        :return: A Pandas Dataframe object (self.results) containing all the result of the predictions
        """

        print(self.results)
        return self.results

    def export_results_to_excel(self, filename):
        writer = ExcelWriter(filename)
        self.results.to_excel(writer, 'Sheet1', index=False)
        writer.save()

    def export_results_to_csv(self, filename):
        self.results.to_csv(filename, index=False)

    def generate_model_performance(self):
        """Generates training performance graphs

        Plots the model performance metrics (MSE and R^2 vs # of epochs) after training and returns a
        base64 encoded image. The NN has to be trained first otherwise the image will be empty.

        Returns: Base64 data stream"""

        '''y_train_norm = self.model.predict(self.predictors_train_normalized)
        y_train = self.targets_scaler.inverse_transform(y_train_norm)

        y_test_norm = self.model.predict(self.predictors_test_normalized)
        y_test = self.targets_scaler.inverse_transform(y_test_norm)

        x_tr = np.squeeze(np.asarray(self.targets_train))
        y_tr = np.squeeze(np.asarray(y_train))
        x_ts = np.squeeze(np.asarray(self.targets_test))
        y_ts = np.squeeze(np.asarray(y_test))

        Rsquared_train = r2_score(x_tr,y_tr)
        Rsquared_test = r2_score(x_ts,y_ts)

        plt. subplot(1, 2, 1)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss, R^2 = %.3f  %.3f' % (Rsquared_train, Rsquared_test))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot([0, 1.5], [0, 1.5], '--', self.targets_train, y_train, 'ro', self.targets_test, y_test, 'bo')
        plt.show()'''

        fig, axs = plt.subplots(1, 2, sharex=True)

        ax = axs[0]
        ax.boxplot([self.total_mse_train, self.total_mse_val, self.total_mse_test], labels=["Training", "Validation", "Testing"])
        ax.set_title("Mean Squared Error")
        tr_legend = "Training Avg MSE: {mse:.4f}".format(mse=self.avg_mse_train)
        val_legend = "Validation Avg MSE: {mse:.4f}".format(mse=self.avg_mse_val)
        ts_legend = "Testing Avg MSE: {mse:.4f}".format(mse=self.avg_mse_test)
        ax.legend([tr_legend, val_legend, ts_legend])

        ax = axs[1]
        ax.boxplot([self.total_rsquared_train, self.total_rsquared_val, self.total_rsquared_test], labels=["Training", "Validation","Testing"])
        ax.set_title("R^2")
        tr_legend = "Training Avg. R^2: {rs:.3f}".format(rs=self.avg_rsq_train)
        val_legend = "Validation Avg. R^2: {rs:.3f}".format(rs=self.avg_rsq_val)
        ts_legend = "Validation Avg. R^2: {rs:.3f}".format(rs=self.avg_rsq_test)
        ax.legend([tr_legend, val_legend, ts_legend])

        fig.suptitle("Performance metrics across 100 training runs on " +
                     str(self.epochs) + " epochs, with " + str(self.layer1_neurons) + " neurons on hidden layer.")
        fig.set_size_inches(12, 8)

        # plt.show()

        # Uncomment the next lines to save the graph to disk
        # plt.savefig("model_metrics\\" + str(self.epochs) + "_epochs_" + str(self.layer1_neurons) + "_neurons.png",
        #            dpi=100)
        # plt.close()



        plt.show()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format='png')
        myStringIOBytes.seek(0)
        my_base_64_pngData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_pngData

    def generate_2d_scatterplot(self):
        """Generate a 2d scatterplot of the predictions

        Plots three, 2-dimensional scatterplots of the predictions as a function of the inputs
        The 3 scatterplots are plotting: predictions vs se1_frc and water temperature, predictions
        vs water conductivity and water temperature, and predictions vs se1_frc and water conductivity.
        A histogram of the prediction set is also generated and plotted. A prediction using the
        predict() method must be made first.

        Returns: a base64 data represenation of the image."""

        df = self.results

        # Uncomment the following line to load the results direclty from an excel file
        # df = pd.read_excel('results.xlsx')

        # Filter out outlier values
        df = df.drop(df[df[FRC_IN] > 2.8].index)

        frc = df[FRC_IN]
        watt = df[WATTEMP]
        cond = df[COND]
        c = df["median"]

        # sort data for the cdf
        sorted_data = np.sort(c)

        # The following lines of code calculate the width of the histogram bars
        # and align the range of the histogram and the pdf
        if min(c) < 0:
            lo_limit = 0
        else:
            lo_limit = round(min(c),2)
            print(lo_limit)

        if max(c) <= 0.75:
            divisions = 16
            hi_limit = 0.75
        elif max(c) < 1:
            divisions = 21
            hi_limit = 1
        elif max(c) <= 1.5:
            divisions = 31
            hi_limit = 1.5
        elif max(c) <= 2:
            divisions = 41
            hi_limit = 2

        divisions = round((hi_limit-lo_limit)/0.05,0) + 1
        print(divisions)

        # Get the data between the limits
        sorted_data = sorted_data[sorted_data > lo_limit]
        sorted_data = sorted_data[sorted_data < hi_limit]

        # create a colorbar for the se4_frc and divide it in 0.2 mg/L intervals
        cmap = plt.cm.jet_r
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(0, 1.4, 8)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        fig = plt.figure()

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

        ax = fig.add_subplot(221)
        img = ax.scatter(frc, watt, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('FRC at tapstand (mg/L)')
        ax.set_ylabel('Water Temperature (' + u"\u00b0" + 'C)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(222)
        img = ax.scatter(frc, cond, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('FRC at tapstand (mg/L)')
        ax.set_ylabel('Water Conductivity (\u03BCS/cm)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(223)
        img = ax.scatter(watt, cond, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('Water Temperature (' + u"\u00b0" + 'C)')
        ax.set_ylabel('Water Conductivity (\u03BCS/cm)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(224)
        img = ax.hist(c, bins=np.linspace(lo_limit,hi_limit,divisions), edgecolor='black', linewidth=0.1)
        ax.grid(linewidth=0.1)
        line02 = ax.axvline(0.2, color='r', linestyle='dashed', linewidth=2)
        line03 = ax.axvline(0.3, color='y', linestyle='dashed', linewidth=2)
        ax.set_xlabel('FRC at household (mg/L)')
        ax.set_ylabel('# of instances')

        axcdf = ax.twinx()
        cdf, = axcdf.step(sorted_data, np.arange(sorted_data.size), color='g')
        ax.legend((line02, line03, cdf), ('0.2 mg/L', '0.3 mg/L', 'CDF'), loc='center right')

        ax2 = fig.add_axes([0.93, 0.1, 0.01, 0.75])
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                       spacing='proportional', ticks=bounds, boundaries=bounds)
        cb.ax.set_ylabel('FRC at se4 (mg/L)', rotation=270, labelpad=20)

        plt.show()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format='png')
        myStringIOBytes.seek(0)
        my_base_64_pngData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_pngData

    def generate_input_info_plots(self, filename):
        """Generates histograms of the inputs to the ANN

        Plots one histogram for each input field on the neural network
        along with the mean and median values."""

        df = self.datainputs

        # df = df.drop(df[df["se1_frc"] > 2.8].index)
        frc = df[FRC_IN]
        watt = df[WATTEMP]
        cond = df[COND]

        dfo = self.file
        dfo = dfo.drop(dfo[dfo[FRC_IN] > 2.8].index)
        frc4 = dfo[FRC_OUT]

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

        #fig.suptitle('Total samples: '+ str(len(frc)))  # +
        #             "\n" + "SWOT version: " + self.software_version +
        #             "\n" + "Input Filename: " + os.path.basename(self.input_filename) +
        #             "\n" + "Generated on: " + self.today)


        axInitialFRC = fig.add_subplot(221)
        axInitialFRC.hist(frc, bins=20, edgecolor='black', linewidth=0.1)
        axInitialFRC.set_xlabel('Initial FRC (mg/L)')
        axInitialFRC.set_ylabel('# of instances')
        mean = round(np.mean(frc), 2)
        median = np.median(frc)
        mean_line = axInitialFRC.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        median_line = axInitialFRC.axvline(median, color='y', linestyle='dashed', linewidth=2)
        axInitialFRC.legend((mean_line, median_line),('Mean: ' + str(mean) + ' mg/L', 'Median: ' + str(median) + ' mg/L'))

        ax = fig.add_subplot(222)
        ax.hist(watt, bins= 20, edgecolor='black', linewidth=0.1)
        ax.set_xlabel('Water Temperature (' + u"\u00b0" + 'C)')
        ax.set_ylabel('# of instances')
        mean = round(np.mean(watt), 2)
        median = np.median(watt)
        mean_line = ax.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        median_line = ax.axvline(median, color='y', linestyle='dashed', linewidth=2)
        ax.legend((mean_line, median_line),('Mean: ' + str(mean) + u"\u00b0" + 'C', 'Median: ' + str(median) + u"\u00b0" + 'C'))

        ax = fig.add_subplot(223)
        ax.hist(cond, bins=20, edgecolor='black', linewidth=0.1)
        ax.set_xlabel('Water Conductivity (\u03BCS/cm)')
        ax.set_ylabel('# of instances')
        mean = round(np.mean(cond), 2)
        median = np.median(cond)
        mean_line = ax.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        median_line = ax.axvline(median, color='y', linestyle='dashed', linewidth=2)
        ax.legend((mean_line, median_line),('Mean: ' + str(mean) + ' (\u03BCS/cm)', 'Median: ' + str(median) + ' (\u03BCS/cm)'))

        axHouseholdFRC = fig.add_subplot(224)
        axHouseholdFRC.hist(frc4, bins=np.linspace(0,2,41), edgecolor='black', linewidth=0.1)
        axHouseholdFRC.set_xlabel('Household FRC (μS/cm)')
        axHouseholdFRC.set_ylabel('# of instances')
        mean = round(np.mean(frc4), 2)
        median = np.median(frc4)
        mean_line = axHouseholdFRC.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        median_line = axHouseholdFRC.axvline(median, color='y', linestyle='dashed', linewidth=2)
        axHouseholdFRC.legend((mean_line, median_line), ('Mean: ' + str(mean) + ' (μS/cm)', 'Median: ' + str(median) + ' (μS/cm)'))

        fig.savefig(os.path.splitext(filename)[0]+'.png', format='png')
        # plt.show()

        # create figure for initial and household FRC separately in a single image
        figFRC = plt.figure(figsize=(19.2, 5.4), dpi=100)

        axInitialFRC = figFRC.add_subplot(221)
        axInitialFRC.hist(frc, bins=20, edgecolor='black', linewidth=0.1)
        axInitialFRC.set_xlabel('Initial FRC (mg/L)')
        axInitialFRC.set_ylabel('# of instances')
        mean = round(np.mean(frc), 2)
        median = np.median(frc)
        mean_line = axInitialFRC.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        median_line = axInitialFRC.axvline(median, color='y', linestyle='dashed', linewidth=2)
        axInitialFRC.legend((mean_line, median_line),('Mean: ' + str(mean) + ' mg/L', 'Median: ' + str(median) + ' mg/L'))

        axHouseholdFRC = figFRC.add_subplot(223)
        axHouseholdFRC.hist(frc4, bins=np.linspace(0,2,41), edgecolor='black', linewidth=0.1)
        axHouseholdFRC.set_xlabel('Household FRC (μS/cm)')
        axHouseholdFRC.set_ylabel('# of instances')
        mean = round(np.mean(frc4), 2)
        median = np.median(frc4)
        mean_line = axHouseholdFRC.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        median_line = axHouseholdFRC.axvline(median, color='y', linestyle='dashed', linewidth=2)
        axHouseholdFRC.legend((mean_line, median_line), ('Mean: ' + str(mean) + ' (μS/cm)', 'Median: ' + str(median) + ' (μS/cm)'))

        figFRC.savefig(os.path.splitext(filename)[0]+'-frc.jpg', format='jpg')

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format='png')
        myStringIOBytes.seek(0)
        my_base_64_pngData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_pngData

    def set_inputs_for_table(self, wt, c):
        frc = np.arange(0.20,2.05,0.05)
        watt = [self.median_wattemp for i in range(0,len(frc))]
        cond = [self.median_cond for i in range(0,len(frc))]
        temp = {"ts_frc": frc, "ts_wattemp": watt, "ts_cond": cond}
        self.predictors = pd.DataFrame(temp)

    def generate_html_report(self, filename):
        """Generates an html report of the SWOT results. The report
        is saved on disk under the name 'filename'."""

        df = self.datainputs
        frc = df[FRC_IN]

        self.generate_input_info_plots(filename).decode('UTF-8')
        # scatterplots_b64 = self.generate_2d_scatterplot().decode('UTF-8')
        html_table = self.prepare_table_for_html_report()

        doc, tag, text, line = Doc().ttl()
        with tag('h1', klass='title'):
            text('SWOT ARTIFICIAL NEURAL NETWORK REPORT')
        with tag('p', klass='swot_version'):
            text('SWOT ANN Code Version: ' + self.software_version)
        with tag('p', klass='input_filename'):
            text('Input File Name: ' + os.path.basename(self.input_filename))
        with tag('p', klass='date'):
            text('Date Generated: ' + self.today)
        with tag('p', klass="time_difference"):
            text("Average time between tapstand and household: " + str(self.avg_time_elapsed.seconds // 3600) + " hours and " +
              str((self.avg_time_elapsed.seconds // 60) % 60) + " minutes")
        with tag('p'):
            text('Total Samples: ' + str(len(frc)))
        with tag('p'):
            text('Inputs specified:')
        with tag('div', id='inputs_graphs'):
            doc.stag('img', src='cid:' +os.path.basename(os.path.splitext(filename)[0]+'.png'))
            #doc.asis('<object data="cid:'+os.path.basename(os.path.splitext(filename)[0]+'.PNG') + '" type="image/jpeg"></object>')

        doc.asis('<object data="'+os.path.basename(os.path.splitext(filename)[0]+'.png') + '" type="image/jpeg"></object>')

        doc.asis(html_table)

        file = open(filename, 'w+')
        file.write(doc.getvalue())
        file.close()

        return doc.getvalue()

    def prepare_table_for_html_report(self):
        """Formats the results into an html table for display."""

        temp = self.results

        table_df = pd.DataFrame()
        table_df['Input FRC (mg/L)'] = self.results[FRC_IN]
        table_df['Water Temperature (oC)'] = self.results[WATTEMP]
        table_df['Water Conductivity (10^-6 S/cm)'] = self.results[COND]
        table_df['Median Predicted FRC level at Household (mg/L)'] = self.results['median']
        table_df['Probability of predicted FRC level to be less than 0.20 mg/L'] = self.results['probability<=0.20']
        table_df['Probability of predicted FRC level to be less than 0.25 mg/L'] = self.results['probability<=0.25']
        table_df['Probability of predicted FRC level to be less than 0.30 mg/L'] = self.results['probability<=0.30']


        str_io = io.StringIO()

        html = table_df.to_html(buf=str_io, classes='tabular_results')
        html_str = str_io.getvalue()
        return html_str

    '''def save_2d_scatterplot_svg(self):

        df = self.results
        df = df.drop(df[df[FRC_IN] > 2.8].index)

        frc = df[FRC_IN]
        watt = df[WATTEMP]
        cond = df[COND]
        c = df["median"]

        sorted_data = np.sort(c)

        if min(c) < 0:
            lo_limit = 0
        else:
            lo_limit = round(min(c),2)
            print(lo_limit)

        if max(c) <= 0.75:
            divisions = 16
            hi_limit = 0.75
        elif max(c) < 1:
            divisions = 21
            hi_limit = 1
        elif max(c) <= 1.5:
            divisions = 31
            hi_limit = 1.5
        elif max(c) <= 2:
            divisions = 41
            hi_limit = 2

        divisions = round((hi_limit-lo_limit)/0.05,0) + 1
        print(divisions)

        sorted_data = sorted_data[sorted_data > lo_limit]
        sorted_data = sorted_data[sorted_data < hi_limit]

        cmap = plt.cm.jet_r
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(0, 1.4, 8)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

        ax = fig.add_subplot(221)
        img = ax.scatter(frc, watt, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('Initial FRC (mg/L)')
        ax.set_ylabel('Water Temperature (' + u"\u00b0" + 'C)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(222)
        img = ax.scatter(frc, cond, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('Initial FRC (mg/L)')
        ax.set_ylabel('Water Conductivity (\u03BCS/cm)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(223)
        img = ax.scatter(watt, cond, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('Water Temperature (' + u"\u00b0" + 'C)')
        ax.set_ylabel('Water Conductivity (\u03BCS/cm)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(224)
        img = ax.hist(c, bins=np.linspace(lo_limit,hi_limit,divisions), edgecolor='black', linewidth=0.1)
        ax.grid(linewidth=0.1)
        line02 = ax.axvline(0.2, color='r', linestyle='dashed', linewidth=2)
        line03 = ax.axvline(0.3, color='y', linestyle='dashed', linewidth=2)
        ax.set_xlabel('FRC at household (mg/L)')
        ax.set_ylabel('# of instances')

        axcdf = ax.twinx()
        cdf, = axcdf.step(sorted_data, np.arange(sorted_data.size), color='g')
        ax.legend((line02, line03, cdf), ('0.2 mg/L', '0.3 mg/L', 'CDF'), loc='center right')

        ax2 = fig.add_axes([0.93, 0.1, 0.01, 0.75])
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                       spacing='proportional', ticks=bounds, boundaries=bounds)
        cb.ax.set_ylabel('FRC at household (mg/L)', rotation=270, labelpad=20)

        fig.savefig('all_Results.svg', format='svg', dpi=2400)

        # plt.show()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format='jpg')
        myStringIOBytes.seek(0)
        my_base_64_jpgData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_jpgData '''

    '''def generate_histogram(self):
        temp = self.results.iloc[0, 0:100].to_numpy()

        j = 0
        for i in temp:
            if i <= 0.2:
                j += 1

        prob = j/len(temp)
        print("Probability: " + str(prob))
        plt.hist(temp, bins=10)
        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format='jpg')
        myStringIOBytes.seek(0)
        my_base_64_jpgData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_jpgData'''

    '''def generate_3d_scatterplot(self):
        """Plots a 3 dimensional scaterrplot of the prediction results.

        Plots a 3 dimensional graph of the results of the prediction made by the ANN. The
        axis of the plot are: FRC at se1, water temperature and water conductivity. The resulting
        FRC at se4 is color encoded making this graph an actual 4-d representation of input vs predicted data.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        df = self.results
        df = df.drop(df[df[FRC_IN] > 2.8].index)

        frc = df[FRC_IN]
        watt = df[WATTEMP]
        cond = df[COND]
        c = df["median"]

        cmap = plt.cm.jet_r
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        bounds = np.linspace(0, 1.4, 8)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        img = ax.scatter(frc, watt, cond, c=c, s=50, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('FRC (mg/L)')
        ax.set_ylabel('Water Temperature (' + u"\u00b0" + 'C)')
        ax.set_zlabel('Water Conductivity (S/cm)')

        ax2 = fig.add_axes([0.90, 0.1, 0.03, 0.8])
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                       spacing='proportional', ticks=bounds, boundaries=bounds)

        plt.show()'''
