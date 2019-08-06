import joblib
import time
import os
import keras
import base64
import io
from yattag import Doc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pandas import ExcelWriter

from mpl_toolkits.mplot3d import Axes3D

class NNetwork:
    def __init__(self):

        self.model = None
        self.pretrained_networks = []

        self.predictors_scaler = MinMaxScaler()
        self.outputs_scaler = MinMaxScaler()

        self.history = None
        self.file = None

        self.layer1_neurons = 5
        self.epochs = 30

        self.predictors = None

        self.predictors_train = None
        self.predictors_test = None
        self.predictors_train_normalized = None
        self.predictors_test_normalized = None
        self.outputs_test_normalized = None

        self.outputs = None
        self.predictions = pd.DataFrame()
        self.results = pd.DataFrame()

        self.total_mse = []
        self.total_rsquared = []
        self.total_mse_val = []
        self.total_rsquared_val = []

        self.avg_mse = 0
        self.avg_rsq = 0
        self.avg_mse_val = 0
        self.avg_rsq_val = 0

        self.outputs_train = None
        self.outputs_test = None
        self.outputs_train_normalized = None

        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(self.layer1_neurons, input_dim=3, activation="tanh"))
        self.model.add(keras.layers.Dense(1))

        self.optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None,
                                                schedule_decay=0.004)
        self.model.compile(loss='mse', optimizer="adam", metrics=['mse'])

    def import_pretrained_model(self, path):
        json_architecture = open(path + '\\' + 'architecture.json', 'r')
        network_architecture = json_architecture.read()
        json_architecture.close()

        pretrained_networks = os.listdir(path + "\\network_weights")
        for network in pretrained_networks:
            temp = keras.models.model_from_json(network_architecture)
            temp.load_weights(path + '\\network_weights' + '\\' + network)
            self.pretrained_networks.append(temp)
            print(network + "loaded")
            del temp
        scalers = joblib.load(path + '\\' + "scaler.save")
        self.predictors_scaler = scalers["input"]
        self.outputs_scaler = scalers["output"]

    def train_network(self):
        self.predictors_scaler = self.predictors_scaler.fit(self.predictors)
        self.outputs_scaler = self.outputs_scaler.fit(self.outputs)

        self.predictors_train, self.predictors_test, self.outputs_train, self.outputs_test =\
            train_test_split(self.predictors, self.outputs, test_size=0.2, shuffle=True)

        self.predictors_train_normalized = self.predictors_scaler.transform(self.predictors_train)
        self.predictors_test_normalized = self.predictors_scaler.transform(self.predictors_test)

        self.outputs_train_normalized = self.outputs_scaler.transform(self.outputs_train)
        self.outputs_test_normalized = self.outputs_scaler.transform(self.outputs_test)

        self.history = self.model.fit(self.predictors_train_normalized,
                                      self.outputs_train_normalized, epochs=self.epochs,
                                      validation_data=(self.predictors_test_normalized, self.outputs_test_normalized),
                                      verbose=0)

        y_train_norm = self.model.predict(self.predictors_train_normalized)
        y_train = self.outputs_scaler.inverse_transform(y_train_norm)

        y_test_norm = self.model.predict(self.predictors_test_normalized)
        y_test = self.outputs_scaler.inverse_transform(y_test_norm)

        x_tr = np.squeeze(np.asarray(self.outputs_train))
        y_tr = np.squeeze(np.asarray(y_train))
        x_ts = np.squeeze(np.asarray(self.outputs_test))
        y_ts = np.squeeze(np.asarray(y_test))

        rsquared_train = r2_score(x_tr, y_tr)
        rsquared_test = r2_score(x_ts, y_ts)

        self.total_mse.append(self.history.history['mean_squared_error'][self.epochs-1])
        self.total_mse_val.append(self.history.history['val_mean_squared_error'][self.epochs-1])
        self.total_rsquared.append(rsquared_train)
        self.total_rsquared_val.append(rsquared_test)

    def load_data(self, data):
        temp = {"se1_frc": data[0], "se1_wattemp": data[1], "se1_cond": data[2]}
        self.predictors = pd.DataFrame(temp, index=[0])

    def import_data_from_excel(self, filename, sheet=0):
        df = pd.read_excel(filename, sheet_name=sheet).dropna()
        # df = df.drop(df[df['sb_sunshade'] == 1].index)
        # df = df.drop(df[df["se1_frc"] > 2].index)
        self.file = df

        # self.inputs = self.file.loc[:,
        # ['se1_frc', 'se1_trc', 'se1_turb', 'se1_airtemp', 'se1_wattemp', 'se1_cond', 'se1_ph', 'se1_orp']]

        self.predictors = self.file.loc[:, ['se1_frc', 'se1_wattemp', 'se1_cond']]
        self.outputs = self.file.loc[:, 'se4_frc'].values.reshape(-1, 1)

    def import_data_from_csv(self, filename):
        self.file = pd.read_csv(filename).dropna()
        self.predictors = self.file.loc[:, ['se1_frc', 'se1_wattemp', 'se1_cond']]
        self.outputs = self.file.loc[:, 'se4_frc'].values.reshape(-1, 1)

    def predict(self):
        prob_02 = []
        prob_025 = []
        prob_03 = []

        results = {}

        input_scaler = self.predictors_scaler
        inputs_norm = input_scaler.transform(self.predictors)

        for j, network in enumerate(self.pretrained_networks):
            key = "se4_frc_net-" + str(j)
            y_norm = network.predict(inputs_norm)
            predictions = self.outputs_scaler.inverse_transform(y_norm).tolist()
            temp = sum(predictions, [])
            results.update({key: temp})
        self.results = pd.DataFrame(results)
        self.results["median"] = self.results.median(axis=1)

        for i in self.predictors.keys():
            self.results.update({i: self.predictors[i].tolist()})
            self.results[i] = self.predictors[i].tolist()

        print(len(self.results["median"]))
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
        print(self.results)
        return self.results

    def export_results_to_excel(self, filename):
        writer = ExcelWriter(filename)
        self.results.to_excel(writer, 'Sheet1', index=False)
        writer.save()

    def export_results_to_csv(self, filename):
        self.results.to_csv(filename, index=False)

    def prepare_results_for_display(self):
        j = self.predictions.tolist()
        temp = sum(j, [])

        for i in self.predictors.keys():
            self.results.update({i: self.predictors[i].tolist()})
        self.results.update({"se4_frc": temp})

    def generate_model_performance(self):

        '''y_train_norm = self.model.predict(self.predictors_train_normalized)
        y_train = self.outputs_scaler.inverse_transform(y_train_norm)

        y_test_norm = self.model.predict(self.predictors_test_normalized)
        y_test = self.outputs_scaler.inverse_transform(y_test_norm)

        x_tr = np.squeeze(np.asarray(self.outputs_train))
        y_tr = np.squeeze(np.asarray(y_train))
        x_ts = np.squeeze(np.asarray(self.outputs_test))
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
        plt.plot([0, 1.5], [0, 1.5], '--', self.outputs_train, y_train, 'ro', self.outputs_test, y_test, 'bo')
        plt.show()'''

        fig, axs = plt.subplots(1, 2, sharex=True)

        ax = axs[0]
        ax.boxplot([self.total_mse, self.total_mse_val], labels=["Training", "Validation"])
        ax.set_title("Mean Squared Error")
        tr_legend = "Training Avg MSE: {mse:.4f}".format(mse=self.avg_mse)
        val_legend = "Validation Avg MSE: {mse:.4f}".format(mse=self.avg_mse_val)
        ax.legend([tr_legend, val_legend])

        ax = axs[1]
        ax.boxplot([self.total_rsquared, self.total_rsquared_val], labels=["Training", "Validation"])
        ax.set_title("R^2")
        tr_legend = "Training Avg. R^2: {rs:.3f}".format(rs=self.avg_rsq)
        val_legend = "Validation Avg. R^2: {rs:.3f}".format(rs=self.avg_rsq_val)
        ax.legend([tr_legend, val_legend])

        fig.suptitle("Performance metrics across 100 training runs on " +
                     str(self.epochs) + " epochs, with " + str(self.layer1_neurons) + " neurons on hidden layer.")
        fig.set_size_inches(12, 8)

        plt.show()

        # plt.savefig("model_metrics\\" + str(self.epochs) + "_epochs_" + str(self.layer1_neurons) + "_neurons.png",
        #            dpi=100)

        self.total_rsquared = []
        self.total_rsquared_val = []
        self.total_mse = []
        self.total_mse_val = []

        plt.close()

    def generate_3d_scatterplot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        df = self.results
        df = df.drop(df[df["se1_frc"] > 2.8].index)

        frc = df["se1_frc"]
        watt = df["se1_wattemp"]
        cond = df["se1_cond"]
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

        plt.show()

    def generate_2d_scatterplot(self):
        df = self.results
        # df = pd.read_excel('results.xlsx')
        df = df.drop(df[df["se1_frc"] > 2.8].index)

        frc = df["se1_frc"]
        watt = df["se1_wattemp"]
        cond = df["se1_cond"]
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

        fig = plt.figure()

        ax = fig.add_subplot(221)
        img = ax.scatter(frc, watt, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('FRC at se1 (mg/L)')
        ax.set_ylabel('Water Temperature (' + u"\u00b0" + 'C)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(222)
        img = ax.scatter(frc, cond, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('FRC at se1 (mg/L)')
        ax.set_ylabel('Water Conductivity (μS/cm)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(223)
        img = ax.scatter(watt, cond, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel('Water Temperature (' + u"\u00b0" + 'C)')
        ax.set_ylabel('Water Conductivity (μS/cm)')
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(224)
        img = ax.hist(c, bins=np.linspace(lo_limit,hi_limit,divisions), edgecolor='black', linewidth=0.1)
        ax.grid(linewidth=0.1)
        line02 = ax.axvline(0.2, color='r', linestyle='dashed', linewidth=2)
        line03 = ax.axvline(0.3, color='y', linestyle='dashed', linewidth=2)
        ax.set_xlabel('FRC at se4 (mg/L)')
        ax.set_ylabel('# of instances')

        axcdf = ax.twinx()
        cdf, = axcdf.step(sorted_data, np.arange(sorted_data.size), color='g')
        ax.legend((line02, line03, cdf), ('0.2 mg/L', '0.3 mg/L', 'CDF'), loc='center right')

        ax2 = fig.add_axes([0.93, 0.1, 0.01, 0.75])
        cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm,
                                       spacing='proportional', ticks=bounds, boundaries=bounds)
        cb.ax.set_ylabel('FRC at se4 (mg/L)', rotation=270, labelpad=20)

        # plt.show()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format='jpg')
        myStringIOBytes.seek(0)
        my_base_64_jpgData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_jpgData

    def generate_input_info_plots(self):

        df = self.predictors
        # df = df.drop(df[df["se1_frc"] > 2.8].index)
        frc = df["se1_frc"]
        watt = df["se1_wattemp"]
        cond = df["se1_cond"]

        fig = plt.figure()

        fig.suptitle('Total samples: '+ str(len(frc)))

        ax = fig.add_subplot(221)
        ax.hist(frc, bins=20, edgecolor='black', linewidth=0.1)
        ax.set_xlabel('FRC at se1 (mg/L)')
        ax.set_ylabel('# of instances')
        mean = round(np.mean(frc), 2)
        median = np.median(frc)
        mean_line = ax.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        median_line = ax.axvline(median, color='y', linestyle='dashed', linewidth=2)
        ax.legend((mean_line, median_line),('Mean: ' + str(mean) + ' mg/L', 'Median: ' + str(median) + ' mg/L'))

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
        ax.set_xlabel('Water Conductivity (μS/cm)')
        ax.set_ylabel('# of instances')
        mean = round(np.mean(cond), 2)
        median = np.median(cond)
        mean_line = ax.axvline(mean, color='r', linestyle='dashed', linewidth=2)
        median_line = ax.axvline(median, color='y', linestyle='dashed', linewidth=2)
        ax.legend((mean_line, median_line),('Mean: ' + str(mean) + ' (μS/cm)', 'Median: ' + str(median) + ' (μS/cm)'))

        # plt.show()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format='jpg')
        myStringIOBytes.seek(0)
        my_base_64_jpgData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_jpgData

    def train_SWOT_network(self, directory='untitled_network'):

        self.total_mse = []
        self.total_rsquared = []
        self.total_mse_val = []
        self.total_rsquared_val = []

        if not os.path.exists(directory):
            os.mkdir(directory)

        if not os.path.exists(directory + '\\' + 'network_weights'):
            os.mkdir(directory + '\\' + 'network_weights')

        model_json = self.model.to_json()
        with open(directory + '\\' + "architecture.json", 'w') as json_file:
            json_file.write(model_json)

        json_file.close()

        for i in range(0, 100):
            self.train_network()
            self.model.save_weights(directory + "\\network_weights" + "\\network" + str(i) + ".h5")
            print('Training network #' + str(i))

        self.avg_mse = np.median(np.array(self.total_mse))
        self.avg_rsq = np.median(np.array(self.total_rsquared))
        self.avg_mse_val = np.median(np.array(self.total_mse_val))
        self.avg_rsq_val = np.median(np.array(self.total_rsquared_val))

        scaler_filename = "scaler.save"
        scalers = {"input": self.predictors_scaler, "output": self.outputs_scaler}
        joblib.dump(scalers, directory + '\\' + scaler_filename)
        print("Model Saved!")

    def generate_histogram(self):
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
        return my_base_64_jpgData

    def set_inputs_for_table(self, wt, c):
        frc = np.linspace(0.2, 2, 37)
        watt = []
        cond = []
        for i in range(len(frc)):
            watt.append(wt)
            cond.append(c)
        temp = {"se1_frc": frc, "se1_wattemp": watt, "se1_cond": cond}
        self.predictors = pd.DataFrame(temp)

    def generate_html_report(self, filename):

        input_plot_b64_graph = self.generate_input_info_plots().decode('UTF-8')
        scatterplots_b64 = self.generate_2d_scatterplot().decode('UTF-8')
        html_table = self.prepare_table_for_html_report()

        doc, tag, text, line = Doc().ttl()
        with tag('h1', klass='title'):
            text('SWOT report')
        with tag('p', klass='summary'):
            text('This is a summary of the SWOT run you requested')
        with tag('p'):
            text('Inputs specified:')
        with tag('div', id='inputs_graphs'):
            doc.stag('img', src='data:image/png;base64, ' + input_plot_b64_graph)
        with tag('div', klass='results_graph'):
            doc.stag('img', src='data:image/png;base64, ' + scatterplots_b64)
        doc.asis(html_table)

        file = open(filename, 'w+')
        file.write(doc.getvalue())
        file.close()

        return doc.getvalue()

    def prepare_table_for_html_report(self):

        temp = self.results

        table_df = pd.DataFrame()
        table_df['Input FRC (mg/L)'] = self.results['se1_frc']
        table_df['Water Temperature (oC)'] = self.results['se1_wattemp']
        table_df['Water Conductivity (10^-6 S/cm)'] = self.results['se1_cond']
        table_df['Median Predicted FRC level at Se4 (mg/L)'] = self.results['median']
        table_df['Probability of predicted FRC level to be less than 0.20 mg/L'] = self.results['probability<=0.20']
        table_df['Probability of predicted FRC level to be less than 0.25 mg/L'] = self.results['probability<=0.25']
        table_df['Probability of predicted FRC level to be less than 0.30 mg/L'] = self.results['probability<=0.30']


        str_io = io.StringIO()

        html = table_df.to_html(buf=str_io, classes='tabular_results')
        html_str = str_io.getvalue()
        return html_str
