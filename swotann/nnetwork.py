import base64
import datetime
import io
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from xlrd.xldate import xldate_as_datetime
from yattag import Doc

plt.rcParams.update({"figure.autolayout": True})
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import logging

"""
TF_CPP_MIN_LOG_LEVEL:
Defaults to 0, so all logs are shown. Set TF_CPP_MIN_LOG_LEVEL to 1 to filter out INFO logs, 2 to additionally filter out WARNING, 3 to additionally filter out ERROR.
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras


class NNetwork(object):
    def __init__(self, network_count=200, epochs=1000):
        logging.getLogger().setLevel(logging.INFO)
        self.xl_dateformat = r"%Y-%m-%dT%H:%M"
        self.model = None
        self.pretrained_networks = []

        self.software_version = "2.0.1"
        self.input_filename = None
        self.today = str(datetime.date.today())
        self.avg_time_elapsed = 0

        self.predictors_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.targets_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.history = None
        self.file = None

        self.skipped_rows = []
        self.ruleset = []

        self.layer1_neurons = 4
        self.network_count = network_count
        self.epochs = epochs

        self.predictors = None

        self.targets = None
        self.predictions = None
        self.avg_case_results_am = None
        self.avg_case_results_pm = None
        self.worst_case_results_am = None
        self.worst_case_results_pm = None
        self.WB_bandwidth = None
        self.post_process_check = False  # Is post-processed better than raw. If False, uses raw results, if true, uses post-processed results

        self.optimizer = keras.optimizers.Nadam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.999
        )
        self.model = keras.models.Sequential()
        self.model.add(
            keras.layers.Dense(self.layer1_neurons, input_dim=5, activation="tanh", name="tanh_init")
        )
        self.model.add(keras.layers.Dense(1, activation="linear", name="linear_init"))
        self.model.compile(loss="mse", optimizer=self.optimizer, metrics=["mse"])

    def import_data_from_csv(self, filename):
        """
        Imports data to the network by a comma-separated values (CSV) file.

        Load data to a network that are stored in .csv file format.
        The data loaded from this method can be used both for training reasons as
        well as to make predictions.

        :param filename: String containing the filename of the .csv file containing the input data (e.g "input_data.csv")
        """

        df = pd.read_csv(filename)
        self.file = df.copy()

        global FRC_IN
        global FRC_OUT
        global WATTEMP
        global COND

        # Locate the fields used as inputs/predictors and outputs in the loaded file
        # and split them

        if "se1_frc" in self.file.columns:
            FRC_IN = "se1_frc"
            WATTEMP = "se1_wattemp"
            COND = "se1_cond"
            FRC_OUT = "se4_frc"
        elif "ts_frc1" in self.file.columns:
            FRC_IN = "ts_frc1"
            WATTEMP = "ts_wattemp"
            COND = "ts_cond"
            FRC_OUT = "hh_frc1"
        elif "ts_frc" in self.file.columns:
            FRC_IN = "ts_frc"
            WATTEMP = "ts_wattemp"
            COND = "ts_cond"
            FRC_OUT = "hh_frc"

        # Standardize the DataFrame by specifying rules
        # To add a new rule, call the method execute_rule with the parameters (description, affected_column, query)
        self.execute_rule("Invalid tapstand FRC", FRC_IN, self.file[FRC_IN].isnull())
        self.execute_rule("Invalid household FRC", FRC_OUT, self.file[FRC_OUT].isnull())
        self.execute_rule(
            "Invalid tapstand date/time",
            "ts_datetime",
            self.valid_dates(self.file["ts_datetime"]),
        )
        self.execute_rule(
            "Invalid household date/time",
            "hh_datetime",
            self.valid_dates(self.file["hh_datetime"]),
        )
        self.skipped_rows = df.loc[df.index.difference(self.file.index)]

        self.file.reset_index(drop=True, inplace=True)  # fix dropped indices in pandas

        # Locate the rows of the missing data
        drop_threshold = 0.90 * len(self.file.loc[:, [FRC_IN]])
        nan_rows_watt = self.file.loc[self.file[WATTEMP].isnull()]
        if len(nan_rows_watt) < drop_threshold:
            self.execute_rule(
                "Missing Water Temperature Measurement",
                WATTEMP,
                self.file[WATTEMP].isnull(),
            )
        nan_rows_cond = self.file.loc[self.file[COND].isnull()]
        if len(nan_rows_cond) < drop_threshold:
            self.execute_rule("Missing EC Measurement", COND, self.file[COND].isnull())
        self.skipped_rows = df.loc[df.index.difference(self.file.index)]

        self.file.reset_index(drop=True, inplace=True)

        start_date = self.file["ts_datetime"]
        end_date = self.file["hh_datetime"]

        durations = []
        all_dates = []
        collection_time = []

        for i in range(len(start_date)):
            try:
                # excel type
                start = float(start_date[i])
                end = float(end_date[i])
                start = xldate_as_datetime(start, datemode=0)
                if start.hour > 12:
                    collection_time = np.append(collection_time, 1)
                else:
                    collection_time = np.append(collection_time, 0)
                end = xldate_as_datetime(end, datemode=0)

            except ValueError:
                # kobo type
                start = start_date[i][:16].replace("/", "-")
                end = end_date[i][:16].replace("/", "-")
                start = datetime.datetime.strptime(start, self.xl_dateformat)
                if start.hour > 12:
                    collection_time = np.append(collection_time, 1)
                else:
                    collection_time = np.append(collection_time, 0)

                end = datetime.datetime.strptime(end, self.xl_dateformat)

            durations.append((end - start).total_seconds())
            all_dates.append(datetime.datetime.strftime(start, self.xl_dateformat))

        self.durations = durations
        self.time_of_collection = collection_time

        self.avg_time_elapsed = np.mean(durations)

        # Extract the column of dates for all data and put them in YYYY-MM-DD format
        self.file["formatted_date"] = all_dates

        predictors = {
            FRC_IN: self.file[FRC_IN],
            "elapsed time": (np.array(self.durations) / 3600),
            "time of collection (0=AM, 1=PM)": self.time_of_collection,
        }
        self.targets = self.file.loc[:, FRC_OUT]
        self.var_names = [
            "Tapstand FRC (mg/L)",
            "Elapsed Time",
            "time of collection (0=AM, 1=PM)",
        ]
        self.predictors = pd.DataFrame(predictors)
        if len(nan_rows_watt) < drop_threshold:
            self.predictors[WATTEMP] = self.file[WATTEMP]
            self.var_names.append("Water Temperature(" + r"$\degree$" + "C)")
            self.median_wattemp = np.median(self.file[WATTEMP].dropna().to_numpy())
            self.upper95_wattemp = np.percentile(
                self.file[WATTEMP].dropna().to_numpy(), 95
            )
        if len(nan_rows_cond) < drop_threshold:
            self.predictors[COND] = self.file[COND]
            self.var_names.append("EC (" + r"$\mu$" + "s/cm)")
            self.median_cond = np.median(self.file[COND].dropna().to_numpy())
            self.upper95_cond = np.percentile(self.file[COND].dropna().to_numpy(), 95)

        self.targets = self.targets.values.reshape(-1, 1)
        self.datainputs = self.predictors
        self.dataoutputs = self.targets
        self.input_filename = filename

    def set_up_model(self):
        self.optimizer = keras.optimizers.Nadam(
            learning_rate=0.01, beta_1=0.9, beta_2=0.999
        )
        self.model = keras.models.Sequential()
        self.model.add(
            keras.layers.Dense(
                self.layer1_neurons,
                input_dim=len(self.datainputs.columns),
                activation="tanh",
                name="tanh_layer"
            )
        )
        self.model.add(keras.layers.Dense(1, activation="linear", name="linear_layer"))
        self.model.compile(loss="mse", optimizer=self.optimizer)

    def train_SWOT_network(self, directory):
        """Train the set of 200 neural networks on SWOT data

        Trains an ensemble of 200 neural networks on se1_frc, water temperature,
        water conductivity."""

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.predictors_scaler = self.predictors_scaler.fit(self.predictors)
        self.targets_scaler = self.targets_scaler.fit(self.targets)

        x = self.predictors
        t = self.targets

        self.calibration_predictions = []
        self.trained_models = {}

        for i in range(self.network_count):
            logging.info("Training Network " + str(i))
            model_out = self.train_network(x, t, directory)

            self.trained_models.update({"model_" + str(i): model_out})

    def train_network(self, x, t, directory):
        """
        Trains a single Neural Network on imported data.

        This method trains Neural Network on data that have previously been imported
        to the network using the import_data_from_csv() method.
        The network used is a Multilayer Perceptron (MLP). Input and Output data are
        normalized using MinMax Normalization.

        The input dataset is split in training and validation datasets, where 80% of the inputs
        are the training dataset and 20% is the validation dataset.

        The training history is stored in a variable called self.history (see keras documentation:
        keras.model.history object)

        Performance metrics are calculated and stored for evaluating the network performance.
        """
        keras.backend.clear_session()
        early_stopping_monitor = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True
        )

        x_norm = self.predictors_scaler.transform(x)
        t_norm = self.targets_scaler.transform(t)

        trained_model = keras.models.clone_model(self.model)

        x_norm_train, x_norm_val, t_norm_train, t_norm_val = train_test_split(
            x_norm, t_norm, train_size=0.333, shuffle=True
        )

        new_weights = [
            np.random.uniform(-0.05, 0.05, w.shape) for w in trained_model.get_weights()
        ]
        trained_model.set_weights(new_weights)
        trained_model.compile(loss="mse", optimizer=self.optimizer)
        trained_model.fit(
            x_norm_train,
            t_norm_train,
            epochs=self.epochs,
            validation_data=(x_norm_val, t_norm_val),
            callbacks=[early_stopping_monitor],
            verbose=0,
            batch_size=len(t_norm_train),
        )

        self.calibration_predictions.append(
            self.targets_scaler.inverse_transform(
                trained_model.predict(x_norm, verbose=0)
            )
        )
        return trained_model

    def calibration_performance_evaluation(self, filename):
        Y_true = np.array(self.targets)
        Y_pred = np.array(self.calibration_predictions)
        FRC_X = self.datainputs[FRC_IN].to_numpy()

        capture_all = (
            np.less_equal(Y_true, np.max(Y_pred, axis=0))
            * np.greater_equal(Y_true, np.min(Y_pred, axis=0))
            * 1
        )
        capture_90 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 95, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 5, axis=0))
            * 1
        )
        capture_80 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 90, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 10, axis=0))
            * 1
        )
        capture_70 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 85, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 15, axis=0))
            * 1
        )
        capture_60 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 80, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 20, axis=0))
            * 1
        )
        capture_50 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 75, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 25, axis=0))
            * 1
        )
        capture_40 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 70, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 30, axis=0))
            * 1
        )
        capture_30 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 65, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 35, axis=0))
            * 1
        )
        capture_20 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 60, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 40, axis=0))
            * 1
        )
        capture_10 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 55, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 45, axis=0))
            * 1
        )

        capture_all_20 = capture_all * np.less(Y_true, 0.2)
        capture_90_20 = capture_90 * np.less(Y_true, 0.2)
        capture_80_20 = capture_80 * np.less(Y_true, 0.2)
        capture_70_20 = capture_70 * np.less(Y_true, 0.2)
        capture_60_20 = capture_60 * np.less(Y_true, 0.2)
        capture_50_20 = capture_50 * np.less(Y_true, 0.2)
        capture_40_20 = capture_40 * np.less(Y_true, 0.2)
        capture_30_20 = capture_30 * np.less(Y_true, 0.2)
        capture_20_20 = capture_20 * np.less(Y_true, 0.2)
        capture_10_20 = capture_10 * np.less(Y_true, 0.2)

        length_20 = np.sum(np.less(Y_true, 0.2))
        test_len = len(Y_true)
        capture_all_sum = np.sum(capture_all)
        capture_90_sum = np.sum(capture_90)
        capture_80_sum = np.sum(capture_80)
        capture_70_sum = np.sum(capture_70)
        capture_60_sum = np.sum(capture_60)
        capture_50_sum = np.sum(capture_50)
        capture_40_sum = np.sum(capture_40)
        capture_30_sum = np.sum(capture_30)
        capture_20_sum = np.sum(capture_20)
        capture_10_sum = np.sum(capture_10)

        capture_all_20_sum = np.sum(capture_all_20)
        capture_90_20_sum = np.sum(capture_90_20)
        capture_80_20_sum = np.sum(capture_80_20)
        capture_70_20_sum = np.sum(capture_70_20)
        capture_60_20_sum = np.sum(capture_60_20)
        capture_50_20_sum = np.sum(capture_50_20)
        capture_40_20_sum = np.sum(capture_40_20)
        capture_30_20_sum = np.sum(capture_30_20)
        capture_20_20_sum = np.sum(capture_20_20)
        capture_10_20_sum = np.sum(capture_10_20)

        capture = [
            capture_10_sum / test_len,
            capture_20_sum / test_len,
            capture_30_sum / test_len,
            capture_40_sum / test_len,
            capture_50_sum / test_len,
            capture_60_sum / test_len,
            capture_70_sum / test_len,
            capture_80_sum / test_len,
            capture_90_sum / test_len,
            capture_all_sum / test_len,
        ]
        capture_20 = [
            capture_10_20_sum / length_20,
            capture_20_20_sum / length_20,
            capture_30_20_sum / length_20,
            capture_40_20_sum / length_20,
            capture_50_20_sum / length_20,
            capture_60_20_sum / length_20,
            capture_70_20_sum / length_20,
            capture_80_20_sum / length_20,
            capture_90_20_sum / length_20,
            capture_all_20_sum / length_20,
        ]

        self.percent_capture_cal = capture_all_sum / test_len

        self.percent_capture_02_cal = capture_all_20_sum / length_20

        self.CI_reliability_cal = (
            (0.1 - capture_10_sum / test_len) ** 2
            + (0.2 - capture_20_sum / test_len) ** 2
            + (0.3 - capture_30_sum / test_len) ** 2
            + (0.4 - capture_40_sum / test_len) ** 2
            + (0.5 - capture_50_sum / test_len) ** 2
            + (0.6 - capture_60_sum / test_len) ** 2
            + (0.7 - capture_70_sum / test_len) ** 2
            + (0.8 - capture_80_sum / test_len) ** 2
            + (0.9 - capture_90_sum / test_len) ** 2
            + (1 - capture_all_sum / test_len) ** 2
        )
        self.CI_reliability_02_cal = (
            (0.1 - capture_10_20_sum / length_20) ** 2
            + (0.2 - capture_20_20_sum / length_20) ** 2
            + (0.3 - capture_30_20_sum / length_20) ** 2
            + (0.4 - capture_40_20_sum / length_20) ** 2
            + (0.5 - capture_50_20_sum / length_20) ** 2
            + (0.6 - capture_60_20_sum / length_20) ** 2
            + (0.7 - capture_70_20_sum / length_20) ** 2
            + (0.8 - capture_80_20_sum / length_20) ** 2
            + (0.9 - capture_90_20_sum / length_20) ** 2
            + (1 - capture_all_20_sum / length_20) ** 2
        )

        # Rank Histogram
        rank = []
        for a in range(0, len(Y_true)):
            n_lower = np.sum(np.greater(Y_true[a], Y_pred[:, a]))
            n_equal = np.sum(np.equal(Y_true[a], Y_pred[:, a]))
            deviate_rank = np.random.random_integers(0, n_equal)
            rank = np.append(rank, n_lower + deviate_rank)

        rank_hist = np.histogram(rank, bins=self.network_count + 1)
        delta = np.sum((rank_hist[0] - (test_len / ((self.network_count + 1)))) ** 2)
        delta_0 = self.network_count * test_len / (self.network_count + 1)
        self.delta_score_cal = delta / delta_0

        c = self.network_count

        alpha = np.zeros((test_len, (c + 1)))
        beta = np.zeros((test_len, (c + 1)))
        low_outlier = 0
        high_outlier = 0

        for a in range(0, test_len):
            observation = Y_true[a]
            forecast = np.sort(Y_pred[:, a])
            for b in range(1, c):
                if observation > forecast[b]:
                    alpha[a, b] = forecast[b] - forecast[b - 1]
                    beta[a, b] = 0
                elif forecast[b] > observation > forecast[b - 1]:
                    alpha[a, b] = observation - forecast[b - 1]
                    beta[a, b] = forecast[b] - observation
                else:
                    alpha[a, b] = 0
                    beta[a, b] = forecast[b] - forecast[b - 1]
            # overwrite boundaries in case of outliers
            if observation < forecast[0]:
                beta[a, 0] = forecast[0] - observation
                low_outlier += 1
            if observation > forecast[c - 1]:
                alpha[a, c] = observation - forecast[c - 1]
                high_outlier += 1

        alpha_bar = np.mean(alpha, axis=0)
        beta_bar = np.mean(beta, axis=0)
        g_bar = alpha_bar + beta_bar
        o_bar = beta_bar / (alpha_bar + beta_bar)

        if low_outlier > 0:
            o_bar[0] = low_outlier / test_len
            g_bar[0] = beta_bar[0] / o_bar[0]
        else:
            o_bar[0] = 0
            g_bar[0] = 0
        if high_outlier > 0:
            o_bar[c] = high_outlier / test_len
            g_bar[c] = alpha_bar[c] / o_bar[c]
        else:
            o_bar[c] = 0
            g_bar[c] = 0

        p_i = np.arange(0 / c, (c + 1) / c, 1 / c)

        self.CRPS_cal = np.sum(
            g_bar * ((1 - o_bar) * (p_i**2) + o_bar * ((1 - p_i) ** 2))
        )

        CI_x = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

        fig = plt.figure(figsize=(15, 10), dpi=100)
        gridspec.GridSpec(2, 3)

        plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        plt.axhline(0.2, c="k", ls="--", label="Point-of-consumption FRC = 0.2 mg/L")
        plt.scatter(
            FRC_X, Y_true, edgecolors="k", facecolors="None", s=20, label="Observed"
        )
        plt.scatter(
            FRC_X,
            np.median(Y_pred, axis=0),
            facecolors="r",
            edgecolors="None",
            s=10,
            label="Forecast Median",
        )
        plt.vlines(
            FRC_X,
            np.min(Y_pred, axis=0),
            np.max(Y_pred, axis=0),
            color="r",
            label="Forecast Range",
        )
        plt.xlabel("Point-of-Distribution FRC (mg/L)")
        plt.ylabel("Point-of-Consumption FRC (mg/L)")
        plt.xlim([0, np.max(FRC_X)])
        plt.legend(
            bbox_to_anchor=(0.001, 0.999),
            shadow=False,
            labelspacing=0.1,
            fontsize="small",
            handletextpad=0.1,
            loc="upper left",
        )
        ax1 = fig.axes[0]
        ax1.set_title("(a)", y=0.88, x=0.05)

        plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
        plt.plot(CI_x, CI_x, c="k")
        plt.scatter(CI_x, capture, label="All observations")
        plt.scatter(CI_x, capture_20, label="Point-of-Consumption FRC below 0.2 mg/L")
        plt.xlabel("Ensemble Confidence Interval")
        plt.ylabel("Percent Capture")
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        plt.legend(
            bbox_to_anchor=(0.001, 0.999),
            shadow=False,
            labelspacing=0.1,
            fontsize="small",
            handletextpad=0.1,
            loc="upper left",
        )
        ax2 = fig.axes[1]
        ax2.set_title("(b)", y=0.88, x=0.05)

        plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)
        plt.hist(rank, bins=(self.network_count + 1), density=True)
        plt.xlabel("Rank")
        plt.ylabel("Probability")
        ax3 = fig.axes[2]
        ax3.set_title("(c)", y=0.88, x=0.05)
        plt.savefig(
            os.path.splitext(filename)[0] + "_Calibration_Diagnostic_Figs.png",
            format="png",
            bbox_inches="tight",
        )

        plt.close()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format="png", bbox_inches="tight")
        myStringIOBytes.seek(0)
        my_base_64_pngData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_pngData

    def get_bw(self):
        Y_true = np.array(self.targets)
        Y_pred = np.array(self.calibration_predictions)[:, :, 0]

        s2 = []
        xt_yt = []

        for a in range(0, len(Y_true)):
            observation = Y_true[a]
            forecast = np.sort(Y_pred[:, a])
            s2 = np.append(s2, np.var(forecast))
            xt_yt = np.append(xt_yt, (np.mean(forecast) - observation) ** 2)
        WB_bw = np.mean(xt_yt) - (1 + 1 / self.network_count) * np.mean(s2)
        return WB_bw

    def post_process_performance_eval(self, bandwidth):
        Y_true = np.squeeze(np.array(self.targets))
        Y_pred = np.array(self.calibration_predictions)[:, :, 0]

        test_len = len(Y_true)

        min_CI = []
        max_CI = []
        CI_90_Lower = []
        CI_90_Upper = []
        CI_80_Lower = []
        CI_80_Upper = []
        CI_70_Lower = []
        CI_70_Upper = []
        CI_60_Lower = []
        CI_60_Upper = []
        CI_50_Lower = []
        CI_50_Upper = []
        CI_40_Lower = []
        CI_40_Upper = []
        CI_30_Lower = []
        CI_30_Upper = []
        CI_20_Lower = []
        CI_20_Upper = []
        CI_10_Lower = []
        CI_10_Upper = []
        CI_median = []

        CRPS = []
        Kernel_Risk = []

        evaluation_range = np.arange(-10, 10.001, 0.001)
        # compute CRPS as well as the confidence intervals of each ensemble forecast
        for a in range(0, test_len):
            scipy_kde = scipy.stats.gaussian_kde(Y_pred[:, a], bw_method=bandwidth)
            scipy_pdf = scipy_kde.evaluate(evaluation_range) * 0.001
            scipy_cdf = np.cumsum(scipy_pdf)
            min_CI = np.append(
                min_CI, evaluation_range[np.max(np.where(scipy_cdf == 0)[0])]
            )
            max_CI = np.append(max_CI, evaluation_range[np.argmax(scipy_cdf)])

            CI_90_Lower = np.append(
                CI_90_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.05)))]
            )
            CI_90_Upper = np.append(
                CI_90_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.95)))]
            )
            CI_80_Lower = np.append(
                CI_80_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.1)))]
            )
            CI_80_Upper = np.append(
                CI_80_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.9)))]
            )
            CI_70_Lower = np.append(
                CI_70_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.15)))]
            )
            CI_70_Upper = np.append(
                CI_70_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.85)))]
            )
            CI_60_Lower = np.append(
                CI_60_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.2)))]
            )
            CI_60_Upper = np.append(
                CI_60_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.8)))]
            )
            CI_50_Lower = np.append(
                CI_50_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.25)))]
            )
            CI_50_Upper = np.append(
                CI_50_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.75)))]
            )
            CI_40_Lower = np.append(
                CI_40_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.3)))]
            )
            CI_40_Upper = np.append(
                CI_40_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.7)))]
            )
            CI_30_Lower = np.append(
                CI_30_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.35)))]
            )
            CI_30_Upper = np.append(
                CI_30_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.65)))]
            )
            CI_20_Lower = np.append(
                CI_20_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.4)))]
            )
            CI_20_Upper = np.append(
                CI_20_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.6)))]
            )
            CI_10_Lower = np.append(
                CI_10_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.45)))]
            )
            CI_10_Upper = np.append(
                CI_10_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.55)))]
            )
            CI_median = np.append(
                CI_median, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.50)))]
            )
            Kernel_Risk = np.append(Kernel_Risk, scipy_kde.integrate_box_1d(-10, 0.2))

            Heaviside = (evaluation_range >= Y_true[a]).astype(int)
            CRPS_dif = (scipy_cdf - Heaviside) ** 2
            CRPS = np.append(CRPS, np.sum(CRPS_dif * 0.001))
        mean_CRPS = np.mean(CRPS)

        capture_all = (
            np.less_equal(Y_true, max_CI) * np.greater_equal(Y_true, min_CI) * 1
        )
        capture_90 = (
            np.less_equal(Y_true, CI_90_Upper)
            * np.greater_equal(Y_true, CI_90_Lower)
            * 1
        )
        capture_80 = (
            np.less_equal(Y_true, CI_80_Upper)
            * np.greater_equal(Y_true, CI_80_Lower)
            * 1
        )
        capture_70 = (
            np.less_equal(Y_true, CI_70_Upper)
            * np.greater_equal(Y_true, CI_70_Lower)
            * 1
        )
        capture_60 = (
            np.less_equal(Y_true, CI_60_Upper)
            * np.greater_equal(Y_true, CI_60_Lower)
            * 1
        )
        capture_50 = (
            np.less_equal(Y_true, CI_50_Upper)
            * np.greater_equal(Y_true, CI_50_Lower)
            * 1
        )
        capture_40 = (
            np.less_equal(Y_true, CI_40_Upper)
            * np.greater_equal(Y_true, CI_40_Lower)
            * 1
        )
        capture_30 = (
            np.less_equal(Y_true, CI_30_Upper)
            * np.greater_equal(Y_true, CI_30_Lower)
            * 1
        )
        capture_20 = (
            np.less_equal(Y_true, CI_20_Upper)
            * np.greater_equal(Y_true, CI_20_Lower)
            * 1
        )
        capture_10 = (
            np.less_equal(Y_true, CI_10_Upper)
            * np.greater_equal(Y_true, CI_10_Lower)
            * 1
        )

        length_20 = np.sum(np.less(Y_true, 0.2))
        capture_all_20 = capture_all * np.less(Y_true, 0.2)
        capture_90_20 = capture_90 * np.less(Y_true, 0.2)
        capture_80_20 = capture_80 * np.less(Y_true, 0.2)
        capture_70_20 = capture_70 * np.less(Y_true, 0.2)
        capture_60_20 = capture_60 * np.less(Y_true, 0.2)
        capture_50_20 = capture_50 * np.less(Y_true, 0.2)
        capture_40_20 = capture_40 * np.less(Y_true, 0.2)
        capture_30_20 = capture_30 * np.less(Y_true, 0.2)
        capture_20_20 = capture_20 * np.less(Y_true, 0.2)
        capture_10_20 = capture_10 * np.less(Y_true, 0.2)

        capture_all_sum = np.sum(capture_all)
        capture_90_sum = np.sum(capture_90)
        capture_80_sum = np.sum(capture_80)
        capture_70_sum = np.sum(capture_70)
        capture_60_sum = np.sum(capture_60)
        capture_50_sum = np.sum(capture_50)
        capture_40_sum = np.sum(capture_40)
        capture_30_sum = np.sum(capture_30)
        capture_20_sum = np.sum(capture_20)
        capture_10_sum = np.sum(capture_10)

        capture_all_20_sum = np.sum(capture_all_20)
        capture_90_20_sum = np.sum(capture_90_20)
        capture_80_20_sum = np.sum(capture_80_20)
        capture_70_20_sum = np.sum(capture_70_20)
        capture_60_20_sum = np.sum(capture_60_20)
        capture_50_20_sum = np.sum(capture_50_20)
        capture_40_20_sum = np.sum(capture_40_20)
        capture_30_20_sum = np.sum(capture_30_20)
        capture_20_20_sum = np.sum(capture_20_20)
        capture_10_20_sum = np.sum(capture_10_20)

        capture_sum_squares = (
            (0.1 - capture_10_sum / test_len) ** 2
            + (0.2 - capture_20_sum / test_len) ** 2
            + (0.3 - capture_30_sum / test_len) ** 2
            + (0.4 - capture_40_sum / test_len) ** 2
            + (0.5 - capture_50_sum / test_len) ** 2
            + (0.6 - capture_60_sum / test_len) ** 2
            + (0.7 - capture_70_sum / test_len) ** 2
            + (0.8 - capture_80_sum / test_len) ** 2
            + (0.9 - capture_90_sum / test_len) ** 2
            + (1 - capture_all_sum / test_len) ** 2
        )
        capture_20_sum_squares = (
            (0.1 - capture_10_20_sum / length_20) ** 2
            + (0.2 - capture_20_20_sum / length_20) ** 2
            + (0.3 - capture_30_20_sum / length_20) ** 2
            + (0.4 - capture_40_20_sum / length_20) ** 2
            + (0.5 - capture_50_20_sum / length_20) ** 2
            + (0.6 - capture_60_20_sum / length_20) ** 2
            + (0.7 - capture_70_20_sum / length_20) ** 2
            + (0.8 - capture_80_20_sum / length_20) ** 2
            + (0.9 - capture_90_20_sum / length_20) ** 2
            + (1 - capture_all_20_sum / length_20) ** 2
        )

        return (
            mean_CRPS,
            capture_sum_squares,
            capture_20_sum_squares,
            capture_all_sum / test_len,
            capture_all_20_sum / length_20,
        )

    def post_process_cal(self):

        self.WB_bandwidth = self.get_bw()
        (
            self.CRPS_post_cal,
            self.CI_reliability_post_cal,
            self.CI_reliability_02_post_cal,
            self.percent_capture_post_cal,
            self.percent_capture_02_post_cal,
        ) = self.post_process_performance_eval(self.WB_bandwidth)

        CRPS_Skill = (self.CRPS_post_cal - self.CRPS_cal) / (0 - self.CRPS_cal)
        CI_Skill = (self.CI_reliability_post_cal - self.CI_reliability_cal) / (
            0 - self.CI_reliability_cal
        )
        CI_20_Skill = (self.CI_reliability_02_post_cal - self.CI_reliability_02_cal) / (
            0 - self.CI_reliability_02_cal
        )
        PC_Skill = (self.percent_capture_post_cal - self.percent_capture_cal) / (
            1 - self.percent_capture_cal
        )
        PC_20_Skill = (
            self.percent_capture_02_post_cal - self.percent_capture_02_cal
        ) / (1 - self.percent_capture_02_cal)

        Net_Score = CRPS_Skill + CI_Skill + CI_20_Skill + PC_Skill + PC_20_Skill

        if Net_Score > 0:
            self.post_process_check = True
        else:
            self.post_process_check = False

    def full_performance_evaluation(self, directory):
        x_norm = self.predictors_scaler.transform(self.predictors)
        t_norm = self.targets_scaler.transform(self.targets)

        base_model = self.model
        network_path = os.path.join(directory, "base_network.h5")
        base_model.save(network_path)

        x_cal_norm, x_test_norm, t_cal_norm, t_test_norm = train_test_split(
            x_norm, t_norm, test_size=0.25, shuffle=False, random_state=10
        )
        self.verifying_observations = self.targets_scaler.inverse_transform(t_test_norm)
        self.test_x_data = self.predictors_scaler.inverse_transform(x_test_norm)

        early_stopping_monitor = keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True
        )

        self.verifying_predictions = []

        for i in range(0, self.network_count):
            keras.backend.clear_session()

            self.model = keras.models.load_model(network_path)

            x_norm_train, x_norm_val, t_norm_train, t_norm_val = train_test_split(
                x_cal_norm,
                t_cal_norm,
                train_size=1 / 3,
                shuffle=True,
                random_state=i**2,
            )

            new_weights = [
                np.random.uniform(-0.05, 0.05, w.shape)
                for w in self.model.get_weights()
            ]
            self.model.set_weights(new_weights)
            self.model.fit(
                x_norm_train,
                t_norm_train,
                epochs=self.epochs,
                validation_data=(x_norm_val, t_norm_val),
                callbacks=[early_stopping_monitor],
                verbose=0,
                batch_size=len(t_norm_train),
            )

            self.verifying_predictions.append(
                self.targets_scaler.inverse_transform(
                    self.model.predict(x_test_norm, verbose=0)
                )
            )

        Y_true = np.array(self.verifying_observations)
        Y_pred = np.array(self.verifying_predictions)
        FRC_X = self.test_x_data[:, 0]

        capture_all = (
            np.less_equal(Y_true, np.max(Y_pred, axis=0))
            * np.greater_equal(Y_true, np.min(Y_pred, axis=0))
            * 1
        )
        capture_90 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 95, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 5, axis=0))
            * 1
        )
        capture_80 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 90, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 10, axis=0))
            * 1
        )
        capture_70 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 85, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 15, axis=0))
            * 1
        )
        capture_60 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 80, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 20, axis=0))
            * 1
        )
        capture_50 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 75, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 25, axis=0))
            * 1
        )
        capture_40 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 70, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 30, axis=0))
            * 1
        )
        capture_30 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 65, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 35, axis=0))
            * 1
        )
        capture_20 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 60, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 40, axis=0))
            * 1
        )
        capture_10 = (
            np.less_equal(Y_true, np.percentile(Y_pred, 55, axis=0))
            * np.greater_equal(Y_true, np.percentile(Y_pred, 45, axis=0))
            * 1
        )

        capture_all_20 = capture_all * np.less(Y_true, 0.2)
        capture_90_20 = capture_90 * np.less(Y_true, 0.2)
        capture_80_20 = capture_80 * np.less(Y_true, 0.2)
        capture_70_20 = capture_70 * np.less(Y_true, 0.2)
        capture_60_20 = capture_60 * np.less(Y_true, 0.2)
        capture_50_20 = capture_50 * np.less(Y_true, 0.2)
        capture_40_20 = capture_40 * np.less(Y_true, 0.2)
        capture_30_20 = capture_30 * np.less(Y_true, 0.2)
        capture_20_20 = capture_20 * np.less(Y_true, 0.2)
        capture_10_20 = capture_10 * np.less(Y_true, 0.2)

        length_20 = np.sum(np.less(Y_true, 0.2))
        test_len = len(Y_true)
        capture_all_sum = np.sum(capture_all)
        capture_90_sum = np.sum(capture_90)
        capture_80_sum = np.sum(capture_80)
        capture_70_sum = np.sum(capture_70)
        capture_60_sum = np.sum(capture_60)
        capture_50_sum = np.sum(capture_50)
        capture_40_sum = np.sum(capture_40)
        capture_30_sum = np.sum(capture_30)
        capture_20_sum = np.sum(capture_20)
        capture_10_sum = np.sum(capture_10)

        capture_all_20_sum = np.sum(capture_all_20)
        capture_90_20_sum = np.sum(capture_90_20)
        capture_80_20_sum = np.sum(capture_80_20)
        capture_70_20_sum = np.sum(capture_70_20)
        capture_60_20_sum = np.sum(capture_60_20)
        capture_50_20_sum = np.sum(capture_50_20)
        capture_40_20_sum = np.sum(capture_40_20)
        capture_30_20_sum = np.sum(capture_30_20)
        capture_20_20_sum = np.sum(capture_20_20)
        capture_10_20_sum = np.sum(capture_10_20)

        capture = [
            capture_10_sum / test_len,
            capture_20_sum / test_len,
            capture_30_sum / test_len,
            capture_40_sum / test_len,
            capture_50_sum / test_len,
            capture_60_sum / test_len,
            capture_70_sum / test_len,
            capture_80_sum / test_len,
            capture_90_sum / test_len,
            capture_all_sum / test_len,
        ]
        capture_20 = [
            capture_10_20_sum / length_20,
            capture_20_20_sum / length_20,
            capture_30_20_sum / length_20,
            capture_40_20_sum / length_20,
            capture_50_20_sum / length_20,
            capture_60_20_sum / length_20,
            capture_70_20_sum / length_20,
            capture_80_20_sum / length_20,
            capture_90_20_sum / length_20,
            capture_all_20_sum / length_20,
        ]

        self.percent_capture_cal = capture_all_sum / test_len

        self.percent_capture_02_cal = capture_all_20_sum / length_20

        self.CI_reliability_cal = (
            (0.1 - capture_10_sum / test_len) ** 2
            + (0.2 - capture_20_sum / test_len) ** 2
            + (0.3 - capture_30_sum / test_len) ** 2
            + (0.4 - capture_40_sum / test_len) ** 2
            + (0.5 - capture_50_sum / test_len) ** 2
            + (0.6 - capture_60_sum / test_len) ** 2
            + (0.7 - capture_70_sum / test_len) ** 2
            + (0.8 - capture_80_sum / test_len) ** 2
            + (0.9 - capture_90_sum / test_len) ** 2
            + (1 - capture_all_sum / test_len) ** 2
        )
        self.CI_reliability_02_cal = (
            (0.1 - capture_10_20_sum / length_20) ** 2
            + (0.2 - capture_20_20_sum / length_20) ** 2
            + (0.3 - capture_30_20_sum / length_20) ** 2
            + (0.4 - capture_40_20_sum / length_20) ** 2
            + (0.5 - capture_50_20_sum / length_20) ** 2
            + (0.6 - capture_60_20_sum / length_20) ** 2
            + (0.7 - capture_70_20_sum / length_20) ** 2
            + (0.8 - capture_80_20_sum / length_20) ** 2
            + (0.9 - capture_90_20_sum / length_20) ** 2
            + (1 - capture_all_20_sum / length_20) ** 2
        )

        # Rank Histogram
        rank = []
        for a in range(0, len(Y_true)):
            n_lower = np.sum(np.greater(Y_true[a], Y_pred[:, a]))
            n_equal = np.sum(np.equal(Y_true[a], Y_pred[:, a]))
            deviate_rank = np.random.random_integers(0, n_equal)
            rank = np.append(rank, n_lower + deviate_rank)

        rank_hist = np.histogram(rank, bins=self.network_count + 1)
        delta = np.sum((rank_hist[0] - (test_len / ((self.network_count + 1)))) ** 2)
        delta_0 = self.network_count * test_len / (self.network_count + 1)
        self.delta_score_cal = delta / delta_0

        CI_x = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

        fig = plt.figure(figsize=(15, 10), dpi=100)
        gridspec.GridSpec(2, 3)

        plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        plt.axhline(0.2, c="k", ls="--", label="Point-of-consumption FRC = 0.2 mg/L")
        plt.scatter(
            FRC_X, Y_true, edgecolors="k", facecolors="None", s=20, label="Observed"
        )
        plt.scatter(
            FRC_X,
            np.median(Y_pred, axis=0),
            facecolors="r",
            edgecolors="None",
            s=10,
            label="Forecast Median",
        )
        plt.vlines(
            FRC_X,
            np.min(Y_pred, axis=0),
            np.max(Y_pred, axis=0),
            color="r",
            label="Forecast Range",
        )
        plt.xlabel("Point-of-Distribution FRC (mg/L)")
        plt.ylabel("Point-of-Consumption FRC (mg/L)")

        plt.subplot2grid((2, 3), (0, 2), colspan=1, rowspan=1)
        plt.plot(CI_x, CI_x, c="k")
        plt.scatter(CI_x, capture)
        plt.scatter(CI_x, capture_20)
        plt.xlabel("Ensemble Confidence Interval")
        plt.ylabel("Percent Capture")
        plt.ylim([0, 1])
        plt.xlim([0, 1])

        plt.subplot2grid((2, 3), (1, 2), colspan=1, rowspan=1)
        plt.hist(rank, bins=(self.network_count + 1), density=True)
        plt.xlabel("Rank")
        plt.ylabel("Probability")
        diag_fig_path = os.path.join(directory, "Verification_Diagnostic_Figs.png")
        plt.savefig(diag_fig_path, format="png")
        plt.close()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format="png")
        myStringIOBytes.seek(0)
        my_base_64_pngData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_pngData

    def set_inputs_for_table(self, storage_target):

        frc = np.arange(0.20, 2.05, 0.05)
        lag_time = [storage_target for i in range(0, len(frc))]
        am_collect = [0 for i in range(0, len(frc))]
        pm_collect = [1 for i in range(0, len(frc))]
        temp_med_am = {
            "ts_frc": frc,
            "elapsed time": lag_time,
            "time of collection (0=AM, 1=PM)": am_collect,
        }
        temp_med_pm = {
            "ts_frc": frc,
            "elapsed time": lag_time,
            "time of collection (0=AM, 1=PM)": pm_collect,
        }
        temp_95_am = {
            "ts_frc": frc,
            "elapsed time": lag_time,
            "time of collection (0=AM, 1=PM)": am_collect,
        }
        temp_95_pm = {
            "ts_frc": frc,
            "elapsed time": lag_time,
            "time of collection (0=AM, 1=PM)": pm_collect,
        }

        if WATTEMP in self.datainputs.columns:
            watt_med = [self.median_wattemp for i in range(0, len(frc))]
            watt_95 = [self.upper95_wattemp for i in range(0, len(frc))]
            temp_med_am.update({"ts_wattemp": watt_med})
            temp_med_pm.update({"ts_wattemp": watt_med})
            temp_95_am.update({"ts_wattemp": watt_95})
            temp_95_pm.update({"ts_wattemp": watt_95})
        if COND in self.datainputs.columns:
            cond_med = [self.median_cond for i in range(0, len(frc))]
            cond_95 = [self.upper95_cond for i in range(0, len(frc))]
            temp_med_am.update({"ts_cond": cond_med})
            temp_med_pm.update({"ts_cond": cond_med})
            temp_95_am.update({"ts_cond": cond_95})
            temp_95_pm.update({"ts_cond": cond_95})

        self.avg_case_predictors_am = pd.DataFrame(temp_med_am)
        self.avg_case_predictors_pm = pd.DataFrame(temp_med_pm)
        self.worst_case_predictors_am = pd.DataFrame(temp_95_am)
        self.worst_case_predictors_pm = pd.DataFrame(temp_95_pm)

    def post_process_predictions(self, results_table_frc):
        # results_table_frc=results_table_frc.to_numpy()
        evaluation_range = np.arange(-10, 10.001, 0.001)
        test1_frc = np.arange(0.2, 2.05, 0.05)
        bandwidth = self.WB_bandwidth
        Max_CI = []
        Min_CI = []
        CI_99_Upper = []
        CI_99_Lower = []
        CI_95_Upper = []
        CI_95_Lower = []
        Median_Results = []
        risk_00_kernel_frc = []
        risk_20_kernel_frc = []
        risk_25_kernel_frc = []
        risk_30_kernel_frc = []

        for a in range(0, len(test1_frc)):
            scipy_kde = scipy.stats.gaussian_kde(
                results_table_frc[a, :], bw_method=bandwidth
            )
            risk_00_kernel_frc = np.append(
                risk_00_kernel_frc, scipy_kde.integrate_box_1d(-10, 0)
            )
            risk_20_kernel_frc = np.append(
                risk_20_kernel_frc, scipy_kde.integrate_box_1d(-10, 0.2)
            )
            risk_25_kernel_frc = np.append(
                risk_25_kernel_frc, scipy_kde.integrate_box_1d(-10, 0.25)
            )
            risk_30_kernel_frc = np.append(
                risk_30_kernel_frc, scipy_kde.integrate_box_1d(-10, 0.3)
            )
            scipy_pdf = scipy_kde.evaluate(evaluation_range) * 0.001
            scipy_cdf = np.cumsum(scipy_pdf)

            Min_CI = np.append(
                Min_CI, evaluation_range[np.max(np.where(scipy_cdf == 0)[0])]
            )
            Max_CI = np.append(Max_CI, evaluation_range[np.argmax(scipy_cdf)])
            CI_99_Upper = np.append(
                CI_99_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.995)))]
            )
            CI_99_Lower = np.append(
                CI_99_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.005)))]
            )
            CI_95_Upper = np.append(
                CI_95_Upper, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.975)))]
            )
            CI_95_Lower = np.append(
                CI_95_Lower, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.025)))]
            )
            Median_Results = np.append(
                Median_Results, evaluation_range[np.argmin(np.abs((scipy_cdf - 0.5)))]
            )
        temp_key = {
            "Tapstand FRC": np.arange(0.20, 2.05, 0.05),
            "median": Median_Results,
            "Ensemble Minimum": Min_CI,
            "Ensemble Maximum": Max_CI,
            "Lower 99 CI": CI_99_Lower,
            "Upper 99 CI": CI_99_Upper,
            "Lower 95 CI": CI_95_Lower,
            "Upper 95 CI": CI_95_Upper,
            "probability==0": risk_00_kernel_frc,
            "probability<=0.20": risk_20_kernel_frc,
            "probability<=0.25": risk_25_kernel_frc,
            "probability<=0.30": risk_30_kernel_frc,
        }
        post_processed_df = pd.DataFrame(temp_key)
        return post_processed_df

    def predict(self):
        """
        To make the predictions, a pretrained model must be loaded using the import_pretrained_model() method.
        The SWOT ANN uses an ensemble of 200 ANNs. All of the 200 ANNs make a prediction on the inputs and the results are
        stored. The median of all the 200 predictions is calculated and stored here.

        The method also calculates the probabilities of the target FRC levels to be less than 0.2, 0.25 and 0.3 mg/L respectively.

        The predictions are target FRC values in  mg/L, and the probability values range from 0 to 1.

        All of the above results are saved in the self.results class field.

        V2.0 Notes: If at least 1 WQ variable is provided, we do a scenario analysis, providing targets for the average case
        (median water quality) and the "worst case" using the upper 95th percentile water quality
        """

        # Initialize empty arrays for the probabilities to be appended in.

        avg_case_results_am = {}
        avg_case_results_pm = {}
        worst_case_results_am = {}
        worst_case_results_pm = {}

        # Normalize the inputs using the input scaler loaded
        input_scaler = self.predictors_scaler
        avg_case_inputs_norm_am = input_scaler.transform(self.avg_case_predictors_am)
        avg_case_inputs_norm_pm = input_scaler.transform(self.avg_case_predictors_pm)
        worst_case_inputs_norm_am = input_scaler.transform(
            self.worst_case_predictors_am
        )
        worst_case_inputs_norm_pm = input_scaler.transform(
            self.worst_case_predictors_pm
        )

        ##AVERAGE CASE TARGET w AM COLLECTION

        # Iterate through all loaded pretrained networks, make predictions based on the inputs,
        # calculate the median of the predictions and store everything to self.results
        for j in range(0, self.network_count):
            key = "se4_frc_net-" + str(j)
            predictions = self.targets_scaler.inverse_transform(
                self.trained_models["model_" + str(j)].predict(
                    avg_case_inputs_norm_am, verbose=0
                )
            ).tolist()
            temp = sum(predictions, [])
            avg_case_results_am.update({key: temp})
        self.avg_case_results_am = pd.DataFrame(avg_case_results_am)
        self.avg_case_results_am["median"] = self.avg_case_results_am.median(axis=1)

        for i in self.avg_case_predictors_am.keys():
            self.avg_case_results_am.update(
                {i: self.avg_case_predictors_am[i].tolist()}
            )
            self.avg_case_results_am[i] = self.avg_case_predictors_am[i].tolist()

        # Include the inputs/predictors in the self.results variable
        for i in self.avg_case_predictors_am.keys():
            self.avg_case_results_am.update(
                {i: self.avg_case_predictors_am[i].tolist()}
            )
            self.avg_case_results_am[i] = self.avg_case_predictors_am[i].tolist()

        if self.post_process_check == False:
            # Calculate all the probability fields and store them to self.results
            # results_table_frc_avg = self.results.iloc[:, 0:(self.network_count - 1)]
            self.avg_case_results_am["probability<=0.20"] = (
                np.sum(
                    np.less_equal(
                        self.avg_case_results_am.iloc[:, 0 : (self.network_count - 1)],
                        0.2,
                    ),
                    axis=1,
                )
                / self.network_count
            )
            self.avg_case_results_am["probability<=0.25"] = (
                np.sum(
                    np.less_equal(
                        self.avg_case_results_am.iloc[:, 0 : (self.network_count - 1)],
                        0.25,
                    ),
                    axis=1,
                )
                / self.network_count
            )
            self.avg_case_results_am["probability<=0.30"] = (
                np.sum(
                    np.less_equal(
                        self.avg_case_results_am.iloc[:, 0 : (self.network_count - 1)],
                        0.3,
                    ),
                    axis=1,
                )
                / self.network_count
            )
        else:
            self.avg_case_results_am_post = self.post_process_predictions(
                self.avg_case_results_am.iloc[
                    :, 0 : (self.network_count - 1)
                ].to_numpy()
            )

        ##AVERAGE CASE TARGET w PM COLLECTION

        # Iterate through all loaded pretrained networks, make predictions based on the inputs,
        # calculate the median of the predictions and store everything to self.results
        for j in range(0, self.network_count):
            key = "se4_frc_net-" + str(j)
            predictions = self.targets_scaler.inverse_transform(
                self.trained_models["model_" + str(j)].predict(
                    avg_case_inputs_norm_pm, verbose=0
                )
            ).tolist()
            temp = sum(predictions, [])
            avg_case_results_pm.update({key: temp})
        self.avg_case_results_pm = pd.DataFrame(avg_case_results_pm)
        self.avg_case_results_pm["median"] = self.avg_case_results_pm.median(axis=1)

        # Include the inputs/predictors in the self.results variable
        for i in self.avg_case_predictors_pm.keys():
            self.avg_case_results_pm.update(
                {i: self.avg_case_predictors_pm[i].tolist()}
            )
            self.avg_case_results_pm[i] = self.avg_case_predictors_pm[i].tolist()

        if self.post_process_check == False:
            # Calculate all the probability fields and store them to self.results
            # results_table_frc_avg = self.results.iloc[:, 0:(self.network_count - 1)]
            self.avg_case_results_pm["probability<=0.20"] = (
                np.sum(
                    np.less(
                        self.avg_case_results_pm.iloc[:, 0 : (self.network_count - 1)],
                        0.2,
                    ),
                    axis=1,
                )
                / self.network_count
            )
            self.avg_case_results_pm["probability<=0.25"] = (
                np.sum(
                    np.less(
                        self.avg_case_results_pm.iloc[:, 0 : (self.network_count - 1)],
                        0.25,
                    ),
                    axis=1,
                )
                / self.network_count
            )
            self.avg_case_results_pm["probability<=0.30"] = (
                np.sum(
                    np.less(
                        self.avg_case_results_pm.iloc[:, 0 : (self.network_count - 1)],
                        0.3,
                    ),
                    axis=1,
                )
                / self.network_count
            )

        else:
            self.avg_case_results_pm_post = self.post_process_predictions(
                self.avg_case_results_pm.iloc[
                    :, 0 : (self.network_count - 1)
                ].to_numpy()
            )

        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            ##WORST CASE TARGET w AM COLLECTION

            for j in range(0, self.network_count):
                key = "se4_frc_net-" + str(j)
                predictions = self.targets_scaler.inverse_transform(
                    self.trained_models["model_" + str(j)].predict(
                        worst_case_inputs_norm_am, verbose=0
                    )
                ).tolist()
                temp = sum(predictions, [])
                worst_case_results_am.update({key: temp})
            self.worst_case_results_am = pd.DataFrame(worst_case_results_am)
            self.worst_case_results_am["median"] = self.worst_case_results_am.median(
                axis=1
            )

            # Include the inputs/predictors in the self.results variable
            for i in self.worst_case_predictors_am.keys():
                self.worst_case_results_am.update(
                    {i: self.worst_case_predictors_am[i].tolist()}
                )
                self.worst_case_results_am[i] = self.worst_case_predictors_am[
                    i
                ].tolist()

            if self.post_process_check == False:
                # Calculate all the probability fields and store them to self.results
                # results_table_frc_avg = self.results.iloc[:, 0:(self.network_count - 1)]
                self.worst_case_results_am["probability<=0.20"] = (
                    np.sum(
                        np.less(
                            self.worst_case_results_am.iloc[
                                :, 0 : (self.network_count - 1)
                            ],
                            0.2,
                        ),
                        axis=1,
                    )
                    / self.network_count
                )
                self.worst_case_results_am["probability<=0.25"] = (
                    np.sum(
                        np.less(
                            self.worst_case_results_am.iloc[
                                :, 0 : (self.network_count - 1)
                            ],
                            0.25,
                        ),
                        axis=1,
                    )
                    / self.network_count
                )
                self.worst_case_results_am["probability<=0.30"] = (
                    np.sum(
                        np.less(
                            self.worst_case_results_am.iloc[
                                :, 0 : (self.network_count - 1)
                            ],
                            0.3,
                        ),
                        axis=1,
                    )
                    / self.network_count
                )
            else:
                self.worst_case_results_am_post = self.post_process_predictions(
                    self.worst_case_results_am.iloc[
                        :, 0 : (self.network_count - 1)
                    ].to_numpy()
                )

            ##WORST CASE TARGET w PM COLLECTION

            for j in range(0, self.network_count):
                key = "se4_frc_net-" + str(j)
                predictions = self.targets_scaler.inverse_transform(
                    self.trained_models["model_" + str(j)].predict(
                        worst_case_inputs_norm_pm, verbose=0
                    )
                ).tolist()
                temp = sum(predictions, [])
                worst_case_results_pm.update({key: temp})
            self.worst_case_results_pm = pd.DataFrame(worst_case_results_pm)
            self.worst_case_results_pm["median"] = self.worst_case_results_pm.median(
                axis=1
            )

            # Include the inputs/predictors in the self.results variable
            for i in self.worst_case_predictors_pm.keys():
                self.worst_case_results_pm.update(
                    {i: self.worst_case_predictors_pm[i].tolist()}
                )
                self.worst_case_results_pm[i] = self.worst_case_predictors_pm[
                    i
                ].tolist()

            if self.post_process_check == False:
                # Calculate all the probability fields and store them to self.results
                # results_table_frc_avg = self.results.iloc[:, 0:(self.network_count - 1)]
                self.worst_case_results_pm["probability<=0.20"] = (
                    np.sum(
                        np.less(
                            self.worst_case_results_pm.iloc[
                                :, 0 : (self.network_count - 1)
                            ],
                            0.2,
                        ),
                        axis=1,
                    )
                    / self.network_count
                )
                self.worst_case_results_pm["probability<=0.25"] = (
                    np.sum(
                        np.less(
                            self.worst_case_results_pm.iloc[
                                :, 0 : (self.network_count - 1)
                            ],
                            0.25,
                        ),
                        axis=1,
                    )
                    / self.network_count
                )
                self.worst_case_results_pm["probability<=0.30"] = (
                    np.sum(
                        np.less(
                            self.worst_case_results_pm.iloc[
                                :, 0 : (self.network_count - 1)
                            ],
                            0.3,
                        ),
                        axis=1,
                    )
                    / self.network_count
                )
            else:
                self.worst_case_results_pm_post = self.post_process_predictions(
                    self.worst_case_results_pm.iloc[
                        :, 0 : (self.network_count - 1)
                    ].to_numpy()
                )

    def results_visualization(self, filename, storage_target):
        test1_frc = np.arange(0.2, 2.05, 0.05)
        # Variables to plot - Full range, 95th percentile, 99th percentile, median, the three risks

        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            if self.post_process_check == False:

                results_table_frc_avg_am = self.avg_case_results_am.iloc[
                    :, 0 : (self.network_count - 1)
                ]
                results_table_frc_avg_pm = self.avg_case_results_pm.iloc[
                    :, 0 : (self.network_count - 1)
                ]
                results_table_frc_worst_am = self.worst_case_results_am.iloc[
                    :, 0 : (self.network_count - 1)
                ]
                results_table_frc_worst_pm = self.worst_case_results_pm.iloc[
                    :, 0 : (self.network_count - 1)
                ]
                preds_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                    2, 2, figsize=(6.69, 6.69), dpi=300
                )
                ax1.fill_between(
                    test1_frc,
                    np.percentile(results_table_frc_avg_am, 97.5, axis=1),
                    np.percentile(results_table_frc_avg_am, 2.5, axis=1),
                    facecolor="#ffa600",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax1.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax1.plot(
                    test1_frc,
                    np.min(results_table_frc_avg_am, axis=1),
                    "#ffa600",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax1.plot(
                    test1_frc,
                    np.max(results_table_frc_avg_am, axis=1),
                    "#ffa600",
                    linewidth=0.5,
                )
                ax1.plot(
                    test1_frc,
                    np.median(results_table_frc_avg_am, axis=1),
                    "#ffa600",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax1.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax1.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax1.set_xlim([0.19, 2.0])
                ax1.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax1.set_xlabel("Tap Stand FRC (mg/L)")
                ax1.set_ylabel("Household FRC (mg/L)")
                ax1.set_title("Average Case - AM Collection")

                ax2.fill_between(
                    test1_frc,
                    np.percentile(results_table_frc_avg_pm, 97.5, axis=1),
                    np.percentile(results_table_frc_avg_pm, 2.5, axis=1),
                    facecolor="#ffa600",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax2.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax2.plot(
                    test1_frc,
                    np.min(results_table_frc_avg_pm, axis=1),
                    "#ffa600",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax2.plot(
                    test1_frc,
                    np.max(results_table_frc_avg_pm, axis=1),
                    "#ffa600",
                    linewidth=0.5,
                )
                ax2.plot(
                    test1_frc,
                    np.median(results_table_frc_avg_pm, axis=1),
                    "#ffa600",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax2.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax2.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax2.set_xlim([0.19, 2.0])
                ax2.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax2.set_xlabel("Tap Stand FRC (mg/L)")
                ax2.set_ylabel("Household FRC (mg/L)")
                ax2.set_title("Average Case - PM Collection")

                ax3.fill_between(
                    test1_frc,
                    np.percentile(results_table_frc_worst_am, 97.5, axis=1),
                    np.percentile(results_table_frc_worst_am, 2.5, axis=1),
                    facecolor="#b80000",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax3.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax3.plot(
                    test1_frc,
                    np.min(results_table_frc_worst_am, axis=1),
                    "#b80000",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax3.plot(
                    test1_frc,
                    np.max(results_table_frc_worst_am, axis=1),
                    "#b80000",
                    linewidth=0.5,
                )
                ax3.plot(
                    test1_frc,
                    np.median(results_table_frc_worst_am, axis=1),
                    "#b80000",
                    linewidth=1,
                    label="Median Prediction",
                )

                ax3.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax3.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax3.set_xlim([0.19, 2.0])
                ax3.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax3.set_xlabel("Tap Stand FRC (mg/L)")
                ax3.set_ylabel("Household FRC (mg/L)")
                ax3.set_title("Worst Case - AM Collection")

                ax4.fill_between(
                    test1_frc,
                    np.percentile(results_table_frc_worst_pm, 97.5, axis=1),
                    np.percentile(results_table_frc_worst_pm, 2.5, axis=1),
                    facecolor="#b80000",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax4.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax4.plot(
                    test1_frc,
                    np.min(results_table_frc_worst_pm, axis=1),
                    "#b80000",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax4.plot(
                    test1_frc,
                    np.max(results_table_frc_worst_pm, axis=1),
                    "#b80000",
                    linewidth=0.5,
                )
                ax4.plot(
                    test1_frc,
                    np.median(results_table_frc_worst_pm, axis=1),
                    "#b80000",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax4.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax4.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax4.set_xlim([0.19, 2.0])
                ax4.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax4.set_xlabel("Tap Stand FRC (mg/L)")
                ax4.set_ylabel("Household FRC (mg/L)")
                ax4.set_title("Worst Case - PM Collection")
                plt.subplots_adjust(wspace=0.25)
                plt.savefig(
                    os.path.splitext(filename)[0] + "_Predictions_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig1.pickle', 'wb'))
                StringIOBytes_preds = io.BytesIO()
                plt.savefig(StringIOBytes_preds, format="png", bbox_inches="tight")
                StringIOBytes_preds.seek(0)
                preds_base_64_pngData = base64.b64encode(StringIOBytes_preds.read())
                plt.close()

                risk_fig = plt.figure(figsize=(6.69, 3.35), dpi=300)
                plt.plot(
                    test1_frc,
                    np.sum(np.less(results_table_frc_avg_am, 0.20), axis=1)
                    / self.network_count,
                    c="#ffa600",
                    label="Risk of Household FRC < 0.20 mg/L - Average Case, AM Collection",
                )
                plt.plot(
                    test1_frc,
                    np.sum(np.less(results_table_frc_avg_pm, 0.20), axis=1)
                    / self.network_count,
                    c="#ffa600",
                    ls="--",
                    label="Risk of Household FRC < 0.20 mg/L - Average Case, PM Collection",
                )
                plt.plot(
                    test1_frc,
                    np.sum(np.less(results_table_frc_worst_am, 0.2), axis=1)
                    / self.network_count,
                    c="#b80000",
                    label="Risk of Household FRC < 0.20 mg/L - Worst Case, AM Collection",
                )
                plt.plot(
                    test1_frc,
                    np.sum(np.less(results_table_frc_worst_pm, 0.2), axis=1)
                    / self.network_count,
                    c="#b80000",
                    ls="--",
                    label="Risk of Household FRC < 0.20 mg/L - Worst Case, PM Collection",
                )
                plt.xlim([0.2, 2])
                plt.xlabel("Tapstand FRC (mg/L)")
                plt.ylim([0, 1])
                plt.ylabel("Risk of Point-of-Consumption FRC < 0.2 mg/L")
                plt.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    ncol=1,
                    labelspacing=0.1,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    loc="upper right",
                )
                plt.subplots_adjust(bottom=0.15, right=0.95)
                plt.savefig(
                    os.path.splitext(filename)[0] + "_Risk_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                StringIOBytes_risk = io.BytesIO()
                plt.savefig(StringIOBytes_risk, format="png", bbox_inches="tight")
                StringIOBytes_risk.seek(0)
                risk_base_64_pngData = base64.b64encode(StringIOBytes_risk.read())
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig2.pickle', 'wb'))
                plt.close()

            elif self.post_process_check == True:

                preds_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                    2, 2, figsize=(6.69, 6.69), dpi=300
                )
                ax1.fill_between(
                    test1_frc,
                    self.avg_case_results_am_post["Upper 95 CI"],
                    self.avg_case_results_am_post["Lower 95 CI"],
                    facecolor="#ffa600",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax1.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax1.plot(
                    test1_frc,
                    self.avg_case_results_am_post["Ensemble Minimum"],
                    "#ffa600",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax1.plot(
                    test1_frc,
                    self.avg_case_results_am_post["Ensemble Maximum"],
                    "#ffa600",
                    linewidth=0.5,
                )
                ax1.plot(
                    test1_frc,
                    self.avg_case_results_am_post["median"],
                    "#ffa600",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax1.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax1.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax1.set_xlim([0.19, 2.0])
                ax1.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax1.set_xlabel("Tap Stand FRC (mg/L)")
                ax1.set_ylabel("Household FRC (mg/L)")
                ax1.set_title("Average Case - AM Collection")

                ax2.fill_between(
                    test1_frc,
                    self.avg_case_results_pm_post["Upper 95 CI"],
                    self.avg_case_results_pm_post["Lower 95 CI"],
                    facecolor="#ffa600",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax2.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax2.plot(
                    test1_frc,
                    self.avg_case_results_pm_post["Ensemble Minimum"],
                    "#ffa600",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax2.plot(
                    test1_frc,
                    self.avg_case_results_pm_post["Ensemble Maximum"],
                    "#ffa600",
                    linewidth=0.5,
                )
                ax2.plot(
                    test1_frc,
                    self.avg_case_results_pm_post["median"],
                    "#ffa600",
                    linewidth=1,
                    label="median Prediction",
                )
                ax2.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax2.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax2.set_xlim([0.19, 2.0])
                ax2.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax2.set_xlabel("Tap Stand FRC (mg/L)")
                ax2.set_ylabel("Household FRC (mg/L)")
                ax2.set_title("Average Case - PM Collection")

                ax3.fill_between(
                    test1_frc,
                    self.worst_case_results_am_post["Upper 95 CI"],
                    self.worst_case_results_am_post["Lower 95 CI"],
                    facecolor="#b80000",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax3.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax3.plot(
                    test1_frc,
                    self.worst_case_results_am_post["Ensemble Minimum"],
                    "#b80000",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax3.plot(
                    test1_frc,
                    self.worst_case_results_am_post["Ensemble Maximum"],
                    "#b80000",
                    linewidth=0.5,
                )
                ax3.plot(
                    test1_frc,
                    self.worst_case_results_am_post["median"],
                    "#b80000",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax3.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax3.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax3.set_xlim([0.19, 2.0])
                ax3.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax3.set_xlabel("Tap Stand FRC (mg/L)")
                ax3.set_ylabel("Household FRC (mg/L)")
                ax3.set_title("Worst Case - AM Collection")

                ax4.fill_between(
                    test1_frc,
                    self.worst_case_results_pm_post["Upper 95 CI"],
                    self.worst_case_results_pm_post["Lower 95 CI"],
                    facecolor="#b80000",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax4.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax4.plot(
                    test1_frc,
                    self.worst_case_results_pm_post["Ensemble Minimum"],
                    "#b80000",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax4.plot(
                    test1_frc,
                    self.worst_case_results_pm_post["Ensemble Maximum"],
                    "#b80000",
                    linewidth=0.5,
                )
                ax4.plot(
                    test1_frc,
                    self.worst_case_results_pm_post["median"],
                    "#b80000",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax4.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax4.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax4.set_xlim([0.19, 2.0])
                ax4.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax4.set_xlabel("Tap Stand FRC (mg/L)")
                ax4.set_ylabel("Household FRC (mg/L)")
                ax4.set_title("Worst Case - PM Collection")
                plt.subplots_adjust(wspace=0.25)
                plt.savefig(
                    os.path.splitext(filename)[0] + "_Predictions_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                StringIOBytes_preds = io.BytesIO()
                plt.savefig(StringIOBytes_preds, format="png", bbox_inches="tight")
                StringIOBytes_preds.seek(0)
                preds_base_64_pngData = base64.b64encode(StringIOBytes_preds.read())
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig1.pickle', 'wb'))
                plt.close()

                risk_fig = plt.figure(figsize=(6.69, 3.35), dpi=300)
                plt.plot(
                    test1_frc,
                    self.avg_case_results_am_post["probability<=0.20"],
                    c="#ffa600",
                    label="Risk of Household FRC < 0.20 mg/L - Average Case, AM Collection",
                )
                plt.plot(
                    test1_frc,
                    self.avg_case_results_pm_post["probability<=0.20"],
                    c="#ffa600",
                    ls="--",
                    label="Risk of Household FRC < 0.20 mg/L - Average Case, PM Collection",
                )
                plt.plot(
                    test1_frc,
                    self.worst_case_results_am_post["probability<=0.20"],
                    c="#b80000",
                    label="Risk of Household FRC < 0.20 mg/L - Worst Case, AM Collection",
                )
                plt.plot(
                    test1_frc,
                    self.worst_case_results_pm_post["probability<=0.20"],
                    c="#b80000",
                    ls="--",
                    label="Risk of Household FRC < 0.20 mg/L - Worst Case, PM Collection",
                )
                plt.xlim([0.2, 2])
                plt.xlabel("Tapstand FRC (mg/L)")
                plt.ylim([0, 1])
                plt.ylabel("Risk of Point-of-Consumption FRC < 0.2 mg/L")
                plt.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    ncol=1,
                    labelspacing=0.1,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    loc="upper right",
                )

                plt.savefig(
                    os.path.splitext(filename)[0] + "_Risk_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                StringIOBytes_risk = io.BytesIO()
                plt.savefig(StringIOBytes_risk, format="png", bbox_inches="tight")
                StringIOBytes_risk.seek(0)
                risk_base_64_pngData = base64.b64encode(StringIOBytes_risk.read())
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig2.pickle', 'wb'))
                plt.close()
            if WATTEMP in self.datainputs.columns and COND in self.datainputs.columns:
                hist_fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
                    6, 1, figsize=(3.35, 6.69), dpi=300
                )

                ax1.set_ylabel("Frequency")
                ax1.set_xlabel("Tapstand FRC (mg/L)")
                ax1.hist(self.datainputs.iloc[:, 0], bins=30, color="grey")

                ax2.set_ylabel("Frequency")
                ax2.set_xlabel("Elapsed Time (hours)")
                ax2.hist(self.datainputs.iloc[:, 1], bins=30, color="grey")

                ax3.set_ylabel("Frequency")
                ax3.set_xlabel("Collection Time (0=AM, 1=PM)")
                ax3.hist(self.datainputs.iloc[:, 2], bins=30, color="grey")

                ax4.set_ylabel("Frequency")
                ax4.set_xlabel("Water Temperature(" + r"$\degree$" + "C)")
                ax4.hist(self.datainputs.iloc[:, 3], bins=30, color="grey")
                ax4.axvline(
                    self.median_wattemp,
                    c="#ffa600",
                    ls="--",
                    label="Average Case Value Used",
                )
                ax4.axvline(
                    self.upper95_wattemp,
                    c="#b80000",
                    ls="--",
                    label="Worst Case Value Used",
                )
                ax4.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    ncol=1,
                    labelspacing=0.1,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    loc="upper right",
                )

                ax5.set_ylabel("Frequency")
                ax5.set_xlabel("Electrical Conductivity (" + r"$\mu$" + "s/cm)")
                ax5.hist(self.datainputs.iloc[:, 4], bins=30, color="grey")
                ax5.axvline(
                    self.median_cond,
                    c="#ffa600",
                    ls="--",
                    label="Average Case Value Used",
                )
                ax5.axvline(
                    self.upper95_cond,
                    c="#b80000",
                    ls="--",
                    label="Worst Case Value used",
                )
                ax5.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    ncol=1,
                    labelspacing=0.1,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    loc="upper right",
                )

                ax6.set_ylabel("Frequency")
                ax6.set_xlabel("Household FRC (mg/L)")
                ax6.hist(self.dataoutputs, bins=30, color="grey")
                plt.subplots_adjust(
                    left=0.18, hspace=0.60, top=0.99, bottom=0.075, right=0.98
                )

                plt.savefig(
                    os.path.splitext(filename)[0] + "_Histograms_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig3.pickle', 'wb'))
                plt.close()

                StringIOBytes_histogram = io.BytesIO()
                plt.savefig(StringIOBytes_histogram, format="png", bbox_inches="tight")
                StringIOBytes_histogram.seek(0)
                hist_base_64_pngData = base64.b64encode(StringIOBytes_histogram.read())

            elif WATTEMP in self.datainputs.columns:
                hist_fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
                    6, 1, figsize=(3.35, 6.69), dpi=300
                )

                ax1.set_ylabel("Frequency")
                ax1.set_xlabel("Tapstand FRC (mg/L)")
                ax1.hist(self.datainputs.iloc[:, 0], bins=30, color="grey")

                ax2.set_ylabel("Frequency")
                ax2.set_xlabel("Elapsed Time (hours)")
                ax2.hist(self.datainputs.iloc[:, 1], bins=30, color="grey")

                ax3.set_ylabel("Frequency")
                ax3.set_xlabel("Collection Time (0=AM, 1=PM)")
                ax3.hist(self.datainputs.iloc[:, 2], bins=30, color="grey")

                ax4.set_ylabel("Frequency")
                ax4.set_xlabel("Water Temperature(" + r"$\degree$" + "C)")
                ax4.hist(self.datainputs.iloc[:, 3], bins=30, color="grey")
                ax4.axvline(
                    self.median_wattemp,
                    c="#ffa600",
                    ls="--",
                    label="Average Case Value Used",
                )
                ax4.axvline(
                    self.upper95_wattemp,
                    c="#b80000",
                    ls="--",
                    label="Worst Case Value Used",
                )
                ax4.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    ncol=1,
                    labelspacing=0.1,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    loc="upper right",
                )

                ax5.set_ylabel("Frequency")
                ax5.set_xlabel("Household FRC (mg/L)")
                ax5.hist(self.dataoutputs, bins=30, color="grey")
                plt.subplots_adjust(
                    left=0.18, hspace=0.60, top=0.99, bottom=0.075, right=0.98
                )

                plt.savefig(
                    os.path.splitext(filename)[0] + "_Histograms_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig3.pickle', 'wb'))
                plt.close()

                StringIOBytes_histogram = io.BytesIO()
                plt.savefig(StringIOBytes_histogram, format="png", bbox_inches="tight")
                StringIOBytes_histogram.seek(0)
                hist_base_64_pngData = base64.b64encode(StringIOBytes_histogram.read())

            elif COND in self.datainputs.columns:
                hist_fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
                    6, 1, figsize=(3.35, 6.69), dpi=300
                )

                ax1.set_ylabel("Frequency")
                ax1.set_xlabel("Tapstand FRC (mg/L)")
                ax1.hist(self.datainputs.iloc[:, 0], bins=30, color="grey")

                ax2.set_ylabel("Frequency")
                ax2.set_xlabel("Elapsed Time (hours)")
                ax2.hist(self.datainputs.iloc[:, 1], bins=30, color="grey")

                ax3.set_ylabel("Frequency")
                ax3.set_xlabel("Collection Time (0=AM, 1=PM)")
                ax3.hist(self.datainputs.iloc[:, 2], bins=30, color="grey")

                ax4.set_ylabel("Frequency")
                ax4.set_xlabel("Electrical Conductivity (" + r"$\mu$" + "s/cm)")
                ax4.hist(self.datainputs.iloc[:, 4], bins=30, color="grey")
                ax4.axvline(
                    self.median_cond,
                    c="#ffa600",
                    ls="--",
                    label="Average Case Value Used",
                )
                ax4.axvline(
                    self.upper95_cond,
                    c="#b80000",
                    ls="--",
                    label="Worst Case Value used",
                )
                ax4.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    ncol=1,
                    labelspacing=0.1,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    loc="upper right",
                )

                ax5.set_ylabel("Frequency")
                ax5.set_xlabel("Household FRC (mg/L)")
                ax5.hist(self.dataoutputs, bins=30, color="grey")
                plt.subplots_adjust(
                    left=0.18, hspace=0.60, top=0.99, bottom=0.075, right=0.98
                )

                plt.savefig(
                    os.path.splitext(filename)[0] + "_Histograms_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig3.pickle', 'wb'))
                plt.close()

                StringIOBytes_histogram = io.BytesIO()
                plt.savefig(StringIOBytes_histogram, format="png", bbox_inches="tight")
                StringIOBytes_histogram.seek(0)
                hist_base_64_pngData = base64.b64encode(StringIOBytes_histogram.read())

        else:
            if self.post_process_check == False:
                results_table_frc_avg_am = self.avg_case_results_am.iloc[
                    :, 0 : (self.network_count - 1)
                ]
                results_table_frc_avg_pm = self.avg_case_results_pm.iloc[
                    :, 0 : (self.network_count - 1)
                ]
                preds_fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(6.69, 3.35), dpi=300
                )
                ax1.fill_between(
                    test1_frc,
                    np.percentile(results_table_frc_avg_am, 97.5, axis=1),
                    np.percentile(results_table_frc_avg_am, 2.5, axis=1),
                    facecolor="#ffa600",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax1.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax1.plot(
                    test1_frc,
                    np.min(results_table_frc_avg_am, axis=1),
                    "#ffa600",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax1.plot(
                    test1_frc,
                    np.max(results_table_frc_avg_am, axis=1),
                    "#ffa600",
                    linewidth=0.5,
                )
                ax1.plot(
                    test1_frc,
                    np.median(results_table_frc_avg_am, axis=1),
                    "#ffa600",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax1.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax1.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax1.set_xlim([0.19, 2.0])
                ax1.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax1.set_xlabel("Tap Stand FRC (mg/L)")
                ax1.set_ylabel("Household FRC (mg/L)")
                ax1.set_title("AM Collection")

                ax2.fill_between(
                    test1_frc,
                    np.percentile(results_table_frc_avg_pm, 97.5, axis=1),
                    np.percentile(results_table_frc_avg_pm, 2.5, axis=1),
                    facecolor="#ffa600",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax2.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax2.plot(
                    test1_frc,
                    np.min(results_table_frc_avg_pm, axis=1),
                    "#ffa600",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax2.plot(
                    test1_frc,
                    np.max(results_table_frc_avg_pm, axis=1),
                    "#ffa600",
                    linewidth=0.5,
                )
                ax2.plot(
                    test1_frc,
                    np.median(results_table_frc_avg_pm, axis=1),
                    "#ffa600",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax2.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax2.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax2.set_xlim([0.19, 2.0])
                ax2.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax2.set_xlabel("Tap Stand FRC (mg/L)")
                ax2.set_ylabel("Household FRC (mg/L)")
                ax2.set_title("PM Collection")

                plt.subplots_adjust(wspace=0.25)
                plt.savefig(
                    os.path.splitext(filename)[0] + "_Predictions_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig1.pickle', 'wb'))
                StringIOBytes_preds = io.BytesIO()
                plt.savefig(StringIOBytes_preds, format="png", bbox_inches="tight")
                StringIOBytes_preds.seek(0)
                preds_base_64_pngData = base64.b64encode(StringIOBytes_preds.read())
                plt.close()

                risk_fig = plt.figure(figsize=(6.69, 3.35), dpi=300)
                plt.plot(
                    test1_frc,
                    np.sum(np.less(results_table_frc_avg_am, 0.20), axis=1)
                    / self.network_count,
                    c="#ffa600",
                    label="Risk of Household FRC < 0.20 mg/L - Average Case, AM Collection",
                )
                plt.plot(
                    test1_frc,
                    np.sum(np.less(results_table_frc_avg_pm, 0.20), axis=1)
                    / self.network_count,
                    c="#ffa600",
                    ls="--",
                    label="Risk of Household FRC < 0.20 mg/L - Average Case, PM Collection",
                )

                plt.xlim([0.2, 2])
                plt.xlabel("Tapstand FRC (mg/L)")
                plt.ylim([0, 1])
                plt.ylabel("Risk of Point-of-Consumption FRC < 0.2 mg/L")
                plt.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    ncol=1,
                    labelspacing=0.1,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    loc="upper right",
                )
                plt.subplots_adjust(bottom=0.15, right=0.95)
                plt.savefig(
                    os.path.splitext(filename)[0] + "_Risk_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig2.pickle', 'wb'))
                StringIOBytes_risk = io.BytesIO()
                plt.savefig(StringIOBytes_risk, format="png", bbox_inches="tight")
                StringIOBytes_risk.seek(0)
                risk_base_64_pngData = base64.b64encode(StringIOBytes_risk.read())
                plt.close()

            elif self.post_process_check == True:
                preds_fig, (ax1, ax2) = plt.subplots(
                    1, 2, figsize=(6.69, 3.35), dpi=300
                )
                ax1.fill_between(
                    test1_frc,
                    self.avg_case_results_am_post["Upper 95 CI"],
                    self.avg_case_results_am_post["Lower 95 CI"],
                    facecolor="#ffa600",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax1.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax1.plot(
                    test1_frc,
                    self.avg_case_results_am_post["Ensemble Minimum"],
                    "#ffa600",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax1.plot(
                    test1_frc,
                    self.avg_case_results_am_post["Ensemble Maximum"],
                    "#ffa600",
                    linewidth=0.5,
                )
                ax1.plot(
                    test1_frc,
                    self.avg_case_results_am_post["median"],
                    "#ffa600",
                    linewidth=1,
                    label="Median Prediction",
                )
                ax1.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax1.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax1.set_xlim([0.19, 2.0])
                ax1.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax1.set_xlabel("Tap Stand FRC (mg/L)")
                ax1.set_ylabel("Household FRC (mg/L)")
                ax1.set_title("AM Collection")

                ax2.fill_between(
                    test1_frc,
                    self.avg_case_results_pm_post["Upper 95 CI"],
                    self.avg_case_results_pm_post["Lower 95 CI"],
                    facecolor="#ffa600",
                    alpha=0.5,
                    label="95th Percentile Range",
                )
                ax2.axhline(0.2, c="k", ls="-.", linewidth=1, label="FRC = 0.2 mg/L")
                ax2.plot(
                    test1_frc,
                    self.avg_case_results_pm_post["Ensemble Minimum"],
                    "#ffa600",
                    linewidth=0.5,
                    label="Minimum/Maximum",
                )
                ax2.plot(
                    test1_frc,
                    self.avg_case_results_pm_post["Ensemble Maximum"],
                    "#ffa600",
                    linewidth=0.5,
                )
                ax2.plot(
                    test1_frc,
                    self.avg_case_results_pm_post["median"],
                    "#ffa600",
                    linewidth=1,
                    label="median Prediction",
                )
                ax2.scatter(
                    self.datainputs[FRC_IN],
                    self.dataoutputs,
                    c="k",
                    s=10,
                    marker="x",
                    label="Testing Observations",
                )
                ax2.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    loc="upper right",
                )
                ax2.set_xlim([0.19, 2.0])
                ax2.set_xticks(np.arange(0.2, 2.2, 0.2))
                ax2.set_xlabel("Tap Stand FRC (mg/L)")
                ax2.set_ylabel("Household FRC (mg/L)")
                ax2.set_title("PM Collection")

                plt.subplots_adjust(wspace=0.25)
                plt.tight_layout()
                plt.savefig(
                    os.path.splitext(filename)[0] + "_Predictions_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig1.pickle', 'wb'))
                StringIOBytes_preds = io.BytesIO()
                plt.savefig(StringIOBytes_preds, format="png", bbox_inches="tight")
                StringIOBytes_preds.seek(0)
                preds_base_64_pngData = base64.b64encode(StringIOBytes_preds.read())

                plt.close()

                risk_fig = plt.figure(figsize=(6.69, 3.35), dpi=300)
                plt.plot(
                    test1_frc,
                    self.avg_case_results_am_post["probability<=0.20"],
                    c="#ffa600",
                    label="Risk of Household FRC < 0.20 mg/L - Average Case, AM Collection",
                )
                plt.plot(
                    test1_frc,
                    self.avg_case_results_pm_post["probability<=0.20"],
                    c="#ffa600",
                    ls="--",
                    label="Risk of Household FRC < 0.20 mg/L - Average Case, PM Collection",
                )
                plt.xlim([0.2, 2])
                plt.xlabel("Tapstand FRC (mg/L)")
                plt.ylim([0, 1])
                plt.ylabel("Risk of Point-of-Consumption FRC < 0.2 mg/L")
                plt.legend(
                    bbox_to_anchor=(0.999, 0.999),
                    shadow=False,
                    fontsize="small",
                    ncol=1,
                    labelspacing=0.1,
                    columnspacing=0.2,
                    handletextpad=0.1,
                    loc="upper right",
                )

                plt.savefig(
                    os.path.splitext(filename)[0] + "_Risk_Fig.png",
                    format="png",
                    bbox_inches="tight",
                )
                # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig2.pickle', 'wb'))
                StringIOBytes_risk = io.BytesIO()
                plt.savefig(StringIOBytes_risk, format="png", bbox_inches="tight")
                StringIOBytes_risk.seek(0)
                risk_base_64_pngData = base64.b64encode(StringIOBytes_risk.read())
                plt.close()

            hist_fig, (ax1, ax2, ax3, ax4) = plt.subplots(
                4, 1, figsize=(3.35, 6.69), dpi=300
            )

            ax1.set_ylabel("Frequency")
            ax1.set_xlabel("Tapstand FRC (mg/L)")
            ax1.hist(self.datainputs.iloc[:, 0], bins=30, color="grey")

            ax2.set_ylabel("Frequency")
            ax2.set_xlabel("Elapsed Time (hours)")
            ax2.hist(self.datainputs.iloc[:, 1], bins=30, color="grey")

            ax3.set_ylabel("Frequency")
            ax3.set_xlabel("Collection Time (0=AM, 1=PM)")
            ax3.hist(self.datainputs.iloc[:, 2], bins=30, color="grey")

            ax4.set_ylabel("Frequency")
            ax4.set_xlabel("Household FRC (mg/L)")
            ax4.hist(self.dataoutputs, bins=30, color="grey")
            plt.subplots_adjust(
                left=0.18, hspace=0.60, top=0.99, bottom=0.075, right=0.98
            )

            plt.savefig(
                os.path.splitext(filename)[0] + "_Histograms_Fig.png",
                format="png",
                bbox_inches="tight",
            )
            # pl.dump(fig, open(os.path.splitext(filename)[0] + 'Fig3.pickle', 'wb'))
            plt.close()

            StringIOBytes_histogram = io.BytesIO()
            plt.savefig(StringIOBytes_histogram, format="png", bbox_inches="tight")
            StringIOBytes_histogram.seek(0)
            hist_base_64_pngData = base64.b64encode(StringIOBytes_histogram.read())
        return hist_base_64_pngData, risk_base_64_pngData, preds_base_64_pngData

    def display_results(self):
        """
        Display the results of the predictions as a console output.

        Display and return all the contents of the self.results variable which is a pandas Dataframe object
        :return: A Pandas Dataframe object (self.results) containing all the result of the predictions
        """
        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            if self.post_process_check == False:
                logging.info(self.avg_case_results_am)
                logging.info(self.worst_case_results_am)
                logging.info(self.avg_case_results_pm)
                logging.info(self.worst_case_results_pm)
                return (
                    self.avg_case_results_am,
                    self.avg_case_results_pm,
                    self.worst_case_results_am,
                    self.worst_case_results_pm,
                )
            else:
                logging.info(self.avg_case_results_am_post)
                logging.info(self.worst_case_results_am_post)
                logging.info(self.avg_case_results_pm_post)
                logging.info(self.worst_case_results_pm_post)
                return (
                    self.avg_case_results_am_post,
                    self.avg_case_results_pm_post,
                    self.worst_case_results_am_post,
                    self.worst_case_results_pm_post,
                )
        else:
            if self.post_process_check == False:
                logging.info(self.avg_case_results_am)
                logging.info(self.avg_case_results_pm)
                return self.avg_case_results_am, self.avg_case_results_pm
            else:
                logging.info(self.avg_case_results_am_post)
                logging.info(self.avg_case_results_pm_post)
                return self.avg_case_results_am_post, self.avg_case_results_pm_post

    def export_results_to_csv(self, filename):
        self.avg_case_results_am.to_csv(
            os.path.splitext(filename)[0] + "_average_case_am.csv", index=False
        )
        self.avg_case_results_pm.to_csv(
            os.path.splitext(filename)[0] + "_average_case_pm.csv", index=False
        )

        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            self.worst_case_results_am.to_csv(
                os.path.splitext(filename)[0] + "_worst_case_am.csv", index=False
            )
            self.worst_case_results_pm.to_csv(
                os.path.splitext(filename)[0] + "_worst_case_pm.csv", index=False
            )
        if self.post_process_check == True:

            self.avg_case_results_am_post.to_csv(
                os.path.splitext(filename)[0] + "_average_case_am.csv", index=False
            )
            self.avg_case_results_pm_post.to_csv(
                os.path.splitext(filename)[0] + "_average_case_pm.csv", index=False
            )
            if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
                self.worst_case_results_am_post.to_csv(
                    os.path.splitext(filename)[0] + "_worst_case_am.csv", index=False
                )
                self.worst_case_results_pm_post.to_csv(
                    os.path.splitext(filename)[0] + "_worst_case_pm.csv", index=False
                )

    def generate_model_performance(self):
        """Generates training performance graphs

        Plots the model performance metrics (MSE and R^2 vs # of epochs) after training and returns a
        base64 encoded image. The NN has to be trained first otherwise the image will be empty.

        Returns: Base64 data stream"""

        fig, axs = plt.subplots(1, 2, sharex=True)

        ax = axs[0]
        ax.boxplot(
            [self.total_mse_train, self.total_mse_val, self.total_mse_test],
            labels=["Training", "Validation", "Testing"],
        )
        ax.set_title("Mean Squared Error")
        tr_legend = "Training Avg MSE: {mse:.4f}".format(mse=self.avg_mse_train)
        val_legend = "Validation Avg MSE: {mse:.4f}".format(mse=self.avg_mse_val)
        ts_legend = "Testing Avg MSE: {mse:.4f}".format(mse=self.avg_mse_test)
        ax.legend([tr_legend, val_legend, ts_legend])

        ax = axs[1]
        ax.boxplot(
            [
                self.total_rsquared_train,
                self.total_rsquared_val,
                self.total_rsquared_test,
            ],
            labels=["Training", "Validation", "Testing"],
        )
        ax.set_title("R^2")
        tr_legend = "Training Avg. R^2: {rs:.3f}".format(rs=self.avg_rsq_train)
        val_legend = "Validation Avg. R^2: {rs:.3f}".format(rs=self.avg_rsq_val)
        ts_legend = "Validation Avg. R^2: {rs:.3f}".format(rs=self.avg_rsq_test)
        ax.legend([tr_legend, val_legend, ts_legend])

        fig.suptitle(
            "Performance metrics across 100 training runs on "
            + str(self.epochs)
            + " epochs, with "
            + str(self.layer1_neurons)
            + " neurons on hidden layer."
        )
        fig.set_size_inches(12, 8)

        # plt.show()

        # Uncomment the next lines to save the graph to disk
        # plt.savefig("model_metrics\\" + str(self.epochs) + "_epochs_" + str(self.layer1_neurons) + "_neurons.png",
        #            dpi=100)
        # plt.close()

        plt.show()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format="png")
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

        # Uncomment the following line to load the results direclty from an csv file
        # df = pd.read_csv('results.csv')

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
            lo_limit = round(min(c), 2)
            logging.info(lo_limit)

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

        divisions = round((hi_limit - lo_limit) / 0.05, 0) + 1
        logging.info(divisions)

        # Get the data between the limits
        sorted_data = sorted_data[sorted_data > lo_limit]
        sorted_data = sorted_data[sorted_data < hi_limit]

        # create a colorbar for the se4_frc and divide it in 0.2 mg/L intervals
        cmap = plt.cm.jet_r
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, cmap.N
        )
        bounds = np.linspace(0, 1.4, 8)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)

        ax = fig.add_subplot(221)
        img = ax.scatter(frc, watt, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel("FRC at tapstand (mg/L)")
        ax.set_ylabel("Water Temperature (" + "\u00b0" + "C)")
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(222)
        img = ax.scatter(frc, cond, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel("FRC at tapstand (mg/L)")
        ax.set_ylabel("Water Conductivity (\u03BCS/cm)")
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(223)
        img = ax.scatter(watt, cond, c=c, s=5, cmap=cmap, norm=norm, alpha=1)
        ax.set_xlabel("Water Temperature (" + "\u00b0" + "C)")
        ax.set_ylabel("Water Conductivity (\u03BCS/cm)")
        ax.grid(linewidth=0.2)

        ax = fig.add_subplot(224)
        img = ax.hist(
            c,
            bins=np.linspace(lo_limit, hi_limit, divisions),
            edgecolor="black",
            linewidth=0.1,
        )
        ax.grid(linewidth=0.1)
        line02 = ax.axvline(0.2, color="r", linestyle="dashed", linewidth=2)
        line03 = ax.axvline(0.3, color="y", linestyle="dashed", linewidth=2)
        ax.set_xlabel("FRC at household (mg/L)")
        ax.set_ylabel("# of instances")

        axcdf = ax.twinx()
        (cdf,) = axcdf.step(sorted_data, np.arange(sorted_data.size), color="g")
        ax.legend(
            (line02, line03, cdf), ("0.2 mg/L", "0.3 mg/L", "CDF"), loc="center right"
        )

        ax2 = fig.add_axes([0.93, 0.1, 0.01, 0.75])
        cb = mpl.colorbar.ColorbarBase(
            ax2,
            cmap=cmap,
            norm=norm,
            spacing="proportional",
            ticks=bounds,
            boundaries=bounds,
        )
        cb.ax.set_ylabel("FRC at se4 (mg/L)", rotation=270, labelpad=20)

        plt.show()

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format="png")
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

        # fig.suptitle('Total samples: '+ str(len(frc)))  # +
        #             "\n" + "SWOT version: " + self.software_version +
        #             "\n" + "Input Filename: " + os.path.basename(self.input_filename) +
        #             "\n" + "Generated on: " + self.today)

        axInitialFRC = fig.add_subplot(221)
        axInitialFRC.hist(frc, bins=20, edgecolor="black", linewidth=0.1)
        axInitialFRC.set_xlabel("Initial FRC (mg/L)")
        axInitialFRC.set_ylabel("# of instances")
        mean = round(np.mean(frc), 2)
        median = np.median(frc)
        mean_line = axInitialFRC.axvline(
            mean, color="r", linestyle="dashed", linewidth=2
        )
        median_line = axInitialFRC.axvline(
            median, color="y", linestyle="dashed", linewidth=2
        )
        axInitialFRC.legend(
            (mean_line, median_line),
            ("Mean: " + str(mean) + " mg/L", "Median: " + str(median) + " mg/L"),
        )

        ax = fig.add_subplot(222)
        ax.hist(watt, bins=20, edgecolor="black", linewidth=0.1)
        ax.set_xlabel("Water Temperature (" + "\u00b0" + "C)")
        ax.set_ylabel("# of instances")
        mean = round(np.mean(watt), 2)
        median = np.median(watt)
        mean_line = ax.axvline(mean, color="r", linestyle="dashed", linewidth=2)
        median_line = ax.axvline(median, color="y", linestyle="dashed", linewidth=2)
        ax.legend(
            (mean_line, median_line),
            (
                "Mean: " + str(mean) + "\u00b0" + "C",
                "Median: " + str(median) + "\u00b0" + "C",
            ),
        )

        ax = fig.add_subplot(223)
        ax.hist(cond, bins=20, edgecolor="black", linewidth=0.1)
        ax.set_xlabel("Water Conductivity (\u03BCS/cm)")
        ax.set_ylabel("# of instances")
        mean = round(np.mean(cond), 2)
        median = np.median(cond)
        mean_line = ax.axvline(mean, color="r", linestyle="dashed", linewidth=2)
        median_line = ax.axvline(median, color="y", linestyle="dashed", linewidth=2)
        ax.legend(
            (mean_line, median_line),
            (
                "Mean: " + str(mean) + " \u03BCS/cm",
                "Median: " + str(median) + " \u03BCS/cm",
            ),
        )

        axHouseholdFRC = fig.add_subplot(224)
        axHouseholdFRC.hist(
            frc4, bins=np.linspace(0, 2, 41), edgecolor="black", linewidth=0.1
        )
        axHouseholdFRC.set_xlabel("Household FRC (\u03BCS/cm)")
        axHouseholdFRC.set_ylabel("# of instances")
        mean = round(np.mean(frc4), 2)
        median = np.median(frc4)
        mean_line = axHouseholdFRC.axvline(
            mean, color="r", linestyle="dashed", linewidth=2
        )
        median_line = axHouseholdFRC.axvline(
            median, color="y", linestyle="dashed", linewidth=2
        )
        axHouseholdFRC.legend(
            (mean_line, median_line),
            (
                "Mean: " + str(mean) + " \u03BCS/cm",
                "Median: " + str(median) + " \u03BCS/cm",
            ),
        )

        fig.savefig(os.path.splitext(filename)[0] + ".png", format="png")
        # plt.show()

        # create figure for initial and household FRC separately in a single image
        figFRC = plt.figure(figsize=(19.2 / 1.45, 6.4), dpi=100)

        axInitialFRC = figFRC.add_subplot(211)
        axInitialFRC.hist(frc, bins=20, edgecolor="black", linewidth=0.1)
        axInitialFRC.set_xlabel("Initial FRC (mg/L)")
        axInitialFRC.set_ylabel("# of instances")
        mean = round(np.mean(frc), 2)
        median = np.median(frc)
        mean_line = axInitialFRC.axvline(
            mean, color="r", linestyle="dashed", linewidth=2
        )
        median_line = axInitialFRC.axvline(
            median, color="y", linestyle="dashed", linewidth=2
        )
        axInitialFRC.legend(
            (mean_line, median_line),
            ("Mean: " + str(mean) + " mg/L", "Median: " + str(median) + " mg/L"),
        )

        axHouseholdFRC = figFRC.add_subplot(212)
        axHouseholdFRC.hist(
            frc4, bins=np.linspace(0, 2, 41), edgecolor="black", linewidth=0.1
        )
        axHouseholdFRC.set_xlabel("Household FRC (mg/L)")
        axHouseholdFRC.set_ylabel("# of instances")
        mean = round(np.mean(frc4), 2)
        median = np.median(frc4)
        mean_line = axHouseholdFRC.axvline(
            mean, color="r", linestyle="dashed", linewidth=2
        )
        median_line = axHouseholdFRC.axvline(
            median, color="y", linestyle="dashed", linewidth=2
        )
        axHouseholdFRC.legend(
            (mean_line, median_line),
            ("Mean: " + str(mean) + " mg/L", "Median: " + str(median) + " mg/L"),
        )

        figFRC.savefig(os.path.splitext(filename)[0] + "-frc.jpg", format="jpg")

        myStringIOBytes = io.BytesIO()
        plt.savefig(myStringIOBytes, format="png")
        myStringIOBytes.seek(0)
        my_base_64_pngData = base64.b64encode(myStringIOBytes.read())
        return my_base_64_pngData

    def generate_html_report(self, filename, storage_target):
        """Generates an html report of the SWOT results. The report
        is saved on disk under the name 'filename'."""

        df = self.datainputs
        frc = df[FRC_IN]

        # self.generate_input_info_plots(filename).decode('UTF-8')
        hist, risk, pred = self.results_visualization(filename, storage_target)
        hist.decode("UTF-8")
        risk.decode("UTF-8")
        pred.decode("UTF-8")
        # scatterplots_b64 = self.generate_2d_scatterplot().decode('UTF-8')
        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            avg_html_table, worst_html_table = self.prepare_table_for_html_report(
                storage_target
            )
        else:
            avg_html_table = self.prepare_table_for_html_report(storage_target)

        skipped_rows_table = self.skipped_rows_html()

        doc, tag, text, line = Doc().ttl()
        with tag("h1", klass="title"):
            text("SWOT ARTIFICIAL NEURAL NETWORK REPORT")
        with tag("p", klass="swot_version"):
            text("SWOT ANN Code Version: " + self.software_version)
        with tag("p", klass="input_filename"):
            text("Input File Name: " + os.path.basename(self.input_filename))
        with tag("p", klass="date"):
            text("Date Generated: " + self.today)
        with tag("p", klass="time_difference"):
            text(
                "Average time between tapstand and household: "
                + str(int(self.avg_time_elapsed // 3600))
                + " hours and "
                + str(int((self.avg_time_elapsed // 60) % 60))
                + " minutes"
            )
        with tag("p"):
            text("Total Samples: " + str(len(frc)))
        if self.post_process_check == False:
            with tag("h2", klass="Header"):
                text("Predicted Risk - Raw Ensemble:")
        else:
            with tag("h2", klass="Header"):
                text("Predicted Risk - Post-Processed Ensemble:")

        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            with tag("p", klass="Predictions Fig Text"):
                text(
                    "Household FRC forecast from an ensemble of "
                    + str(self.network_count)
                    + " ANN models. The predictions of each model are grouped into a probability density function to predict the risk of FRC below threshold values of 0.20 mg/L using a fixed input variable set for worst case and average case scenarios (shown in the risk tables below). Note that if FRC is collected using pool testers instead of a cholorimeter, the predicted FRC may be disproportionately clustered towards the centre of the observations, resulting in some observations with low FRC to not be captured within the ensemble forecast. In these cases, the predicted risk in the next figure and in the subsequent risk tables may be underpredicted. Average case predictions use median collected values for conductivity and water temperature; worst-case scenario uses 95th percentile values for conductivity and water temeperature"
                )
            with tag("div", id="Predictions Graphs"):
                doc.stag(
                    "img",
                    src=os.path.basename(
                        os.path.splitext(filename)[0] + "_Predictions_Fig.png"
                    ),
                )
                # doc.asis('<object data="cid:'+os.path.basename(os.path.splitext(filename)[0]+'.PNG') + '" type="image/jpeg"></object>')
            # doc.asis(
            #     '<object data="'
            #     + os.path.basename(
            #         os.path.splitext(filename)[0] + "_Predictions_Fig.png"
            #     )
            #     + '" type="image/jpeg"></object>'
            # )
            with tag("p", klass="Risk Fig Text"):
                text(
                    "Figure and tables showing predicted risk of household FRC below 0.2 mg/L for average and worst case scenarios for both AM and PM collection. Risk obtained from forecast pdf (above) and taken as cumulative probability of houeshold FRC below 0.2 mg/L. Note that 0% predicted risk of household FRC below 0.2 mg/L does not mean that there is no possibility of household FRC being below 0.2 mg/L, simply that the predicted risk is too low to be measured. The average case target may, in some, cases be more conservative than the worst case targets as the worst case target is derived on the assumption that higher conductivity and water temperature will lead to greater decay (as confirmed by FRC decay chemisty and results at past sites). However, this may not be true in all cases, so the most conservative target is always recommended."
                )
            with tag("div", id="Risk Graphs"):
                doc.stag(
                    "img",
                    src=os.path.basename(
                        os.path.splitext(filename)[0] + "_Risk_Fig.png"
                    ),
                )
                # doc.asis('<object data="cid:'+os.path.basename(os.path.splitext(filename)[0]+'.PNG') + '" type="image/jpeg"></object>')
            # doc.asis(
            #     '<object data="'
            #     + os.path.basename(os.path.splitext(filename)[0] + "_Risk_Fig.png")
            #     + '" type="image/jpeg"></object>'
            # )
            with tag("h2", klass="Header"):
                text("Average Case Targets Table")
            with tag("table", id="average case table"):
                doc.asis(avg_html_table)
            with tag("h2", klass="Header"):
                text("Worst Case Targets Table")
            with tag("table", id="worst case table"):
                doc.asis(worst_html_table)
            with tag("p", klass="Histograms Text"):
                text(
                    "Histograms for the input variables used to generate predictions and risk recommendations above. Average case and worst case conductivity and water temperature are shown for context of values used to generate targets."
                )
            with tag("div", id="Histograms"):
                doc.stag(
                    "img",
                    src=os.path.basename(
                        os.path.splitext(filename)[0] + "_Histograms_Fig.png"
                    ),
                )
                # doc.asis('<object data="cid:'+os.path.basename(os.path.splitext(filename)[0]+'.PNG') + '" type="image/jpeg"></object>')
            # doc.asis(
            #     '<object data="'
            #     + os.path.basename(
            #         os.path.splitext(filename)[0] + "_Histograms_Fig.png"
            #     )
            #     + '" type="image/jpeg"></object>'
            # )
        else:
            with tag("p", klass="Predictions Fig Text"):
                text(
                    "Household FRC forecast from an ensemble of "
                    + str(self.network_count)
                    + " ANN models. The predictions of each model are grouped into a probability density function to predict the risk of FRC below threshold values of 0.20 mg/L using a fixed input variable set(shown in the risk table below). Note that if FRC is collected using pool testers instead of a cholorimeter, the predicted FRC may be disproportionately clustered towards the centre of the observations, resulting in some observations with low FRC to not be captured within the ensemble forecast. In these cases, the predicted risk in the next figure and in the subsequent risk table may be underpredicted."
                )
            with tag("div", id="Predictions Graphs"):
                doc.stag(
                    "img",
                    src=os.path.basename(
                        os.path.splitext(filename)[0] + "_Predictions_Fig.png"
                    ),
                )
                # doc.asis('<object data="cid:'+os.path.basename(os.path.splitext(filename)[0]+'.PNG') + '" type="image/jpeg"></object>')
            # doc.asis(
            #     '<object data="'
            #     + os.path.basename(
            #         os.path.splitext(filename)[0] + "_Predictions_Fig.png"
            #     )
            #     + '" type="image/jpeg"></object>'
            # )
            with tag("p", klass="Risk Fig Text"):
                text(
                    "Figure and tables showing predicted risk of household FRC below 0.2 mg/L for both AM and PM collection. Risk obtained from forecast probability density function (above) and taken as cumulative probability of houeshold FRC below 0.2 mg/L. Note that 0% predicted risk of household FRC below 0.2 mg/L does not mean that there is no possibility of household FRC being below 0.2 mg/L, simply that the predicted risk is too low to be measured."
                )
            with tag("div", id="Risk Graphs"):
                doc.stag(
                    "img",
                    src=os.path.basename(
                        os.path.splitext(filename)[0] + "_Risk_Fig.png"
                    ),
                )
                # doc.asis('<object data="cid:'+os.path.basename(os.path.splitext(filename)[0]+'.PNG') + '" type="image/jpeg"></object>')
            # doc.asis(
            #     '<object data="'
            #     + os.path.basename(os.path.splitext(filename)[0] + "_Risk_Fig.png")
            #     + '" type="image/jpeg"></object>'
            # )
            with tag("h2", klass="Header"):
                text("Targets Table")
            with tag("table", id="average case table"):
                doc.asis(avg_html_table)
            with tag("p", klass="Histograms Text"):
                text(
                    "Histograms for the input variables used to generate predictions and risk recommendations above."
                )
            with tag("div", id="Histograms"):
                doc.stag(
                    "img",
                    src=os.path.basename(
                        os.path.splitext(filename)[0] + "_Histograms_Fig.png"
                    ),
                )
                # doc.asis('<object data="cid:'+os.path.basename(os.path.splitext(filename)[0]+'.PNG') + '" type="image/jpeg"></object>')
            # doc.asis(
            #     '<object data="'
            #     + os.path.basename(
            #         os.path.splitext(filename)[0] + "_Histograms_Fig.png"
            #     )
            #     + '" type="image/jpeg"></object>'
            # )

        with tag("h2", klass="Header"):
            text("Model Diagnostic Figures")
        with tag("p", klass="Performance Indicator General Text"):
            text(
                "These figures evaluate the performance of the ANN ensemble model after training. These figures serve as an indicator of the similarity between the distribution of forecasts produced by the ANN ensembles and the observed data and can be used to evaluate the soundness of the models, and of the confidence we can have in the targets."
            )
        with tag("p", klass="Performance annotation 1"):
            text(
                "Subplot A: Household FRC forecasts from an ensemble of"
                + str(self.network_count)
                + " neural networks using the full provided dataset."
            )
        with tag("p", klass="Performance annotation 2"):
            text(
                "Subplot B: Confidence Interval (CI) reliability diagram. Each point shows the percentage of observations captured within each ensemble CI. An ideal model will have all points on the 1:1 line. If points are below the line, indicates forecast underdispersion (may lead to overly optimistic targets). If points are above the line, indicates overdispersion (may result in overly conservative targets)."
            )
        with tag("p", klass="Performance annotation 3"):
            text(
                "Subplot C: Rank Histogram. This creates a histogram of the relative location of all recorded observations relative to each ensemble member. An ideal model has a flat rank histogram. A U-shaped rank histogram indicates forecast underdispersion (may lead to overly optimistic targets). An arch-shaped rank histogram indicates overdispersion (may result in overly conservative targets)."
            )
        with tag("div", id="diagnostic_graphs"):
            doc.stag(
                "img",
                src=os.path.basename(
                    os.path.splitext(filename)[0] + "_Calibration_Diagnostic_Figs.png"
                ),
            )
        # doc.asis(
        #     '<object data="'
        #     + os.path.basename(
        #         os.path.splitext(filename)[0] + "_Calibration_Diagnostic_Figs.png"
        #     )
        #     + '" type="image/jpeg"></object>'
        # )

        doc.asis(skipped_rows_table)

        totalmatches = 0
        if len(self.ruleset):
            with tag("ul", id="ann_ruleset"):
                for rule in self.ruleset:
                    totalmatches += rule[2]
                    line("li", "%s. Matches: %d" % (rule[0], rule[2]))

        with tag("div", id="pythonSkipped_count"):
            text(totalmatches)

        file = open(filename, "w+")
        file.write(doc.getvalue())
        file.close()

        return doc.getvalue()

    def generate_metadata(self):
        metadata = {}
        metadata["average_time"] = self.avg_time_elapsed  # in seconds
        return metadata

    def prepare_table_for_html_report(self, storage_target):
        """Formats the results into an html table for display."""

        avg_table_df = pd.DataFrame()
        avg_table_df["Input FRC (mg/L)"] = self.avg_case_results_am[FRC_IN]
        avg_table_df["Storage Duration for Target"] = storage_target
        if WATTEMP in self.datainputs.columns:
            avg_table_df["Water Temperature (Degrees C)"] = self.avg_case_results_am[
                WATTEMP
            ]
        if COND in self.datainputs.columns:
            avg_table_df[
                "Electrical Conductivity (s*10^-6/cm)"
            ] = self.avg_case_results_am[COND]

        if self.post_process_check == False:
            avg_table_df[
                "Median Predicted Household FRC Concentration (mg/L) - AM Collection"
            ] = np.round(self.avg_case_results_am["median"], decimals=3)
            avg_table_df[
                "Median Predicted Household FRC Concentration (mg/L) - PM Collection"
            ] = np.round(self.avg_case_results_pm["median"], decimals=3)
            avg_table_df[
                "Predicted Risk of Household FRC below 0.20 mg/L - AM Collection"
            ] = np.round(self.avg_case_results_am["probability<=0.20"], decimals=3)
            avg_table_df[
                "Predicted Risk of Household FRC below 0.20 mg/L - PM Collection"
            ] = np.round(self.avg_case_results_pm["probability<=0.20"], decimals=3)
            # avg_table_df['Predicted Risk of Household FRC below 0.30 mg/L'] = self.avg_case_results['probability<=0.30']
        else:
            avg_table_df[
                "Median Predicted Household FRC Concentration (mg/L) - AM Collection"
            ] = np.round(self.avg_case_results_am_post["median"], decimals=3)
            avg_table_df[
                "Median Predicted Household FRC Concentration (mg/L) - PM Collection"
            ] = np.round(self.avg_case_results_pm_post["median"], decimals=3)
            avg_table_df[
                "Predicted Risk of Household FRC below 0.20 mg/L - AM Collection"
            ] = np.round(self.avg_case_results_am_post["probability<=0.20"], decimals=3)
            avg_table_df[
                "Predicted Risk of Household FRC below 0.20 mg/L - PM Collection"
            ] = np.round(self.avg_case_results_pm_post["probability<=0.20"], decimals=3)
            # avg_table_df['Predicted Risk of Household FRC below 0.30 mg/L'] = self.avg_case_results['probability<=0.30']

        str_io = io.StringIO()

        avg_table_df.to_html(buf=str_io, table_id="annTable")
        avg_html_str = str_io.getvalue()

        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            worst_table_df = pd.DataFrame()
            worst_table_df["Input FRC (mg/L)"] = self.worst_case_results_am[FRC_IN]
            worst_table_df["Storage Duration for Target"] = storage_target
            if WATTEMP in self.datainputs.columns:
                worst_table_df[
                    "Water Temperature(" + r"$\degree$" + "C)"
                ] = self.worst_case_results_am[WATTEMP]
            if COND in self.datainputs.columns:
                worst_table_df[
                    "Electrical Conductivity (" + r"$\mu$" + "s/cm)"
                ] = self.worst_case_results_am[COND]
            worst_table_df["Storage Duration for Target"] = storage_target
            if self.post_process_check == False:
                worst_table_df[
                    "Median Predicted FRC level at Household (mg/L) - AM Collection"
                ] = np.round(self.worst_case_results_am["median"], decimals=3)
                worst_table_df[
                    "Median Predicted FRC level at Household (mg/L) - PM Collection"
                ] = np.round(self.worst_case_results_pm["median"], decimals=3)
                worst_table_df[
                    "Predicted Risk of Household FRC below 0.20 mg/L - AMM Collection"
                ] = np.round(
                    self.worst_case_results_am["probability<=0.20"], decimals=3
                )
                worst_table_df[
                    "Predicted Risk of Household FRC below 0.20 mg/L - PM Collection"
                ] = np.round(
                    self.worst_case_results_pm["probability<=0.20"], decimals=3
                )
            else:
                worst_table_df[
                    "Median Predicted FRC level at Household (mg/L) - AM Collection"
                ] = np.round(self.worst_case_results_am_post["median"], decimals=3)
                worst_table_df[
                    "Median Predicted FRC level at Household (mg/L) - PM Collection"
                ] = np.round(self.worst_case_results_pm_post["median"], decimals=3)
                worst_table_df[
                    "Predicted Risk of Household FRC below 0.20 mg/L - AMM Collection"
                ] = np.round(
                    self.worst_case_results_am_post["probability<=0.20"], decimals=3
                )
                worst_table_df[
                    "Predicted Risk of Household FRC below 0.20 mg/L - PM Collection"
                ] = np.round(
                    self.worst_case_results_pm_post["probability<=0.20"], decimals=3
                )
            # worst_table_df['Predicted Risk of Household FRC below 0.30 mg/L'] = self.worst_case_results['probability<=0.30']

            str_io = io.StringIO()

            worst_table_df.to_html(buf=str_io, table_id="annTable")
            worst_html_str = str_io.getvalue()

            return avg_html_str, worst_html_str
        else:
            return avg_html_str

    def skipped_rows_html(self):
        if self.skipped_rows.empty:
            return ""

        printable_columns = [
            "ts_datetime",
            FRC_IN,
            "hh_datetime",
            FRC_OUT,
            WATTEMP,
            COND,
        ]
        required_columns = [rule[1] for rule in self.ruleset]

        doc, tag, text = Doc().tagtext()

        with tag(
            "table",
            klass="table center fill-whitespace",
            id="pythonSkipped",
            border="1",
        ):
            with tag("thead"):
                with tag("tr"):
                    for col in printable_columns:
                        with tag("th"):
                            text(col)
            with tag("tbody"):
                for (_, row) in self.skipped_rows[printable_columns].iterrows():
                    with tag("tr"):
                        for col in printable_columns:
                            with tag("td"):
                                # check if required value in cell is nan
                                if col in required_columns and (
                                    not row[col] or row[col] != row[col]
                                ):
                                    with tag("div", klass="red-cell"):
                                        text("")
                                else:
                                    text(row[col])

        return doc.getvalue()

    def valid_dates(self, series):
        mask = []
        for i in series.index.to_numpy():
            row = series[i]
            if row == None:
                mask.append(True)
                continue
            if isinstance(row, str) and not row.replace(".", "", 1).isdigit():
                try:
                    datetime.datetime.strptime(
                        row[:16].replace("/", "-"), self.xl_dateformat
                    )
                    mask.append(False)
                except:
                    mask.append(True)
            else:
                try:
                    start = float(row)
                    start = xldate_as_datetime(start, datemode=0)
                    mask.append(False)
                except:
                    mask.append(True)
        return mask

    def execute_rule(self, description, column, matches):
        rule = (description, column, sum(matches))
        self.ruleset.append(rule)
        if sum(matches):
            self.file.drop(self.file.loc[matches].index, inplace=True)

    def run_swot(self, input_file, results_file, report_file, storage_target):
        now = datetime.datetime.now()
        directory = os.path.join(
            "model_retraining",
            f'{now.strftime(r"%m%d%Y_%H%M%S")}_{os.path.basename(input_file)}',
        )

        # Uncommentfor Excel processing
        # file = pd.read_excel(input_file)

        file = pd.read_csv(input_file)

        # Support from 3 different input templates se1_frc, ts_frc, and ts frc1
        if "se1_frc" in file.columns:
            FRC_IN = "se1_frc"
            WATTEMP = "se1_wattemp"
            COND = "se1_cond"
            FRC_OUT = "se4_frc"
        elif "ts_frc1" in file.columns:
            FRC_IN = "ts_frc1"
            WATTEMP = "ts_wattemp"
            COND = "ts_cond"
            FRC_OUT = "hh_frc1"
        elif "ts_frc" in file.columns:
            FRC_IN = "ts_frc"
            WATTEMP = "ts_wattemp"
            COND = "ts_cond"
            FRC_OUT = "hh_frc"

        self.import_data_from_csv(input_file)
        self.set_up_model()
        self.train_SWOT_network(directory)
        self.calibration_performance_evaluation(report_file)
        self.post_process_cal()
        # self.full_performance_evaluation(directory)
        self.set_inputs_for_table(storage_target)
        self.predict()
        self.display_results()
        self.export_results_to_csv(results_file)
        self.generate_html_report(report_file, storage_target)
        metadata = self.generate_metadata()
        return metadata
