import base64
import datetime
import io
import os
import tempfile
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from xlrd.xldate import xldate_as_datetime
from yattag import Doc
from statsmodels.api import OLS
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from swotann import QuantReg_Functions
from swotann import QuantReg_Models
import quadprog
import cvxopt
cvxopt.solvers.options['maxiters'] = 100
cvxopt.solvers.options['feastol'] = 1e-7
cvxopt.solvers.options['reltol'] = 1e-6
cvxopt.solvers.options['abstol'] = 1e-7

plt.rcParams.update({"figure.autolayout": True})


"""
TF_CPP_MIN_LOG_LEVEL:
Defaults to 0, so all logs are shown. Set TF_CPP_MIN_LOG_LEVEL to 1 to filter out INFO logs, 2 to additionally filter out WARNING, 3 to additionally filter out ERROR.
"""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras


class SWOT_ML(object):
    def __init__(self):
        logging.getLogger().setLevel(logging.INFO)
        self.xl_dateformat = r"%Y-%m-%dT%H:%M"
        self.model = None
        self.pretrained_networks = []

        self.software_version = "3.0.1"
        self.input_filename = None
        self.today = str(datetime.date.today())
        self.avg_time_elapsed = 0

        self.predictors_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.targets_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.history = None
        self.file = None

        self.skipped_rows = []
        self.ruleset = []

        self.predictors = None

        self.targets = None
        self.predictions = None
        self.avg_case_results_am = None
        self.avg_case_results_pm = None
        self.worst_case_results_am = None
        self.worst_case_results_pm = None
        return


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
            self.average_case_wattemp = np.median(self.file[WATTEMP].dropna().to_numpy())

        if len(nan_rows_cond) < drop_threshold:
            self.predictors[COND] = self.file[COND]
            self.var_names.append("EC (" + r"$\mu$" + "s/cm)")
            self.average_case_cond = np.median(self.file[COND].dropna().to_numpy())

        if WATTEMP in self.predictors.columns:
            p=self.partial_corr(WATTEMP,FRC_OUT,self.predictors.columns.drop([WATTEMP]))
            if p>0:
                self.worst_case_wattemp = np.percentile(
                    self.file[WATTEMP].dropna().to_numpy(), 95
                )
            elif p<0:
                self.worst_case_wattemp = np.percentile(
                    self.file[WATTEMP].dropna().to_numpy(), 5
                )
        if COND in self.predictors.columns:
            p=self.partial_corr(COND,FRC_OUT,self.predictors.columns.drop([COND]))
            if p>0:
                self.worst_case_cond = np.percentile(
                    self.file[COND].dropna().to_numpy(), 95
                )
            elif p<0:
                self.worst_case_cond = np.percentile(
                    self.file[COND].dropna().to_numpy(), 5
                )

        self.targets = self.targets.values.reshape(-1, 1)
        self.datainputs = self.predictors
        self.dataoutputs = self.targets
        self.input_filename = filename
        return

    def partial_corr(self, target_name,y_name, other_names):
        """
        :param target_name: name of target variable for which partial corr is calculated
        :param y_name: name of the y_variable for the partial correlation
        :param other_names: list of names of variables to control for
        :return:
        """

        lmy=OLS(self.targets,self.predictors[other_names]).fit()
        residuals_y=lmy.resid

        lmx=OLS(self.predictors[target_name],self.predictors[other_names]).fit()
        residuals_x=lmx.resid

        pc=np.corrcoef([residuals_y,residuals_x])[0,1]
        return pc

    def set_up_model(self):
        """"
        Straightforward logic: if number of samples below DoF CQRNN, of if only available vars are FRC and Time, use
        SVQR. Otherwise, use CQRNN, with hyperparameters as selected
        Also, initialize the quantiles and scalers
        """
        self.predictors_scaler = self.predictors_scaler.fit(self.predictors)
        self.targets_scaler = self.targets_scaler.fit(self.targets)

        quantiles = np.arange(0.05, 1, 0.05)
        quantiles = np.append(0.0001, quantiles)
        quantiles = np.append(quantiles, 0.9999)
        self.quantiles=quantiles

        Ni=len(self.predictors.columns)
        Nh=8
        No=len(quantiles)
        CQRNN_dof=Ni*Nh+Nh+Nh*Nh+Nh+Nh*No+No
        if len(self.predictors.columns)==3:
            self.model=svqr(quantiles=self.quantiles,
                                            kernel='rbf',
                                            C=10)
        elif len(self.targets)<CQRNN_dof:
            self.model = svqr(quantiles=self.quantiles,
                                              kernel='rbf',
                                              C=10)

        else:
            self.model=cqrann(quantiles=self.quantiles,
                                              loss='smoothed',
                                              epsilon=10**-32,
                                              hidden_activation='tanh',
                                              kernel_initializer='GlorotUniform',
                                              n_hidden=2,
                                              hl_size=8,
                                              left_censor=None)

        return

    def train_ML_models(self, directory):
        """
        New idea here is to use the simplified "fit" methods baked into these models directly instead of creating an
        overly complex script here. Process is to scale the data (and save the scaler), fit the models, and output
        the model predictions, labelling the predictions by their quantile level.

        :param directory: save directory for models and modelling outputs
        :return:
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        self.calibration_predictions = []

        x_norm = self.predictors_scaler.transform(self.predictors)
        t_norm = self.targets_scaler.transform(self.targets)

        self.model.fit(x_norm,t_norm.flatten())
        self.calibration_predictions = self.model.predict(x_norm)
        for key in self.calibration_predictions.keys():
            self.calibration_predictions[key] = self.targets_scaler.inverse_transform(self.calibration_predictions[key].reshape(-1, 1)).flatten()
        return

    def calibration_performance_evaluation(self, filename):

        perf_df=pd.DataFrame(self.calibration_predictions)
        perf_df=perf_df.where(perf_df>0,0)
        perf_df[FRC_IN]=self.datainputs[FRC_IN].values
        perf_df[FRC_OUT]=self.targets.flatten()

        self.scores=QuantReg_Functions.evaluate_model(perf_df,self.quantiles,FRC_IN,FRC_OUT,os.path.splitext(filename)[0]
                                                      ,save=True)

        return

    def full_performance_evaluation(self, filename):
        x_norm = self.predictors_scaler.transform(self.predictors)
        t_norm = self.targets_scaler.transform(self.targets)

        self.eval_model=self.model

        x_cal_norm, x_test_norm, t_cal_norm, t_test_norm = train_test_split(
            x_norm, t_norm, test_size=0.25, shuffle=False, random_state=10
        )
        self.verifying_observations = self.targets_scaler.inverse_transform(t_test_norm)
        self.test_x_data = self.predictors_scaler.inverse_transform(x_test_norm)

        self.verifying_predictions = []

        self.eval_model.fit(x_cal_norm,t_cal_norm.flatten())
        self.verifying_predictions = self.eval_model.predict(x_test_norm)
        for key in self.verifying_predictions.keys():
            self.verifying_predictions[key]=self.targets_scaler.inverse_transform(self.verifying_predictions[key].reshape(-1, 1)).flatten()

        perf_df = pd.DataFrame(self.verifying_predictions)
        perf_df = perf_df.where(perf_df > 0, 0)
        perf_df[FRC_IN] = self.test_x_data[:,0]
        perf_df[FRC_OUT] = self.verifying_observations.flatten()

        QuantReg_Functions.evaluate_model(perf_df, self.quantiles, FRC_IN, FRC_OUT, os.path.splitext(filename)[0], save=True)
        return

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
            watt_med = [self.average_case_wattemp for i in range(0, len(frc))]
            watt_95 = [self.worst_case_wattemp for i in range(0, len(frc))]
            temp_med_am.update({"ts_wattemp": watt_med})
            temp_med_pm.update({"ts_wattemp": watt_med})
            temp_95_am.update({"ts_wattemp": watt_95})
            temp_95_pm.update({"ts_wattemp": watt_95})
        if COND in self.datainputs.columns:
            cond_med = [self.average_case_cond for i in range(0, len(frc))]
            cond_95 = [self.worst_case_cond for i in range(0, len(frc))]
            temp_med_am.update({"ts_cond": cond_med})
            temp_med_pm.update({"ts_cond": cond_med})
            temp_95_am.update({"ts_cond": cond_95})
            temp_95_pm.update({"ts_cond": cond_95})

        self.avg_case_predictors_am = pd.DataFrame(temp_med_am)
        self.avg_case_predictors_pm = pd.DataFrame(temp_med_pm)
        self.worst_case_predictors_am = pd.DataFrame(temp_95_am)
        self.worst_case_predictors_pm = pd.DataFrame(temp_95_pm)

    def risk_eval(self):
        """
        V3 Notes: With QR models, goal is no longer to count number of networks above/below the threshold,
        but instead to find the lowest quantile beyond which there are no predictions below the threshold
        """

        # Normalize the inputs using the input scaler loaded
        avg_case_inputs_norm_am = self.predictors_scaler.transform(self.avg_case_predictors_am)
        avg_case_inputs_norm_pm = self.predictors_scaler.transform(self.avg_case_predictors_pm)
        worst_case_inputs_norm_am = self.predictors_scaler.transform(
            self.worst_case_predictors_am
        )
        worst_case_inputs_norm_pm = self.predictors_scaler.transform(
            self.worst_case_predictors_pm
        )

        ##AVERAGE CASE TARGET w AM COLLECTION

        # Iterate through all loaded pretrained networks, make predictions based on the inputs,
        # calculate the median of the predictions and store everything to self.results

        temp_results=self.model.predict(avg_case_inputs_norm_am)
        for key in temp_results.keys():
            temp_results[key]=self.targets_scaler.inverse_transform(temp_results[key].reshape(-1, 1)).flatten()

        temp_results = pd.DataFrame(temp_results)
        temp_results = temp_results.where(temp_results > 0, 0)
        proba_0=self.quantiles[np.argmin(np.array(np.abs(0.0-temp_results)),axis=1)]
        proba_20=self.quantiles[np.argmin(np.array(np.abs(0.20-temp_results)),axis=1)]
        proba_25=self.quantiles[np.argmin(np.array(np.abs(0.25-temp_results)),axis=1)]
        proba_30=self.quantiles[np.argmin(np.array(np.abs(0.30-temp_results)),axis=1)]

        temp_results["probability=0"]=proba_0
        temp_results["probability<=0.20"]=proba_20
        temp_results["probability<=0.25"] = proba_25
        temp_results["probability<=0.30"] = proba_30
        temp_results["median"]=temp_results['0.5'] #Don't love this since it is just a duplication of the same column, should try to find where this is used

        temp_results=pd.concat([temp_results,self.avg_case_predictors_am],axis=1)#Not sure if this is really necessary
        self.avg_case_results_am=temp_results

        ##AVERAGE CASE TARGET w PM COLLECTION

        # Iterate through all loaded pretrained networks, make predictions based on the inputs,
        # calculate the median of the predictions and store everything to self.results
        temp_results = self.model.predict(avg_case_inputs_norm_pm)
        for key in temp_results.keys():
            temp_results[key] = self.targets_scaler.inverse_transform(temp_results[key].reshape(-1, 1)).flatten()

        temp_results = pd.DataFrame(temp_results)
        temp_results = temp_results.where(temp_results > 0, 0)
        proba_0 = self.quantiles[np.argmin(np.array(np.abs(0.0 - temp_results)), axis=1)]
        proba_20 = self.quantiles[np.argmin(np.array(np.abs(0.20 - temp_results)), axis=1)]
        proba_25 = self.quantiles[np.argmin(np.array(np.abs(0.25 - temp_results)), axis=1)]
        proba_30 = self.quantiles[np.argmin(np.array(np.abs(0.30 - temp_results)), axis=1)]

        temp_results["probability=0"] = proba_0
        temp_results["probability<=0.20"] = proba_20
        temp_results["probability<=0.25"] = proba_25
        temp_results["probability<=0.30"] = proba_30
        temp_results["median"] = temp_results[
            '0.5']  # Don't love this since it is just a duplication of the same column, should try to find where this is used

        temp_results = pd.concat([temp_results, self.avg_case_predictors_am],
                                 axis=1)  # Not sure if this is really necessary
        self.avg_case_results_pm = temp_results


        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            ##WORST CASE TARGET w AM COLLECTION

            temp_results = self.model.predict(worst_case_inputs_norm_am)
            for key in temp_results.keys():
                temp_results[key] = self.targets_scaler.inverse_transform(temp_results[key].reshape(-1, 1)).flatten()

            temp_results = pd.DataFrame(temp_results)
            temp_results = temp_results.where(temp_results > 0, 0)
            proba_0 = self.quantiles[np.argmin(np.array(np.abs(0.0 - temp_results)), axis=1)]
            proba_20 = self.quantiles[np.argmin(np.array(np.abs(0.20 - temp_results)), axis=1)]
            proba_25 = self.quantiles[np.argmin(np.array(np.abs(0.25 - temp_results)), axis=1)]
            proba_30 = self.quantiles[np.argmin(np.array(np.abs(0.30 - temp_results)), axis=1)]

            temp_results["probability=0"] = proba_0
            temp_results["probability<=0.20"] = proba_20
            temp_results["probability<=0.25"] = proba_25
            temp_results["probability<=0.30"] = proba_30
            temp_results["median"] = temp_results[
                '0.5']  # Don't love this since it is just a duplication of the same column, should try to find where this is used

            temp_results = pd.concat([temp_results, self.avg_case_predictors_am],
                                     axis=1)  # Not sure if this is really necessary
            self.worst_case_results_am = temp_results

            ##WORST CASE TARGET w PM COLLECTION

            temp_results = self.model.predict(worst_case_inputs_norm_pm)
            for key in temp_results.keys():
                temp_results[key] = self.targets_scaler.inverse_transform(temp_results[key].reshape(-1, 1)).flatten()

            temp_results = pd.DataFrame(temp_results)
            temp_results = temp_results.where(temp_results > 0, 0)
            proba_0 = self.quantiles[np.argmin(np.array(np.abs(0.0 - temp_results)), axis=1)]
            proba_20 = self.quantiles[np.argmin(np.array(np.abs(0.20 - temp_results)), axis=1)]
            proba_25 = self.quantiles[np.argmin(np.array(np.abs(0.25 - temp_results)), axis=1)]
            proba_30 = self.quantiles[np.argmin(np.array(np.abs(0.30 - temp_results)), axis=1)]

            temp_results["probability=0"] = proba_0
            temp_results["probability<=0.20"] = proba_20
            temp_results["probability<=0.25"] = proba_25
            temp_results["probability<=0.30"] = proba_30
            temp_results["median"] = temp_results[
                '0.5']  # Don't love this since it is just a duplication of the same column, should try to find where this is used

            temp_results = pd.concat([temp_results, self.avg_case_predictors_am],
                                     axis=1)  # Not sure if this is really necessary
            self.worst_case_results_pm = temp_results

            bands = pd.concat(
                [self.avg_case_results_am["probability<=0.20"], self.avg_case_results_pm["probability<=0.20"],
                 self.worst_case_results_am["probability<=0.20"], self.worst_case_results_pm["probability<=0.20"]],
                axis=1)  # Space here to be a bit more flexible...look at possibility of 0 FRC

            lb=bands.min(axis=1)
            ub=bands.max(axis=1)
            bands["Lower Bound"]=lb
            bands["Upper Bound"]=ub
            bands.index=self.avg_case_predictors_am[FRC_IN]
            bands=bands[["Lower Bound","Upper Bound"]]
            self.risk_bands_20=bands

            bands = pd.concat(
                [self.avg_case_results_am["probability=0"], self.avg_case_results_pm["probability=0"],
                 self.worst_case_results_am["probability=0"], self.worst_case_results_pm["probability=0"]],
                axis=1)
            lb = bands.min(axis=1)
            ub = bands.max(axis=1)
            bands["Lower Bound"] = lb
            bands["Upper Bound"] = ub
            bands.index = self.avg_case_predictors_am[FRC_IN]
            bands = bands[["Lower Bound", "Upper Bound"]]
            self.risk_bands_0 = bands
        else:
            bands = pd.concat(
                [self.avg_case_results_am["probability<=0.20"], self.avg_case_results_pm["probability<=0.20"]],
                axis=1)

            lb = bands.min(axis=1)
            ub = bands.max(axis=1)
            bands["Lower Bound"] = lb
            bands["Upper Bound"] = ub
            bands.index = self.avg_case_predictors_am[FRC_IN]
            bands = bands[["Lower Bound", "Upper Bound"]]
            self.risk_bands_20 = bands

            bands = pd.concat(
                [self.avg_case_results_am["probability=0"], self.avg_case_results_pm["probability=0"]],
                axis=1)
            lb = bands.min(axis=1)
            ub = bands.max(axis=1)
            bands["Lower Bound"] = lb
            bands["Upper Bound"] = ub
            bands.index = self.avg_case_predictors_am[FRC_IN]
            bands = bands[["Lower Bound", "Upper Bound"]]
            self.risk_bands_0 = bands

        return

    def results_visualization(self, filename, storage_target):
        # Variables to plot - Full range, 95th percentile, 99th percentile, median, the three risks
        risk_fig = plt.figure(figsize=(6.69, 3.35), dpi=300)
        plt.fill_between(self.risk_bands_20.index, self.risk_bands_20["Lower Bound"], self.risk_bands_20["Upper Bound"], alpha=0.5,
                         facecolor="#b80000",
                         label='Risk Range - FRC below 0.2 mg/L')
        plt.plot(self.risk_bands_20.index, self.risk_bands_20["Lower Bound"],c="#b80000")
        plt.plot(self.risk_bands_20.index, self.risk_bands_20["Upper Bound"], c="#b80000")


        '''plt.fill_between(self.risk_bands_0.index, self.risk_bands_0["Lower Bound"], self.risk_bands_0["Upper Bound"], alpha=0.5,
                         facecolor="#b80000",
                         label='Risk Range - 0 mg/L FRC')'''

        plt.xlim([0.2, 2])
        plt.xlabel("Tapstand FRC (mg/L)")
        plt.ylim([0, 1])
        plt.ylabel("Risk")
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

        return risk_base_64_pngData

    def display_results(self):
        """
        Display the results of the predictions as a console output.

        Display and return all the contents of the self.results variable which is a pandas Dataframe object
        :return: A Pandas Dataframe object (self.results) containing all the result of the predictions
        """
        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:

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
            logging.info(self.avg_case_results_am)
            logging.info(self.avg_case_results_pm)
            return self.avg_case_results_am, self.avg_case_results_pm

    def export_results_to_csv(self, filename):
        self.avg_case_results_am.to_csv(
            os.path.splitext(filename)[0] + "_average_case_am.csv", index=False
        )
        self.avg_case_results_pm.to_csv(
            os.path.splitext(filename)[0] + "_average_case_pm.csv", index=False
        )

        self.risk_bands_20.to_csv(os.path.splitext(filename)[0] + "Risk_Bands_02.csv", index=True)
        self.risk_bands_0.to_csv(os.path.splitext(filename)[0] + "Risk_Bands_0.csv", index=True)

        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            self.worst_case_results_am.to_csv(
                os.path.splitext(filename)[0] + "_worst_case_am.csv", index=False
            )
            self.worst_case_results_pm.to_csv(
                os.path.splitext(filename)[0] + "_worst_case_pm.csv", index=False
            )
        return

    def generate_html_report(self, filename, storage_target):
        """Generates an html report of the SWOT results. The report
        is saved on disk under the name 'filename'."""

        df = self.datainputs
        frc = df[FRC_IN]

        #risk = self.results_visualization(filename, storage_target)
        #risk.decode("UTF-8")

        str_io = io.StringIO()
        pd.DataFrame(self.scores).to_html(buf=str_io, table_id="ScoresTable")
        scores_html_str = str_io.getvalue()


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

        with tag("h2", klass="Header"):
            text("Predicted Risk")
        '''with tag("p", klass="Risk Fig Text"):
            text(
                "Figure and tables showing predicted risk of household FRC below 0.2 mg/L for average and worst case scenarios for both AM and PM collection. Risk obtained from forecast pdf (above) and taken as cumulative probability of houeshold FRC below 0.2 mg/L. Note that 0% predicted risk of household FRC below 0.2 mg/L does not mean that there is no possibility of household FRC being below 0.2 mg/L, simply that the predicted risk is too low to be measured. The average case target may, in some, cases be more conservative than the worst case targets as the worst case target is derived on the assumption that higher conductivity and water temperature will lead to greater decay (as confirmed by FRC decay chemisty and results at past sites). However, this may not be true in all cases, so the most conservative target is always recommended."
            )
        with tag("div", id="Risk Graphs"):
            doc.stag(
                "img",
                src=os.path.basename(
                    os.path.splitext(filename)[0] + "_Risk_Fig.png"
                ),
            )'''

        if WATTEMP in self.datainputs.columns or COND in self.datainputs.columns:
            with tag("h2", klass="Header"):
                text("Average Case Targets Table")
            with tag("table", id="average case table"):
                doc.asis(avg_html_table)
            with tag("h2", klass="Header"):
                text("Worst Case Targets Table")
            with tag("table", id="worst case table"):
                doc.asis(worst_html_table)

        else:
            with tag("h2", klass="Header"):
                text("Targets Table")
            with tag("table", id="average case table"):
                doc.asis(avg_html_table)


        with tag("h2", klass="Header"):
            text("Model Diagnostics")
        with tag("p", klass="Performance Indicator General Text"):
            text(
                "This table summarizes the scores obtained by the model during training. Percent Capture and Percent "
                "Capture (HH FRC < 0.2 mg/L) should be higher (ideal is 100% for both) and the remaining scores should"
                "be lower. The best Delta Score is 1, the rest have a best score of 0. Note high Delta scores occur at "
                "times when there are many observations."
            )
        with tag("table", klass="Performance Table"):
            doc.asis(scores_html_str)
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
        avg_table_df[
            "Predicted Risk of No Household FRC - AM Collection"
        ] = np.round(self.avg_case_results_am["probability=0"], decimals=3)
        avg_table_df[
            "Predicted Risk of No Household FRC - PM Collection"
        ] = np.round(self.avg_case_results_pm["probability=0"], decimals=3)

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
            worst_table_df[
                "Predicted Risk of No Household FRC - AMM Collection"
            ] = np.round(
                self.worst_case_results_am["probability=0"], decimals=3
            )
            worst_table_df[
                "Predicted Risk of No Household FRC - PM Collection"
            ] = np.round(
                self.worst_case_results_pm["probability=0"], decimals=3
            )

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

    def run_swot(
        self, input_file, results_file, report_file, storage_target, usetmpdir=False
    ):
        now = datetime.datetime.now()
        if usetmpdir:
            tmp_dirpath = tempfile.gettempdir()
        else:
            tmp_dirpath = ""
        directory = os.path.join(
            tmp_dirpath,
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
        #self.full_performance_evaluation(directory)
        self.train_ML_models(directory)
        self.calibration_performance_evaluation(results_file)
        self.set_inputs_for_table(storage_target)
        self.risk_eval()
        self.display_results()
        self.export_results_to_csv(results_file)
        self.generate_html_report(report_file, storage_target)
        metadata = self.generate_metadata()
        return metadata

class cqrann:
    def __init__(self, quantiles, hidden_activation='tanh',kernel_initializer='GlorotUniform', output_activation='linear', loss='pinball',
                 epsilon=0, n_hidden=1, hl_size=4, optimizer='Nadam', validation_percent=0.1,left_censor=None):
        self.quantiles = quantiles
        self.No = len(quantiles)

        if loss == 'pinball':
            self.loss = loss
        elif loss == 'smoothed':
            self.loss = loss
            self.epsilon=epsilon
        else:
            raise ValueError("Acceptable loss functions are 'pinball' or 'smoothed'")

        self.n_hidden = n_hidden
        self.Nh = hl_size
        self.hidden_activation = hidden_activation
        self.kernel_initializer = kernel_initializer
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.left_censor = None
        if left_censor is not None:
            self.left_censor = left_censor
        self.build_status = 0
        self.train_status = 0


        self.Ni = None
        self.val = validation_percent
        self.models = {}
        return


    def build_model(self):
        model = keras.models.Sequential()
        for i in range(self.n_hidden):

            model.add(keras.layers.Dense(self.Nh, activation=self.hidden_activation, kernel_initializer="uniform",
                                     bias_initializer="zeros"))
        model.add(keras.layers.Dense(self.No, kernel_initializer="uniform", bias_initializer="zeros", activation=self.output_activation))
        if self.left_censor is not None:
            model.add(keras.layers.ThresholdedReLU(theta=self.left_censor))
        model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=None)
        self.build_status=1
        self.base_model=model
        return


    def fit(self, X, y):
        if self.build_status==0:
            self.build_model()

        early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001,
                                                               patience=100,
                                                               restore_best_weights=True)

        tf.keras.backend.clear_session()
        model = keras.models.clone_model(self.base_model)

        if self.loss == 'smoothed':
            model.cost = QuantReg_Functions.simultaneous_loss_keras(self.quantiles, self.epsilon)
        else:
            #model.cost=QuantReg_Functions.pinball_loss_keras(self.quantiles)
            model.cost = QuantReg_Functions.simultaneous_loss_keras(self.quantiles, 0.0000000000000000000000000000001)

        model.compile(optimizer=self.optimizer, loss=model.cost)
        model.fit(X, y, validation_split=self.val, epochs=500, callbacks=[early_stopping_monitor],verbose=False)
        self.model=model
        self.train_status = 1
        return

    def predict(self,X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")

        predarray=self.model.predict(X)
        preds={}
        for q in range(len(self.quantiles)):
            preds.update({str(np.round(self.quantiles[q],decimals=4)):predarray[:,q]})
        return preds

class svqr:
    def __init__(self, quantiles, C=1, kernel='linear', degree=3, gamma='auto', c0=1):
        self.C = C
        self.kernel = kernel
        self.quantiles = quantiles
        self.gamma = gamma
        self.c0 = c0
        self.degree = degree
        self.train_status = 0

        self.kernel_dict = {
            'linear': self.linear_kernel_matrix,
            'rbf': self.rbf_kernel_matrix,
            'polynomial': self.polynomial_kernel_matrix,
            'sigmoid': self.sigmoid_kernel_matrix
        }

        if self.kernel not in self.kernel_dict.keys():
            Warning(
                "Value Error: " + self.kernel + " is not a valid kernel. Valid kernels are: 'linear', 'rbf', 'polynomial', or 'sigmoid'")
        self.models = {}
        return

    def fit(self, X, y):
        '''From Takeuchi et al. (2006), using the dual problem, the minimization is 1/1*alphaTKalpha-alphaTy subject to
        C(tau-1)<=alpha<=C*tau
        where w=sum(alpha_i, phi(xi)'''

        n_samples, n_features = X.shape

        if self.gamma == 'scale':
            self.gamma = 1 / (n_features * X.var())
        if self.gamma == 'auto':
            self.gamma = 1 / n_features

        kern = self.kernel_dict[self.kernel]
        K = kern(X, X)
        G = K
        try:
            np.linalg.cholesky(G)
        except np.linalg.linalg.LinAlgError:
            G = G + np.eye(n_samples) * 1e-10

        a = y
        C = np.vstack((np.eye(n_samples), -1 * np.eye(n_samples)))
        C = np.vstack((np.ones(n_samples), C))
        for q in self.quantiles:
            b0 = 0.0
            b0 = np.append(b0, np.array([self.C * (q - 1) for i in range(n_samples)]))
            b0 = np.append(b0, np.array([-1 * self.C * q for i in range(n_samples)]))
            try:
                res = quadprog.solve_qp(G=G, a=a, C=C.T, b=b0, meq=1)
            except ValueError:
                res = quadprog.solve_qp(G=G + np.eye(n_samples) * 1e-10, a=a, C=C.T, b=b0, meq=1)
            alpha = res[0]
            f = np.matmul(alpha, K)
            offshift = np.argmin(
                (np.round(alpha, 3) - (self.C * q)) ** 2 + (np.round(alpha, 3) - (self.C * (q - 1))) ** 2)

            model = {'alpha': alpha, 'b':y[offshift] - f[offshift]}

            self.models.update({str(q): model})
        self.sv = X
        self.train_status = 1
        return

    def linear_kernel_matrix(self, x_mat, y_mat):
        n_samp_x = x_mat.shape[0]
        n_samp_y = y_mat.shape[0]

        kern_mat = np.zeros((n_samp_x, n_samp_y))
        for i in range(n_samp_x):
            for j in range(n_samp_y):
                kern_mat[i, j] = np.dot(x_mat[i], y_mat[j])
        return kern_mat

    def rbf_kernel_matrix(self, x_mat, y_mat):
        n_samp_x = x_mat.shape[0]
        n_samp_y = y_mat.shape[0]

        kern_mat = np.zeros((n_samp_x, n_samp_y))
        for i in range(n_samp_x):
            for j in range(n_samp_y):
                kern_mat[i, j] = np.exp(-1 * self.gamma * (np.linalg.norm(x_mat[i, :] - y_mat[j, :]) ** 2))
        return kern_mat

    def sigmoid_kernel_matrix(self, x_mat, y_mat):
        n_samp_x = x_mat.shape[0]
        n_samp_y = y_mat.shape[0]

        kern_mat = np.zeros((n_samp_x, n_samp_y))
        for i in range(n_samp_x):
            for j in range(n_samp_y):
                kern_mat[i, j] = np.tanh(self.gamma * np.dot(x_mat[i], y_mat[j]) + self.c0)
        return kern_mat

    def polynomial_kernel_matrix(self, x_mat, y_mat):
        n_samp_x = x_mat.shape[0]
        n_samp_y = y_mat.shape[0]

        kern_mat = np.zeros((n_samp_x, n_samp_y))
        for i in range(n_samp_x):
            for j in range(n_samp_y):
                kern_mat[i, j] = (self.gamma * np.dot(x_mat[i], y_mat[j]) + self.c0) ** self.degree
        return kern_mat

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")
        kern = self.kernel_dict[self.kernel]

        x_proj = kern(X, self.sv)

        preds = {}
        for q in self.quantiles:
            model = self.models[str(q)]
            preds.update({str(np.round(q, decimals=4)): np.dot(x_proj, model['alpha']) + model['b']})
        return preds