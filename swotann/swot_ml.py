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
import logging
from statsmodels.api import OLS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from swotann import QuantReg_Functions
from swotann import QuantReg_Models

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
        #self.execute_rule(
        #    "Invalid tapstand date/time",
        #    "ts_datetime",
        #    self.valid_dates(self.file["ts_datetime"]),
        #)
        #self.execute_rule(
        #    "Invalid household date/time",
        #    "hh_datetime",
        #    self.valid_dates(self.file["hh_datetime"]),
        #)
        self.skipped_rows = df.loc[df.index.difference(self.file.index)]

        self.file.reset_index(drop=True, inplace=True)  # fix dropped indices in pandas

        # Locate the rows of the missing data

        drop_threshold = np.maximum(0.10 * len(self.file.loc[:, [FRC_IN]]),200)
        nan_rows_watt = self.file.loc[self.file[WATTEMP].isnull()]

        if len(self.file.loc[:, [FRC_IN]])-len(nan_rows_watt) > drop_threshold:
            self.execute_rule(
                "Missing Water Temperature Measurement",
                WATTEMP,
                self.file[WATTEMP].isnull(),
            )

        nan_rows_cond = self.file.loc[self.file[COND].isnull()]
        if len(self.file.loc[:, [FRC_IN]])-len(nan_rows_cond) > drop_threshold:
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
        if len(self.file.loc[:, [FRC_IN]])-len(nan_rows_watt) > drop_threshold:
            self.predictors[WATTEMP] = self.file[WATTEMP]
            self.var_names.append("Water Temperature(" + r"$\degree$" + "C)")
            self.average_case_wattemp = np.median(self.file[WATTEMP].dropna().to_numpy())
            p = self.partial_corr(WATTEMP,  self.predictors.columns.drop([WATTEMP]))
            if p > 0:
                self.worst_case_wattemp = np.percentile(
                    self.file[WATTEMP].dropna().to_numpy(), 95
                )
            elif p < 0:
                self.worst_case_wattemp = np.percentile(
                    self.file[WATTEMP].dropna().to_numpy(), 5
                )

        if len(self.file.loc[:, [FRC_IN]])-len(nan_rows_cond) > drop_threshold:
            self.predictors[COND] = self.file[COND]
            self.var_names.append("EC (" + r"$\mu$" + "s/cm)")
            self.average_case_cond = np.median(self.file[COND].dropna().to_numpy())
            p = self.partial_corr(COND, self.predictors.columns.drop([COND]))
            if p > 0:
                self.worst_case_cond = np.percentile(
                    self.file[COND].dropna().to_numpy(), 95
                )
            elif p < 0:
                self.worst_case_cond = np.percentile(
                    self.file[COND].dropna().to_numpy(), 5
                )

        self.targets = self.targets.values.reshape(-1, 1)
        self.datainputs = self.predictors
        self.dataoutputs = self.targets
        self.input_filename = filename
        return

    def partial_corr(self, target_name, other_names):
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


        self.model=QuantReg_Models.cqrann(quantiles=self.quantiles,
                                          loss='smoothed',
                                          epsilon=10**-32,
                                          hidden_activation='tanh',
                                          kernel_initializer='GlorotUniform',
                                          n_hidden=5,
                                          hl_size=4,
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

        self.scores=QuantReg_Functions.evaluate_model(perf_df,self.quantiles,FRC_OUT,os.path.splitext(filename)[0],save=True)

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

        QuantReg_Functions.evaluate_model(perf_df, self.quantiles, FRC_IN,  os.path.splitext(filename)[0], save=True)
        return

    def set_inputs_for_table(self, storage_target):

        self.frc = np.arange(0.20, 2.05, 0.05)
        if storage_target<24:
            self.lag_time=np.arange(3,24.1,3)
        else:
            self.lag_time=np.arange(3,storage_target+1,3)
        if WATTEMP in self.predictors.columns and COND in self.predictors.columns:
            self.full_pred_array=np.array(
                np.meshgrid(self.frc,
                            self.lag_time,
                            np.array([0,1]),
                            np.array([self.average_case_wattemp,
                                      self.worst_case_wattemp]),
                            np.array([self.average_case_cond,
                                      self.worst_case_cond]))).T.reshape(-1,len(self.predictors.columns))
        elif WATTEMP in self.predictors.columns:
            self.full_pred_array = np.array(
                np.meshgrid(self.frc,
                            self.lag_time,
                            np.array([0, 1]),
                            np.array([self.average_case_wattemp,
                                      self.worst_case_wattemp]))).T.reshape(-1, len(self.predictors.columns))
        elif COND in self.predictors.columns:
            self.full_pred_array = np.array(
                np.meshgrid(self.frc,
                            self.lag_time,
                            np.array([0, 1]),
                            np.array([self.average_case_cond,
                                      self.worst_case_cond]))).T.reshape(-1, len(self.predictors.columns))
        else:
            self.full_pred_array = np.array(
                np.meshgrid(self.frc,
                            self.lag_time,
                            np.array([0, 1]))).T.reshape(-1, len(self.predictors.columns))

        self.full_pred_array=pd.DataFrame(data=self.full_pred_array,columns=self.predictors.columns)

        '''am_collect = [0 for i in range(0, len(frc))]
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
        self.worst_case_predictors_pm = pd.DataFrame(temp_95_pm)'''

    def risk_eval(self):
        """
        V3 Notes: With QR models, goal is no longer to count number of networks above/below the threshold,
        but instead to find the lowest quantile beyond which there are no predictions below the threshold
        """

        # Normalize the inputs using the input scaler loaded
        pred_array_scaled=self.predictors_scaler.transform(self.full_pred_array)

        temp_results = self.model.predict(pred_array_scaled)
        for key in temp_results.keys():
            temp_results[key] = self.targets_scaler.inverse_transform(temp_results[key].reshape(-1, 1)).flatten()

        temp_results = pd.DataFrame(temp_results)
        temp_results = temp_results.where(temp_results > 0, 0)


        proba_0 = 1-self.quantiles[np.argmin(np.array(np.abs(0.0 - temp_results)), axis=1)]
        proba_20 = 1-self.quantiles[np.argmin(np.array(np.abs(0.20 - temp_results)), axis=1)]
        proba_25 = 1-self.quantiles[np.argmin(np.array(np.abs(0.25 - temp_results)), axis=1)]
        proba_30 = 1-self.quantiles[np.argmin(np.array(np.abs(0.30 - temp_results)), axis=1)]

        self.full_results = pd.concat([self.full_pred_array, temp_results], axis=1)
        self.full_results["probability=0"] = proba_0
        self.full_results["probability<=0.20"] = proba_20
        self.full_results["probability<=0.25"] = proba_25
        self.full_results["probability<=0.30"] = proba_30

        grid_size =len(self.frc) * len(self.lag_time)
        total_grids=int(len(pred_array_scaled)/grid_size)
        grids=[]

        for i in range(total_grids):
            start_idx=i*grid_size
            end_idx=(i+1)*grid_size
            grids.append(proba_20[start_idx:end_idx].reshape(len(self.frc), len(self.lag_time)))
        self.grids=np.array(grids)
        self.max_grid=pd.DataFrame(data=np.max(self.grids,axis=0),index=self.frc,columns=self.lag_time)
        self.min_grid=pd.DataFrame(data=np.min(self.grids,axis=0),index=self.frc,columns=self.lag_time)

        '''avg_case_inputs_norm_am = self.predictors_scaler.transform(self.avg_case_predictors_am)
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
            self.risk_bands_0 = bands'''

        return

    def generate_metadata(self):
        metadata = {}
        metadata["average_time"] = self.avg_time_elapsed  # in seconds
        return metadata

    def display_results(self):
        """
        Display the results of the predictions as a console output.

        Display and return all the contents of the self.results variable which is a pandas Dataframe object
        :return: A Pandas Dataframe object (self.results) containing all the result of the predictions
        """
        logging.info(self.full_results)
        logging.info(self.min_grid)
        logging.info(self.max_grid)
        return (
            self.full_results,
            self.min_grid,
            self.max_grid
        )


    def export_results_to_csv(self, filename):
        self.full_results.to_csv(
            os.path.splitext(filename)[0] + "_full_prediction_results.csv", index=False
        )
        self.min_grid.to_csv(
            os.path.splitext(filename)[0] + "_min_predicted_safety.csv", index=True
        )
        self.max_grid.to_csv(
            os.path.splitext(filename)[0] + "_max_predicted_safety.csv", index=True
        )
        return

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
        metadata = self.generate_metadata()
        return metadata

