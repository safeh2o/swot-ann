import cvxopt
import keras
import numpy as np
import quadprog
import statsmodels.api as sm
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from swotann import QuantReg_Functions
from memory_profiler import profile

# from quadprog import solve_qp
cvxopt.solvers.options['maxiters'] = 100
cvxopt.solvers.options['feastol'] = 1e-7
cvxopt.solvers.options['reltol'] = 1e-6
cvxopt.solvers.options['abstol'] = 1e-7


class qrann:
    def __init__(self, quantiles, hidden_activation='tanh', kernel_initializer='GlorotNormal',
                 output_activation='linear', loss='pinball', n_hidden=2, hl_size=4, optimizer='Nadam',
                 validation_percent=0.1, epsilon=10, left_censor=None):
        self.quantiles = quantiles
        self.No = 1

        if loss == 'pinball':
            self.loss = loss
        elif loss == 'smoothed':
            self.loss = loss
            self.epsilon = epsilon
        else:
            raise ValueError("Acceptable loss functions are 'pinball' or 'smoothed'")
        self.n_hidden = n_hidden
        self.Nh = hl_size
        self.left_censor = None
        if left_censor is not None:
            self.left_censor = left_censor
        self.hidden_activation = hidden_activation
        self.kernel_initializer = kernel_initializer
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.build_status = 0
        self.train_status = 0
        self.left_censor = left_censor
        self.Ni = None
        self.val = validation_percent
        self.models = {}
        return

    def build_model(self):
        model = tf.keras.models.Sequential()
        for i in range(self.n_hidden):
            model.add(tf.keras.layers.Dense(self.Nh, activation=self.hidden_activation,
                                            kernel_initializer=self.kernel_initializer,
                                            bias_initializer="zeros"))
            # model.add(keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(self.No, kernel_initializer="uniform", bias_initializer="zeros",
                                        activation=self.output_activation))
        if self.left_censor is not None:
            model.add(tf.keras.layers.ThresholdedReLU(theta=self.left_censor))
        model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=None)
        self.build_status = 1
        self.base_model = model
        return

    def fit(self, X, y):
        if self.build_status == 0:
            self.build_model()

        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001,
                                                                  patience=100,
                                                                  restore_best_weights=True)
        for q in self.quantiles:
            tf.keras.backend.clear_session()
            model = tf.keras.models.clone_model(self.base_model)

            if self.loss == 'smoothed':
                model.cost = QuantReg_Functions.smoothed_pinball_keras(q, self.epsilon)


            else:
                # model.cost=QuantReg_Functions.pinball_loss_keras(q)
                model.cost = QuantReg_Functions.smoothed_pinball_keras(q, 0.000000000000000000000000001)

            model.compile(optimizer=self.optimizer, loss=model.cost)
            model.fit(X, y, validation_split=self.val, epochs=500, callbacks=[early_stopping_monitor], verbose=False)
            self.models.update({str(q): model})
        self.train_status = 1
        return

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")
        preds = {}
        for q in self.quantiles:
            model = self.models[str(q)]
            preds.update({str(np.round(q, decimals=4)): model.predict(X)})
        return preds


class cqrann:
    def __init__(self, quantiles, hidden_activation='tanh', kernel_initializer='GlorotUniform',
                 output_activation='linear', loss='pinball',
                 epsilon=0, n_hidden=1, hl_size=4, optimizer='Nadam', validation_percent=0.1, left_censor=None):
        self.quantiles = quantiles
        self.No = len(quantiles)

        if loss == 'pinball':
            self.loss = loss
        elif loss == 'smoothed':
            self.loss = loss
            self.epsilon = epsilon
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
        model = tf.keras.models.Sequential()
        for i in range(self.n_hidden):
            model.add(tf.keras.layers.Dense(self.Nh, activation=self.hidden_activation, kernel_initializer="uniform",
                                            bias_initializer="zeros"))
        model.add(tf.keras.layers.Dense(self.No, kernel_initializer="uniform", bias_initializer="zeros",
                                        activation=self.output_activation))
        if self.left_censor is not None:
            model.add(tf.keras.layers.ThresholdedReLU(theta=self.left_censor))
        model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=None)
        self.build_status = 1
        self.base_model = model
        return

    def fit(self, X, y):
        if self.build_status == 0:
            self.build_model()

        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001,
                                                                  patience=100,
                                                                  restore_best_weights=True)

        tf.keras.backend.clear_session()
        model = tf.keras.models.clone_model(self.base_model)

        if self.loss == 'smoothed':
            model.cost = QuantReg_Functions.simultaneous_loss_keras(self.quantiles, self.epsilon)
        else:
            # model.cost=QuantReg_Functions.pinball_loss_keras(self.quantiles)
            model.cost = QuantReg_Functions.simultaneous_loss_keras(self.quantiles, 0.0000000000000000000000000000001)

        model.compile(optimizer=self.optimizer, loss=model.cost)
        model.fit(X, y, validation_split=self.val, epochs=500, callbacks=[early_stopping_monitor], verbose=False)
        self.model = model
        self.train_status = 1
        return

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")

        predarray = self.model.predict(X)
        preds = {}
        for q in range(len(self.quantiles)):
            preds.update({str(np.round(self.quantiles[q], decimals=4)): predarray[:, q]})
        return preds


class mqrann:
    def __init__(self, quantiles, hidden_activation='tanh', kernel_initializer='GlorotUniform',
                 output_activation='linear', loss='pinball', n_hidden=1, hl_size=4, optimizer='Nadam',
                 validation_percent=0.1, epsilon=10, left_censor=None, monotone_indices=[0]):
        self.quantiles = quantiles
        self.No = 1

        if loss == 'pinball':
            self.loss = loss
        elif loss == 'smoothed':
            self.loss = loss
            self.epsilon = epsilon
        else:
            raise ValueError("Acceptable loss functions are 'pinball' or 'smoothed'")
        self.n_hidden = n_hidden
        self.Nh = hl_size
        self.left_censor = None
        if left_censor is not None:
            self.left_censor = left_censor
        self.hidden_activation = hidden_activation
        self.kernel_initializer = kernel_initializer
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.monotone_indices = monotone_indices
        self.build_status = 0
        self.train_status = 0
        self.left_censor = left_censor
        self.Ni = None
        self.val = validation_percent
        self.models = {}
        return

    def build_model(self):
        if self.n_hidden <= 2:
            model = tf.keras.models.Sequential()
            model.add(QuantReg_Functions.MonotoneHidden(units=self.Nh, activation=self.hidden_activation,
                                                        kernel_initializer="uniform",
                                                        bias_initializer="zeros", monotone=len(self.monotone_indices)))
            model.add(tf.keras.layers.Dense(self.Nh, activation=self.hidden_activation, kernel_initializer="uniform",
                                            bias_initializer="zeros"))
            model.add(tf.keras.layers.Dense(self.No, kernel_initializer="uniform", bias_initializer="zeros",
                                            activation=self.output_activation))
            if self.left_censor is not None:
                model.add(tf.keras.layers.ThresholdedReLU(theta=self.left_censor))
            model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=None)
            self.build_status = 1
            # self.base_model = model


        else:
            model = tf.keras.models.Sequential()
            model.add(QuantReg_Functions.MonotoneHidden(units=self.Nh, activation=self.hidden_activation,
                                                        kernel_initializer="uniform",
                                                        bias_initializer="zeros",
                                                        monotone=len(self.monotone_indices)))
            for i in range(self.n_hidden - 1):
                model.add(
                    tf.keras.layers.Dense(self.Nh, activation=self.hidden_activation, kernel_initializer="uniform",
                                          bias_initializer="zeros"))
            model.add(tf.keras.layers.Dense(self.No, kernel_initializer="uniform", bias_initializer="zeros",
                                            activation=self.output_activation))
            if self.left_censor is not None:
                model.add(tf.keras.layers.ThresholdedReLU(theta=self.left_censor))
            model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=None)
            self.build_status = 1
            # self.base_model = model

        return model

    def fit(self, X, y):
        if self.build_status == 0:
            self.build_model()

        early_stopping_monitor = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001,
                                                                  patience=100,
                                                                  restore_best_weights=True)
        new_X = np.zeros(np.shape(X))
        non_monotone_indices = np.delete(np.arange(len(X[0, :])), self.monotone_indices)

        for j in range(len(self.monotone_indices)):
            new_X[:, j] = X[:, self.monotone_indices[j]]
        for j in range(len(non_monotone_indices)):
            new_X[:, len(self.monotone_indices) + j] = X[:, non_monotone_indices[j]]
        custom_object = {'MonotoneHidden': QuantReg_Functions.MonotoneHidden}
        keras.utils.get_custom_objects().update(custom_object)

        for q in self.quantiles:
            tf.keras.backend.clear_session()
            # model = tf.keras.models.clone_model(self.base_model)
            model = self.build_model()

            if self.loss == 'smoothed':
                model.cost = QuantReg_Functions.smoothed_pinball_keras(q, self.epsilon)
            else:
                model.cost = QuantReg_Functions.smoothed_pinball_keras(q, 0.000000000000000000000000001)

            model.compile(optimizer=self.optimizer, loss=model.cost)
            model.fit(new_X, y, validation_split=self.val, epochs=500, callbacks=[early_stopping_monitor],
                      verbose=False)
            self.models.update({str(q): model})
        self.train_status = 1
        return

    def predict(self, X):

        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")
        new_X = np.zeros(np.shape(X))
        non_monotone_indices = np.delete(np.arange(len(X[0, :])), self.monotone_indices)

        for j in range(len(self.monotone_indices)):
            new_X[:, j] = X[:, self.monotone_indices[j]]
        for j in range(len(non_monotone_indices)):
            new_X[:, len(self.monotone_indices) + j] = X[:, non_monotone_indices[j]]
        preds = {}
        for q in self.quantiles:
            model = self.models[str(q)]
            preds.update({str(np.round(q, decimals=4)): model.predict(new_X)})
        return preds


class mcqrann:
    def __init__(self, quantiles, hidden_activation='tanh', output_activation='linear', loss='pinball',
                 epsilon=0, n_hidden=1, hl_size=4, optimizer='Nadam', validation_percent=0.1, left_censor=None,
                 monotone_indices=[]):
        self.quantiles = quantiles
        self.No = len(quantiles)

        if loss == 'pinball':
            self.loss = loss
        elif loss == 'smoothed':
            self.loss = loss
            self.epsilon = epsilon
        else:
            raise ValueError("Acceptable loss functions are 'pinball' or 'smoothed'")

        self.n_hidden = n_hidden
        self.Nh = hl_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.monotone_indices = monotone_indices
        self.build_status = 0
        self.train_status = 0
        self.left_censor = left_censor

        self.Ni = None
        self.val = validation_percent
        self.models = {}
        return

    def build_model(self, in_shape=None):
        if self.n_hidden <= 2:
            model = keras.models.Sequential()
            model.add(keras.layers.InputLayer(input_shape=(in_shape,)))
            model.add(QuantReg_Functions.MonotoneHidden(units=self.Nh, activation=self.hidden_activation,
                                                        kernel_initializer="uniform",
                                                        bias_initializer="zeros",
                                                        monotone=len(self.monotone_indices) + 1))
            model.add(keras.layers.Dense(self.Nh, activation=self.hidden_activation, kernel_initializer="uniform",
                                         bias_initializer="zeros"))
            model.add(keras.layers.Dense(1, kernel_initializer="uniform", bias_initializer="zeros",
                                         activation=self.output_activation))
            if self.left_censor is not None:
                model.add(keras.layers.ThresholdedReLU(theta=self.left_censor))
            # model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=None)
            self.build_status = 1
            self.base_model = model


        else:
            model = keras.models.Sequential()
            model.add(keras.layers.InputLayer(input_shape=(in_shape,)))
            model.add(QuantReg_Functions.MonotoneHidden(units=self.Nh, activation=self.hidden_activation,
                                                        kernel_initializer="uniform",
                                                        bias_initializer="zeros",
                                                        monotone=len(self.monotone_indices) + 1))
            for i in range(self.n_hidden - 1):
                model.add(
                    keras.layers.Dense(self.Nh, activation=self.hidden_activation, kernel_initializer="uniform",
                                       bias_initializer="zeros"))
            model.add(keras.layers.Dense(1, kernel_initializer="uniform", bias_initializer="zeros",
                                         activation=self.output_activation))
            if self.left_censor is not None:
                model.add(keras.layers.ThresholdedReLU(theta=self.left_censor))
            # model.compile(optimizer=self.optimizer, loss=self.loss, loss_weights=None)
            self.build_status = 1
            self.base_model = model

        return

    def fit(self, X, y):
        if self.build_status == 0:
            self.build_model(in_shape=len(X[0, :]) + 1)

        new_X = np.zeros(np.shape(X))
        non_monotone_indices = np.delete(np.arange(len(X[0, :])), self.monotone_indices)

        for j in range(len(self.monotone_indices)):
            new_X[:, j] = X[:, self.monotone_indices[j]]
        for j in range(len(non_monotone_indices)):
            new_X[:, len(self.monotone_indices) + j] = X[:, non_monotone_indices[j]]

        inp_quant = np.repeat(self.quantiles, len(y))
        inp_rep = np.tile(new_X.T, len(self.quantiles))
        X_use = np.vstack((inp_quant, inp_rep))
        y_use = np.repeat(y, len(self.quantiles))

        early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001,
                                                               patience=100,
                                                               restore_best_weights=True)
        # x_train,x_val,y_train,y_val=train_test_split(X_use,y_use,test_size=self.val,shuffle=True)

        tf.keras.backend.clear_session()
        model = keras.models.clone_model(self.base_model)

        if self.loss == 'smoothed':
            model.cost = QuantReg_Functions.mcqrnn_loss(model.input, self.epsilon)


        else:
            # model.cost=QuantReg_Functions.pinball_loss_keras(self.quantiles)

            model.cost = QuantReg_Functions.mcqrnn_loss(model.input, 0.0000000000000000000000000000001)

        model.compile(optimizer=self.optimizer, loss=model.cost)
        # model.fit(X_use.T, y_use, validation_split=self.val, callbacks=[early_stopping_monitor], epochs=500, verbose=False)
        model.fit(X_use.T, y_use, epochs=500, verbose=False)
        self.model = model
        self.train_status = 1
        return

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")

        new_X = np.zeros(np.shape(X))
        non_monotone_indices = np.delete(np.arange(len(X[0, :])), self.monotone_indices)

        for j in range(len(self.monotone_indices)):
            new_X[:, j] = X[:, self.monotone_indices[j]]
        for j in range(len(non_monotone_indices)):
            new_X[:, len(self.monotone_indices) + j] = X[:, non_monotone_indices[j]]

        preds = {}
        for q in self.quantiles:
            inp_quant = np.array([q for i in range(len(X[:, 0]))])
            X_use = np.vstack((inp_quant, new_X.T))
            preds.update({str(np.round(q, decimals=4)): self.model.predict(X_use.T)})

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
            res = quadprog.solve_qp(G=G, a=a, C=C.T, b=b0, meq=1)
            alpha = res[0]
            f = np.matmul(alpha, K)
            offshift = np.argmin(
                (np.round(alpha, 3) - (self.C * q)) ** 2 + (np.round(alpha, 3) - (self.C * (q - 1))) ** 2)

            model = {'alpha': alpha, 'b': y[offshift] - f[offshift]}

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


class lm:
    def __init__(self, quantiles, kernel=None, bw=None):
        self.quantiles = quantiles
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = 'epa'
        if bw is not None:
            self.bandwidth = bw
        else:
            self.bandwidth = 'hsheather'
        self.models = {}
        self.train_status = 0

    def fit(self, X, y):
        for q in self.quantiles:
            model = sm.QuantReg(y, X, q=self.quantiles).fit(q=q, kernel=self.kernel, bandwidth=self.bandwidth)
            self.models.update({str(q): model})
        self.train_status = 1

        return

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")

        preds = {}
        for q in self.quantiles:
            model = self.models[str(q)]
            preds.update({str(np.round(q, decimals=4)): model.predict(X)})
        return preds


class gbm():
    def __init__(self, quantiles, learning_rate=0.1, trees=100, min_samples_split=2, min_samples_leaf=1,
                 max_depth=None):
        self.quantiles = quantiles
        self.learning_rate = learning_rate
        self.n_estimators = trees
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.models = {}
        self.train_status = 0

    def fit(self, X, y):
        base_model = GradientBoostingRegressor(loss='quantile',
                                               learning_rate=self.learning_rate,
                                               n_estimators=self.n_estimators,
                                               min_samples_split=self.min_samples_split,
                                               min_samples_leaf=self.min_samples_leaf,
                                               max_depth=self.max_depth)
        for q in self.quantiles:
            model = GradientBoostingRegressor(loss='quantile',
                                              learning_rate=self.learning_rate,
                                              n_estimators=self.n_estimators,
                                              min_samples_split=self.min_samples_split,
                                              min_samples_leaf=self.min_samples_leaf,
                                              max_depth=self.max_depth,
                                              alpha=q)
            model.fit(X, y)
            self.models.update({str(q): model})
        self.train_status = 1
        return

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")

        preds = {}
        for q in self.quantiles:
            model = self.models[str(q)]
            preds.update({str(np.round(q, decimals=4)): model.predict(X)})
        return preds


class rf():
    def __init__(self, quantiles, trees=100, min_samples_split=2, min_samples_leaf=1, max_depth=None):

        self.quantiles = quantiles
        self.n_estimators = trees
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.train_status = 0

    def fit(self, X, y):
        model = RandomForestRegressor(n_estimators=self.n_estimators,
                                      min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf,
                                      max_depth=self.max_depth
                                      )

        model.fit(X, y)
        self.models = model
        self.train_status = 1

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")

        all_preds = []
        for estimator in self.models.estimators_:
            all_preds.append(estimator.predict(X))
        all_preds = np.array(all_preds)
        preds = {}
        for q in self.quantiles:
            preds.update({str(np.round(q, decimals=4)): np.percentile(all_preds, q * 100, axis=0)})
        return preds


class log_linear():
    def __init__(self):
        self.train_status = 0

    def fit(self, X, y):
        model = sm.ols(y, X).fit()
        self.models = model
        self.train_status = 1

        return

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")

        model = self.models
        return model.predict(X)


class proba_ann:
    def __init__(self, hidden_activation='tanh', output_activation='linear', loss='pinball', n_hidden=2, hl_size=4,
                 optimizer='Nadam', validation_percent=0.1, epsilon=10, left_censor=None):

        self.No = 1

        if loss == 'pinball':
            self.loss = loss
        elif loss == 'smoothed':
            self.loss = loss
            self.epsilon = epsilon
        else:
            raise ValueError("Acceptable loss functions are 'pinball' or 'smoothed'")
        self.n_hidden = n_hidden
        self.Nh = hl_size
        self.left_censor = None
        if left_censor is not None:
            self.left_censor = left_censor
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.build_status = 0
        self.train_status = 0
        self.left_censor = left_censor
        self.Ni = None
        self.val = validation_percent
        self.models = {}
        return

    def build_model(self):
        model = keras.models.Sequential()
        for i in range(self.n_hidden):
            model.add(keras.layers.Dense(self.Nh, activation=self.hidden_activation, kernel_initializer="uniform",
                                         bias_initializer="zeros"))
            # model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(self.No, kernel_initializer="uniform", bias_initializer="zeros",
                                     activation=self.output_activation))
        if self.left_censor is not None:
            model.add(keras.layers.ThresholdedReLU(theta=self.left_censor))
        model.compile(optimizer=self.optimizer, loss='mean_squared_error', loss_weights=None)
        self.build_status = 1
        self.base_model = model
        return

    def fit(self, X, y):
        if self.build_status == 0:
            self.build_model()

        early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00000001, patience=100,
                                                               restore_best_weights=True)

        model = keras.models.clone_model(self.base_model)
        model.compile(optimizer=self.optimizer, loss='mean_squared_error')
        model.fit(X, y, validation_split=self.val, epochs=500, callbacks=[early_stopping_monitor], verbose=False)
        self.models = model
        self.train_status = 1
        return

    def predict(self, X):
        if self.train_status == 0:
            raise ValueError("Model must be trained before predicting")
        preds = self.models.predict(X)
        return preds