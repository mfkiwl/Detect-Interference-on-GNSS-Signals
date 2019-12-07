import os
import pandas as pd
import numpy as np
import georinex as gr
import matplotlib.pyplot as plt
from functools import reduce
from keras.models import load_model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def parse_rinex(kind, name=None):
    """ Parse observation files. If 'name' is given, only
        that specific file is parse, otherwise parse all files
        in 'kind' folder.
    """
    path = kind + '/'
    if not name:
        X = {"E":[], "G":[]}
        for f in os.listdir(path):
            try:
                print('Parsing ' + f)
                gps = gr.load(path + f, use='G').to_dataframe()
                gal = gr.load(path + f, use='E').to_dataframe()
                if not gps.empty:
                    X["G"].append(gps)
                if not gal.empty:
                    X["E"].append(gal)
            except:
                print("Something went wrong!!!")
        return X
    else:
        X = {}
        try:
            print('Parsing ' + name)
            gps = gr.load(path + name, use='G').to_dataframe()
            gal = gr.load(path + name, use='E').to_dataframe()
            if not gps.empty:
                X["G"] = gps
            if not gal.empty:
                X["E"] = gal
        except:
            print("Something went wrong!!!")
        return X

def get_features(df):
    """ Get features from observation data.
        That is, compute the average signal strength off all
        satellites in a constellation.
    """
    satellite = df.index.get_level_values('sv').unique()
    tmp = []
    avg = []
    num_off = []
    for sat in satellite:
        tmp.append(df['S1C'].loc[sat].diff())
    _X = pd.concat(tmp, axis=1)
    for index, row in _X.iterrows():
        avg.append(row.mean(skipna=True))
        num_off.append(row.isna().sum())
    return pd.DataFrame(np.array([avg, num_off]).T,
            index=_X.index, columns=['Average Signal Strength', \
                                     'Number of Off Satellites'])

def get_feature_window(array, window):
    """ Split the features into slices of length 'window'.
    """
    outer = []
    inner = []
    for i in range(len(array)):
        inner.append(array[i])
        if (len(inner) == window):
            outer.append(inner)
            inner = []
    return np.array(outer)

def get_target_window(array, window):
    """ Split the targets into slices of length 'window'.
    """
    tmp = [array[i:i+window, :] for i in range(0, array.shape[0], window)]
    if len(array) % window == 0:
        return [int(i.sum() > (window / 2)) for i in tmp]
    else:
        return [int(i.sum() > (window / 2)) for i in tmp[:-1]]

def prepare_train_data(kind, dic, window):
    """ Compute new features from training data, label
        each time instance as either interfered or not, finally
        split both features and targets into chunks of equal length.
    """
    features = {'E': [], 'G': []}
    targets = {'E': [], 'G': []}

    for k, v in dic.items():
        features[k] = [get_features(i) for i in v]

    for k, v in features.items():
        for i in v:
            if kind == 'static':
                targets[k].append([label_static(j) for j in i.index])
            elif kind == 'kinematic':
                targets[k].append([label_kinematic(j) for j in i.index])
            else:
                targets[k].append([label_natural(j) for j in i.index])

    for k, v in targets.items():
        tmp = []
        for i in v:
            tmp.append(get_target_window(np.array(i).reshape((len(i), 1)), 5))
        targets[k] = tmp

    for k, v in features.items():
        features[k] = [get_feature_window(i.fillna(0).to_numpy(), 5) for i in v]
    return features, targets

def prepare_test_data(dic, window):
    """ Compute new features for test data and split into chunks.
    """
    features = {'E': [], 'G': []}
    for k, v in dic.items():
        feature_array = get_features(v).fillna(0).to_numpy()
        features[k] = get_feature_window(feature_array, window)
    return features

def stack_features(dic):
    """ Stack features from different files into one.
    """
    X_dic = {}
    for k, v in dic.items():
        X_dic[k] = reduce(lambda x, y: np.vstack((x, y)), v)
    return X_dic

def stack_target(dic):
    """ Stack target from different files into one.
    """
    y_dic = {}
    for k, v in dic.items():
        y_dic[k] = reduce(lambda x, y: x + y, v)
    return y_dic

def label_static(time):
    """ Label time instances as either 1 (interfered) or 0 (normal)
        according to reference data description for static interference.
    """
    if 9 <= time.minute < 10:
        return 1
    if 11 <= time.minute < 12:
        return 1
    if time.minute == 12:
        if 30 <= time.second <= 35 or 40 <= time.second <= 45 or 50 <= time.second <= 55:
            return 1
    if time.minute == 13:
        if 0 <= time.second <= 5:
            return 1
    if 14 <= time.minute < 15:
        return 1
    if time.minute == 15:
        if 30 <= time.second <= 35 or 40 <= time.second <= 45 or 50 <= time.second <= 55:
            return 1
    if time.minute == 16:
        if 0 <= time.second <= 5:
            return 1
    if 17 <= time.minute < 18:
        return 1
    if time.minute == 18:
        if 45 <= time.second <= 50 or 55 <= time.second:
            return 1
    if time.minute == 19:
        if 5 <= time.second <= 40:
            return 1
    if time.minute == 20:
        return 1
    if time.minute == 21:
        if time.second <= 40 or 45 <= time.second <= 50 or 55 <= time.second:
            return 1
    if time.minute == 23:
        if time.second <= 5 or 10<= time.second <= 15 or 20 <= time.second <= 25 or 30 <= time.second <= 35 or 40 <= time.second <=45:
            return 1
    return 0

def label_kinematic(time):
    """ Label time instances as either 1 (interfered) or 0 (normal)
        according to reference data description for kinematic interference.
    """
    if time.minute == 21:
        return 1
    if time.minute == 22:
        if 0 <= time.second < 5 or 10 <= time.second < 15 or 20 <= time.second < 25:
            return 1
    if time.minute == 23:
        if 10 <= time.second < 45:
            return 1
    if time.minute == 24:
        if 0 <= time.second < 5 or 10 <= time.second < 15 or 20 <= time.second < 25 or 30 <= time.second < 35 or 40 <= time.second < 45:
            return 1
    if time.minute == 25:
        if 15 <= time.second < 40:
            return 1
    if time.minute == 26:
        if 20 <= time.second <= 25 or 30 <= time.second <= 35 or 40 <= time.second <= 45 or 50 <= time.second < 55:
            return 1
    if time.minute == 29:
        if 0 <= time.second < 5 or 10 <= time.second < 15 or 20 <= time.second < 25 or 30 <= time.second < 35 or 40 <= time.second < 45:
            return 1
    if time.minute == 31:
        if 0 <= time.second < 5 or 10 <= time.second < 15 or 20 <= time.second < 25 or 30 <= time.second < 35 or 40 <= time.second < 45 or 50 <= time.second < 55:
            return 1
    return 0

def label_natural(time):
    """ Label time instances as either 1 (interfered) or 0 (normal)
        according to reference data description for natural interference.
    """
    return 0

def build_models(X, y, lr, epoch, batch, lstm, verbose=0):
    """ Build RNN models for each constellation (GPS and Galileo).
    """
    models = {}
    for k, v in X.items():
        if k == 'G':
            print('Training model for GPS ....')
        else:
            print('Training model for Galileo ....')
        model = Sequential()
        model.add(LSTM(lstm, input_shape=(5, 2), kernel_initializer='he_normal'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # try using different optimizers and different optimizer configs
        adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='binary_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])

        # fit the model:
        model.fit(v, y[k], epochs=epoch, batch_size=batch, verbose=verbose)
        models[k] = model
    return models

def save_models(models):
    """ Save trained models for later analysis.
    """
    for k, v in models.items():
        v.save('models/model_' + k + '.h5')

def load_models():
    """ Load trained models for analysis.
    """
    models = {}
    for f in os.listdir('models'):
        constell = f[6:-3]
        models[constell] = load_model('models/'+f)
    return models

def visualize_prediction(data, models, window, scale):
    """ Visualize signal data and interference prediction intervals.
    """
    titles = {'G': 'GPS', 'E': 'Galileo'}
    data_processed = prepare_test_data(data, window)
    visual = {}
    for constell, data_p in data_processed.items():
        predict = []
        for pre in models[constell].predict(data_p):
            for _ in range(window):
                predict.append(np.round(pre[0])*scale)
        vis = get_features(data[constell]).iloc[: len(predict), :]
        vis['Intentional Interference Detection'] = predict
        visual[constell] = vis
    visual['G'].plot(figsize=(12, 5), title='GPS')
    visual['E'].plot(figsize=(12, 5), title='Galileo')
