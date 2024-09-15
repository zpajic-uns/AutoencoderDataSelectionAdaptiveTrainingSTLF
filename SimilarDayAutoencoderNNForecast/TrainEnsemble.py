import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
# from tensorflow import set_random_seed
from SimilarDayAutoencoderNNForecast.Constants import DAY_TO_PREDICT, NN_INPUT_DIM, SELECT_SORTED_DAYS, \
    NMB_SELECTED_DAYS, NMB_EPOCHS, LOAD_SAVED_MODEL, USE_DISTANCES, USE_WEIGHTS, MIN_LR, IS_FINE_TUNING
from SimilarDayAutoencoderNNForecast.PreprocessDayData import DayData
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model, load_model
from keras import activations
from pandas import DataFrame
from StringToDatetime import dayInWeek, string2date

def set_seed(s):
    np.random.seed(s)
    tf.random.set_seed(s)
    tf.keras.utils.set_random_seed(s)


if __name__ == '__main__':
    set_seed(2)

    input_dict = np.load('InputData/Input_dict.npy', allow_pickle='TRUE').item()
    day_to_predict_data: DayData = input_dict.pop(DAY_TO_PREDICT)
    # The first day without previous data should be removed from the dictionary too.
    key_to_remove = None
    for key, data in input_dict.items():
        if (data.minusOneDay == None):
            key_to_remove = key
        break
    input_dict.pop(key_to_remove)
    # Define distances for weighted training
    sorted_days_list_all = np.load('InputData/SortedByValueCodes_list.npy', allow_pickle='TRUE')
    sorted_days_list = []
    week_day_to_predict = dayInWeek(string2date(DAY_TO_PREDICT))
    for k in range(0, len(sorted_days_list_all)):
        date_string = sorted_days_list_all[k][0]
        weekDay = dayInWeek(string2date(date_string))
        if weekDay == week_day_to_predict:
            sorted_days_list.append(sorted_days_list_all[k])
        else:
            del input_dict[date_string]

    sorted_days_list_values = []
    for tup in sorted_days_list:
        sorted_days_list_values.append(float(tup[1]))

    distances = np.array(1. / activations.relu(sorted_days_list_values))

    if (SELECT_SORTED_DAYS):
        for i in range(NMB_SELECTED_DAYS, len(sorted_days_list)):
            tup = sorted_days_list[i]
            input_dict.pop(tup[0])
        distances = distances[:NMB_SELECTED_DAYS]
    distances_list = distances.tolist()

    #distances = distances ** 2
    x_data_list = []
    y_data_list = []
    for key, data in input_dict.items():
        # In more complicated input data, day attributes should be concatenated to one array and then put to x_data_list
        #x_entry = np.concatenate((data.temp, data.weekDay))
        #x_entry = np.concatenate((x_entry, data.daylight))
        #x_entry = np.concatenate((x_entry, data.year))
        x_data_list.append(data.temp)
        print(data.temp)
        print(data.load)
        y_data_list.append(data.load)

    x_data_ensemble_list = []
    y_data_ensemble_list = []
    x_val_data_ensemble_list = []
    y_val_data_ensemble_list = []
    distances_ensemble = []
    for ensemble_index in range(0, 4):
        temp_x = []
        temp_y = []
        temp_x_val = []
        temp_y_val = []
        temp_distances = []
        for i in range(len(x_data_list)):
            if i % 4 == ensemble_index:
                temp_x_val.append(x_data_list[i])
                temp_y_val.append(y_data_list[i])
            else:
                temp_x.append(x_data_list[i])
                temp_y.append(y_data_list[i])
                temp_distances.append(distances_list[i])
        #temp_x.extend(temp_x * 1)
        #temp_y.extend(temp_y * 1)
        #temp_distances.extend(temp_distances * 1)
        x_data_ensemble_list.append(temp_x)
        y_data_ensemble_list.append(temp_y)
        distances_ensemble.append(temp_distances)
        x_val_data_ensemble_list.append(temp_x_val)
        y_val_data_ensemble_list.append(temp_y_val)

    for ensemble_index in range(0, 4):
        x_data_train = np.array(x_data_ensemble_list[ensemble_index])
        y_data_train = np.array(y_data_ensemble_list[ensemble_index])
        distances_data_train = np.array(distances_ensemble[ensemble_index])
        x_data_val = np.array(x_val_data_ensemble_list[ensemble_index])
        y_data_val = np.array(y_val_data_ensemble_list[ensemble_index])

        model = None
        model_name = 'SavedModel/model-00004743-12.667326-10.262156.h5'
        model = load_model(model_name)
        assert isinstance(model, Model)

        optimizer = keras.optimizers.Adam(lr=0.001)
        if USE_WEIGHTS:
            model.compile(optimizer=optimizer,
                          loss='mse',
                          loss_weights=distances_data_train,
                          weighted_metrics=['mse', 'mae', 'mean_absolute_percentage_error'])
        else:
            model.compile(optimizer=optimizer,
                          loss='mse',
                          metrics=['mse', 'mae', 'mean_absolute_percentage_error'])

        monitor_string = 'val_mse'
        checkpoint = ModelCheckpoint(
            'SavedModel/' + str(ensemble_index) + '_model.h5',
            verbose=1, monitor=monitor_string, save_best_only=True,
            mode='auto')
        num_epochs = 0
        if SELECT_SORTED_DAYS:
            num_epochs = NMB_EPOCHS
        else:
            num_epochs = NMB_EPOCHS
        earlyStopping = EarlyStopping(monitor='loss', patience=1000)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, min_delta=0.0,
                                      patience=500, min_lr=MIN_LR, verbose=1)
        print('Ensemble training started...')

        if USE_DISTANCES:
            history = model.fit(x_data_train, y_data_train, sample_weight=distances_data_train, batch_size=0,
                                epochs=num_epochs, verbose=1, validation_data=(x_data_val, y_data_val),
                                callbacks=[checkpoint, earlyStopping, reduce_lr])
        else:
            history = model.fit(x_data_train, y_data_train, batch_size=0,
                                epochs=num_epochs, verbose=1, validation_data=(x_data_val, y_data_val),
                                callbacks=[checkpoint, earlyStopping, reduce_lr])

    print('Ensemble training finished.')

