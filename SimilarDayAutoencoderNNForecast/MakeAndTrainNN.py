import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
# from tensorflow import set_random_seed
from SimilarDayAutoencoderNNForecast.Constants import DAY_TO_PREDICT, NN_INPUT_DIM, SELECT_SORTED_DAYS, \
    NMB_SELECTED_DAYS, NMB_EPOCHS, LOAD_SAVED_MODEL, USE_DISTANCES, USE_WEIGHTS, MIN_LR, IS_FINE_TUNING, \
    IS_ENSEMBLE_PRETRAINING, PRETRAINED_MODEL_NAME
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
    sorted_days_list_temp = []
    week_day_to_predict = dayInWeek(string2date(DAY_TO_PREDICT))
    for k in range(0, len(sorted_days_list_all)):
        date_string = sorted_days_list_all[k][0]
        weekDay = dayInWeek(string2date(date_string))
        if weekDay == week_day_to_predict:
            sorted_days_list_temp.append(sorted_days_list_all[k])
        else:
            del input_dict[date_string]
    # delete first n days which will be used in ensemble training
    val_dict = {}
    if IS_ENSEMBLE_PRETRAINING:
        for k in range(0, len(sorted_days_list_temp)):
            date_string = sorted_days_list_temp[k][0]
            if k >= 20:
                sorted_days_list.append(sorted_days_list_temp[k])
            else:
                val_dict[date_string] = input_dict[date_string]
                del input_dict[date_string]
    else:
        sorted_days_list = sorted_days_list_temp

    sorted_days_list_values = []
    for tup in sorted_days_list:
        sorted_days_list_values.append(float(tup[1]))

    pyplot.plot(sorted_days_list_values, color='olive', label="Distances")
    # pyplot.plot(1./activations.relu(list_values), color='red', label="Distances scaled")
    pyplot.legend(loc="upper left")
    pyplot.xlabel("Sorted days")
    pyplot.ylabel("Distance to forecast day")
    pyplot.show()

    distances = np.array(1. / activations.relu(sorted_days_list_values))

    if (SELECT_SORTED_DAYS):
        for i in range(NMB_SELECTED_DAYS, len(sorted_days_list)):
            tup = sorted_days_list[i]
            input_dict.pop(tup[0])
        distances = distances[:NMB_SELECTED_DAYS]

    df_selected_days = DataFrame.from_dict(sorted_days_list)
    df_selected_days = df_selected_days.head(NMB_SELECTED_DAYS)
    df_selected_days.columns = ['Day', 'Distance']
    csv_name = 'SavedModel/' + str(NMB_SELECTED_DAYS) + '_SelectedDays_' + DAY_TO_PREDICT + '.csv'
    df_selected_days.to_csv(csv_name)

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

    x_data_list.extend(x_data_list * 100)
    y_data_list.extend(y_data_list * 100)
    if (USE_DISTANCES):
        distances_list = distances.tolist()
        distances_list.extend(distances_list * 100)
        distances = np.array(distances_list)

    x_data = np.array(x_data_list)
    y_data = np.array(y_data_list)
    predict_input = []
    #predict_input = np.concatenate((day_to_predict_data.temp, day_to_predict_data.weekDay))
    #predict_input = np.concatenate((predict_input, day_to_predict_data.daylight))
    #predict_input = np.concatenate((predict_input, day_to_predict_data.year))
    predict_input.append(day_to_predict_data.temp)
    predict_input = np.reshape(predict_input, (1, NN_INPUT_DIM))
    real_load = np.reshape(day_to_predict_data.load, (1, 24))

    x_val_data_list = []
    y_val_data_list = []
    if IS_ENSEMBLE_PRETRAINING:
        for key, data in val_dict.items():
            # In more complicated input data, day attributes should be concatenated to one array and then put to x_data_list
            # x_entry = np.concatenate((data.temp, data.weekDay))
            # x_entry = np.concatenate((x_entry, data.daylight))
            # x_entry = np.concatenate((x_entry, data.year))
            x_val_data_list.append(data.temp)
            print(data.temp)
            print(data.load)
            y_val_data_list.append(data.load)

    x_data_train = x_data
    y_data_train = y_data
    distances_data_train = distances
    if IS_ENSEMBLE_PRETRAINING:
        x_data_val = np.array(x_val_data_list)
        y_data_val = np.array(y_val_data_list)
    else:
        x_data_val = predict_input
        y_data_val = real_load

    model = None
    if (LOAD_SAVED_MODEL is False):
        model = Sequential()
        model.add(Dense(NN_INPUT_DIM, activation='sigmoid', input_shape=(NN_INPUT_DIM,)))
        #model.add(Dropout(rate=0.05))
        model.add(Dense(NN_INPUT_DIM * 2, activation='sigmoid'))
        #model.add(Dropout(rate=0.05))
        model.add(Dense(NN_INPUT_DIM * 2, activation='sigmoid'))
        # model.add(Dropout(rate=0.05))
        model.add(Dense(NN_INPUT_DIM * 2, activation='sigmoid'))
        # model.add(Dropout(rate=0.05))
        model.add(Dense(24, activation='relu'))
    else:
        model_name = 'SavedModel/' + PRETRAINED_MODEL_NAME
        model = load_model(model_name)
        assert isinstance(model, Model)

    model.summary()
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
    monitor_string = 'mean_absolute_percentage_error'
    if IS_FINE_TUNING:
        monitor_string = 'val_mean_absolute_percentage_error'
    elif IS_ENSEMBLE_PRETRAINING:
        monitor_string = 'val_mse'
    checkpoint = ModelCheckpoint(
        'SavedModel/model-{epoch:08d}-{mean_absolute_percentage_error:03f}-{val_mean_absolute_percentage_error:03f}.h5',
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
    print('Training started...')
    if USE_DISTANCES:
        history = model.fit(x_data_train, y_data_train, sample_weight=distances_data_train, batch_size=16,
                            epochs=num_epochs, verbose=1, validation_data=(x_data_val, y_data_val),
                            callbacks=[checkpoint, earlyStopping, reduce_lr])
    else:
        history = model.fit(x_data_train, y_data_train, batch_size=16,
                            epochs=num_epochs, verbose=1, validation_data=(x_data_val, y_data_val),
                            callbacks=[checkpoint, earlyStopping, reduce_lr])
    print('Training finished.')

    predicted_load = model.predict(predict_input)
    predicted_load = np.reshape(predicted_load, (24,))
    real_load = np.reshape(real_load, (24,))
    pyplot.plot(predicted_load, color='blue')
    pyplot.plot(real_load, color='olive')
    pyplot.show()
    print('test')
