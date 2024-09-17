import keras
import numpy as np
import tensorflow as tf
from SimilarDayAutoencoderNNForecast.Constants import DAY_TO_PREDICT, NN_INPUT_DIM, MAX_LOAD, MIN_LOAD, \
    PRETRAINED_MODEL_NAME
from SimilarDayAutoencoderNNForecast.PreprocessDayData import DayData
from matplotlib import pyplot
from keras.models import Model, load_model

if __name__ == '__main__':
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    print(tf.config.list_physical_devices('GPU'))

    # model_name = 'SavedModel/!all-days-model-00001352-4.890618-6.759103.h5'
    #
    # model_all = load_model(model_name)
    # assert isinstance(model_all, Model)
    #
    # model_name = 'SavedModel/!200-selected-days-linear-weights-weighted-training-model-00004272-0.025805-3.571944.h5'
    # model_selected = load_model(model_name)
    # assert isinstance(model_selected, Model)

    model_name = 'SavedModel/' + PRETRAINED_MODEL_NAME
    model_selected_linear_weights = load_model(model_name)
    assert isinstance(model_selected_linear_weights, Model)

    input_dict = np.load('InputData/Input_dict.npy', allow_pickle='TRUE').item()
    day_to_predict_data: DayData = input_dict.pop(DAY_TO_PREDICT)
    predict_input = []
    predict_input.append(day_to_predict_data.temp)
    #predict_input = np.concatenate((day_to_predict_data.temp, day_to_predict_data.weekDay))
    #predict_input = np.concatenate((predict_input, day_to_predict_data.daylight))
    #predict_input = np.concatenate((predict_input, day_to_predict_data.year))
    predict_input = np.reshape(predict_input, (1, NN_INPUT_DIM))
    real_load = np.reshape(day_to_predict_data.load, (1, 24))

    # predicted_load_all = model_all.predict(predict_input)
    # predicted_load_all = np.reshape(predicted_load_all, (24,))*(MAX_LOAD-MIN_LOAD) + MIN_LOAD
    #
    # predicted_load_selected = model_selected.predict(predict_input)
    # predicted_load_selected = np.reshape(predicted_load_selected, (24,)) * (MAX_LOAD - MIN_LOAD) + MIN_LOAD

    predicted_load_selected_linear_weights = model_selected_linear_weights.predict(predict_input)
    predicted_load_selected_linear_weights = np.reshape(predicted_load_selected_linear_weights, (24,)) * (MAX_LOAD - MIN_LOAD) + MIN_LOAD

    real_load = np.reshape(real_load, (24,))*(MAX_LOAD-MIN_LOAD) + MIN_LOAD

    # mape_all = np.mean(np.abs((real_load - predicted_load_all) / real_load)*100)
    # print('All days MAPE % = ' + str(mape_all))
    #
    # mape_selected = np.mean(np.abs((real_load - predicted_load_selected) / real_load) * 100)
    # print('Selected days MAPE % = ' + str(mape_selected))

    mape_selected_linear_weights = np.mean(np.abs((real_load - predicted_load_selected_linear_weights) / real_load) * 100)
    print('Selected days (linear weights) MAPE % = ' + str(mape_selected_linear_weights))

    pyplot.plot(real_load, color='black', label='Actual load')
    # pyplot.plot(predicted_load_all, linestyle=(0, (1, 1)) , color='red', label='Predicted laod - ANN for all days')
    # pyplot.plot(predicted_load_selected, '-.', color='green', label='Predicted laod - ANN for selected days')
    pyplot.plot(predicted_load_selected_linear_weights, '-.', color='blue', label='Predicted laod - ANN for selected days (linear weights)')
    pyplot.legend(loc="lower right")
    pyplot.xlabel("Hours")
    pyplot.ylabel("Load, [MW]")
    pyplot.show()