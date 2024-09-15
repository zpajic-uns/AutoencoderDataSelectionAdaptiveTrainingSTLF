import keras
import numpy as np
import tensorflow as tf
from SimilarDayAutoencoderNNForecast.Constants import DAY_TO_PREDICT, NN_INPUT_DIM, MAX_LOAD, MIN_LOAD
from SimilarDayAutoencoderNNForecast.PreprocessDayData import DayData
from matplotlib import pyplot
from keras.models import Model, load_model

if __name__ == '__main__':
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    print(tf.config.list_physical_devices('GPU'))

    model_name = 'SavedModel/0_model.h5'
    model_0 = load_model(model_name)
    assert isinstance(model_0, Model)
    #
    model_name = 'SavedModel/1_model.h5'
    model_1 = load_model(model_name)
    assert isinstance(model_1, Model)

    model_name = 'SavedModel/2_model.h5'
    model_2 = load_model(model_name)
    assert isinstance(model_2, Model)

    model_name = 'SavedModel/3_model.h5'
    model_3 = load_model(model_name)
    assert isinstance(model_3, Model)

    input_dict = np.load('InputData/Input_dict.npy', allow_pickle='TRUE').item()
    day_to_predict_data: DayData = input_dict.pop(DAY_TO_PREDICT)
    predict_input = []
    predict_input.append(day_to_predict_data.temp)
    #predict_input = np.concatenate((day_to_predict_data.temp, day_to_predict_data.weekDay))
    #predict_input = np.concatenate((predict_input, day_to_predict_data.daylight))
    #predict_input = np.concatenate((predict_input, day_to_predict_data.year))
    predict_input = np.reshape(predict_input, (1, NN_INPUT_DIM))
    real_load = np.reshape(day_to_predict_data.load, (1, 24))

    predicted_load_0 = model_0.predict(predict_input)
    predicted_load_0 = np.reshape(predicted_load_0, (24,))*(MAX_LOAD-MIN_LOAD) + MIN_LOAD
    #
    predicted_load_1 = model_1.predict(predict_input)
    predicted_load_1 = np.reshape(predicted_load_1, (24,)) * (MAX_LOAD - MIN_LOAD) + MIN_LOAD

    predicted_load_2 = model_2.predict(predict_input)
    predicted_load_2 = np.reshape(predicted_load_2, (24,)) * (MAX_LOAD - MIN_LOAD) + MIN_LOAD
    #
    predicted_load_3 = model_3.predict(predict_input)
    predicted_load_3 = np.reshape(predicted_load_3, (24,)) * (MAX_LOAD - MIN_LOAD) + MIN_LOAD

    predicted_load_ensemble = (predicted_load_0 + predicted_load_1 + predicted_load_2 + predicted_load_3) / 4

    real_load = np.reshape(real_load, (24,))*(MAX_LOAD-MIN_LOAD) + MIN_LOAD

    mape_0 = np.mean(np.abs((real_load - predicted_load_0) / real_load)*100)
    print('Model 0 MAPE % = ' + str(mape_0))
    mape_1 = np.mean(np.abs((real_load - predicted_load_1) / real_load) * 100)
    print('Model 1 MAPE % = ' + str(mape_1))
    mape_2 = np.mean(np.abs((real_load - predicted_load_2) / real_load) * 100)
    print('Model 2 MAPE % = ' + str(mape_2))
    mape_3 = np.mean(np.abs((real_load - predicted_load_3) / real_load) * 100)
    print('Model 3 MAPE % = ' + str(mape_3))

    mape_ensemble = np.mean(np.abs((real_load - predicted_load_ensemble) / real_load) * 100)
    print('Model Ensemble MAPE % = ' + str(mape_ensemble))

    pyplot.plot(real_load, color='black', label='Actual load')
    pyplot.plot(predicted_load_0, '-.', color='red', label='Model 0')
    pyplot.plot(predicted_load_1, '-.', color='orange', label='Model 1')
    pyplot.plot(predicted_load_2, '-.', color='green', label='Model 2')
    pyplot.plot(predicted_load_3, '-.', color='blue', label='Model 3')
    pyplot.legend(loc="lower right")
    pyplot.xlabel("Hours")
    pyplot.ylabel("Load, [MW]")
    pyplot.xlim(6, 23)
    pyplot.ylim(3000, 4600)
    pyplot.savefig('SavedModel/Figure.png', dpi=1200)
    pyplot.show()

    pyplot.plot(real_load, color='black', label='Actual load')
    pyplot.plot(predicted_load_ensemble, '-.', color='blue', label='Model Ensemble')
    pyplot.legend(loc="lower right")
    pyplot.xlabel("Hours")
    pyplot.ylabel("Load, [MW]")
    pyplot.show()