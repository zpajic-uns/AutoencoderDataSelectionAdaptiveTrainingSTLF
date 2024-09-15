import numpy as np
import pandas as pd
from SimilarDayAutoencoderNNForecast.PreprocessDayData import DayData
from keras.models import Model, load_model
from SimilarDayAutoencoderNNForecast.Constants import AUTOENCODER_INPUT_ATTRIBUTES, AUTOENCODER_CODE_DIMENSION

if __name__ == '__main__':
    input_dict = np.load('InputData/Input_dict.npy', allow_pickle='TRUE').item()

    model_name = 'SavedModel\Autoencoder_dim_40_MinusOneDay_Load_Temp.h5'
    # if SELECT_DAYS:
    #     model_name = model_name + '_SelectedDays'
    # for i in range(len(input_atributes)):
    #     model_name = model_name + '_' + input_atributes[i]
    #model_name = model_name + '.h5'

    model = load_model(model_name)
    assert isinstance(model, Model)
    attributes = AUTOENCODER_INPUT_ATTRIBUTES
    coded_days_dict = {}
    for key, data in input_dict.items():
        #print(data.temp.shape)
        if (data.minusOneDay == None):
            continue
        x_entry = None
        dimension = 0
        for i in range(0, len(attributes)):
            if i == 0:
                if attributes[i] == 'MinusOneDay_Load':
                    x_entry = np.array(data.minusOneDay.load)
                    dimension = 24
                elif (attributes[i] == 'Temp'):
                    x_entry = np.array(data.temp)
                    dimension = 24
                else:
                    print('Unknown case!')
            else:
                if attributes[i] == 'MinusOneDay_Load':
                    x_entry = np.concatenate((x_entry, data.minusOneDay.load))
                    dimension = dimension + 24
                elif (attributes[i] == 'Temp'):
                    x_entry = np.concatenate((x_entry, data.temp))
                    dimension = dimension + 24
                else:
                    print('Unknown case!')

        data_input = np.reshape(x_entry, (1, dimension)) # this sould be changed when coding is changed
        #print (data_input)
        code = model.predict(data_input)
        code = np.reshape(code, (AUTOENCODER_CODE_DIMENSION,))
        coded_days_dict[key] = code
    np.save('InputData/CodedDays_dict.npy', coded_days_dict)
    print('test')