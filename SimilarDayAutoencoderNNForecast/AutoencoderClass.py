import keras
from keras.layers import Input, Dense
from keras.models import Model
#from tensorflow import set_random_seed
import tensorflow as tf
import pandas as pd
from ReadCSVinDataFrame import ReadCSVSheets
from SimilarDayAutoencoderNNForecast.PreprocessDayData import preprocessDataFrame, prepareInputsForEncoderDecoder
from SimilarDayAutoencoderNNForecast.Constants import AUTOENCODER_INPUT_ATTRIBUTES, AUTOENCODER_CODE_DIMENSION
from SimilarDayAutoencoderNNForecast.PreprocessDayData import DayData
#from keras.callbacks import TensorBoard
import numpy as np
import os


def set_seed(s):
    np.random.seed(s)
    tf.keras.utils.set_random_seed(s)


class AutoEncoder:
    def __init__(self, main_df, input_atributes, encoding_dim=24):


        x_data, dimension = prepareInputsForEncoderDecoder(main_df, input_atributes)

        self.x = x_data[:-100]
        self.val_data = x_data[-100:]
        #Hidden layer has same dimension line input/output layers
        self.input_output_layer_dim = dimension
        self.hidden_layer_dim = dimension
        self.encoding_dim = encoding_dim

        #self.input_dict = dict


    def _encoder(self):
        inputs = Input(shape=(self.x[0].shape))
        hidden = Dense(self.hidden_layer_dim, activation='sigmoid')(inputs)
        encoded = Dense(self.encoding_dim, activation='sigmoid')(hidden)
        model = Model(inputs, encoded)
        self.encoder = model
        return model

    def _decoder(self):
        inputs = Input(shape=(self.encoding_dim,))
        hidden = Dense(self.hidden_layer_dim, activation='sigmoid')(inputs)
        decoded = Dense(self.input_output_layer_dim, activation='sigmoid')(hidden)
        model = Model(inputs, decoded)
        self.decoder = model
        return model

    def encoder_decoder(self):
        ec = self._encoder()
        dc = self._decoder()

        inputs = Input(shape=(self.x[0].shape))
        ec_out = ec(inputs)
        dc_out = dc(ec_out)
        model = Model(inputs, dc_out)

        self.model = model
        return model

    def fit(self, batch_size=10, epochs=300):
        opt = keras.optimizers.Adam(lr=0.00005)
        self.model.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', 'mape'])
        #log_dir = './log/'
        #tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
        history = self.model.fit(self.x, self.x,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_data=(self.val_data,self.val_data))
        return history

    def save(self):
        if not os.path.exists(r'./weights'):
            os.mkdir(r'./weights')
        else:
            self.encoder.save(r'./weights/encoder_weights.h5')
            self.decoder.save(r'./weights/decoder_weights.h5')
            self.model.save(r'./weights/ae_weights.h5')


if __name__ == '__main__':
    set_seed(2)
    input_df = pd.read_csv('InputData/NormalizedInputValues.csv')
    main_df = input_df[['DateTime', 'Load', 'Temp', 'Hour']]
    print(main_df.head(48))
    input_attributes = AUTOENCODER_INPUT_ATTRIBUTES
    for encoding_dimension in range(AUTOENCODER_CODE_DIMENSION, AUTOENCODER_CODE_DIMENSION+1):
        #encoding_dimension = 20
        ae = AutoEncoder(main_df, input_attributes, encoding_dim=encoding_dimension)
        ae.encoder_decoder()
        history = ae.fit(batch_size=5, epochs=2000)
        hist_df = pd.DataFrame(history.history)

        model_name = 'SavedModel/Autoencoder'
        model_name = model_name + '_dim_' + str(encoding_dimension)
        # if SELECT_DAYS:
        #     model_name = model_name + '_SelectedDays'
        for i in range(len(input_attributes)):
            model_name = model_name + '_' + input_attributes[i]
        history_log_name = model_name + 'history_log.csv'
        hist_df.to_csv(history_log_name)
        model_name = model_name + '.h5'
        ae.encoder.save(model_name)