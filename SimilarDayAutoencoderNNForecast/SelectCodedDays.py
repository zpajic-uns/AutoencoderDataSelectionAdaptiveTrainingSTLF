import numpy as np
from scipy.spatial import distance
from SimilarDayAutoencoderNNForecast.Constants import DAY_TO_PREDICT
from matplotlib import pyplot
from SimilarDayAutoencoderNNForecast.PreprocessDayData import DayData
from keras import activations
if __name__ == '__main__':
    coded_dict = np.load('InputData/CodedDays_dict.npy', allow_pickle='TRUE').item()

    predict_day_code = coded_dict[DAY_TO_PREDICT]
    distances_dict = {}
    #Euclidian norm of codes
    for key, data in coded_dict.items():
        vector_difference = data-predict_day_code
        dst = distance.euclidean(data, predict_day_code)
        distances_dict[key] = dst
    distances_dict.pop(DAY_TO_PREDICT)
    sorted_by_value = sorted(distances_dict.items(), key=lambda kv: kv[1])
    #sorted_list_of_codes = sorted_by_value.
    np.save('InputData/SortedByValueCodes_list.npy', sorted_by_value)
    list_values = []
    for tup in sorted_by_value:
        list_values.append(tup[1])
    pyplot.plot(list_values, color='olive', label="Distances")
    #pyplot.plot(1./activations.relu(list_values), color='red', label="Distances scaled")
    pyplot.legend(loc="upper left")
    pyplot.xlabel("Sorted days")
    pyplot.ylabel("Distance to forecast day")
    pyplot.show()

    input_dict = np.load('InputData/Input_dict.npy', allow_pickle='TRUE').item()
    day_to_predict_data: DayData = input_dict.pop(DAY_TO_PREDICT)
    real_temp = np.reshape(day_to_predict_data.temp, (24,))
    best_temp = np.reshape(input_dict[sorted_by_value[0][0]].temp, (24,))
    pyplot.plot(real_temp, color='blue')
    pyplot.plot(best_temp, color='olive')
    pyplot.show()

    real_load = np.reshape(day_to_predict_data.load, (24,))
    best_load = np.reshape(input_dict[sorted_by_value[0][0]].load, (24,))

    mape = np.mean(np.abs((real_load - best_load) / real_load) * 100)
    print('Best day load MAPE % = ' + str(mape))

    pyplot.plot(real_load, color='blue')
    pyplot.plot(best_load, color='olive')
    pyplot.show()