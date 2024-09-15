from DayLightCalculator import daylight_difference, LATITUDE_BELGRADE
from StringToDatetime import string2datetime, dayInYear, dayInWeek, string2date
import numpy as np
import pandas as pd
from SimilarDayAutoencoderNNForecast.Constants import MAX_TEMP,MIN_TEMP, MAX_LOAD, MAX_WIND, MAX_DAYLIGHT_DIFF, DAY_TO_PREDICT



def prepareInputs(dataframe_in, atributes):
    # 1. segment: hours - length: 24
    # 2. segment: day in week, if present - length: 7
    # 3. segment: month in year, if present - length: 12
    # 4. segment: temperature, if present - length: 1
    # 5. segment: daylight, if present - length: 1
    # 6. segment: wind, if present - length: 1




    dim_hours = 24
    dim_temp = 0
    dim_wind = 0
    dim_daylight = 0
    dim_weekdays = 0
    dim_months = 0
    if 'Temp' in atributes:
        dim_temp = 1
    if 'Wind' in atributes:
        dim_wind = 1
    if 'DayLight' in atributes:
        dim_daylight = 1
    if 'DayInWeek' in atributes:
        dim_weekdays = 7
    if 'MonthInYear' in atributes:
        dim_months = 12
    dimension = int(dim_hours + dim_weekdays + dim_months + dim_temp + dim_daylight + dim_wind)
    data = np.zeros(shape=(len(dataframe_in),dimension))
    #data = np.zeros(shape=(len(dataframe), 1 + 1 + 1))
    res = np.zeros(shape=(len(dataframe_in)))
    i = 0
    for index, row in dataframe_in.iterrows():
        #print('Row: ', row)
        data[i, int(row['Hour'])] = 1
        if 'DayInWeek' in atributes:
            data[i, dim_hours + int(row['DayInWeek'])] = 1
        if 'MonthInYear' in atributes:
            data[i, dim_hours + dim_weekdays + int(row['MonthInYear']) - 1] = 1 # index must be int(row['MonthInYear']) - 1 because months are numbered as 1-12
        if 'Temp' in atributes:
            data[i, dim_hours + dim_weekdays + dim_months] = row['Temp']
        if 'DayLight' in atributes:
            data[i, dim_hours + dim_weekdays + dim_months + dim_temp] = row['DayLight']
        if 'Wind' in atributes:
            data[i, dim_hours + dim_weekdays + dim_months + dim_temp + dim_daylight] = row['Wind']
        if 'Load' in atributes:
            res[i] = row['Load']
        #print(data[i,], 'res:', res[i])
        i = i + 1
    #print_to_csv_df = pd.DataFrame(data)
    #print_to_csv_df.to_csv('TestData.csv')
    return data, res, dimension



def preprocessDataFrame(main_df, atributes, shiftTemp = 0):
    #null_df = main_df.isnull()
    #print('Is null any? Answer:', main_df.isnull().any())
    #null_df.to_csv('IspisZaNULL.csv')
    if 'Temp' in atributes:
        main_df['Temp'] = (main_df['Temp'] + shiftTemp - MIN_TEMP)/(MAX_TEMP-MIN_TEMP)
    if 'Load' in atributes:
        main_df['Load'] = (main_df['Load'] - 2000)/(MAX_LOAD-2000)
    #maxWind = main_df['Wind'].max()
    if 'Wind' in atributes:
        main_df['Wind'] = main_df['Wind'] / MAX_WIND
    main_df.reset_index(inplace=True)
    #print(main_df.head())
    # Transform string to relevant data numbers
    datetime_strings = main_df['DateTime']
    #print(datetime_strings[0])
    #datetime_objects = np.empty(shape=len(datetime_strings))
    day_in_year_array = np.empty(shape=len(datetime_strings))
    month_in_year_array = np.empty(shape=len(datetime_strings))
    day_in_week_array = np.empty(shape=len(datetime_strings))
    hour_in_day_array = np.empty(shape=len(datetime_strings))
    day_differences = np.empty(shape=len(datetime_strings))
    for i in range(len(datetime_strings)):
        #print(i)
        string = datetime_strings[i]
        dt_obj = string2datetime(string)
        #datetime_objects[i] = dt_obj
        day_in_year_array[i] = dayInYear(dt_obj)
        month_in_year_array[i] = dt_obj.month
        day_in_week_array[i] = dayInWeek(dt_obj)
        hour_in_day_array[i] = dt_obj.hour
        day_differences[i] = daylight_difference(LATITUDE_BELGRADE,day_in_year_array[i])
    # Hour is used always
    main_df['Hour'] = hour_in_day_array
    if 'DayLight' in atributes:
        main_df['DayLight'] = day_differences
        #maxDayLightDiff = main_df['DayLight'].max()
        main_df['DayLight'] = main_df['DayLight']/MAX_DAYLIGHT_DIFF
    if 'DayInYear' in atributes:
        main_df['DayInYear'] = day_in_year_array
    if 'MonthInYear' in atributes:
        main_df['MonthInYear'] = month_in_year_array
    if 'DayInWeek' in atributes:
        main_df['DayInWeek'] = day_in_week_array
    #print('MAX_DIFF: ', main_df['DayLight'].max())
    #print(main_df.head())
    #print(MAX_LOAD)
    return main_df

class DayData:
    load = []
    temp = []
    weekDay = None
    daylight = None
    year = None
    minusOneDay = None

def prepareInputsForEncoderDecoder(dataframe_in, attributes):
    input_dict = np.load('InputData/Input_dict.npy', allow_pickle='TRUE').item()

    input_dict.pop(DAY_TO_PREDICT)
    x_data_list = []
    dimension = 0
    for key, data in input_dict.items():
        # In more complicated input data, day attributes should be concatenated to one array and then put to x_data_list
        if(data.minusOneDay == None):
            continue
        x_entry = None
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

        x_data_list.append(x_entry)
    x_data = np.array(x_data_list)
    return x_data, dimension

def createDayDataDict(main_df):
    loadData_df: pd.DataFrame = main_df['Load']
    load_data = loadData_df.values.reshape(-1, 24)
    tempData_df: pd.DataFrame = main_df['Temp']
    temp_data = tempData_df.values.reshape(-1, 24)

    format = '%Y-%m-%d'
    dictionary = {}
    i = 0
    prev_day = None
    for index, row in main_df.iterrows():
        datetime = string2datetime(row['DateTime'])
        key = datetime.strftime(format)
        if key in dictionary:
            continue
        else:
            data = DayData()
            data.load = load_data[i]
            data.temp = temp_data[i]
            data.weekDay = np.zeros((7,))
            data.weekDay[dayInWeek(datetime)] = 1
            data.daylight = np.zeros((1,))
            data.daylight[0] = daylight_difference(LATITUDE_BELGRADE,dayInYear(datetime))/MAX_DAYLIGHT_DIFF
            year = (datetime.year - 2012)/(2020-2012)
            data.year = np.zeros((1,))
            data.year[0] = year
            data.minusOneDay = prev_day # just point to the previous day
            prev_day = data
            i = i + 1
            dictionary[key] = data

    for k in list(dictionary.keys()):
        current = string2date(k)
        day_to_predict = string2date(DAY_TO_PREDICT);
        if current > day_to_predict:
            del dictionary[k]
    return dictionary

if __name__ == '__main__':
    #Load and Temp input files are filtered for missing and duplicate data
    input_files = ['Load', 'Temp']
    #input_atributes = ['Load', 'Temp']
    #input_atributes.append('Wind')
    additional_atributes = list(input_files)
    #additional_atributes.append('DayLight')
    #atributes.append('DayInYear')
    # atributes.append('MonthInYear')
    #additional_atributes.append('DayInWeek')
    # Read input files into dataframe
    main_df = pd.DataFrame()
    file_prefix = 'InputFiles/EMS_DataHistory_'
    from ReadCSVinDataFrame import ReadCSVSheets
    main_df = ReadCSVSheets(file_prefix, input_files)
    print(main_df.tail(48))
    # Check if there is missing data
    # print('Is null any? Answer:', main_df.isnull().any())
    # #In case there is missing data, next few lines help to detect and replase missing data.
    # null_df = main_df.isnull()
    # null_df.to_csv('IspisZaNULL.csv')
    # filter_null_df = null_df.loc[null_df['Temp']==True]
    # print(filter_null_df.tail(48))
    # main_df.dropna(inplace=True)
    # for row in main_df.iterrows():
    #     if(row[1].Load < 100.0):
    #         print(row[1])
    #         #break
    # Normalize input data and add needed data for NN training
    main_df = preprocessDataFrame(main_df, additional_atributes)
    #In case there is duplicate data, indicates what row is wrong
    dayHourIterator = 0.0
    for row in main_df.iterrows():
        #print(row)
        if (row[1].Hour!= dayHourIterator):
            print(row[1].DateTime)
            break
        if (dayHourIterator < 23):
            dayHourIterator = dayHourIterator + 1
        else:
            dayHourIterator = 0
    print(main_df.tail(48))
    dict = createDayDataDict(main_df)
    np.save('InputData/Input_dict.npy', dict)

    main_df.to_csv('InputData/NormalizedInputValues.csv')