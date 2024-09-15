import numpy as np
import pandas as pd

def ReadCSV(csv_filename, column):
    #csv_filename = 'EMS_DataHistory_Load.csv'

    df = pd.read_csv(csv_filename)
    #print(df.head())
    df.columns = ['DateTime', str(column)]
    #print(df.head())
    df.set_index('DateTime', inplace=True)  # set time as index so we can join them on this shared time
    #print(df.head())
    return df

def ReadCSVSheets(file_prefix, atributes):
    #atributes = ['Load', 'Temp']
    main_df = pd.DataFrame() # begin empty
    for atribute in atributes:
        filename = f'{file_prefix}{atribute}.csv'
        local_df = ReadCSV(filename,atribute)
        #print(local_df.head())

        if len(main_df) == 0:  # if the dataframe is empty
            main_df = local_df  # then it's just the current df
        else:  # otherwise, join this data to the main one
            main_df = main_df.join(local_df)
    #print(main_df.head())
    return main_df
