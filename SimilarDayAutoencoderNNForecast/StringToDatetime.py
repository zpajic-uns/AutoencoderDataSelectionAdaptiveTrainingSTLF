from datetime import datetime as dt
import calendar

def string2date(datetime):
    #datetime_object = datetime.strptime('2013-03-03 03:00:00.000', '%Y-%m-%d %H:%M:%S.%f')
    datetime_object = dt.strptime(str(datetime), '%Y-%m-%d')
    return datetime_object.date()

def string2datetime(datetime):
    #datetime_object = datetime.strptime('2013-03-03 03:00:00.000', '%Y-%m-%d %H:%M:%S.%f')
    datetime_object = dt.strptime(str(datetime), '%Y-%m-%d %H:%M:%S.%f')

    #print('Day: ', datetime_object.day)
    #print('Month', datetime_object.month)
    #print('Hour:', datetime_object.hour)
    #print(datetime_object)
    return datetime_object

def isLeap(year):
    return calendar.isleap(year)

def dayInYear(datetime_object):
    month_days = [0, 31, 31+28, 31+28+31, 31+28+31+30, 31+28+31+30+31, 31+28+31+30+31+30, 31+28+31+30+31+30+31, 31+28+31+30+31+30+31+31, 31+28+31+30+31+30+31+31+30, 31+28+31+30+31+30+31+31+30+31, 31+28+31+30+31+30+31+31+30+31+30]
    previous_month = datetime_object.month-1
    leapYearAddition = 0
    if isLeap(datetime_object.year) and datetime_object.month>2:
        leapYearAddition = 1
    day_in_year = month_days[previous_month] + datetime_object.day + leapYearAddition

    #print('day in year:', day_in_year)
    return day_in_year

def dayInWeek(datetime_object):
    return datetime_object.weekday()
#daylight_hours = daylight_difference(latitude_belgrade, day_in_year)
#print('Difference between current day and shortest day:', daylight_hours)