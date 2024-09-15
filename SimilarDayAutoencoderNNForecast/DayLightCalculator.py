import math
LATITUDE_BELGRADE = 44.787197
def daylight(latitude,day):
    P = math.asin(0.39795 * math.cos(0.2163108 + 2 * math.atan(0.9671396 * math.tan(.00860 * (day - 186)))))
    pi = math.pi
    #daylightamount = 24 - (24 / pi) * math.acos((math.sin((0.8333 * pi / 180) + math.sin(latitude * pi / 180) * math.sin(P)) / (math.cos(latitude * pi / 180) * math.cos(P))))
    daylightamount = 24 - (24 / pi) * math.acos((math.sin(0.8333 * pi / 180) + math.sin(latitude * pi / 180) * math.sin(P)) / ( math.cos(latitude * pi / 180) * math.cos(P)))
    return daylightamount

def daylight_difference(latitude, day):
    shortest_day = 356
    difference = daylight(latitude, day) - daylight(latitude,shortest_day)
    return difference