import datetime
import sys
def setStdoutToFile(filePath):
    f = open(filePath, 'w')
    sys.stdout = f
    return f

def setStdoutToDefault(f):
    f.close()
    sys.stdout = sys.__stdout__

def getCurrentTime():
    # return string type of current date and time
    loc_dt = datetime.datetime.today() 
    loc_dt_format = loc_dt.strftime("%Y/%m/%d %H:%M:%S")
    return loc_dt_format

def getCurrentTime1():
    # return string type of current date and time
    loc_dt = datetime.datetime.today() 
    loc_dt_format = loc_dt.strftime("%H_%M_%S")
    return loc_dt_format