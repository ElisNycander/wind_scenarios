"""
Small help functions common to all modules
"""
import datetime

def str_to_date(strdate):
    """ Take a string with a date and return datetime object
    Allowed formats:
        'YYYYMMDD'
        'YYYY-MM-DD'
        'YYYYMMDD:HH'
        'YYYY-MM-DD:HH'
        'YYYYMMDD:HHMM
    """
    year = int(strdate[0:4])
    if strdate[4] == '-':
        month = int(strdate[5:7])
        day = int(strdate[8:10])
        idx = 10
    else:
        month = int(strdate[4:6])
        day = int(strdate[6:8])
        idx = 8
    if strdate.__len__() > idx:
        hour = int(strdate[idx+1:idx+3])
        if strdate.__len__() - idx > 3:
            min = int(strdate[idx+3:idx+5])
        else:
            min = 0
        if hour == 24:
            hour = 0
            day += 1
        return datetime.datetime(year,month,day,hour,min)
    else:
        return datetime.datetime(year,month,day)

def create_select_list(l):
    """ Create a string containing all elements in the list l:
        '("l[0]", "l[1]", ... , "l[-1]")'
        Used for conditional selects in Sqlite
    """
    s = '('
    for idx, cat in enumerate(l):
        if idx > 0:
            if type(cat) is str:
                s += ",'{0}'".format(cat)
            else:
                s += f",{cat}"
        else:
            if type(cat) is str:
                s += "'{0}'".format(cat)
            else:
                s += f"{cat}"

    s += ')'
    return s