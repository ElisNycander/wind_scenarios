import requests
from bs4 import BeautifulSoup
import urllib.request
from pathlib import Path
import os
import zipfile
import sqlite3
import csv
import datetime
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from help_functions import str_to_date, create_select_list
""" 
# List of wind farms:
NOTE: This list was made by obtaining all DUIDs with WF in the name, and then manually checking similarity between
the DUID and wind farm names from Wikipedia. Thus it is not certain that the DUID belongs to the identified 
wind farm for all cases. 
## VICTORIA ##
ARWF1 - Ararat Wind Farm
BALDHWF1 - Bald Hills Wind Farm
CHALLHWF - Challicum Hills Wind Farm
CROWLWF1 - Crowlands Wind Farm
MACARTH1 - Macarthur wind farm
MLWF1 - Mortons Lane Wind Farm
MTGELWF1 - Mount Gellibrand Wind Farm
MEWF1 - Mount Mercer Wind Farm??
MUWAWF1 - Murra Warra Wind Farm
OAKLAND1 - Oaklands Hill Wind Farm ??
PORTWF - Portland Wind Farm
WAUBRAWF - Waubra Wind Farm
## NEW SOUTH WALES ##
BOCORWF1 - Boco Rock Wind Farm
BODWF1 - Bodangora Wind Farm 1
CAPTL_WF - Capital Wind Farm
GULLRWF1 - Gullen Range Wind Farm
SAPHWF1 - Sapphire Wind Farm
STWF1 - Silverton Wind Farm
TARALGA1 - Taralga Wind Farm
WRWF1 - White Rock Stage 1
Woodlawn Wind Farm - ??
## SOUTH AUSTRALIA ##
CELMGPWF - Clements Gap Wind Farm
HALLWF1/HALLWF2 - Hallett Group
HDWF1/HDWF2/HDWF3 - Hornsdale Wind Farm
LGAPWF1 - Lincoln Gap Wind Farm
STWF1 - Starfish Hill Wind Farm
WATERLWF - Waterloo Wind Farm
WGWF1 - Willogoleche Wind Farm ??
## QUEENSLAND ##
COOPGWF1 - Coopers Gap Wind Farm
## UNKOWN ##
CROOKWF2
KIATAWF1 
NBHWF1
YENDWF1

"""
# %%
# lists of wind farms, used when inserting forecast data into database
wind_duid = {
    'victoria': ['ARWF1', 'BALDHWF1', 'CHALLHWF', 'CROWLWF1', 'MACARTH1', 'MLWF1', 'MTGELWF1', 'MEWF1', 'OAKLAND1',
                 'PORTWF', 'WAUBRAWF','MUWAWF1'],
    'new south wales': ['BOCORWF1', 'BODWF1', 'CAPTL_WF', 'GULLRWF1', 'SAPHWF1', 'STWF1', 'TARALGA1', 'WRWF1'],
    'queensland':['COOPGWF1'],
    'south australia':['CELMGPWF','HALLWF1','HALLWF2','HDWF1','HDWF2','HDWF3','LGAPWF1','STWF1','WATERLWF','WGWF1'],
   # 'unknown':['CROOKWF2','KIATAWF1','NBHWF1','YENDWF1']
}
# capacity of wind farms, used for normalization (unless maximum production value is used)
wind_cap = {
    'ARWF1':240,
    'MACARTH1':420,
    'PORTWF':195,
    'BALDHWF1':107,
}

cm_per_inch = 2.5

wind_latlon = {
    'ARWF1':[-37.26228480761284, 143.07848659783988],
    'MACARTH1':[-38.009698109704246, 142.19470124339418],
    'BALDHWF1':[-38.6973409404194, 145.95180550500424],
}


def _time2table_(timestr,table_type='dispatch'):
    """ Given timestr of format 'YYYYMMDD:HHMM' produce string of format 'YYYYMM'
    This function is used to map a time to a table, to be able to divide data
    into multiple tables for faster search. May be updated if further data
    division is required, e.g. one table per day
    """
    if table_type == 'dispatch' or table_type == 'forecast_full':
        return timestr[0:8]  # currently one table per day
    elif table_type == 'forecast':
        return timestr[0:6] # one table per month
    else:
        return None

def _create_table_(c, name,table_type='dispatch'):
    """ Drop table if it exists, then create new table with given name """

    c.execute('DROP TABLE IF EXISTS {0}'.format(name))
    if table_type == 'dispatch':
        c.execute('CREATE TABLE {0} ('.format(name) + \
                  'time TEXT NOT NULL,' + \
                  'name TEXT NOT NULL,' + \
                  'val REAL NOT NULL' + \
                  ')')
    elif table_type == 'forecast':
        c.execute('CREATE TABLE {0} ('.format(name) + \
                  'time TEXT NOT NULL,' + \
                  'name TEXT NOT NULL,' + \
                  'val REAL NOT NULL' + \
                  ')')
    elif table_type == 'forecast_full':
        c.execute('CREATE TABLE {0} ('.format(name) + \
                  'time TEXT NOT NULL,' + \
                  'name TEXT NOT NULL,' + \
                  'lead INTEGER NOT NULL,' + \
                  'val REAL NOT NULL' + \
                  ')')
    else:
        print(f'_create_table_(): Unknown table_type "{table_type}"')


def _execute_(c, cmd):
    try:
        c.execute(cmd)
    except sqlite3.Error as e:
        print('Executing command ''{0}'' returned error: {1}'.format(cmd, e))


class Database:

    def __init__(self, db='C:/Data/aemo.db'):
        # Note: database is large, hence store it offline
        self.db = db

    def add_dispatch_data(self, data_path='D:/Data/AEMO/Dispatch/',nfiles=1000,startdate=None,enddate=None,units=None):

        # db = Database(db='D:/Data/aemo.db')
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        # read csv files
        files = [f for f in os.listdir(data_path) if '.CSV' in f]

        # %% enter dispatch data into sqlite database
        existing_tables = []  # tables in db created in this function call
        # Note: all tables into which data is inserted are recreated, but existing tables that are
        # note needed to insert data into are untouched
        table_type = 'dispatch'
        file_counter = 0
        if nfiles is not None:
            read_files = files[0:nfiles]
        elif startdate is not None and enddate is not None:
            read_files = [f for f in files if f.split('_')[2] >= startdate and f.split('_')[2] <= enddate]
        else:
            read_files = files
        start_time = time.time()
        for f in read_files:
            # print(f'{f}')
            with open(Path(data_path) / f) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                for row in spamreader:
                    if row[0] == 'D':  # data value
                        # extract: time, name, value
                        val = row[6]
                        name = row[5]
                        time_f1 = row[4]  # time format: YYYY/MM/DD HH:MM:00
                        day = time_f1.split(' ')[0].replace('/', '')
                        hour = time_f1.split(' ')[1].replace(':', '')[0:4]
                        time_f2 = f'{day}:{hour}'

                        table_name = f'{table_type}_{_time2table_(time_f2)}'
                        if table_name not in existing_tables:
                            # replace table and append to list
                            existing_tables.append(table_name)
                            _create_table_(c, table_name, table_type=table_type)
                            print(f'Table: {table_name}')
                        # add data to db:
                        cmd = f'INSERT INTO {table_name} (time,name,val) values("{time_f2}","{name}",{val})'
                        if units is None or name in units:
                            _execute_(c, cmd)

                # commit after reading each file
            conn.commit()
            file_counter += 1
            if np.remainder(file_counter,50) == 0:
                print(f'File nr. {file_counter}')
                avg_time = (time.time()-start_time)/file_counter
                rem_time = avg_time * (read_files.__len__()-file_counter)
                min = np.remainder(rem_time,3600)/60
                hours = int(np.floor_divide(rem_time,3600))
                print(f'Estimated time remaining: {hours} hours and {min:0.1f} min')

        conn.close()


    def add_forecast_data_full(self,data_path = 'D:/Data/AEMO/PreDispatch/',
        nfiles=10,startdate=None,enddate=None):
        """
        Add full forecast data with all forecast horizons

        Note: lead is integer variable denoting when the forecast
        was made. It is number of 30-minute intervals prior to the
        start of the day at 04:30
        E.g.:
        lead
        32  - first forecast made at 12.30 day ahead for all periods
        0   - forecast made at 04:30 for remaining 47 periods 05:00-04:00
        -46 - last forecast made at 03:30 for period 04:00

        """
        # db = Database(db='D:/Data/aemo.db')

        # data_path = 'D:/Data/AEMO/PreDispatch/'
        # nfiles = 1

        # db = Database(db='D:/Data/aemo.db')

        # data_path = 'D:/Data/AEMO/PreDispatch/'
        # nfiles = 100
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        # read csv files
        files = [f for f in os.listdir(data_path) if '.CSV' in f]
        time_fmt1 = '%Y/%m/%d %H:%M:%S'
        time_fmt2 = '%Y%m%d:%H%M'
        time_fmt3 = '%H%M'

        units = []
        for a in wind_duid:
            units += wind_duid[a]



        # %% enter dispatch data into sqlite database
        existing_tables = []  # tables in db created in this function call
        # Note: all tables into which data is inserted are recreated, but existing tables that are
        # not needed to insert data into are untouched
        table_type = 'forecast_full'
        file_counter = 0
        if nfiles is not None:
            read_files = files[0:nfiles]
        elif startdate is not None and enddate is not None:
            read_files = [f for f in files if f.split('_')[4] >= startdate and f.split('_')[4] <= enddate]
        else:
            read_files = files

        start_time = time.time()

        for file_name in read_files:

            file_date_str = file_name.split('_')
            sidx = 0
            while sidx <= file_date_str.__len__():
                try:
                    s = file_date_str[sidx]
                    int(s)
                    break
                except:
                    sidx += 1
            file_date_start = datetime.datetime(year=int(s[:4]),
                                                month=int(s[4:6]), day=int(s[6:]), hour=4, minute=30)

            # file_name = files[0]
            csvfile = open(Path(data_path) / file_name)
            while True:
                row = [c.strip('\n') for c in csvfile.readline().split(',')]
                if row[0] == 'I' and row[2] == 'UNIT_SOLUTION':
                    break
            csv_idx2col = row
            csv_col2idx = {}
            for idx, col in enumerate(row):
                csv_col2idx[col] = idx

            # read lines for given units
            ridx = 0
            current_unit = ''
            lead_time = 32
            while True:
                row_old = row
                row = [c.strip('\n') for c in csvfile.readline().split(',')]
                name = row[csv_col2idx['DUID']]

                if name in units and row[2] == 'UNIT_SOLUTION':

                    # enter data from row into database
                    val = row[csv_col2idx['AVAILABILITY']]

                    time_f1 = row[csv_col2idx['DATETIME']].strip('"')
                    time_dt = datetime.datetime.strptime(time_f1, time_fmt1)
                    time_f2 = time_dt.strftime(time_fmt2)

                    fc_time_f1 = row[csv_col2idx['LASTCHANGED']].strip('"')
                    fc_time_dt = datetime.datetime.strptime(fc_time_f1, time_fmt1)
                    fc_time_f2 = fc_time_dt.strftime(time_fmt2)

                    if name != current_unit:
                        # then this is the first entry of a 24-hour forecast
                        # may be shorter time period if forecast is made after
                        # 04:30
                        current_unit = name
                        # round forecast time to 30 min
                        if fc_time_dt.minute < 30:
                            fc_time_dt = fc_time_dt \
                                         - datetime.timedelta(seconds=60 * fc_time_dt.minute) \
                                         - datetime.timedelta(seconds=fc_time_dt.second)
                        else:
                            fc_time_dt = fc_time_dt \
                                         - datetime.timedelta(seconds=60 * (fc_time_dt.minute - 30)) \
                                         - datetime.timedelta(seconds=fc_time_dt.second)
                        # calculate lead time for this forecast
                        td = file_date_start - fc_time_dt
                        lead_time = int(2 * (td.days * 24 + td.seconds / 3600))

                    table_name = f'{table_type}_{file_date_start.strftime("%Y%m%d")}'

                    if table_name not in existing_tables:
                        # replace table and append to list
                        existing_tables.append(table_name)
                        _create_table_(c, table_name, table_type=table_type)
                        # print(f'Table: {table_name}')
                    # add data to db:
                    cmd = f'INSERT INTO {table_name} (time,name,lead,val) values("{time_f2}","{name}",{lead_time},{val})'
                    _execute_(c, cmd)

                if row[0] != 'D':  # stop when reach end of current data section
                    break

            csvfile.close()
            conn.commit()
            file_counter += 1
            if np.remainder(file_counter, 5) == 0:
                print(f'File nr. {file_counter}')
                avg_time = (time.time() - start_time) / file_counter
                rem_time = avg_time * (read_files.__len__() - file_counter)
                min = np.remainder(rem_time, 3600) / 60
                hours = int(np.floor_divide(rem_time, 3600))
                print(f'Estimated time remaining: {hours} hours and {min:0.1f} min')

        conn.close()

    def query_names(self,table='20190706',table_type='dispatch'):

        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        cmd = f"SELECT DISTINCT name FROM {table_type}_{table}"
        _execute_(c, cmd)
        names = []
        for row in c.fetchall():
            names.append(row[0])
        conn.close()
        return names

    def query_max_values(self,table_type='dispatch',categories=None):

        # %% find relevant tables
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        c.execute(cmd)
        # tables of right type
        rel_tables = [t[0] for t in c.fetchall() if table_type in t[0]]


        if categories is None:
            categories = self.query_names(table=rel_tables[0].split('_')[1], table_type=table_type)
            select_plants = False
        else:
            select_plants = True
            cat_str = create_select_list(categories)

        data = pd.Series(0.0, dtype=float, index=categories)
        for t in rel_tables[:]:
            if select_plants:
                cmd = f"SELECT name, MAX(val) FROM {t} WHERE name in {cat_str} GROUP BY name"
            else:
                cmd = f"SELECT name, MAX(val) FROM {t} GROUP BY name"

            _execute_(c, cmd)
            for row in c.fetchall():
                if row[0] in categories:
                    if row[1] > data.at[row[0]]:
                        data.at[row[0]] = row[1]

        return data

    def select_data(self,starttime='20190706:00',endtime='20190706:23',table_type='dispatch',categories=None):

        # %% select data
        # db.add_dispatch_data()
        # starttime = '20190706:00'
        # endtime = '20190707:00'
        # plants=['YWPS1','YWPS2']
        # categories = None
        # table_type = 'dispatch'

        if table_type == 'dispatch':
            time_freq = '5min'
            first_table = starttime[0:8]
        elif table_type == 'forecast':
            time_freq = '30min'
            first_table = starttime[0:6]

        else:
            time_freq = '5min'
            first_table = starttime[0:8]


        if categories is None:
            categories = self.query_names(table=first_table, table_type=table_type)
            select_plants = False
        else:
            select_plants = True
            cat_str = create_select_list(categories)


        time_idx = pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq=time_freq)
        # time_idx_30min = pd.date_range(start=str_to_date(starttime),end=str_to_date(endtime), freq='30min')

        # time_idx_quarter = pd.date_range(start=str_to_date(starttime), end=str_to_date(endtime), freq='15min')

        # %% find relevant tables
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        c.execute(cmd)
        # tables of right type
        rel_tables = [t[0] for t in c.fetchall() if table_type in t[0]]
        # tables for specified time range

        get_tables = [t for t in rel_tables if t.split('_')[1] >= _time2table_(starttime,table_type=table_type) \
                      and t.split('_')[1] <= _time2table_(endtime,table_type=table_type)]

        # %% initialize dataframes

        data = pd.DataFrame(dtype=float, index=time_idx, columns=categories)

        # %% get data
        for t in get_tables:
            if starttime.__len__() < 13:  # time format YYYYMMDD:HH
                cmd = f"SELECT time,name,val FROM {t} WHERE time >= '{starttime}00' AND time <= '{endtime}00'"
            else:
                cmd = f"SELECT time,name,val FROM {t} WHERE time >= '{starttime}' AND time <= '{endtime}'"
            if select_plants:
                cmd += f" AND name IN {cat_str}"

            _execute_(c, cmd)
            for row in c.fetchall():
                # convert DD:2400 to D+1:0000
                data.at[str_to_date(row[0]), row[1]] = row[2]
        conn.close()
        return data

    def select_forecast_data_full(self,startdate='20190706',enddate='20190707',
                            lead_times=[32,16,1],categories=None):

        # startdate = '20190706'
        # enddate = '20190707'
        # lead_times = [32, 31, 10, -10]
        # categories = ['ARWF1', 'MACARTH1']

        time_freq = '30min'
        first_table = startdate
        table_type = 'forecast_full'

        if categories is None:
            categories = self.query_names(table=first_table, table_type=table_type)
            select_plants = False
        else:
            select_plants = True
            cat_str = create_select_list(categories)

        lead_str = create_select_list(lead_times)
        # %% find relevant tables
        conn = sqlite3.connect(self.db)
        c = conn.cursor()
        cmd = "SELECT name FROM sqlite_master WHERE type='table'"
        c.execute(cmd)
        # tables of right type
        rel_tables = [t[0] for t in c.fetchall() if table_type in t[0]]
        # tables for specified time range

        get_tables = [t for t in rel_tables if t.split('_')[2] >= _time2table_(startdate, table_type=table_type) \
                      and t.split('_')[2] <= _time2table_(enddate, table_type=table_type)]

        # %% initialize dataframes
        # Note: planning day from 04:30 - 04:00

        starttime = datetime.datetime(year=int(startdate[:4]),
                                      month=int(startdate[4:6]),
                                      day=int(startdate[6:8]),
                                      hour=4,
                                      minute=30)
        endtime = datetime.datetime(year=int(enddate[:4]),
                                    month=int(enddate[4:6]),
                                    day=int(enddate[6:8]),
                                    hour=4)
        endtime = endtime + datetime.timedelta(days=1)
        time_idx = pd.date_range(start=starttime, end=endtime, freq=time_freq)

        col_idx = pd.MultiIndex.from_product([categories, lead_times],
                                             names=['unit', 'lead'])

        data = pd.DataFrame(dtype=float, index=time_idx, columns=col_idx)

        # %% get data
        for t in get_tables:
            cmd = f"SELECT time,name,lead,val FROM {t} WHERE lead IN {lead_str}"
            if select_plants:
                cmd += f" AND name IN {cat_str}"
            _execute_(c, cmd)
            for row in c.fetchall():
                data.at[str_to_date(row[0]), (row[1], row[2])] = row[3]
        conn.close()

        return data

def download_aemo_data(data_path = 'D:/Data/AEMO/Dispatch',
    dir='http://nemweb.com.au/Reports/Archive/Dispatch_SCADA/',
    zip_levels=2,startdate='20190801',enddate='20190802'):

    # data_path = 'D:/Data/AEMO/PreDispatch'
    # dir = 'https://nemweb.com.au/Reports/Current/Next_Day_PreDispatch/'
    # zip_levels = 1
    # startdate = '20190801'
    # enddate = '20190802'
    """ Download CSV files from AESO webpage

    Data type:          Link:                                                   Folder:
    Dispatch SCADA      'http://nemweb.com.au/Reports/Archive/Dispatch_SCADA/'  'D:/Data/AEMO/Dispatch'
    Forecast (RES)      'http://nemweb.com.au/Reports/Archive/Next_Day_Intermittent_DS/' 'D:/Data/AEMO/Forecast'
    PreDispatchD         http://nemweb.com.au/Reports/Current/Next_Day_PreDispatchD/ 'D/Data/AEMO/PreDispatchD'
    PreDispatch         https://nemweb.com.au/Reports/Current/Next_Day_PreDispatch/  'D/Data/AEMO/PreDispatch'
    """
    data_path = Path(data_path)
    # data_path = Path('D:/Data/AEMO/Dispatch')
    # dispatch_dir = 'http://nemweb.com.au/Reports/Archive/Dispatch_SCADA/'
    parser = 'html.parser'

    (data_path / 'zip').mkdir(exist_ok=True, parents=True)
    (data_path / 'zip2').mkdir(exist_ok=True, parents=True)
    resp = urllib.request.urlopen(dir)
    soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))
    #
    all_files = [l.text for l in soup.find_all('a', href=True) if '.zip' in l.text]
    print(all_files)

    # %%
    # find index of date

    split_str = [s.strip('.zip') for s in all_files[0].split('_')]
    print(split_str)
    idx = 0
    while True:
        try:
            int(split_str[idx])
            break
        except:
            idx += 1
    print(idx)

    dl_files = [f for f in all_files if f.split('_')[idx] >= startdate and f.split('_')[idx] <= enddate]
    print(dl_files)

    for idx, fname in enumerate(dl_files):
        print(f'{fname}')
        file = requests.get(dir + fname, allow_redirects=True)
        with open(data_path / 'zip' / fname, 'wb') as f:
            f.write(file.content)

    # %% unzip files (creates many zip files)
    if zip_levels == 2:  # zip files contain other zip files which contain csv files
        zipfiles = [f for f in os.listdir(data_path / 'zip') if '.zip' in f]
        for f in zipfiles:
            with zipfile.ZipFile(data_path / 'zip' / f, "r") as zip_ref:
                zip_ref.extractall(path=data_path / 'zip2')

        # unzip files (to csv files)
        zipfiles = [f for f in os.listdir(data_path / 'zip2') if '.zip' in f]
        for f in zipfiles:
            with zipfile.ZipFile(data_path / 'zip2' / f, "r") as zip_ref:
                zip_ref.extractall(path=data_path)
    else:  # zip files only contain csv files
        zipfiles = [f for f in os.listdir(data_path / 'zip') if '.zip' in f]
        for f in zipfiles:
            with zipfile.ZipFile(data_path / 'zip' / f, "r") as zip_ref:
                zip_ref.extractall(path=data_path)

def find_duid_predispatch(file_path='D:/Data/AEMO/PreDispatch',file_nr=0):
    """
    Parse single PreDispatch file to find all DUIDs

    :param file_path: Path to folder with PreDispatch csv files
    :param file_nr: Index of file to parse
    :return: list of unique DUIDs
    """
    files = [f for f in os.listdir(file_path) if '.CSV' in f]

    file_name = files[file_nr]

    csvfile = open(Path(file_path) / file_name)

    # increment until data_type is found
    ridx = 0
    while True:
        row = [c.strip('\n') for c in csvfile.readline().split(',')]
        if row[0] == 'I' and row[2] == 'UNIT_SOLUTION':
            break

    csv_idx2col = row
    csv_col2idx = {}
    for idx,col in enumerate(row):
        csv_col2idx[col] = idx

    # read lines and find new DUID
    ridx = 0
    duids = []
    while True:
        row_old = row
        row = [c.strip('\n') for c in csvfile.readline().split(',')]

        if row[2] != 'UNIT_SOLUTION':
            break
        # save duid
        id = row[csv_col2idx['DUID']]
        if id not in duids:
            duids.append(id)
        ridx += 1

    return duids

def plot_wind_farm_map():

    # fig_path = 'C:/Users/elisn/Box Sync/Papers/C4 - Wind scenarios/Figures'
    fig_path = 'D:/wind_scenarios/Figures'
    Path(fig_path).mkdir(exist_ok=True,parents=True)
    units = ['ARWF1','MACARTH1','BALDHWF1']
    """ Create Basemap projection corresponding to efas data """
    proj = 'laea'
    lat_0 = -38.95
    lat_0 = -39.4
    lon_0 = 144.5
    width = 1*1e6
    height = 0.6*1e6
    # return m
    msize = 10
    offs1 = 2.5e4

    from mpl_toolkits.basemap import Basemap

    m = Basemap(projection=proj, resolution='l',
                width=width, height=height,
                lat_0=lat_0, lon_0=lon_0)

    f,ax=plt.subplots()
    f.set_size_inches(6,3.5)
    m.drawcoastlines()
    m.fillcontinents()
    # m.drawmeridians()
    # m.drawparallels()
    # m.plot()

    for u in units:
        xy = m(wind_latlon[u][1],wind_latlon[u][0])
        plt.plot(xy[0],xy[1],marker='o',markersize=msize,color='green',markeredgecolor='black')
        # m.scatter(xy[0],xy[1],marker='o',edgecolors='black',color='green')
        plt.text(xy[0]+offs1,xy[1],u)

    bass_latlon = [-39.686821378896616, 145.25306912813292]
    tasm_latlon = [-41.976073008838834, 145.80081762407574]
    melb_latlon = [-37.81300776034128, 144.9677902316384]


    xy = m(melb_latlon[1],melb_latlon[0])
    plt.plot(xy[0],xy[1],marker='s',markersize=msize,color='red',markeredgecolor='black')
    plt.text(xy[0]+offs1,xy[1],'Melbourne')

    xy = m(bass_latlon[1],bass_latlon[0])
    plt.text(xy[0],xy[1],'Bass Strait')

    xy = m(tasm_latlon[1],tasm_latlon[0])
    plt.text(xy[0],xy[1],'Tasmania')

    m.drawmapscale(141,-41,0,0,100)

    plt.savefig(Path(fig_path)/f'wind_farms_map.png')
    plt.savefig(Path(fig_path)/f'wind_farms_map.eps')

if __name__ == "__main__":

    pd.set_option('display.max_rows',20)
    pd.set_option('display.max_columns',None)


    plot_wind_farm_map()

    #%% build database, from August 01 - October 31

    # db = Database('D:/Data/aemo_small.db')
    #
    # db.add_forecast_data_full(data_path = 'D:/Data/AEMO/PreDispatch/zip2',nfiles=None,startdate='20190801',enddate='20191031')
    #
    # # add dispatch data for wind units only
    # units = []
    # for a in wind_duid:
    #     units += wind_duid[a]
    #
    # db.add_dispatch_data(data_path='D:/Data/AEMO/Dispatch',nfiles=None,startdate='20190801',enddate='20191031',units=units)


    # db = Database('D:/Data/aemo_new.db')
    #
    # df = db.select_forecast_data_full(lead_times=[1],startdate='20190901',enddate='20191030',categories=['ARWF1','MACARTH1','BALDHWF1'])
    # df.plot()





