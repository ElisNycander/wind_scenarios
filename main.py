############ MAIN FILE TO GENERATE SCENARIOS ##################
"""
Simplest method to generate scenarios

Specify options below, then run file to create scenarios
Scenarios will be exported to an excel file: scenarios.xlsx
Scenarios will also be plotted in wind_model folder

scenarios.xlsx contains following sheets:
scenarios: the scenarios generated for all wind farms
data: the original forecast used to generate scenarios, the actual production during the day,
the low-pass filtered production, as well as the min and max ranges obtained for the specified quantiles
"""
###### OPTIONS ##########

# path to database with production and forecast data
db = 'D:/Data/aemo_small.db'

# string name which is attached to model name; full name is wind_model_{name}
name = 'v1'

# path to data directory
# 1. the fitted model file must be stored as path/wind_model_{name}/wind_model_{name}.pkl
# 2. excel file generated will be stored in path
path = 'C:/Users/elisn/Box Sync/Python/wind_scenarios/data'

# date for which to create scenarios
date = '20190801'

# number of scenarios to generate
nscen = 10

# seed for random number generator, set to None to avoid using seed
seed = 1

# quantile range for which to get uncertainty range
qrange = (0.05,0.95)

# option to create hourly scenarios instead of 5-min
# all time series are simply averaged over each hour
hourly = False

noise_scale = 0.3
# use this to scale the high-frequency noise added to the scenarios, may be set to 0

####### END OPTIONS ##########


from wind_model import  WindModel, plot_scenarios_minmax
import pandas as pd

m = WindModel(name='v1',path=path,wpd_db=db,wfc_db=db)
m.load_model()
m.hf_scale = noise_scale
rls,data = m.generate_scenarios(nscen=nscen,date=date,qrange=qrange,seed=seed,hourly_values=hourly)
plot_scenarios_minmax(rls,data,m.path,tag='default',qrange=qrange)

# save scenarios to excel file
writer = pd.ExcelWriter(f'{path}/scenarios.xlsx',engine='xlsxwriter')
rls.to_excel(writer,sheet_name='scenarios')
data.to_excel(writer,sheet_name='data')
writer.save()
