# wind_scenarios
A model for generating wind power production scenarios based on open data from AEMO

To use the model, do the following steps:
1. Clone git repository. It has the python code and the fitted model (saved as a pickle file).
2. Download the AEMO sqlite database from https://drive.google.com/file/d/1xUQofxWEfB4xt7bSo8iAFnzuzHLNI2D6/view?usp=sharing
   This has 3 months of data starting 2019-08-01 and is 300 MB. 
3. Open main.py and specify the options you want, such as number of scenarios and which date to use for the original forecast. 
   You have to specify the path to the database from 2 above. 
4. Run main.py, the scenarios will be generated and exported to an excel file.

If you wish to fit your own model, see wind_model.fit_model() for an example. 

Notes:
1. The fitting of the quantile curves in fit_quantiles() has been implemented using the Gurobi python API, so currently
   you need to have Gurobi installed to fit the model. It would be straightforward to replace the optimization problem 
   with a Pyomo implementation.