# A Simple Implementation of Bayesian Improved Surname Geocoding in Python

The goal of this repo is to provide a simple, easily import-able, BISG class that is implemented in Python. Users will be able to import the `BISG` class to handle all of the BISG race prediction computations on any input data with a surname and a GEOID or ZCTA5 column. This package ports the functionality from a series of legacy stata scripts, whose functionality has been documented [here](https://files.consumerfinance.gov/f/201409_cfpb_report_proxy-methodology.pdf).

## Usage of Key Objects

The `BISG` class can be imported and instantiated as so

**(this is currently desired behavior and not yet implemented, right now you still have to pass arguments to BISG's constructor, which we will lock up later)**

```{python}
from BISG import BISG

#instantiate BISG object
myBISG=BISG()
```

In the instantiation of the `BISG` object, a series of probability look-up tables are constructed from 2000 and 2010 census data. These tables will be referenced when a user uses the predict method of their `BISG` instance. The primary public method of a `BISG` instance is `predict`. The `predict` method takes as arguments 
1. `input_data`, a `pandas.DataFrame` class or a string path to a `.csv` or `.dta` file.

2. `name_column`, a string which provides the name of the column which contains surname information

3. `zip_column`, a string which provides the name of the column which contains ZCTA5 information

4. `GEOID_column`, a string which provides the name of the column which contains GEOID information 

5. `blk_grp`, a boolean value which indicates whether the GEOID is 12 digits or longer (contains information on block group). This could be found automatically in later iterations. If true, the `BISG.predict` method will return predictions at the block group level.

6. `only_final_probs`, a boolean value which indicates whether intermediate probabilities used to calculate final probabilities should be returned as well.

The predict method then will return a dictionary of dataframes of predicted race values for each level that of specificity that was provided:

1. `'zip'` if `zip_column` was provided, 
2. `'tract'` if `GEOID_column` was provided, and 
3. `'blk_grp'` if `GEOID_column` was provided and `blk_grp = True`.

The dataframes of predictions returned contain one column for the row number and then columns for the probability of each race in the race list.