import pandas as pd
import re
from pathlib import Path

dirname = Path(__file__).parent.parent


def input_to_dataframe(dataframe, id_columns=None
                       ):
    if isinstance(dataframe, pd.DataFrame):
        return dataframe
    if not isinstance(dataframe, str):
        raise TypeError(
            "Error: Inputted dataframes must be of type str (path) or pd.DataFrame.")

    # see if there are columns that must be read in as strings
    if id_columns is None:
        id_columns = []
    elif isinstance(id_columns, str):
        id_columns = [id_columns]
    elif not isinstance(id_columns, list):
        raise TypeError(
            "ID Columns provided must be of either list or string type.")

    dtype_dict = {col: str for col in id_columns}

    if dataframe[-3:] == 'csv':
        return (pd.read_csv(dataframe, dtype=dtype_dict, keep_default_na=False, na_values=['']))
    elif dataframe[-3:] == 'dta':
        return (pd.read_stata(dataframe))
    else:
        raise Exception(
            "Error: Please provide input file path to a .csv or .dta file.")


def best_subname(lname: str, name_set: set):
    if pd.isna(lname):
        return lname

    subnames = re.split("[\\-\\-\\-\\–\\—\\−]+", lname)
    match_names = [subname for subname in subnames if subname in name_set]
    if len(match_names) > 0:
        return match_names[0]
    return subnames[0]


def clean_name(lname: str):

    # do NOT coerce type if pandas recognizes it as missing
    if pd.isna(lname):
        return lname

    # otherwise coerce
    lname = str(lname)

    # strips spaces at the beg and end of string, so we can remove spaces
    lname = lname.strip()
    # add in spaces for suffix matching
    lname = ' '+lname+' '

    # Deletes non-letter, non-hyphen characters and turns upper case into lowercase
    # Question: Should I replace with a space or just remove?
    # Question: What about names that include $,%,^ etc.... what is the cut off for removing non-letter, non-hyphen characters?
    lname = re.sub(r'[\`\{}\\,.0-9"]', ' ', lname.lower())

    # Remove double quotes.
    lname = lname.replace('"', ' ',)

    # Remove common suffixes with spaces!
    suffix_list = [" jr ", " sr ", " iii ",
                   " ii ", " iv ", " dds ", " md ", " phd "]
    for suffix in suffix_list:
        lname = re.sub(suffix, ' ', lname)

    # Removes Apostrophes separately in order to avoid potential problems with names like D'Angelo, O'Leary, etc., having the first letter dropped.
    lname = lname.replace("'", ' ',)

    # Removes any lone letters in lname are most likely initials (in most cases, middle initials); remove them unless they are the letters "o" or "d".
    lname = re.sub(r' [a-ce-np-z] ', ' ', lname)

    lname = lname.strip()

    # Removes any Lone letters -- at the beg of a string
    lname = re.sub(r'^[a-ce-np-z] ', ' ', lname)
    # Removes any Lone letters -- at the end of a string
    lname = re.sub(r' [a-ce-np-z]$', ' ', lname)
    # ^ Perhaps we ought we to include o's and d's here
    # clean up lone letters 'o' or 'd'

    # Last Step: Remove all spaces, no matter how long!
    lname = lname.replace(' ', '',)

    return lname
