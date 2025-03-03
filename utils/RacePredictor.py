from DataFrameUtils import clean_name, best_subname, input_to_dataframe
from LookupTables import LookUpTables
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Union

dirname = Path(__file__).parent.parent


class RacePredictor:
    """
    The RacePredictor class uses lookup tables from the LookUpTables class to predict the race 
    probabilities of individuals based on their surname and geographic location. The class
    has one primary predict method, which accepts input data along with the columns for geographic 
    identifiers and surname, and then merges that information with lookup tables to performing the 
    Bayesian Improved Surname Geocoding (BISG) method to infer a vector of posterior race probavilities.
    The predictions can be made at 3 different geographic levels: block group, tract, and ZIP code, 
    and the results ca nbe customized to include only the final probabilities or intermediate values for debugging.
    """

    def __init__(self,
                 lookuptables: LookUpTables = None,
                 debug=False
                 ) -> None:

        if lookuptables is None:
            raise Exception(
                "Race Predictor expected to be passed a LookupTables object")
        self.debug_mode = debug
        if debug:
            self.internal_dfs = dict()
        # declare attributes
        self.race_list = lookuptables.race_list
        self.LookUpTables = lookuptables

    # this fcn allows for 3 scenarios and perhaps we should make it allow for fewer
    def _add_row_id_column(self, input_data: pd.DataFrame, row_id_column: Union[str, None]):
        # if the row_id column provided is already in the dataset, return the dataset
        if row_id_column in input_data.columns:
            return input_data

        # else if no row_id was provided, make one
        input_data[row_id_column] = np.arange(input_data.shape[0])+1
        return input_data

    def _add_names_to_input_data(self, input_data=None, name_column: str = None, surname_table=None):
        # Clean input data names
        name_set = set(
            self.LookUpTables.pr_race_given_surname['name'].unique())
        input_data[name_column] = input_data[name_column].apply(clean_name)
        input_data[name_column] = input_data[name_column].apply(
            lambda x: best_subname(x, name_set))

        # Delete names from census that are null and nan so they are not read in and ruin the merge
        # bandaid solution for the null/na problem outlined on the project board -- df_census[df_census["name"].isna()]
        surname_table = surname_table.dropna(subset=["name"])
        full_table = pd.merge(input_data,
                              surname_table[[
                                  'name']+[f'pr_{race}_given_surname'for race in self.race_list]],
                              left_on=name_column, right_on='name', how='left')

        # return all but duplicative name column
        return full_table[[column for column in full_table.columns if column != "name"]]

    # return only one type of bisg probs (tract, zip, blkgrp)
    # does only one of the iterations of the for loop in geo_name_merger (one of the original stata scripts)
    # predict for a given geoid type

    def _typed_predict(self,
                       input_data_with_names,
                       row_id_column,
                       lookup_table,
                       lookup_type,
                       geo_column_name,
                       only_final_probs: bool = False
                       ):
        if lookup_table is None or lookup_type is None:
            raise Exception(
                "Lookup Table and Lookup Type must both be specified")

        # start the pandas-coded way to do this
        # begin computation

        race_geo_merged_df = pd.merge(input_data_with_names, lookup_table,
                                      left_on=geo_column_name, right_on="GeoInd", how="left", indicator=True)
        final_race_list = self.race_list

        # Use Bayesian updating to combine name and geo probabilities.
        # Follow the method and notation in Elliott et al. (2009).
        # u_white=P(white|name)*P(this tract|white), and so on for other races.
        # print(race_geo_merged_df.columns)
        for x in final_race_list:
            race_geo_merged_df["u_" +
                               x] = race_geo_merged_df[f'pr_{x}_given_surname'] * race_geo_merged_df["pr_geo_given_" + x]
        dummy_cols_list = ["u_" + x for x in final_race_list]

        race_geo_merged_df['u_sum'] = race_geo_merged_df[dummy_cols_list].sum(
            axis=1)
        for x in final_race_list:
            race_geo_merged_df["pr_" + x] = race_geo_merged_df["u_" +
                                                               x] / race_geo_merged_df['u_sum']

        # Begin a series of checks
        pr_race_list = ['pr_'+race for race in final_race_list]
        race_geo_merged_df['prtotal'] = race_geo_merged_df[pr_race_list].sum(
            axis=1)

        ####################################################

        # interesting that we are choosing to just overwrite if we have unexpected behavior!
        # And that we only overwrite too low

        ####################################################
        # Replace pr_race if the prtotal is less than < .99
        # *.99 chosen as these should sum to approximately 1, give or take rounding error.
        for x in final_race_list:
            race_geo_merged_df['pr_'+x] = np.where(
                race_geo_merged_df['prtotal'] < .99, pd.NA, race_geo_merged_df['pr_'+x])

        # Start the Assertions: 1. All Probabilities should be between 0 and 1

        # check that we are outputting probabilities
        for x in final_race_list:
            assert ((race_geo_merged_df['pr_'+x].isnull()) | (
                (race_geo_merged_df['pr_'+x] >= 0) & (race_geo_merged_df['pr_'+x] <= 1))).all()

        # Assert that the total probabilities are sensible (probs sum to 1 or zero (could be 0 if census out of date?))
        race_geo_merged_df['check_name'] = race_geo_merged_df[[
            f"pr_{race}_given_surname" for race in final_race_list]].sum(axis=1)
        race_geo_merged_df['check_geo'] = race_geo_merged_df[[
            f"pr_{race}_given_geo" for race in final_race_list]].sum(axis=1)
        race_geo_merged_df['check_pr'] = race_geo_merged_df[[
            "pr_"+race for race in final_race_list]].sum(axis=1)

        for y in ['name', 'geo', 'pr']:
            assert ((race_geo_merged_df['check_'+y] == 0.0) | ((race_geo_merged_df['check_'+y] >= .99) & (race_geo_merged_df['check_'+y] <= 1.01)) | (
                race_geo_merged_df['check_'+y].isna())).all(), f'Not all {y} probabilities sum to one. Throwing an error.'

        # I would rather say for file in ('blkgrp','tract','zip'), but that is not the way i set up the file
        # Rename BISG proxy variables ot reflect geography used                                                            #do we want to do this?

        # fix?

        full_probability_dictionary = {
            f'pr_{race}': f'pr_{race}_given_geo_and_surname' for race in final_race_list}
        geo_probability_dictionary = {
            f'pr_{race}_given_geo': f'pr_{race}_given_geo' for race in final_race_list}
        # race_geo_merged_df.rename(columns = geo_probability_dictionary, inplace = True)
        race_geo_merged_df.rename(
            columns=full_probability_dictionary, inplace=True)

        surname_columns = [
            f'pr_{race}_given_surname'for race in final_race_list]
        geo_columns = list(geo_probability_dictionary.values())
        full_probability_columns = list(full_probability_dictionary.values())

        # add in a column which indicates what geo_indicator we are using
        race_geo_merged_df['GeoInd'] = lookup_type

        # store intermediate columns if debugging
        if self.debug_mode:
            self.internal_dfs[lookup_type] = race_geo_merged_df

        # remove all intermediate probability columns if requested
        if only_final_probs:
            return race_geo_merged_df[[row_id_column, 'GeoInd']+full_probability_columns]
        probability_and_id_columns = [
            row_id_column, 'GeoInd']+full_probability_columns+geo_columns+surname_columns
        return race_geo_merged_df[probability_and_id_columns]

    def predict(self,
                input_data,
                name_column,
                row_id_column=None,
                zip_column=None,
                GEOID_column=None,
                blk_grp: bool = False,
                only_final_probs: bool = True):

        # warnings go here. Raise warning if predicting on geoid but not zip
        if zip_column == None and GEOID_column is not None:
            warnings.warn(
                'Predicting at tract level without predicting at zip level. Please make sure this is intentional behavior.')

        if isinstance(input_data, str):
            if input_data[-3:] == 'dta':
                warnings.warn(
                    '.dta data type supplied for input data. Make certain that GEOID and ZCTA5 are read in as str type, otherwise merge will fail.')

        # enforce types on parameters
        if not isinstance(name_column, str):
            raise TypeError("name_column must be of type str")

        if not isinstance(only_final_probs, bool):
            raise TypeError("only_final_probs should be boolean value")

        if not isinstance(blk_grp, bool):
            raise TypeError("blk_grp should be boolean value")
        if blk_grp and GEOID_column is None:
            raise Exception(
                "blk_grp can only be True if GEOID_column is provided")

        if (not isinstance(row_id_column, str)) and (row_id_column is not None):
            raise TypeError("Provided row_id_column must be of type str")

        # Parse the lookup types that are being asked for
        lookup_types = []
        if zip_column is None and GEOID_column is None:
            raise Exception("Must provide atleast zip column or GEOID column")

        # read in the zip column and raise error if it is not a string or None
        if zip_column is None:
            zip_column = []
        elif isinstance(zip_column, str):
            lookup_types = lookup_types+['zip']
            zip_column = [zip_column]
        else:
            raise TypeError("zip_column must be of type str")

        # same for the GEOID column
        if GEOID_column is None:
            GEOID_column = []
        elif isinstance(GEOID_column, str):
            lookup_types = lookup_types+['tract']
            GEOID_column = [GEOID_column]
        else:
            raise TypeError("GEOID_column must be of type str")

        # add blk_grp to the lookup types if it is requested
        if blk_grp:
            lookup_types = lookup_types+['blk_grp']

        # These are the columns that should be read in as str
        id_cols = [name_column]+zip_column+GEOID_column

        # read in input data
        input_data = input_to_dataframe(input_data, id_cols)

        # add names and rownum
        if row_id_column is None:
            row_id_column = 'rownum'
            warnings.warn(
                "No row_id_column passed. Creating a column 'rownum' for unique row id")
        input_data = self._add_row_id_column(
            input_data, row_id_column=row_id_column)
        input_data_with_names = self._add_names_to_input_data(input_data=input_data,
                                                              name_column=name_column,
                                                              surname_table=self.LookUpTables.pr_race_given_surname)

        # take stock of the geoid column
        # raise error if the entries of the geoid column are of problematic length, given lookup types specified
        if GEOID_column is not None:
            assert input_data_with_names[GEOID_column[0]].apply(lambda x: len(x) >= 11 if type(x) == str else pd.NA).all(
                # this could be disabled
            ), "Not all GEOIDs of length atleast eleven, but GEOID_column argument given"
            input_data_with_names['tract'] = input_data_with_names[GEOID_column[0]].apply(
                lambda x: x[:11] if type(x) == str else pd.NA)

        if blk_grp:
            # check to see that the geoid is greater than or equal to 12 or na
            assert input_data_with_names[GEOID_column[0]].apply(lambda x: len(x) >= 12 if type(x) == str else pd.NA).all(
                # this could be disabled
            ), "Not all GEOIDs of length atleast twelve, but blk_grp = True"
            input_data_with_names['blk_grp'] = input_data_with_names[GEOID_column[0]].apply(
                lambda x: x[:12] if type(x) == str else pd.NA)

        return_dict = dict()
        type_to_table = {"blk_grp": self.LookUpTables.pr_block_given_race,
                         "tract": self.LookUpTables.pr_tract_given_race,
                         "zip": self.LookUpTables.pr_zip_given_race}

        # map name of the column in the input data which contains the relevant geo tag to
        # the name of the column in the lookup table
        # we are hardcoding in only supporting blk grp, tract, and zip
        geo_column_names = {"tract": "tract",
                            "blk_grp": "blk_grp",
                            "zip": zip_column[0]

                            }

        # loop over all specified lookup types
        # single function that takes in a col, merges on that col, then performs BISG
        for lookup_type in lookup_types:
            # we should never throw this error.
            assert type_to_table[lookup_type] is not None, f"""Prediction on geolocation type {lookup_type} was requested, but no {lookup_type} table was provided in construction of BISG. 
                                                               \nPlease construct a new BISG object with all required tables, or use a prediction type that uses only tables given to the BISG object."""

            return_dict[lookup_type] = self._typed_predict(input_data_with_names=input_data_with_names[[row_id_column, geo_column_names[lookup_type]]+[f'pr_{race}_given_surname' for race in self.race_list]],
                                                           row_id_column=row_id_column,
                                                           lookup_table=type_to_table[lookup_type],
                                                           lookup_type=lookup_type,
                                                           geo_column_name=geo_column_names[lookup_type],
                                                           only_final_probs=only_final_probs)
        return return_dict


#############################
        # USER NEEDS TO SPECIFY IF THEY WANT TO USE BLKGRP
#######################
        raise Exception("read this")
        block_group = self._typed_predict(input_data_with_names=input_data_with_names,
                                          lookup_table=self.LookUpTables.pr_block_given_race, lookup_type="blkgrp", GeoIndNames=GeoIndNames)
        tract_group = self._typed_predict(input_data_with_names=input_data_with_names,
                                          lookup_table=self.LookUpTables.pr_tract_given_race, lookup_type="tract", GeoIndNames=GeoIndNames)
        zip_group = self._typed_predict(input_data_with_names=input_data_with_names,
                                        lookup_table=self.LookUpTables.pr_zip_given_race, lookup_type="zip", GeoIndNames=GeoIndNames)

        # Merge Block and TRact files together on rownum
        merged_df1 = pd.merge(block_group, tract_group,
                              on="rownum", how="left")
        # Merge file with both block and tract files with zip on rownum
        merged_df2 = pd.merge(merged_df1, zip_group, on="rownum", how="left")
        # Merge in file containing geocodes (fictitious data sample) and keep the variable containing the geography precision code.
        # keep only the rownum and precision code from the merged file from fictitious data sample
        bisg = pd.merge(merged_df2, input_data_with_names[[
                        row_id_column, 'geo_code_precision']], on="rownum", how="left")

        # Create Final BISG Proxy
        # For records geocoded to the street ("USAStreetName"), 9-digit ZIP code ("USAZIP4"), and 5-digit ZIP code ("USAZipcode"), use 5-digit ZIP code demographics.
        # For records geocoded to the rooftop ("USAStreetAddr"), use first available: block group, tract, or 5-digit ZIP code demographics.

        # The pr_ variables are the final BISG probabilities:

        # white - non-Hispanic White
        # black - non-Hispanic Black
        # hispanic - Hispanic
        # api - Asian/Pacific Islander
        # aian - American Indian/Alaska Native
        # mult_other - Multiracial/Other

        # blkgrp18_pr_ is the BISG probability based on block group level demographics
        # tract18_pr_ is the BISG probability based on tract level demographics
        # zip18_pr_ is the BISG probabilitiy based on 5-digit ZIP code level demographics

        for g in ['blkgrp18', 'tract18', 'zip18']:
            bisg['sum_'+g+'_pr'] = bisg[[x for x in bisg.columns if g +
                                         '_pr_' in x]].sum(axis=1)

        bisg['pr_precision'] = np.nan
        bisg['pr_precision'] = np.where((bisg['sum_zip18_pr'].notna()) & (bisg['sum_zip18_pr'] > 0) & (bisg['pr_precision'].isna()) & (
            bisg['geo_code_precision'].isin(['USAStreetName', 'USAZIP4', 'USAZipcode'])), 2,  bisg['pr_precision'])
        bisg['pr_precision'] = np.where((bisg['sum_blkgrp18_pr'].notna()) & (bisg['sum_blkgrp18_pr'] > 0) & (
            bisg['pr_precision'].isna()) & (bisg['geo_code_precision'].isin(['USAStreetAddr'])), 3,  bisg['pr_precision'])
        bisg['pr_precision'] = np.where((bisg['sum_tract18_pr'].notna()) & (bisg['sum_tract18_pr'] > 0) & (
            bisg['pr_precision'].isna()) & (bisg['geo_code_precision'].isin(['USAStreetAddr'])), 4,  bisg['pr_precision'])
        bisg['pr_precision'] = np.where((bisg['sum_zip18_pr'].notna()) & (bisg['sum_zip18_pr'] > 0) & (
            bisg['pr_precision'].isna()) & (bisg['geo_code_precision'].isin(['USAStreetAddr'])), 5,  bisg['pr_precision'])
        bisg['pr_precision'] = np.where(
            bisg['pr_precision'].isna(), 1,  bisg['pr_precision'])

        # Question: This doesn't return an error right now, becuase it is true. but my code would break if there was a missing error -- so how do i prevent it from breaking
        # and just using the assert as a check?
        assert bisg['pr_precision'].isnull().sum() == 0

        # Dictionary to map assigned values above ot meanings in BISG
        prob_labels = {1: 'NO FINAL PROB ASSIGNED', 2: 'ZIP (not rooftop lat/long)', 3: 'BLKGRP (has rooftop lat/long)',
                       4: 'TRACT (has rooftop lat/long)', 5: 'ZIP (has rooftop lat/long'}
        bisg['prob_lables'] = bisg['pr_precision'].map(prob_labels)

        # display frequencies
        freq_table = bisg['pr_precision'].value_counts(dropna=False)
        print(freq_table)
        freq_table2 = pd.crosstab(
            bisg['pr_precision'], bisg['geo_code_precision'], margins=True)
        print(freq_table2)

        # choose what probability to use based on geography
        for r in ['white', 'black', 'hispanic', 'api', 'aian', 'mult_other']:
            bisg['pr_'+r] = np.nan
            bisg['pr_'+r] = np.where((bisg['pr_precision'] == 2),
                                     bisg['zip18_pr_'+r], bisg['pr_'+r])
            bisg['pr_'+r] = np.where((bisg['pr_precision'] == 3),
                                     bisg['blkgrp18_pr_'+r], bisg['pr_'+r])
            bisg['pr_'+r] = np.where((bisg['pr_precision'] == 4),
                                     bisg['tract18_pr_'+r], bisg['pr_'+r])
            bisg['pr_'+r] = np.where((bisg['pr_precision'] == 5),
                                     bisg['zip18_pr_'+r], bisg['pr_'+r])

        # check that final probabilities sum to one
        bisg['check_pr'] = bisg[['pr_white', 'pr_black', 'pr_hispanic',
                                 'pr_api', 'pr_aian', 'pr_mult_other']].sum(axis=1)

        # Selects the values in pr_precision where check_pr is 0
        # then it chekcs if all the selectes values in col2 are 0
        assert (bisg.loc[bisg['check_pr'] == 0,  'pr_precision'] == 1).all()

        # final df needs: `matchvars' `geoprecvar' pr_* blkgrp18_pr_* tract18_pr_* zip18_pr_* blkgrp18_geo_pr_* tract18_geo_pr_* zip18_geo_pr_* name_pr*
        # bisg_df0 = bisg[[row_id_column, 'geo_code_precision']]
        # bisg_df1 = bisg[[col for col in bisg if col.startswith('pr')]]
        # bisg_df2 = bisg[[col for col in bisg if col.startswith('blkgrp18_pr_')]]
        # bisg_df3 = bisg[[col for col in bisg if col.startswith('tract18_pr_')]]
        # bisg_df4 = bisg[[col for col in bisg if col.startswith('zip18_pr_')]]
        # bisg_df5 = bisg[[col for col in bisg if col.startswith('blkgrp18_geo_pr_')]]
        # bisg_df6 = bisg[[col for col in bisg if col.startswith('tract18_geo_pr_')]]
        # bisg_df7 = bisg[[col for col in bisg if col.startswith('zip18_geo_pr_')]]
        # bisg_df8 = bisg[[col for col in bisg if col.startswith('name_pr')]]
        # final_bisg = pd.concat([bisg_df0, bisg_df1, bisg_df2, bisg_df3, bisg_df4, bisg_df5, bisg_df6, bisg_df7, bisg_df8], axis = 1)
        final_bisg = bisg[[row_id_column, 'geo_code_precision']+[col for col in bisg if col.startswith(
            ('pr', 'blkgrp18_pr_', 'tract18_pr_', 'zip18_pr_', 'blkgrp18_geo_pr_', 'tract18_geo_pr_', 'zip18_geo_pr_', 'name_pr'))]]
        # print(final_bisg)
        # Export to CSV
        # final_bisg.to_csv(sourcedata + 'test_proxied_final_jan20.csv', index = False)
        return (final_bisg)
