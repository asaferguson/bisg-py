from DataFrameUtils import input_to_dataframe
import pandas as pd
import numpy as np
from pathlib import Path

dirname = Path(__file__).parent.parent


class LookUpTables:
    """
    The LookUpTables class is responsible for creating and storing four specific lookup tables
    that map geographic loction and surname to race probabilities. These tables are:

    1. pr_race_given_surname: Maps surname to race probabilities based on census data.
    2. pr_block_given_race: Maps block group geographic identifier to race probabilities.
    3. pr_tract_given_race: Maps tract geographic identifier to race probabilities.
    4. pr_zip_given_race: Maps ZIP code geographic identifier to race probabilities.

    The class processes census data to generate these tables, and stores them for use by the 
    RacePredictor class. It includes on internal method for updating the tables with new data.
    The reason we store this class and its inputs and not the processed lookup tables is that
    we want to have the assumptions of the data processing stored in the class, and easily 
    accessible to any future user.
    """

    def __init__(self,

                 surname_data,
                 tract_data=None,
                 zip_data=None,
                 block_group_data=None,
                 race_list=['white', 'black', 'api',
                            'aian', '2prace', 'hispanic'],
                 # census_name_column="name",
                 id_columns=['GEOID20_BlkGrp', 'GEOID20_Tract', 'ZCTA5']
                 # missing_behavior=None
                 ) -> None:

       # think about if we also want to add a surname column argument, which would be used to indicate which column the wrangling functions should treat as surname
       # we could do simething similar for GEOID. This would be for adapting this code to more datasets later.

        self.pr_race_given_surname = None
        self.pr_block_given_race = None
        self.pr_tract_given_race = None
        self.pr_zip_given_race = None
        self.race_list = race_list

        self._update_tables(surname_data=surname_data,
                            tract_data=tract_data,
                            zip_data=zip_data,
                            block_group_data=block_group_data,
                            # surname_column=surname_column
                            id_columns=id_columns)

        pass

    # keeping this as its own method in case we want to have a rewrite tables method later
    def _update_tables(self,
                       surname_data,
                       tract_data,
                       zip_data,
                       block_group_data,
                       # surname_column=None,
                       id_columns=['GEOID20_BlkGrp', 'GEOID20_Tract', 'ZCTA5']):

        # run primary table wrangling and writing functions
        self.pr_race_given_surname = self._census_surnames(surname_data)
        self.pr_block_given_race, self.pr_tract_given_race, self.pr_zip_given_race = self._census_geoids(
            tract_data,
            zip_data,
            block_group_data,
            id_columns
        )
        pass

    # convert census geo information to lookup tables
    # robust to passing one, two, or all three tables
    # this function is entirely specific to the census data format we currently have, and would basically need to be rebuilt for anything else
    def _census_geoids(self, tract_data: pd.DataFrame = None, zip_data: pd.DataFrame = None, block_group_data: pd.DataFrame = None, id_columns=['GEOID20_BlkGrp', 'GEOID20_Tract', 'ZCTA5']) -> list:

        # we do this to allow for partial census data in the future
        # make sure we have at least one dataset
        list_of_datasets = [block_group_data, tract_data, zip_data]
        if [1 if dft is None else 0 for dft in list_of_datasets] == [1, 1, 1]:
            raise Exception("No Census Data Inputted")

        files = ['blkgrp', 'tract', 'zip']
        final_dfs = []
        for i in range(3):
            file = files[i]

            if list_of_datasets[i] is None:
                final_dfs.append(None)
                continue

            df = input_to_dataframe(list_of_datasets[i], id_columns)

            # Step 1: From the SF1, retain population counts for the contiguous U.S., Alaska, and Hawaii in order to ensure consistency with the population covered by the census surname list.
            df = df.drop(df[df['State_FIPS20'] == '72'].index)
            if file == 'zip':
                df = df.drop(df[df['ZCTA5'].str.startswith(
                    ('006', '007', '008', '009'))].index)

            # Step 2: Address "Other" category from 2010 Census; what is done here follows Word(2008).
            # combine with the identified race
            for x in ['NH_White', 'NH_Black', 'NH_AIAN', 'NH_API']:
                df[x+'_alone'] = df[x+'_alone'] + df[x+'_Other']

            # Census breaks out Asian and PI separately; since we consider them as one, we correct for this.
            df['NH_API_alone'] = df['NH_API_alone'] + \
                df['NH_Asian_HPI']+df['NH_Asian_HPI_Other']

            # * Replace multiracial total to account for the fact that we have suppressed the Other category.
            for x in ['NH_White_Other', 'NH_Black_Other', 'NH_AIAN_Other', 'NH_Asian_HPI', 'NH_API_Other', 'NH_Asian_HPI_Other']:
                df['NH_Mult_Total'] = df['NH_Mult_Total'] - df[x]

            # * Verify the steps above by confirming that the Total Population still matches.
            assert list(df['Total_Pop']) == list(df[['NH_White_alone', 'NH_Black_alone', 'NH_API_alone',
                                                     'NH_AIAN_alone', 'NH_Mult_Total', 'NH_Other_alone', 'Hispanic_Total']].sum(axis=1).astype(int))

            # Step 3: Proportionally redistribute Non-Hispanic Other population to remaining Non-Hispanic groups within each block.
            for x in ['NH_White_alone', 'NH_Black_alone', 'NH_API_alone', 'NH_AIAN_alone', 'NH_Mult_Total']:
                df[x] = df[x] + (df[x]/(df["Total_Pop"] - df["Hispanic_Total"] -
                                 df["NH_Other_alone"])) * df["NH_Other_alone"]

                # cover your bases in the div by 0 case
                df[x] = np.where(df["Total_Pop"] == 0, 0, df[x])
                df[x] = np.where(df["Non_Hispanic_Total"] ==
                                 df["NH_Other_alone"], df["NH_Other_alone"]/5, df[x])

            # * Verify the loop above by confirming that the Total Population still matches
            # Numbers are the same but assert is false I think becuase the type is different -- I am trying to change it by astype()
            df['pop_check'] = df[["NH_White_alone", "NH_Black_alone", "NH_AIAN_alone",
                                  "NH_API_alone", "NH_Mult_Total", "Hispanic_Total"]].sum(axis=1)
            # assert df['Total_Pop'].equals(df['pop_check'])
            # print(list(df['Total_Pop'].astype(float).round(2))[:3], list( df['pop_check'].astype(float).round(2))[:3])
            assert list(df['Total_Pop'].astype(float).round(2)) == list(
                df['pop_check'].astype(float).round(2))

            # * Collapse dataset to get Population Totals for each group.
            # store as a dictionary -- A dictionary is a collection which is ordered*, changeable and do not allow duplicates.
            GROUP3 = ["NH_White_alone", "NH_Black_alone", "NH_AIAN_alone",
                      "NH_API_alone", "NH_Mult_Total", "Hispanic_Total", "Total_Pop"]
            pop_sum = {'national_'+x: df[x].sum() for x in GROUP3}

            pop_sum["national_NH_Hawn_alone"] = 0  # Why????

            # Generates percentage dividing all the races
            GROUP4 = ["white", "black", "aian", "api", "mult_other",
                      "hispanic"]  # These are the final races

            for x, y in zip(GROUP4, GROUP3[:6]):
                df["pr_" + x +
                    "_given_geo"] = df[y].astype(float)/df["Total_Pop"].astype(float)

            # * When updating geocoded race probabilities, we require the probability that someone of a particular race lives in that block group, tract, or ZIP code. Our race counts are single race
            # reported counts, therefore we divide the single race population within each block by the total single race population for each group.

            GROUP5 = ["NH_White_alone", "NH_Black_alone", "NH_AIAN_alone",
                      "NH_Mult_Total", "Hispanic_Total", "NH_API_alone", "NH_Hawn_alone"]
            for x in GROUP5:
                # "Number of other-race or multiple-race non-Hispanics: `national_nh_mult_other'"
                pop_sum["national_NH_Mult_other"] = pop_sum["national_Total_Pop"] - \
                    pop_sum["national_" + x]

            # Dividing by national populations
            # p(geoID | race) = n_{geoID and race}/n_race_total
            # list(zip(GROUP4,GROUP3[:6]))
            for x, y in zip(GROUP4, GROUP3[:6]):
                # dividing vector by scalar here
                df["pr_geo_given_" +
                    x] = df[y].astype(float)/pop_sum["national_"+y]

            # p(geoID)=n_geoID/n_total
            df['pr_geo'] = df["Total_Pop"]/pop_sum['national_Total_Pop']

            # Renaming Columns before save
            # This is broken and does not rename!
            if file == "blkgrp":
                df = df.rename(columns={'GEOID20_BlkGrp': 'GeoInd'})

            if file == "tract":
                df = df.rename(columns={'GEOID20_Tract': 'GeoInd'})

            if file == "zip":
                df = df.rename(columns={'ZCTA5': 'GeoInd'})

            # Identifying which columns to keep in the output
            # [col for col in df if col.startswith('geo_pr')]
            geo_col = [f"pr_{x}_given_geo" for x in self.race_list]
            # [col for col in df if col.startswith('here')]
            here_col = [f"pr_geo_given_{x}" for x in self.race_list]

            # Takes one list and adds on the other created from the list comprehension abov
            df_output = df[['GeoInd']+geo_col+here_col].copy()  # why copy?

            # convert key to string
            df_output['GeoInd'] = df_output['GeoInd'].astype(str)

            final_dfs = final_dfs+[df_output]

        return final_dfs

    # convert a list of census surname information to a single lookup table
    # this function is entirely specific to the census data format we currently have, and would basically need to be rebuilt for anything else
    def _census_surnames(self, surname_data: pd.DataFrame) -> pd.DataFrame:
        census_race_list = ['white', 'black',
                            'api', 'aian', '2prace', 'hispanic']
        # start by creating the full dataframe of all surname census data from the list of dataframes passed
        # (this is robust to only passing one)

        # Part 1: Combining all dataframes passed into one
        if not isinstance(surname_data, (list, tuple)):
            full_surname_dataframe = input_to_dataframe(surname_data)
        elif len(surname_data) == 1:
            full_surname_dataframe = input_to_dataframe(surname_data[0])
        else:
            surname_data = [input_to_dataframe(
                dataframe) for dataframe in surname_data]

            #######################################################################################
            #
            # (this is where we could add more cleaning)
            #     We would have to add functionality to merge rows that have the same name after cleaning
            #
            #######################################################################################

            for dataframe in surname_data:
                # Strip white space around the name and make the first letter Uppercase
                # Why do we capitalize if we are just going to lowercase later?
                dataframe["name"] = dataframe["name"].str.capitalize().str.strip()

            # concatenate the dataframes in order of priority, only keeping the dupe names from highest priority DataFrames
            full_surname_dataframe = surname_data[0]
            for lower_priority_dataframe in surname_data[1:]:
                full_surname_dataframe = pd.concat(
                    [full_surname_dataframe, lower_priority_dataframe[~lower_priority_dataframe["name"].isin(full_surname_dataframe["name"])]])

        # very specific to our given data
        assert pd.DataFrame({"Nessary_Columns": ["name", "count"]+["pct"+race for race in census_race_list]}).isin(list(
            full_surname_dataframe.columns)).all().iloc[0], "Surnames Data Passed Does not Contain All the Necessary Columns"

        # part 2: turn our dataframe into lookup table using helper functions or just a procedural data pipeline

        # add in the dist remaining percentages function here
        # full_surname_dataframe = self._percent_to_proportion(full_surname_dataframe)
        # could be abstracted as this function above
        for x in ["pct"+race for race in census_race_list]:
            full_surname_dataframe[x] = pd.to_numeric(
                full_surname_dataframe[x], errors='coerce')
            full_surname_dataframe[x] = full_surname_dataframe[x]/100
        ###########

        # calculate a field for how many of the race/eth percentages are missing

        # full_surname_dataframe = self._distribute_remaining_percentages(full_surname_dataframe)
        # could be abstracted as this function above
        race_list_pct = ["pct"+race for race in census_race_list]
        full_surname_dataframe.replace('(S)', np.nan, inplace=True)

        # add a field for how many of the race/eth percentages are missing
        full_surname_dataframe["countmiss"] = full_surname_dataframe[race_list_pct].isna(
        ).sum(axis=1)

        # calculate a field that sums the percentages of all races for each name
        full_surname_dataframe["remaining"] = full_surname_dataframe[race_list_pct].sum(
            axis=1)

        # replace the remaining values with the following funciton from stata: replace remaining = count * (1 - remaining)
        # this is a count of the number of individuals not yet assigned
        full_surname_dataframe["remaining"] = full_surname_dataframe["count"] * (
            1 - full_surname_dataframe["remaining"])

        # replace the percentgae with remaining over the
        # divide proportions equally (1/n_missing)*(total_remaining_proportion)=(1/n_missing)*(n_remaining/n_total)

        for x in race_list_pct:
            full_surname_dataframe[x] = np.where(full_surname_dataframe[x].isna(),
                                                 (full_surname_dataframe["remaining"]/(
                                                     (full_surname_dataframe["countmiss"]) * full_surname_dataframe["count"])),
                                                 full_surname_dataframe[x])

        full_surname_dataframe.drop(
            columns=["remaining", "countmiss"], inplace=True)
        ##################################

        # Make all names lowercase
        full_surname_dataframe["name"] = full_surname_dataframe["name"].str.lower(
        )

        # rename columns to final versions (kinda messy but we will only ever run this once per instance)
        races_to_final = {race: race for race in census_race_list}
        races_to_final['2prace'] = 'mult_other'

        # Do some renaming to final vals
        full_surname_dataframe.rename(columns={
                                      f'pct{race}': f'pr_{races_to_final[race]}_given_surname' for race in census_race_list}, inplace=True)

        return full_surname_dataframe

    # Defunct functions that lead to extra abstraction and make the pipeline hard to read imo
    # we can go back to abstracting if we must

    def _percent_to_proportion(self,
                               full_surname_dataframe,
                               race_list=None  # ,prefix='pct'
                               ):
        if race_list is None:
            race_list = self.race_list
        for x in ["pct"+race for race in race_list]:
            full_surname_dataframe[x] = pd.to_numeric(
                full_surname_dataframe[x], errors='coerce')
            full_surname_dataframe[x] = full_surname_dataframe[x]/100

        return full_surname_dataframe

    # Input: Data Frame that has race percentages from the census
    # Output: Data Frame that has race percentages from the census that are completely filled out
    # Function: For each row, the funciton takes the the missing percentage from 100 and divides it among the missing race categories
    # why is len(df) relevant? should it be taken out
    def _distribute_remaining_percentages(self, df: pd.DataFrame, race_list=None):
        if race_list is None:
            race_list_pct = ["pct"+race for race in self.race_list]

        # Change (S) values into Nan Values
        df.replace('(S)', np.nan, inplace=True)

        # add a field for how many of the race/eth percentages are missing
        df["countmiss"] = df[race_list_pct].isna().sum(axis=1)

        # calculate a field that sums the percentages of all races for each name
        df["remaining"] = df[race_list_pct].sum(axis=1)

        # replace the remaining values with the following funciton from stata: replace remaining = count * (1 - remaining)
        # this is a count of the number of individuals not yet assigned
        df["remaining"] = df["count"] * (1 - df["remaining"])

        # replace the percentgae with remaining over the
        # divide proportions equally (1/n_missing)*(total_remaining_proportion)=(1/n_missing)*(n_remaining/n_total)
    # print(race_list)
        for x in race_list_pct:
            df[x] = np.where(df[x].isna(), (df["remaining"] /
                             ((df["countmiss"]) * df["count"])), df[x])

        df.drop(columns=["remaining", "countmiss"], inplace=True)

        return df


# x=LookUpTables(surname_data= [str(dirname.absolute())+'/input_files/Names_2010Census.csv',str(dirname.absolute())+'/input_files/app_c.csv'],
# block_group_data=str(dirname.absolute())+'/input_files/blkgrp_over18_race_jan20.dta',
#  tract_data=str(dirname.absolute())+'/input_files/tract_over18_race_jan20.dta',
#  zip_data=str(dirname.absolute())+'/input_files/zip_over18_race_jan20.dta')
# for item in x.__dict__:
#     print(item, type(x.__dict__[item]))
