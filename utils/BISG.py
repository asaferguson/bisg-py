from LookupTables import LookUpTables
from RacePredictor import RacePredictor
from pathlib import Path

dirname = Path(__file__).parent.parent


class BISG:
    """
    The BISG (Bayesian Improved Surname Geocoding) class is the main class that orchestrates
    the process of predicting race probabilities based on surname and geographic location.
    It initializes the necessary lookup tables and race predictor components using census data.

    The class uses the following components:
    - LookUpTables: Creates and manages lookup tables that map geographic and surname data to race probabilities.
    - RacePredictor: Uses the lookup tables to predict race probabilities for individuals based on their surname and geographic location.

    The BISG class provides a predict method that accepts input data along with the columns for geographic identifiers and surname,
    and then uses the RacePredictor to perform the BISG method to infer a vector of posterior race probabilities.
    The predictions can be made at different geographic levels: block group, tract, and ZIP code, and the results can be customized to include
    only the final probabilities or intermediate values for debugging.
    """

    def __init__(self,

                 # Does not need inputs as we only ever going to use the 2020 census data for our MVP

                 # census_surname_data,
                 # census_tract_data=None,
                 # census_zip_data=None,
                 # census_block_group_data=None,
                 debug=False


                 ) -> None:

        # Set parameters to be passed to the LookupTables class
        # Thse are the source census data files
        census_surname_data = [
            str(dirname.absolute()) + '/input_files/Names_2010Census.csv',
            str(dirname.absolute()) + '/input_files/app_c.csv'
        ]
        census_block_group_data = str(
            dirname.absolute())+'/input_files/blkgrp_over_18_race_jan20.csv'
        census_tract_data = str(dirname.absolute()) + \
            '/input_files/tract_over_18_race_jan20.csv'
        census_zip_data = str(dirname.absolute()) + \
            '/input_files/zip_over_18_race_jan20.csv'
        race_list = ['white', 'black', 'api', 'aian', 'mult_other', 'hispanic']

        # These are the columns which identify the geographic level of the input data
        census_id_columns = ['GEOID20_BlkGrp', 'GEOID20_Tract', 'ZCTA5']

        self.LookupTables = LookUpTables(
            census_surname_data,
            census_tract_data,
            census_zip_data,
            census_block_group_data,
            race_list,
            census_id_columns
        )
        self.RacePredictor = RacePredictor(
            self.LookupTables,
            debug
        )

    def predict(
        self,
        input_data=None,
        name_column: str = None,
        zip_column: str = None,
        GEOID_column: str = None,
        blk_grp: bool = False,
        only_final_probs: bool = False
    ):
        return self.RacePredictor.predict(
            input_data=input_data,
            name_column=name_column,
            zip_column=zip_column,
            GEOID_column=GEOID_column,
            blk_grp=blk_grp,
            only_final_probs=only_final_probs)


if __name__ == "__main__":
    print("start")
    myBISG = BISG(debug=True)

    print(
        myBISG.predict(
            str(dirname.absolute()) +
            '/test_files/fictitious_sample_data_str_datatypes.dta',
            name_column='name1',
            zip_column='ZCTA5',
            GEOID_column='GEOID10_BlkGrp',
            blk_grp=True,
            only_final_probs=True
        )
    )
    print("end")
    # print(
    #     myBISG.LookupTables.pr_race_given_surname['name'][
    #         (myBISG.LookupTables.pr_race_given_surname['name'] == 'nan') |
    #         (myBISG.LookupTables.pr_race_given_surname['name'] == 'null')
    #     ]
    # )
