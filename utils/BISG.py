from pathlib import Path

dirname =  Path(__file__).parent.parent

from RacePredictor import RacePredictor
from LookupTables import LookUpTables


class BISG:
    ######################## wrap it all up!
    def __init__(self,
                    census_surname_data, 
                    census_tract_data=None,
                    census_zip_data=None,
                    census_block_group_data=None,
                    debug=False
                    #surname_column=None,
                    
                    ) -> None:
        
        #consider implementing a surname column argument?
        #never in a wprld where these are not these values for this census data
        race_list=['white', 'black', 'api', 'aian', 'mult_other', 'hispanic']
        census_id_columns=['GEOID20_BlkGrp','GEOID20_Tract','ZCTA5']


        self.LookupTables = LookUpTables(census_surname_data,census_tract_data,census_zip_data,census_block_group_data,race_list,census_id_columns)
        self.RacePredictor= RacePredictor(self.LookupTables,debug)
        pass

    def predict(self,input_data=None, name_column:str=None,zip_column:str=None, GEOID_column:str=None, blk_grp:bool=False,only_final_probs:bool=False): 
        return self.RacePredictor.predict(input_data=input_data, 
                                            name_column=name_column,
                                            zip_column=zip_column, 
                                            GEOID_column=GEOID_column, 
                                            blk_grp=blk_grp,
                                            only_final_probs=only_final_probs) 


if __name__ == "__main__":
    print("start")
    myBISG=BISG(
                census_surname_data= [str(dirname.absolute())+'/input_files/Names_2010Census.csv',str(dirname.absolute())+'/input_files/app_c.csv'],
                #surname_data=[str(dirname.absolute())+'/input_files/Names_2010Census.csv'],
                census_block_group_data=str(dirname.absolute())+'/input_files/blkgrp_over18_race_jan20.dta',
                census_tract_data=str(dirname.absolute())+'/input_files/tract_over18_race_jan20.dta',
                census_zip_data=str(dirname.absolute())+'/input_files/zip_over18_race_jan20.dta',
                debug=True
                )

    print(myBISG.predict(str(dirname.absolute())+'/test_output/fictitious_sample_data_str_datatypes.dta',name_column='name1',zip_column='ZCTA5',GEOID_column='GEOID10_BlkGrp',blk_grp=True,only_final_probs=True))
    #print(myBISG.predict(str(dirname.absolute())+'/test_output/fictitious_sample_data_str_datatypes.dta',name_column='name1',GEOID_column='GEOID10_BlkGrp',blk_grp=True,only_final_probs=True))
    #print(myBISG.RacePredictor.internal_dfs)
    print("end")
    print(myBISG.LookupTables.pr_race_given_surname['name'][ (myBISG.LookupTables.pr_race_given_surname['name']=='nan') |(myBISG.LookupTables.pr_race_given_surname['name']=='null')])
    
    