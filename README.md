# RL-for-sepsis
A RL research for sepsis, use new assessment methods
Notice, due to copyright issues, I am unable to directly provide you with data from MIMIC IV. You will need to obtain it on your own. Once you have obtained access to MIMIC IV, you will need to retrieve the following two tables from it:
Extracting data as data_extraxct.pdf from mimic iv.
Table_sofa: Only contains 3 col:hadm_id, timestep and sofa_24hours.
Table_heparin:Only contains 3 col:hadm_id, timestep and u/kg/h(the dose of heparin).
You need to use Preprocessing1 to process Table_sofa, and get the Table_1.
Then, You need to use Preprocessing2 to process Table_1 and Table_heparin, and get the Table_data.
Finally, you only need to run the main.
TD_lambda is the program for our algorithm, it can give the value function.
similarity, action_similarity and relative_gain can give the assessment result, more details for our paper.
