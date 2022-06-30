# Prescriptive-Analytics

## Installation
### Create conda environment
conda env create -f env.yml -p ./env \
conda activate ./env 

### Run the code
Write your command line script inside run.sh, then run it: \
./run.sh

### Parameters
Mandatory parameters: \
--filename_completed --> log file containing all completed cases \
--case_id_name, activity_name, start_date_name --> self explanatory, name of the columns containing the specified information \
--pred_column --> kpi to be predicted (remaining_time | lead_time | total_cost | independent_activity) 

Optional parameters: \
--end_date_name \
--resource_name, role_name \
--predict_activities --> name of the activity to be predcited (mandatory if pred_column is independent_activity) \
--experiment_name --> folder in which results will be saved at the end of the experiment \
--override (default True) --> deletes temporary files from previous run \
--shap (default False) --> if you want to calculate also the explanations of predictions \
--explain (default False) --> if you want to calculate the explanations of recommendations
--outlier_thrs (default 0.001) --> if you want to set the threshold for outlier's frequence

## Code Example (run it from shell, if you have a Windows machine, remove the "\" and run the complete command )

## Recommendations and explanations for VINST case study on total time
python main_recsys.py --filename_completed 'data/VINST cases incidents.csv' --case_id_name SR_Number --activity_name ACTIVITY --start_date_name Change_Date+Time \
--pred_column lead_time --resource_name Involved_ST --role_name Involved_Org_line_3 --experiment_name exp_time_VINST --explain True

### Recommendations and explanations for BAC case study on Activity "Pending Liquidation Request"
python main_recsys.py --filename_completed data/completed.csv --case_id_name REQUEST_ID --activity_name ACTIVITY --start_date_name START_DATE \
--resource_name CE_UO --role_name ROLE --end_date_name END_DATE --pred_column independent_activity --predict_activities "Pending Liquidation Request" \
--experiment_name prova_activity_BAC_PLR --explain True

## Contributors
Riccardo Galanti
Alessandro Padella
