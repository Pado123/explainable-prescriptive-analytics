"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""
import os.path
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import tqdm
import pickle
import seaborn as sns

import matplotlib.pyplot as plt

import shap
from catboost import Pool
from catboost._catboost import CatBoostError

from explainable import find_explanations_for_running_cases
from write_results import prepare_csv_results, histogram_median_events_per_dataset
from ml import prepare_data_for_ml_model_and_predict
from logme import log_it

# REFACTOR
from IO import read, write, folders
from os.path import join, exists


def calculateTimeFromMidnight(actual_datetime):
    midnight = actual_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = (actual_datetime - midnight).total_seconds()
    return timesincemidnight


def createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date):
    activityTimestamp = line[1]
    activity = []
    activity.append(caseID)
    for feature in line[2:]:
        activity.append(feature)

    # We get timestamps in ms
    activityTimestamp = datetime.fromtimestamp(int(activityTimestamp) / 1_000)
    starttime = datetime.fromtimestamp(int(starttime) / 1_000)
    lastevtime = datetime.fromtimestamp(int(lastevtime) / 1_000)

    # add features: time from trace start, time from last_startdate_event, time from midnight, weekday
    activity.append((activityTimestamp - starttime).total_seconds())
    activity.append((activityTimestamp - lastevtime).total_seconds())
    activity.append(calculateTimeFromMidnight(activityTimestamp))
    activity.append(activityTimestamp.weekday())
    # if there is also end_date add features time from last_enddate_event and activity_duration.
    # hotfix: sometimes "missing" comes from myinvenio
    if current_activity_end_date is not None:
        current_activity_end_date = datetime.fromtimestamp(int(current_activity_end_date) / 1_000)
        activity.append((current_activity_end_date - activityTimestamp).total_seconds())
        # add timestamp end or start to calculate remaining time later
        activity.append(current_activity_end_date)
    else:
        activity.append(activityTimestamp)
    return activity


def move_essential_columns(df, case_id_name, start_date_name):
    columns = df.columns.to_list()
    # move case_id column and start_date column to always know their position
    columns.pop(columns.index(case_id_name))
    columns.pop(columns.index(start_date_name))
    df = df[[case_id_name, start_date_name] + columns]
    return df


def convert_strings_to_datetime(df, date_format):
    # convert string columns that contain datetime to datetime
    for column in df.columns:
        try:
            # if a number do nothing
            if np.issubdtype(df[column], np.number):
                continue
            df[column] = pd.to_datetime(df[column], format=date_format)
        # exception means it is really a string
        except (ValueError, TypeError, OverflowError):
            pass
    return df


def find_case_finish_time(trace, num_activities):
    # we find the max finishtime for the actual case
    for i in range(num_activities):
        if i == 0:
            finishtime = trace[-i - 1][-1]
        else:
            if trace[-i - 1][-1] > finishtime:
                finishtime = trace[-i - 1][-1]
    return finishtime


def calculate_remaining_time_for_actual_case(traces, num_activities):
    finishtime = find_case_finish_time(traces, num_activities)
    for i in range(num_activities):
        # calculate remaining time to finish the case for every activity in the actual case
        traces[-(i + 1)][-1] = (finishtime - traces[-(i + 1)][-1]).total_seconds()
    return traces


def fill_missing_end_dates(df, start_date_position, end_date_position):
    df[df.columns[end_date_position]] = df.apply(lambda row: row[start_date_position]
    if row[end_date_position] == 0 else row[end_date_position], axis=1)
    return df


def convert_datetime_columns_to_seconds(df):
    for column in df.columns:
        try:
            if not np.issubdtype(df[column], np.number):
                df[column] = pd.to_datetime(df[column])
                df[column] = (df[column] - pd.to_datetime('1970-01-01 00:00:00')).dt.total_seconds()
        except (ValueError, TypeError, OverflowError):
            pass
    return df


@log_it
def add_features(df, end_date_name):
    dataset = df.values
    if end_date_name is not None:
        end_date_position = df.columns.to_list().index(end_date_name)
    else:
        end_date_position = None
    traces = []
    # analyze first dataset line
    caseID = dataset[0][0]
    activityTimestamp = dataset[0][1]
    starttime = activityTimestamp
    lastevtime = activityTimestamp
    current_activity_end_date = None
    line = dataset[0]
    if end_date_position is not None:
        # at the begin previous and current end time are the same
        current_activity_end_date = dataset[0][end_date_position]
        line = np.delete(line, end_date_position)
    num_activities = 1
    activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)
    traces.append(activity)
    for line in dataset[1:, :]:
        case = line[0]
        if case == caseID:
            # continues the current case
            activityTimestamp = line[1]
            if end_date_position is not None:
                current_activity_end_date = line[end_date_position]
                line = np.delete(line, end_date_position)
            activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)

            # lasteventtimes become the actual
            lastevtime = activityTimestamp
            traces.append(activity)
            num_activities += 1
        else:
            caseID = case
            traces = calculate_remaining_time_for_actual_case(traces, num_activities)

            activityTimestamp = line[1]
            starttime = activityTimestamp
            lastevtime = activityTimestamp
            if end_date_position is not None:
                current_activity_end_date = line[end_date_position]
                line = np.delete(line, end_date_position)
            activity = createActivityFeatures(line, starttime, lastevtime, caseID, current_activity_end_date)
            traces.append(activity)
            num_activities = 1

    # last case
    traces = calculate_remaining_time_for_actual_case(traces, num_activities)
    # construct df again with new features
    columns = df.columns
    if end_date_position is not None:
        columns = columns.delete(end_date_position)
    columns = columns.delete(1)
    columns = columns.to_list()
    if end_date_position is not None:
        # columns.extend(["time_from_previous_event(start)", "time_from_midnight",
        #                 "weekday", "activity_duration", "remaining_time"])
        columns.extend(["time_from_start", "time_from_previous_event(start)", "time_from_midnight",
                        "weekday", "activity_duration", "remaining_time"])
    else:
        # columns.extend(["time_from_previous_event(start)", "time_from_midnight",
        #                 "weekday", "remaining_time"])
        columns.extend(["time_from_start", "time_from_previous_event(start)", "time_from_midnight",
                        "weekday", "remaining_time"])
    df = pd.DataFrame(traces, columns=columns)
    print("Features added")
    return df


def pad_columns_in_real_data(df, case_ids):
    # fill real test df with empty missing columns (one-hot encoding can generate much more columns in train)
    # IO
    train_columns = read(folders['model']['data_info'])['columns']
    # if some columns never seen in train set, now we just drop them
    # we should retrain the model with also this new columns (that will be 0 in the original train)
    columns_not_in_test = [x for x in train_columns if x not in df.columns]
    df2 = pd.DataFrame(columns=columns_not_in_test)
    # enrich test df with missing columns seen in train
    df = pd.concat([df, df2], axis=1)
    df = df.fillna(0)
    # reorder data as in train
    df = df[train_columns]
    # add again the case id column
    df = pd.concat([case_ids, df], axis=1)
    return df


def sort_df(df, case_id_name, start_date_name):
    df.sort_values([case_id_name, start_date_name], axis=0, ascending=True, inplace=True, kind='quicksort',
                   na_position='last')
    return df


def fillna(df):
    for i, column in enumerate(df.columns):
        if df[column].dtype == 'object':
            df[column] = df[column].fillna("missing")
    return df


@log_it
def prepare_data_and_add_features(df, case_id_name, start_date_name, date_format, end_date_name):
    # df = fillna(df, date_format)
    if end_date_name is not None:
        df[end_date_name] = df[end_date_name].fillna(df[start_date_name])
        # df = fill_missing_end_dates(df, start_date_position, end_date_position)
    # df = convert_strings_to_datetime(df, date_format)
    df = fillna(df)
    df = move_essential_columns(df, case_id_name, start_date_name)
    df = sort_df(df, case_id_name, start_date_name)
    df = add_features(df, end_date_name)
    df["weekday"].replace({0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                           4: "Friday", 5: "Saturday", 6: "Sunday"}, inplace=True)
    return df


def new_case_level_attribute_detection(df, case_id_name, mode):
    # needed for explanations later
    if mode == "train":
        if exists(folders['model']['data_info']):
            info = read(folders['model']['data_info'])
            info["case_level_attributes"] = []
        else:
            info = {"case_level_attributes": []}
        case_level_attributes = []
        for column in df.columns[1:]:
            # if there is at least 1 False in the series it is an event-level attribute
            if False not in df.groupby(case_id_name)[column].nunique().eq(1).values:
                case_level_attributes.append(column)
                info["case_level_attributes"].append(column)
        write(info, folders['model']['data_info'])
    else:
        case_level_attributes = read(folders['model']['data_info'])["case_level_attributes"]
    return case_level_attributes


def detect_case_level_attribute(df, pred_column):
    # take some sample cases and understand if the attribute to be predicted is event-level or case-level
    # we assume that the column is numeric or a string
    event_level = 0
    case_ids = df[df.columns[0]].unique()
    case_ids = case_ids[:int((len(case_ids) / 100))]
    for case in case_ids:
        df_reduced = df[df[df.columns[0]] == case]
        if len(df_reduced[pred_column].unique()) == 1:
            continue
        else:
            event_level = 1
            break
    return event_level


def bin_numeric(df, mode):
    if mode == 'train':
        columns = df.columns[1:-1]
    else:
        columns = df.columns[1:]
    labs = ['very low', 'low', 'high', 'very_high']
    for column in columns:
        if (df[column].dtype != 'object') and (column != "time_from_midnight") and (df[column].notna().any()):
            df[column] = (pd.cut(df[column], 4, labels=labs)
                          .cat
                          .add_categories('missing')
                          .fillna('missing'))
    return df


def bin_daytime(time_from_midnight):
    time_from_midnight = (time_from_midnight / (3600)).round(0)
    labs = ['night', 'morning', 'afternoon', 'evening']
    return (pd.cut(time_from_midnight, [0, 8, 14, 20, 24], labels=labs)
            .cat
            .add_categories('missing')
            .fillna('missing'))


def bin_features(df, mode):
    # df = bin_numeric(df, mode)
    # df['time_from_midnight'] = bin_daytime(df['time_from_midnight'])
    df.columns = df.columns.str.replace('time_from_midnight', 'daytime')
    return df


@log_it
def calculate_costs(df, costs, working_times, activity_column_name, resource_column_name, role_column_name,
                    case_id_name):
    """
    cost is activity cost + resource or role cost(hour)*working time

    resource is present:
    activity_cost + resource_cost (if present) * working_time
    only role cost:
    activity_cost + role_cost (if present) * working_time
    no specific resource or role cost:
    activity_cost + default_resource (if present, otherwise default_role) * working_time

    Note that in MyInvenio, costs can vary for different periods of time, but this is not currently implemented here.

    """

    # preallocate case cost column
    import ipdb;
    ipdb.set_trace()
    df["case_cost"] = 0
    activities = df[activity_column_name].unique()
    roles = df[role_column_name].unique()
    resources = df[resource_column_name].unique()
    # assign role cost* working time since we don't have resource cost
    for resource in costs["resourceCost"].keys():
        if resource != "__DEFAULT__":
            df.loc[df[resource_column_name] == resource, "case_cost"] = costs["resourceCost"][resource]
    for role in costs["roleCost"].keys():
        if role != "__DEFAULT__":
            df.loc[(df[role_column_name] == role) & (df["case_cost"] == 0), "case_cost"] = costs["roleCost"][role]
    if "__DEFAULT__" in costs["resourceCost"]:
        df.loc[df["case_cost"] == 0, "case_cost"] = costs["resourceCost"]["__DEFAULT__"]
    elif "__DEFAULT__" in costs["roleCost"]:
        df.loc[df["case_cost"] == 0, "case_cost"] = costs["roleCost"]["__DEFAULT__"]

    # multiply by working time and then sum the cost for the activity
    for activity in activities:
        if activity in working_times:
            df.loc[df[activity_column_name] == activity, "case_cost"] *= working_times[activity]
        elif "__DEFAULT__" in working_times:
            df.loc[df[activity_column_name] == activity, "case_cost"] *= working_times["__DEFAULT__"]

        if activity in costs["activityCost"]:
            df.loc[df[activity_column_name] == activity, "case_cost"] += costs["activityCost"][activity]
        elif "__DEFAULT__" in costs["activityCost"]:
            df.loc[df[activity_column_name] == activity, "case_cost"] += costs["activityCost"]["__DEFAULT__"]

    # at this point you sum the cost for previous events of the case (is a case cost, not event cost)
    df["case_cost"] = df.groupby(case_id_name)["case_cost"].cumsum()
    import ipdb;
    ipdb.set_trace()
    return df


def write_leadtime_reference_mean(df, case_id_name, start_date_name, end_date_name):
    # avg of all the completed cases to be passed as a reference value
    if end_date_name is not None:
        avg_duration_days = (df.groupby(case_id_name)[end_date_name].max() -
                             df.groupby(case_id_name)[start_date_name].min()).mean()
    else:
        avg_duration_days = (df.groupby(case_id_name)[start_date_name].max() -
                             df.groupby(case_id_name)[start_date_name].min()).mean()
    mean = {'completedMean': round(avg_duration_days, 2)}
    print(f'"Average completed lead time (days): {mean["completedMean"] / (3600 * 24 * 1000)}"')
    write(mean, folders["results"]["mean"])
    return round(avg_duration_days, 2)


def write_costs_reference_mean(df, case_id_name):
    avg_cost = (df.groupby(case_id_name)["case_cost"].max() -
                df.groupby(case_id_name)["case_cost"].min()).mean()
    mean = {'completedMean': round(avg_cost, 2)}
    print(f"Average completed cost: {mean}")
    write(mean, folders["results"]["mean"])
    return round(avg_cost, 2)


def add_aggregated_history(df, case_id_name, activity_column_name):
    for activity in df[activity_column_name].unique():
        df[f"# {activity_column_name}={activity}"] = 0
        # first put 1 in correspondence to each activity
        df.loc[df[activity_column_name] == activity, f"# {activity_column_name}={activity}"] = 1
        # sum the count from the previous events
        df[f"# {activity_column_name}={activity}"] = \
            df.groupby(case_id_name)[f"# {activity_column_name}={activity}"].cumsum()
    return df


def get_split_indexes(df, case_id_name, start_date_name, train_size=float):
    print('Starting splitting procedure..')
    start_end_couple = list()
    for idx in df[case_id_name].unique():
        df_ = df[df[case_id_name] == idx].reset_index(drop=True)
        start_end_couple.append([idx, df_[start_date_name].values[0], df_[start_date_name].values[len(df_) - 1]])
    start_end_couple = pd.DataFrame(start_end_couple, columns=['idx', 'start', 'end'])
    print(f'The min max range is {start_end_couple.start.min()}, {start_end_couple.end.max()}')
    print(f'With length {start_end_couple.end.max() - start_end_couple.start.min()}')

    # Initialize pdf of active cases and cdf of closed cases
    times_dict_pdf = dict()
    times_dict_cdf = dict()
    split = int((start_end_couple.end.max() - start_end_couple.start.min()) / 10000)  # In order to get a 10000 dotted graph
    for time in range(int(start_end_couple.start.min()), int(start_end_couple.end.max()), split):
        times_dict_pdf[time] = 0
        times_dict_cdf[time] = 0

    for time in tqdm.tqdm(range(int(start_end_couple.start.min()), int(start_end_couple.end.max()), split)):
        for line in np.array(start_end_couple[['start', 'end']]):
            line = np.array(line)
            if (line[0] <= time) and (line[1] >= time):
                times_dict_pdf[time] += 1
    for time in tqdm.tqdm(range(int(start_end_couple.start.min()), int(start_end_couple.end.max()), split)):
        for line in np.array(start_end_couple[['start', 'end']]):
            line = np.array(line)
            if (line[1] <= time):  # Keep just k closes cases
                times_dict_cdf[time] += 1

    sns.set_style('darkgrid')
    plt.title('Number of active operations')
    plt.xlabel('Time')
    plt.ylabel('Count')

    sns.lineplot(times_dict_pdf.keys(), times_dict_pdf.values())
    sns.lineplot(times_dict_cdf.keys(), times_dict_cdf.values())
    plt.savefig('Active and completed cases distribution.png')
    times_dist = pd.DataFrame(columns=['times', 'pdf_active', 'cdf_closed'])
    times_dist['times'] = times_dict_pdf.keys()
    times_dist['pdf_active'] = times_dict_pdf.values()
    times_dist['cdf_closed'] = np.array(list(times_dict_cdf.values())) / (len(df[case_id_name].unique()))
    # Set threshold after 60 of closed activities (it'll be the train set)

    test_dim = times_dist[times_dist.cdf_closed > train_size].pdf_active.max()
    thrs = times_dist[times_dist.pdf_active == test_dim].times.values[0]
    train_idxs = start_end_couple[start_end_couple['end'] <= thrs]['idx'].values
    test_idxs = start_end_couple[start_end_couple['end'] >= thrs][start_end_couple['start'] <= thrs]['idx'].values

    pickle.dump(train_idxs, open(f'indexes/train_idx_{case_id_name}.pkl', 'wb'))
    pickle.dump(test_idxs, open(f'indexes/test_idx_{case_id_name}.pkl', 'wb'))
    print('Split done')


def apply_history_to_df(df, case_id_name, activity_column_name, timestep, case_level_attributes):
    remaining_time = df["remaining_time"]
    if timestep == "no history":
        return df
    elif timestep == "aggr hist":
        df = add_aggregated_history(df, case_id_name, activity_column_name)
        print("Added history")
        return df
    # we delete remaining_time and later we reapply it in order to not duplicate it
    del df["remaining_time"]
    df_original = df.copy()
    float_columns = df.select_dtypes(include=['float64']).columns
    int_columns = df.select_dtypes(include=['int']).columns
    int_columns = [x for x in int_columns if (x != case_id_name) and (x not in case_level_attributes)]
    float_columns = [x for x in float_columns if (x != case_id_name) and (x not in case_level_attributes)]

    # at step 2 you need to append also step 1
    for i in range(1, int(timestep) + 1):
        df_shifted = df_original.copy()
        # don't shift case_level columns
        df_shifted.drop(case_level_attributes, axis=1, inplace=True)
        df_shifted = df_shifted.groupby(case_id_name).shift(i, fill_value="No previous activity").drop([case_id_name],
                                                                                                       axis=1)
        # keep numerical columns as numbers for history
        df_shifted.loc[df_shifted[activity_column_name] == "No previous activity", float_columns] = -1
        df_shifted.loc[df_shifted[activity_column_name] == "No previous activity", int_columns] = -1
        df_shifted[float_columns] = df_shifted[float_columns].astype("float64")
        df_shifted[int_columns] = df_shifted[int_columns].astype("float64")
        df_shifted.columns = df_shifted.columns + f' (-{i})'
        df = df.merge(df_shifted, left_index=True, right_index=True)

    # put to missing categorical nan columns
    for i, column in enumerate(df.columns):
        if df[column].dtype == 'object':
            df[column] = df[column].fillna("missing")
    # df = add_aggregated_history(df, case_id_name, activity_column_name)
    df["remaining_time"] = remaining_time
    print("Added history")
    return df


@log_it
def prepare_dataset(df, case_id_name, activity_column_name, start_date_name, date_format,
                    end_date_name, pred_column, mode, experiment_name, override=False,
                    pred_attributes=None, costs=None,
                    working_times=None, resource_column_name=None, role_column_name=None,
                    use_remaining_for_num_targets=False, predict_activities=None, lost_activities=None,
                    retained_activities=None, custom_attribute_column_name=None, grid=False, shap=False):

    activity_name = activity_column_name
    mean_reference_target = None
    #If there are not a folder for contain indexes, create it
    if not os.path.exists('indexes'):
        os.mkdir('indexes')

    if not (os.path.exists(f'indexes/test_idx_{case_id_name}.pkl') and os.path.exists(f'indexes/train_idx_{case_id_name}.pkl')):
        get_split_indexes(df, case_id_name, start_date_name, train_size=.65)
    else:
        print('reading indexes')

    if mode == "train" and pred_column == "remaining_time":
        mean_reference_target = write_leadtime_reference_mean(df, case_id_name, start_date_name, end_date_name)
        histogram_median_events_per_dataset(df, case_id_name, activity_column_name, start_date_name,
                                            end_date_name)
    df = prepare_data_and_add_features(df, case_id_name, start_date_name, date_format, end_date_name)

    if "activity_duration" in df.columns:
        df_completed_cases = df.groupby(case_id_name).agg("last")[
            [activity_column_name, "time_from_start", "activity_duration"]].reset_index()
        df_completed_cases["current"] = df_completed_cases["time_from_start"] + df_completed_cases["activity_duration"]
        df_completed_cases.drop(["time_from_start", "activity_duration"], axis=1, inplace=True)
    else:
        df_completed_cases = df.groupby(case_id_name).agg("last")[
            [activity_column_name, "time_from_start"]].reset_index()
        df_completed_cases["current"] = df_completed_cases["time_from_start"]
        df_completed_cases.drop(["time_from_start"], axis=1, inplace=True)
    df_completed_cases.rename(columns={case_id_name: "CASE ID", activity_column_name: "Activity"}, inplace=True)
    if costs is not None:
        try:
            df = calculate_costs(df, costs, working_times, activity_column_name, resource_column_name,
                                 role_column_name, case_id_name)
            if pred_column == "case_cost" and mode == "train":
                mean_reference_target = write_costs_reference_mean(df, case_id_name)
        except Exception as e:
            print(traceback.format_exc(), '\nContinuing')
            pass
    if mode == "train":
        mean_events = round(np.mean(df.groupby(case_id_name).count()[activity_column_name]))
        if grid is True:
            history = ["no history", "aggr hist"]
            for i in range(1, mean_events + 1):
                history.append(i)
            # end is needed if we try all models and validation curve is still decreasing (edge case)
            history.append("end")
        else:
            history = ["aggr hist"]
        df_original = df.copy()
        end = False
    else:
        history = [read(folders['model']['params'])["history"]]

    case_level_attributes = new_case_level_attribute_detection(df, case_id_name, mode)
    for model_type in history:
        if mode == "train":
            if exists(folders['model']['params']):
                if "history" in read(folders['model']['params']) or model_type == "end":
                    model_type = read(folders['model']['params'])["history"]
                    end = True
            df = df_original.copy()
        df = apply_history_to_df(df, case_id_name, activity_column_name, model_type, case_level_attributes)
        # if target column != remaining time exclude target column
        if pred_column != 'remaining_time' and mode == "train":
            if pred_column == "independent_activity":
                event_level = 1
                pred_attributes = predict_activities[0]
                pred_column = activity_column_name
            elif pred_column == "churn_activity":
                event_level = 1
                pred_attributes = retained_activities
                pred_column = activity_column_name
            elif pred_column == "custom_attribute":
                # this follows the same path as independent_activity
                event_level = 1
                pred_attributes = pred_attributes[0]
                pred_column = custom_attribute_column_name
            else:
                event_level = detect_case_level_attribute(df, pred_column)
            if event_level == 0:
                # case level - test column as is
                if np.issubdtype(df[pred_column], np.number):
                    # take target numeric column as is
                    column_type = 'Numeric'
                    target_column = df[pred_column].reset_index(drop=True)
                    target_column_name = pred_column
                    # add temporary column to know which rows delete later (remaining_time=0)
                    target_column = pd.concat([target_column, df['remaining_time']], axis=1)
                    del df[pred_column]
                else:
                    # case level string (you want to discover if a client recess will be performed)
                    column_type = 'Categorical'
                    # add one more test column for every value to be predicted
                    for value in pred_attributes:
                        df[value] = 0
                    # assign 1 to the column corresponding to that value
                    for value in pred_attributes:
                        df.loc[df[pred_column] == value, value] = 1
                    # eliminate old test column and take columns that are one-hot encoded test
                    del df[pred_column]
                    target_column = df[pred_attributes]
                    target_column_name = pred_attributes
                    df.drop(pred_attributes, axis=1, inplace=True)
                    target_column = target_column.join(df['remaining_time'])
            else:
                # event level attribute prediction
                if np.issubdtype(df[pred_column], np.number):
                    column_type = 'Numeric'
                    # if a number you want to discover the final value of the attribute (ex invoice amount)
                    df_last_attribute = df.groupby(case_id_name)[pred_column].agg(['last']).reset_index()
                    target_column = df[case_id_name].map(df_last_attribute.set_index(case_id_name)['last'])
                    if use_remaining_for_num_targets:
                        # now we predict remaining attribute (e.g. remaining cost)
                        target_column = target_column - df[pred_column]
                    # if you don't add y to the name you already have the same column in x, when you add the y-column
                    # after normalizing
                    target_column_name = 'Y_COLUMN_' + pred_column
                    target_column = pd.concat([target_column, df['remaining_time']], axis=1)
                else:
                    # TODO: target column could be calculated only once for all history models when using grid
                    # you want to discover if a certain activity will be performed
                    # set 1 for each case until that activity happens, the rest 0
                    if pred_column == activity_column_name and not type(pred_attributes) == np.str:
                        target_column_name = "retained_activity"
                        df[target_column_name] = 0
                    else:
                        # this is the case for single independent activity or custom attribute
                        df[pred_attributes] = 0
                    # multiple end activities (churn) are monitored together
                    if not type(pred_attributes) == np.str:
                        case_ids = []
                        for pred_attribute in pred_attributes:
                            case_ids.extend(
                                df.loc[df[activity_column_name] == pred_attribute][case_id_name].unique().tolist())
                    else:
                        case_ids = df.loc[df[pred_column] == pred_attributes][case_id_name].unique()
                    if type(pred_attributes) == np.str:
                        df.reset_index(inplace=True)
                        # take start indexes of cases that contain that activity
                        # and the index where there is the last target activity for that case
                        start_case_indexes = \
                            df.loc[df[case_id_name].isin(case_ids)].groupby(case_id_name, as_index=False).agg('first')[
                                'index']
                        last_observed_activity_indexes = \
                            df.loc[(df[case_id_name].isin(case_ids)) & (
                                        df[activity_column_name] == pred_attributes)].groupby(
                                case_id_name).agg('last')['index']
                        df_indexes = pd.concat([start_case_indexes.reset_index(drop=True),
                                                last_observed_activity_indexes.reset_index(drop=True).rename(
                                                    "index_1")],
                                               axis=1)
                        index_list = []
                        for x, y in zip(df_indexes['index'], df_indexes['index_1']):
                            for index in range(x, y):
                                index_list.append(index)
                        del df["index"]
                        df.loc[df.index.isin(index_list), pred_attributes] = 1
                    else:
                        # TODO: make this more efficient also for churn prediction
                        for case_id in case_ids:
                            for pred_attribute in pred_attributes:
                                index = df.loc[
                                    (df[case_id_name] == case_id) & (df[activity_column_name] == pred_attribute)].index
                                # each case id corresponds to one end activity only
                                if len(index) != 0:
                                    break
                        if len(index) == 1:
                            index = index[0]
                        else:
                            # if activity is performed more than once, take only the last
                            index = index[-1]
                        # put 1 to all y_targets before that activity in the case
                        df.loc[(df[case_id_name] == case_id) & (df.index < index), target_column_name] = 1
                    column_type = 'Categorical'

                    # we take only the columns we are interested in
                    if not type(pred_attributes) == np.str:
                        target_column = df[target_column_name]
                        df.drop(target_column_name, axis=1, inplace=True)
                    else:
                        target_column = df[pred_attributes]
                        target_column_name = pred_attributes
                        df.drop(pred_attributes, axis=1, inplace=True)
                    target_column = pd.concat([target_column, df["remaining_time"]], axis=1)
            print("Calculated target column")
        elif pred_column == 'remaining_time' and mode == "train":
            column_type = 'Numeric'
            if use_remaining_for_num_targets:
                event_level = 1
                target_column_name = 'remaining_time'
            else:
                event_level = 0
                if end_date_name is not None:
                    leadtime_per_case = df.groupby(case_id_name).agg("first")["activity_duration"] \
                                        + df.groupby(case_id_name).agg("first")["remaining_time"]
                else:
                    leadtime_per_case = df.groupby(case_id_name).agg("first")["remaining_time"]
                df["lead_time"] = df[case_id_name].map(leadtime_per_case)
                target_column_name = 'lead_time'

            # remove rows where remaining_time=0
            df = df[df.loc[:, 'remaining_time'] != 0].reset_index(drop=True)

            if use_remaining_for_num_targets:
                target_column = df.loc[:, 'remaining_time'].reset_index(drop=True)
            else:
                target_column = df.loc[:, 'lead_time'].reset_index(drop=True)
                del df["remaining_time"]
            del df[target_column_name]
            print("Calculated target column")
        else:
            # case when you have true data to be predicted (running cases)
            cases = read(folders['model']['data_info'])
            type_test = cases['test']
            target_column_name = cases['y_columns']
            if type_test == "event":
                # clean name (you have this when you save event-level column)
                cleaned_names = []
                for name in target_column_name:
                    cleaned_names.append(name.replace('Y_COLUMN_', ''))
                target_column_name = cleaned_names
            # we don't have target column in true test
            target_column = None
            # if attribute is of type case remove it from the dataset (in the train it hasn't been seen, but here you
            # have it)
            # if type_test == 'case':
            #     del df[pred_column]
            # hotfix
            try:
                del df['remaining_time']
            except KeyError:
                pass
        # eliminate rows where remaining time = 0 (nothing to predict) - only in train
        if mode == "train" and pred_column != 'remaining_time':
            df = df[df.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
            target_column = target_column[target_column.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
            del df['remaining_time']
            del target_column['remaining_time']

        # convert case id series to string and cut spaces if we have dirty data (both int and string)
        df.iloc[:, 0] = df.iloc[:, 0].apply(str)
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.strip())
        test_case_ids = df.iloc[:, 0]
        # convert back to int in order to use median (if it is a string convert to categoric int values)
        try:
            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
        except ValueError:
            pass
            # df.iloc[:, 0] = pd.Series(df.iloc[:, 0]).astype('category').cat.codes.values
        import time
        print(f'final preprocessing time is {time.time()}')
        if mode == "train":
            prepare_data_for_ml_model_and_predict(df, target_column, target_column_name, event_level, column_type, mode,
                                                  experiment_name, override, activity_column_name, pred_column,
                                                  pred_attributes, model_type, mean_events, mean_reference_target,
                                                  history, df_completed_cases, case_id_name, grid, shap)
            if end is True:
                break

        else:
            # here we have only the true running cases to predict
            info = read(folders['model']['data_info'])
            column_type = info['column_type']
            categorical_features = info['categorical_features']
            df.columns = df.columns.str.replace('time_from_midnight', 'daytime')
            df = pad_columns_in_real_data(df.iloc[:, 1:], df.iloc[:, 0])
            df[categorical_features] = df[categorical_features].astype(str)

            # keep for every case the last activity and delete the multicolumn
            df = df.groupby(case_id_name).agg(["last"]).reset_index(drop=True)
            df.columns = df.columns.droplevel(1)

            current_activities = df[activity_column_name]
            if pred_column == "remaining_time" or pred_column == "independent_activity" \
                    or pred_column == "churn_activity" or pred_column == "custom_attribute":
                if "activity_duration" in df:
                    current = df["time_from_start"] + df["activity_duration"]
                else:
                    current = df["time_from_start"]
            elif pred_column == "case_cost":
                current = df["case_cost"]
            else:
                raise NotImplementedError(f"pred_column {pred_column} cannot be handled.")
            current.rename("current", inplace=True)
            X_test = df
            running_data = Pool(df, cat_features=categorical_features)

            model = read(folders['model']['model'])
            print('Reloaded model')

            print('Starting predicting running cases...')
            try:
                if pred_column == "independent_activity" or pred_column == "churn_activity" \
                        or pred_column == "custom_attribute":
                    predictions = model.predict_proba(running_data)
                    predictions = [0 if x[1] < info["decision_threshold"] else 1 for x in predictions]
                else:
                    predictions = model.predict(running_data)
            except CatBoostError:
                # hotfix: if a categorical variable in running has only missing values,
                #  it is wrongly typed as numeric, which leads to a mismatch in categorical
                #  columns; this should be properly fixed in the new version of the pipeline.
                df[df.columns[df.isna().all(axis=0)]] = df[df.columns[df.isna().all(axis=0)]].fillna("missing")
                # rerun the data prep and predict
                categorical_features = df.select_dtypes(exclude=np.number).columns
                df[categorical_features] = df[categorical_features].astype(str)
                X_test = df
                running_data = Pool(df, cat_features=categorical_features)
                if pred_column == "independent_activity" or pred_column == "churn_activity" \
                        or pred_column == "custom_attribute":
                    predictions = model.predict_proba(running_data)
                    predictions = [0 if x[1] < info["decision_threshold"] else 1 for x in predictions]
                else:
                    predictions = model.predict(running_data)

            df = prepare_csv_results(predictions, test_case_ids, current_activities, target_column_name, pred_column,
                                     mode,
                                     column_type, current)

            if shap is True:
                # compute explanations
                explainer = shap.TreeExplainer(model)
                shapley_test = explainer.shap_values(running_data)
                # if column_type == "Categorical":
                #     pred_column = pred_attributes
                explanations_running_new_logic = find_explanations_for_running_cases(shapley_test, X_test, df,
                                                                                     pred_column)
                write(explanations_running_new_logic, folders['results']['explanations_running'])
            df.to_csv(folders['results']['running'], index=False)
            print("Generated predictions for running cases along with explanations")

def preprocess_df(df, case_id_name, activity_column_name, start_date_name, date_format,
                    end_date_name, pred_column, experiment_name, mode='train', override=False,
                    pred_attributes=None, costs=None, working_times=None, resource_column_name=None,
                    role_column_name=None, use_remaining_for_num_targets=False, predict_activities=None,
                    lost_activities=None, retained_activities=None, custom_attribute_column_name=None, grid=False, shap=False):

    activity_name = activity_column_name
    mean_reference_target = None
    #If there are not a folder for contain indexes, create it
    if not os.path.exists('indexes'):
        os.mkdir('indexes')

    if not (os.path.exists(f'indexes/test_idx_{case_id_name}.pkl') and os.path.exists(f'indexes/train_idx_{case_id_name}.pkl')):
        get_split_indexes(df, case_id_name, start_date_name, train_size=.65)
    else:
        print('reading indexes')

    if mode == "train" and pred_column == "remaining_time":
        mean_reference_target = write_leadtime_reference_mean(df, case_id_name, start_date_name, end_date_name)
        # histogram_median_events_per_dataset(df, case_id_name, activity_column_name, start_date_name,
        #                                     end_date_name)
    df = prepare_data_and_add_features(df, case_id_name, start_date_name, date_format, end_date_name)

    if "activity_duration" in df.columns:
        df_completed_cases = df.groupby(case_id_name).agg("last")[
            [activity_column_name, "time_from_start", "activity_duration"]].reset_index()
        df_completed_cases["current"] = df_completed_cases["time_from_start"] + df_completed_cases["activity_duration"]
        df_completed_cases.drop(["time_from_start", "activity_duration"], axis=1, inplace=True)
    else:
        df_completed_cases = df.groupby(case_id_name).agg("last")[
            [activity_column_name, "time_from_start"]].reset_index()
        df_completed_cases["current"] = df_completed_cases["time_from_start"]
        df_completed_cases.drop(["time_from_start"], axis=1, inplace=True)
    df_completed_cases.rename(columns={case_id_name: "CASE ID", activity_column_name: "Activity"}, inplace=True)
    if costs is not None:
        try:
            df = calculate_costs(df, costs, working_times, activity_column_name, resource_column_name,
                                 role_column_name, case_id_name)
            if pred_column == "case_cost" and mode == "train":
                mean_reference_target = write_costs_reference_mean(df, case_id_name)
        except Exception as e:
            print(traceback.format_exc(), '\nContinuing')
            pass
    if mode == "train":
        mean_events = round(np.mean(df.groupby(case_id_name).count()[activity_column_name]))
        if grid is True:
            history = ["no history", "aggr hist"]
            for i in range(1, mean_events + 1):
                history.append(i)
            # end is needed if we try all models and validation curve is still decreasing (edge case)
            history.append("end")
        else:
            history = ["aggr hist"]
        df_original = df.copy()
        end = False
    else:
        history = [read(folders['model']['params'])["history"]]

    case_level_attributes = new_case_level_attribute_detection(df, case_id_name, mode)
    for model_type in history:
        if mode == "train":
            if exists(folders['model']['params']):
                if "history" in read(folders['model']['params']) or model_type == "end":
                    model_type = read(folders['model']['params'])["history"]
                    end = True
            df = df_original.copy()
        df = apply_history_to_df(df, case_id_name, activity_column_name, model_type, case_level_attributes)
        # if target column != remaining time exclude target column
        if pred_column != 'remaining_time' and mode == "train":
            if pred_column == "independent_activity":
                event_level = 1
                pred_attributes = predict_activities[0]
                pred_column = activity_column_name
            elif pred_column == "churn_activity":
                event_level = 1
                pred_attributes = retained_activities
                pred_column = activity_column_name
            elif pred_column == "custom_attribute":
                # this follows the same path as independent_activity
                event_level = 1
                pred_attributes = pred_attributes[0]
                pred_column = custom_attribute_column_name
            else:
                event_level = detect_case_level_attribute(df, pred_column)
            if event_level == 0:
                # case level - test column as is
                if np.issubdtype(df[pred_column], np.number):
                    # take target numeric column as is
                    column_type = 'Numeric'
                    target_column = df[pred_column].reset_index(drop=True)
                    target_column_name = pred_column
                    # add temporary column to know which rows delete later (remaining_time=0)
                    target_column = pd.concat([target_column, df['remaining_time']], axis=1)
                    del df[pred_column]
                else:
                    # case level string (you want to discover if a client recess will be performed)
                    column_type = 'Categorical'
                    # add one more test column for every value to be predicted
                    for value in pred_attributes:
                        df[value] = 0
                    # assign 1 to the column corresponding to that value
                    for value in pred_attributes:
                        df.loc[df[pred_column] == value, value] = 1
                    # eliminate old test column and take columns that are one-hot encoded test
                    del df[pred_column]
                    target_column = df[pred_attributes]
                    target_column_name = pred_attributes
                    df.drop(pred_attributes, axis=1, inplace=True)
                    target_column = target_column.join(df['remaining_time'])
            else:
                # event level attribute prediction
                if np.issubdtype(df[pred_column], np.number):
                    column_type = 'Numeric'
                    # if a number you want to discover the final value of the attribute (ex invoice amount)
                    df_last_attribute = df.groupby(case_id_name)[pred_column].agg(['last']).reset_index()
                    target_column = df[case_id_name].map(df_last_attribute.set_index(case_id_name)['last'])
                    if use_remaining_for_num_targets:
                        # now we predict remaining attribute (e.g. remaining cost)
                        target_column = target_column - df[pred_column]
                    # if you don't add y to the name you already have the same column in x, when you add the y-column
                    # after normalizing
                    target_column_name = 'Y_COLUMN_' + pred_column
                    target_column = pd.concat([target_column, df['remaining_time']], axis=1)
                else:
                    # TODO: target column could be calculated only once for all history models when using grid
                    # you want to discover if a certain activity will be performed
                    # set 1 for each case until that activity happens, the rest 0
                    if pred_column == activity_column_name and not type(pred_attributes) == np.str:
                        target_column_name = "retained_activity"
                        df[target_column_name] = 0
                    else:
                        # this is the case for single independent activity or custom attribute
                        df[pred_attributes] = 0
                    # multiple end activities (churn) are monitored together
                    if not type(pred_attributes) == np.str:
                        case_ids = []
                        for pred_attribute in pred_attributes:
                            case_ids.extend(
                                df.loc[df[activity_column_name] == pred_attribute][case_id_name].unique().tolist())
                    else:
                        case_ids = df.loc[df[pred_column] == pred_attributes][case_id_name].unique()
                    if type(pred_attributes) == np.str:
                        df.reset_index(inplace=True)
                        # take start indexes of cases that contain that activity
                        # and the index where there is the last target activity for that case
                        start_case_indexes = \
                            df.loc[df[case_id_name].isin(case_ids)].groupby(case_id_name, as_index=False).agg('first')[
                                'index']
                        last_observed_activity_indexes = \
                            df.loc[(df[case_id_name].isin(case_ids)) & (
                                        df[activity_column_name] == pred_attributes)].groupby(
                                case_id_name).agg('last')['index']
                        df_indexes = pd.concat([start_case_indexes.reset_index(drop=True),
                                                last_observed_activity_indexes.reset_index(drop=True).rename(
                                                    "index_1")],
                                               axis=1)
                        index_list = []
                        for x, y in zip(df_indexes['index'], df_indexes['index_1']):
                            for index in range(x, y):
                                index_list.append(index)
                        del df["index"]
                        df.loc[df.index.isin(index_list), pred_attributes] = 1
                    else:
                        # TODO: make this more efficient also for churn prediction
                        for case_id in case_ids:
                            for pred_attribute in pred_attributes:
                                index = df.loc[
                                    (df[case_id_name] == case_id) & (df[activity_column_name] == pred_attribute)].index
                                # each case id corresponds to one end activity only
                                if len(index) != 0:
                                    break
                        if len(index) == 1:
                            index = index[0]
                        else:
                            # if activity is performed more than once, take only the last
                            index = index[-1]
                        # put 1 to all y_targets before that activity in the case
                        df.loc[(df[case_id_name] == case_id) & (df.index < index), target_column_name] = 1
                    column_type = 'Categorical'

                    # we take only the columns we are interested in
                    if not type(pred_attributes) == np.str:
                        target_column = df[target_column_name]
                        df.drop(target_column_name, axis=1, inplace=True)
                    else:
                        target_column = df[pred_attributes]
                        target_column_name = pred_attributes
                        df.drop(pred_attributes, axis=1, inplace=True)
                    target_column = pd.concat([target_column, df["remaining_time"]], axis=1)
            print("Calculated target column")
        elif pred_column == 'remaining_time' and mode == "train":
            column_type = 'Numeric'
            if use_remaining_for_num_targets:
                event_level = 1
                target_column_name = 'remaining_time'
            else:
                event_level = 0
                if end_date_name is not None:
                    leadtime_per_case = df.groupby(case_id_name).agg("first")["activity_duration"] \
                                        + df.groupby(case_id_name).agg("first")["remaining_time"]
                else:
                    leadtime_per_case = df.groupby(case_id_name).agg("first")["remaining_time"]
                df["lead_time"] = df[case_id_name].map(leadtime_per_case)
                target_column_name = 'lead_time'

            # remove rows where remaining_time=0
            df = df[df.loc[:, 'remaining_time'] != 0].reset_index(drop=True)

            if use_remaining_for_num_targets:
                target_column = df.loc[:, 'remaining_time'].reset_index(drop=True)
            else:
                target_column = df.loc[:, 'lead_time'].reset_index(drop=True)
                del df["remaining_time"]
            del df[target_column_name]
            print("Calculated target column")
        else:
            # case when you have true data to be predicted (running cases)
            cases = read(folders['model']['data_info'])
            type_test = cases['test']
            target_column_name = cases['y_columns']
            if type_test == "event":
                # clean name (you have this when you save event-level column)
                cleaned_names = []
                for name in target_column_name:
                    cleaned_names.append(name.replace('Y_COLUMN_', ''))
                target_column_name = cleaned_names
            # we don't have target column in true test
            target_column = None
            # if attribute is of type case remove it from the dataset (in the train it hasn't been seen, but here you
            # have it)
            # if type_test == 'case':
            #     del df[pred_column]
            # hotfix
            try:
                del df['remaining_time']
            except KeyError:
                pass
        # eliminate rows where remaining time = 0 (nothing to predict) - only in train
        if mode == "train" and pred_column != 'remaining_time':
            df = df[df.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
            target_column = target_column[target_column.loc[:, 'remaining_time'] != 0].reset_index(drop=True)
            del df['remaining_time']
            del target_column['remaining_time']

        # convert case id series to string and cut spaces if we have dirty data (both int and string)
        df.iloc[:, 0] = df.iloc[:, 0].apply(str)
        df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x.strip())
        test_case_ids = df.iloc[:, 0]
        # convert back to int in order to use median (if it is a string convert to categoric int values)
        try:
            df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0])
        except ValueError:
            pass

        return df