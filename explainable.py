import numpy as np
import pandas as pd
from catboost import CatboostError

import shap
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

#REFACTOR
from os.path import exists
from IO import read, write, folders


def plot_histogram(explanation_histogram, experiment_name, index_name=None):
    # Fixing random state for reproducibility
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(15, 15))

    # Example data
    shap_columns = explanation_histogram.keys()
    y_pos = np.arange(len(shap_columns))
    values = np.array(list(explanation_histogram.values()))
    error = np.random.rand(len(shap_columns))

    ix_pos = values > 0

    ax.barh(y_pos[ix_pos], values[ix_pos], align='center', color='r')
    ax.barh(y_pos[~ix_pos], values[~ix_pos], align='center', color='b')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(shap_columns)
    ax.invert_yaxis()  # labels read top-to-bottom
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
    ax.set_title('How are predictions going?', fontsize=22)
    if index_name is None:
        plt.savefig("/experiment_files" + "/plots/shap_histogram.png",
                    dpi=300, bbox_inches="tight")
    else:
        plt.savefig(
            experiment_name + f"/plots/shap_histogram_{index_name}.png", dpi=300, bbox_inches="tight")

def plot_histogram_new(df_explanations, pred_column):
    # Fixing random state for reproducibility
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(15, 15))

    shap_columns = df_explanations["explanation"]
    y_pos = np.arange(len(shap_columns))
    values = df_explanations["value"]

    ix_pos = values > 0

    ax.barh(y_pos[ix_pos], values[ix_pos], align='center', color='r')
    ax.barh(y_pos[~ix_pos], values[~ix_pos], align='center', color='b')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(shap_columns)
    ax.invert_yaxis()  # labels read top-to-bottom
    if "time" in pred_column:
        plt.xlabel("Days", fontsize=15)
    elif "cost" in pred_column:
        plt.xlabel("Euros")
    else:
        plt.xlabel("Feature importance")
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
    ax.set_title('How are predictions going?', fontsize=22)
    plt.savefig(f"{os.getcwd()}/experiment_files/plots/shap_histogram.png", dpi=300, bbox_inches="tight")


def compare_best_validation_curves(pred_column="remaining_time", mean_reference_target=None):
    x = []
    y_train_percentage = []
    y_validation_percentage = []
    y_validation = []
    y_train = []
    lw = 2
    # overall_scores = read(folders['model']['scores'])
    import json
    import matplotlib.pyplot as plt
    overall_scores = json.load(open('model/models_scores.json', ))
    for name, score in overall_scores.items():
        if name != "Best" and "_std" not in name:
            if "_validation" in name:
                x.append(name.replace("_validation", ""))
                if pred_column == "remaining_time":
                    y_validation.append(score / 3600)
                    if mean_reference_target is not None:
                        y_validation_percentage.append(round((score / (mean_reference_target / 1000)) * 100, 2))
                else:
                    y_validation.append(score)
                    if mean_reference_target is not None:
                        y_validation_percentage.append(round((score / mean_reference_target)*100, 2))
            else:
                if pred_column == "remaining_time":
                    y_train.append(score / 3600)
                    if mean_reference_target is not None:
                        y_train_percentage.append(round((score / (mean_reference_target / 1000)) * 100, 2))
                else:
                    y_train.append(score)
                    if mean_reference_target is not None:
                        y_train_percentage.append(round((score / mean_reference_target)*100, 2))

    if mean_reference_target is not None:
        plt.clf()
        plt.xlabel("History timesteps")
        plt.ylabel("Error %")
        # plt.plot(x, y_train_percentage, color="red", lw=lw, label="Training score")
        plt.plot(x, y_validation_percentage, color="blue", lw=lw, label="Validation score")
        plt.xticks(rotation=30)
        plt.legend(loc="best")
        plt.savefig(f"plots/compare_best_error_percentage_validation.png", dpi=300, bbox_inches="tight")

    plt.clf()
    plt.xlabel("History timesteps")
    if pred_column == "remaining_time":
        plt.ylabel("Error hours (MAE)")
    elif pred_column == "case_cost":
        plt.ylabel("Error Euros (MAE)")
    else:
        plt.ylabel("F1 score")
    # plt.plot(x, y_train, color="red", lw=lw, label="Training score")
    plt.plot(x, y_validation, color="blue", lw=lw, label="Validation score")
    plt.xticks(rotation=30)
    plt.legend(loc="best")
    if pred_column == "remaining_time":
        plt.savefig(f"plots/compare_best_error_hours_validation.png", dpi=300, bbox_inches="tight")
    # else:
    #     plt.savefig(f"{os.getcwd()}/experiment_files/plots/compare_best_error.png", dpi=300, bbox_inches="tight")
    print("Plotted validation curve")



def translate(column, Min_original_range, Max_original_range, Min_new_range, Max_new_range):
    # function to convert probabilities from 0/1 to -1/+1
    # Figure out how 'wide' each range is
    original_span = Max_original_range - Min_original_range
    new_span = Max_new_range - Min_new_range

    # Convert the left range into a 0-1 range (float)
    column_scaled = (column - Min_original_range) / float(original_span)

    # Convert the 0-1 range into a value in the right range.
    return Min_new_range + (column_scaled * new_span)

#TODO: link this function to the "normal workflow" when producing shapley values for activity prediction!
def plot_histogram_activity(pred_column="activity"):
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    historical_explanations = json.load(open('results/explanations_completed.json',))
    # plot shap only for paper
    df_shap = pd.DataFrame(historical_explanations)
    del df_shap[df_shap.columns[0]]
    df_shap["value"] = df_shap["value"].astype("float")
    df_shap_grouped = df_shap.groupby("explanation").mean().reset_index()
    df_explanations = df_shap_grouped.iloc[(-df_shap_grouped["value"].abs()).argsort()].iloc[:15]

    # convert log odds into probabilities
    df_explanations["value"] = np.exp(df_explanations["value"])
    df_explanations["value"] = df_explanations["value"] / (1 + df_explanations["value"])

    # transform range to show strength of positive and negative influencers
    df_explanations["value"] = translate(df_explanations["value"], 0, 1, -1, +1)

    # Fixing random state for reproducibility
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(15, 15))

    shap_columns = df_explanations["explanation"]
    y_pos = np.arange(len(shap_columns))
    values = df_explanations["value"]

    #here below 0.5 means that it is unlikely to happen
    ix_pos = values > 0

    ax.barh(y_pos[ix_pos], values[ix_pos], align='center', color='r')
    ax.barh(y_pos[~ix_pos], values[~ix_pos], align='center', color='b')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(shap_columns)
    ax.invert_yaxis()  # labels read top-to-bottom
    if "time" in pred_column:
        plt.xlabel("Days", fontsize=15)
    elif "cost" in pred_column:
        plt.xlabel("Euros")
    else:
        plt.xlabel("Feature importance")
    plt.setp(ax.get_yticklabels(), fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), fontsize=12, fontweight="bold")
    ax.set_title('How are predictions going?', fontsize=22)
    plt.savefig(f"plots/shap_histogram_prob_new_range.png", dpi=300, bbox_inches="tight")


def plot_heatmap(explanation_histogram, experiment_name, index_name=None):
    df = pd.DataFrame(explanation_histogram)
    df = df.set_index(["feature", "ts"]).unstack()
    df["sort"] = df.abs().max(axis=1)
    # bigger values are on top
    df = df.sort_values("sort", ascending=False).drop("sort", axis=1)
    df.columns.rename(["#", "Timesteps"], inplace=True)
    df = df.iloc[:15]
    fig, ax = plt.subplots(figsize=(40, 15))
    heatmap = sns.heatmap(df, cbar=True, cmap="RdBu_r", center=0, robust=True,
                          annot=True, fmt='g', ax=ax, annot_kws={"fontsize": 13})
    heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=13)
    heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=13)

    if index_name is None:
        plt.savefig(experiment_name + "/plots/shap_heatmap.png",
                    dpi=300, bbox_inches="tight")
    else:
        plt.savefig(
            experiment_name + f"/plots/shap_heatmap_{index_name}.png", dpi=300, bbox_inches="tight")


def refine_explanation_name(x_test_instance, explanation_index, explanation_name):
    #with this technique we don't have explanation != and need to think about numerical explanations
    explanation_name = explanation_name + "=" + str(x_test_instance[explanation_index])
    return explanation_name


def add_explanation_to_histogram(x_test_instance, shap_values_instance, 
                                 feature_columns, shapley_value, i, 
                                 explanation_histogram):
    # take the column name for every explanation
    explanation_index = np.where(shap_values_instance == shapley_value)[0][0]
    explanation_name = feature_columns[explanation_index]

    explanation_name = refine_explanation_name(x_test_instance, explanation_index, explanation_name)
    # print("Instance {} --> {} : {}".format(i + 1, explanation_name, shapley_value))

    if explanation_name not in explanation_histogram:
        if shapley_value > 0:
            explanation_histogram[explanation_name] = 1
        else:
            explanation_histogram[explanation_name] = -1
    else:
        if shapley_value > 0:
            explanation_histogram[explanation_name] += 1
        else:
            explanation_histogram[explanation_name] -= 1

    return explanation_histogram


def find_instance_explanation_values(X_test, shapley_test, i):
    """
    Train phase (historical): take all shapley values for explaining
    Predict phase (running): take only the most relevant ones
    :param n_std: number of standard deviations to define significance threshold
    :return:
    """
    x_test_instance = X_test.iloc[i]
    shap_values_instance = shapley_test[i]

    # if mode == "predict":
    #     mean = shap_values_instance[np.nonzero(shap_values_instance)].mean()
    #     sd = shap_values_instance[np.nonzero(shap_values_instance)].std()
    #
    #     upper_threshold = mean + n_std * sd
    #     lower_threshold = mean - n_std * sd
    #
    #     # calculate most important values for explanation (+-3sd from the mean)
    #     explanation_values = shap_values_instance[(
    #         shap_values_instance > upper_threshold) | (shap_values_instance < lower_threshold)]
    #     # order shapley values for decrescent importance
    #     explanation_values = sorted(explanation_values, key=abs, reverse=True)

    explanation_values = shap_values_instance
    return x_test_instance, shap_values_instance, explanation_values


def convert(obj):
    """function needed because numpy int is not recognized by the serializer"""
    if isinstance(obj, np.int64):
        return int(obj)
    raise TypeError

def refine_explanations_name_and_aggregate_history(x_test_instance, shap_values_instance):
    # the list of the name of the explanation and the value should be updated only after aggregating the history
    columns = []
    for column_name, value in zip(x_test_instance.keys(), x_test_instance.values):
        # replace characters representing history and group together columns based on key value pair
        if "# " not in column_name:
            if "<" in str(value) or ">" in str(value):
                # here you have the full explanation name already in the value (bins)
                columns.append(value)
            else:
                # same feature different timestep will have the same name and will be aggregated
                columns.append(column_name.split(" (-")[0] + "=" + str(value))
        else:
            # activity count will be grouped with the correspondent activity name
            # if activity count is >= 1 (happening) add the column (replace #)
            if value >= 1:
                columns.append(column_name.replace('# ', ''))
            else:
                #particular case #act=0
                columns.append(column_name.replace('# ', '').replace('=', ' ') + " not performed")
    df = pd.DataFrame([shap_values_instance], columns=columns)
    # sum shapley values of identical columns (ex. same feature in different points in time)
    df = df.groupby(df.columns, axis=1).sum()
    return df

def update_completed_explanations(case, historical_explanations, case_explanations, pred_column):
    for explanation in case_explanations.keys():
        if pred_column == 'remaining_time':
            value = np.mean(case_explanations[explanation])*1000
        else:
            value = np.mean(case_explanations[explanation])
        if value != 0:
            historical_explanations.append({"caseId": str(case),
                                            "value": str(value),
                                            "explanation": explanation})
    return historical_explanations

def update_case_explanations(df_current_explanations, case_explanations):
    """In the end for each explanation you will have the list of its shapley values"""
    for column in df_current_explanations.columns:
        #do not insert missing explanations (noise)
        if "missing" not in column and "No previous activity" not in column and "-1" not in column:
            if column not in case_explanations:
                case_explanations[column] = [df_current_explanations.loc[0, column]]
            else:
                case_explanations[column].append(df_current_explanations.loc[0, column])
    return case_explanations

def update_running_explanations(df, i, df_current_explanations, explanations_running, pred_column):
    for column in df_current_explanations.columns:
        if pred_column == 'remaining_time':
            value = df_current_explanations.loc[0, column] * 1000
        else:
            value = df_current_explanations.loc[0, column]
        # do not insert missing explanations (noise) or null contributions
        if value != 0:
            if "missing" not in column and "No previous activity" not in column:
                explanations_running.append({"caseId": df.loc[i, "CASE ID"],
                                             "value": str(value),
                                             "explanation": column})
    return explanations_running


def keep_most_relevant_explanations(explanations, pred_column):
    essential_explanations = {}
    for i, key in enumerate(explanations.keys()):
        if i == 40:
            break
        else:
            #myinvenio needs in milliseconds, in case we can convert to days here
            if pred_column == "remaining_time":
                #for local test convert to days (usually done by MyInvenio)
                # essential_explanations[key] = np.mean(explanations[key]) / (3600*24)
                essential_explanations[key] = np.mean(explanations[key]) * 1000
            else:
                essential_explanations[key] = np.mean(explanations[key])
    return essential_explanations

def bin_numerical_variables_with_trees_for_explanation(X_test, column_type):
    from catboost import Pool, CatBoostRegressor, CatBoostClassifier
    column_types = read(folders['model']['data_info'])["column_types"]
    dfTrain = read(folders['model']['dfTrain'], dtype=column_types)

    params = {
        'depth': 3,
        'learning_rate': 0.01,
        'iterations': 1,
        'early_stopping_rounds': 5,
        'thread_count': 4,
        'logging_level': 'Silent',
        'task_type': "CPU"  # "GPU" if int(os.environ["USE_GPU"]) else "CPU"
    }

    X_test_original = X_test.copy()
    for column in X_test_original.columns:
        if X_test_original[column].dtype != 'object':
            # from num features representing the present we bin also the past
            if "(-" not in column and "#" not in column:
                if "daytime" in column or "time_from_start" in column or "time_from_previous_event" in column or "activity_duration" in column:
                    # convert time columns in hours
                    resizer = 3600
                else:
                    resizer = 1

                train_data = Pool(dfTrain.loc[:, column], dfTrain.iloc[:, -1])
                #TODO: I THINK IT'S BREAKING WITH THE datetime (always 0, as well as activity duration) AGGR HISTORY
                try:
                    if column_type != "Categorical":
                        params["loss_function"] = "MAE"
                        model = CatBoostRegressor(**params)
                        model.fit(train_data)
                    else:
                        params["loss_function"] = "Logloss"
                        model = CatBoostClassifier(**params)
                        model.fit(train_data)
                except CatboostError:
                    #in case a feature is always constant
                    X_test[column] = "missing"
                    past_columns_to_be_binned = X_test_original.columns[X_test_original.columns.str.startswith(f'{column} (-')]
                    for col in past_columns_to_be_binned:
                        X_test[col] = "missing"
                    continue

                bins = model.calc_feature_statistics(dfTrain.loc[:, column], dfTrain.iloc[:, -1], feature=0, plot=False)["borders"]
                #obtain splits made by the model using this feature
                original_bins = bins.copy()
                if resizer == 3600:
                    bins = [x / 3600 for x in bins]
                    # TODO: for times round to the upper number (more readable) and slightly modify the bins?
                    bins = [f"{round(x, 2)}h" if x > 1 else f"{round(x * 60, 2)}m" if x > 0 else '0s' for x in bins]
                else:
                    bins = ["{:.2f}".format(round(x, 2)) if x > 0 else '0' for x in bins]
                # in some cases the cut divides in less bins (values skewed towards 0)
                if len(bins) == 5:
                    labels = [f"{column} < {bins[0]}", f"{bins[0]} < {column} < {bins[1]}",
                              f"{bins[1]} < {column} < {bins[2]}", f"{bins[2]} < {column} < {bins[3]}" 
                              f"{bins[3]} < {column} < {bins[4]}", f"{column} > {bins[4]}"]
                elif len(bins) == 4:
                    labels = [f"{column} < {bins[0]}", f"{bins[0]} < {column} < {bins[1]}",
                              f"{bins[1]} < {column} < {bins[2]}", f"{bins[2]} < {column} < {bins[3]}" 
                              f"{column} > {bins[3]}"]
                elif len(bins) == 3:
                    labels = [f"{column} < {bins[0]}", f"{bins[0]} < {column} < {bins[1]}",
                              f"{bins[1]} < {column} < {bins[2]}", f"{column} > {bins[2]}"]
                elif len(bins) == 2:
                    labels = [f"{column} < {bins[0]}", f"{bins[0]} < {column} < {bins[1]}",
                              f"{column} > {bins[1]}"]
                elif len(bins) == 1:
                    labels = [f"{column} < {bins[0]}", f"{column} > {bins[0]}"]
                else:
                    continue

                #we need this because otherwise the column becomes categorical and no more num filters allowed
                X_test_encoded = X_test[column].copy()
                #apply each label obtained with the split to the correspondent values (present column)
                for i, original_bin in enumerate(original_bins):
                    if i == 0:
                        X_test_encoded.iloc[X_test.loc[X_test[column] < original_bin, column].index] = labels[i]
                        #if there is only one threshold
                        if len(original_bins) == 1:
                            X_test_encoded.iloc[X_test.loc[X_test[column] > original_bin, column].index] = labels[i+1]
                    elif i < (len(original_bins) - 1):
                        X_test_encoded.iloc[X_test.loc[((X_test[column] > original_bins[i - 1]) &
                                                        (X_test[column] < original_bins[i])), column].index] = labels[i]
                    else:
                        X_test_encoded.iloc[X_test.loc[((X_test[column] > original_bins[i - 1]) &
                                                        (X_test[column] < original_bins[i])), column].index] = labels[i]
                        X_test_encoded.iloc[X_test.loc[X_test[column] > original_bins[i], column].index] = labels[i+1]

                # assign the new categorical column to the df column
                X_test[column] = X_test_encoded.copy()
                X_test.loc[X_test[column].isnull(), column] = "missing"

                #apply also to the same column in the past
                past_columns_to_be_binned = X_test_original.columns[
                    X_test_original.columns.str.startswith(f'{column} (-')]
                for col in past_columns_to_be_binned:
                    # before splitting you must exclude and then reapply the "No previous activity (-1)"
                    no_activity_indexes = X_test_original.loc[X_test_original[col] == -1, col].index
                    X_test_encoded = X_test[col].copy()
                    for i, original_bin in enumerate(original_bins):
                        if i == 0:
                            X_test_encoded.iloc[X_test.loc[X_test[col] < original_bin, col].index] = labels[i]
                            # if there is only one threshold
                            if len(original_bins) == 1:
                                X_test_encoded.iloc[X_test.loc[X_test[col] > original_bin, col].index] = labels[i+1]
                        elif i < (len(original_bins) - 1):
                            X_test_encoded.iloc[X_test.loc[((X_test[col] > original_bins[i - 1]) &
                                                            (X_test[col] < original_bins[i])), col].index] = labels[i]
                        else:
                            X_test_encoded.iloc[X_test.loc[((X_test[col] > original_bins[i - 1]) &
                                                            (X_test[col] < original_bins[i])), col].index] = labels[i]
                            X_test_encoded.iloc[X_test.loc[X_test[col] > original_bins[i], col].index] = labels[i+1]

                    X_test[col] = X_test_encoded.copy()
                    X_test.loc[no_activity_indexes, col] = -1
                    X_test.loc[X_test[col].isnull(), col] = "missing"

    return X_test

def bin_numerical_variables_for_explanation(X_test):
    X_test_original = X_test.copy()
    for column in X_test_original.columns:
        if X_test_original[column].dtype != 'object':
            # from num features representing the present we bin also the past
            if "(-" not in column and "#" not in column:
                if "daytime" in column or "time_from_start" in column or "time_from_previous_event" in column or "activity_duration" in column:
                    # convert time columns in hours
                    resizer = 3600
                else:
                    resizer = 1
                bins = pd.qcut(X_test_original[column] / resizer, q=4, labels=False, retbins=True, duplicates="drop")[1]
                original_bins = bins.copy()
                if resizer == 3600:
                    # TODO: for times round to the upper number (more readable) and slightly modify the bins?
                    bins = [f"{round(x, 2)}h" if x > 1 else f"{round(x * 60, 2)}m" for x in bins]
                else:
                    bins = [round(x, 2) for x in bins]
                #in some cases the cut divides in less bins (values skewed towards 0)
                if len(bins) == 5:
                    labels = [f"{column} < {bins[1]}", f"{bins[1]} < {column} < {bins[2]}",
                              f"{bins[2]} < {column} < {bins[3]}", f"{column} > {bins[3]}"]
                elif len(bins) == 4:
                    labels = [f"{column} < {bins[1]}", f"{bins[1]} < {column} < {bins[2]}",
                              f"{bins[2]} < {column} < {bins[3]}"]
                elif len(bins) == 3:
                    labels = [f"{column} < {bins[1]}", f"{bins[1]} < {column} < {bins[2]}"]
                elif len(bins) == 2:
                    labels = [f"{column} < {bins[1]}"]
                else:
                    continue

                # now we can cut again but naming the bins as we want
                X_test[column] = pd.qcut(X_test_original[column] / resizer, q=4, labels=labels, duplicates="drop") \
                        .cat.add_categories('missing').fillna('missing')
                # usa gli stessi bins per binnare le stesse variabili, ma storiche
                columns_to_be_binned = X_test_original.columns[X_test_original.columns.str.startswith(f'{column} (-')]
                for col in columns_to_be_binned:
                    # include also 0 in the lower bound
                    bins = np.where(original_bins == 0, -float("inf"), original_bins)
                    # before splitting you must exclude and then reapply the "No previous activity (-1)"
                    no_activity_indexes = X_test_original.loc[X_test_original[col] == -1, col].index
                    X_test_original.loc[X_test_original[col] == -1, col] = np.nan
                    X_test_original[col] = X_test_original[col].astype("float")
                    # use pd.cut in order to replicate the same bins, otherwise duplicate label error
                    X_test[col] = pd.cut(X_test_original[col] / resizer, bins, labels=labels, duplicates="drop") \
                        .cat.add_categories('missing').fillna('missing')
                    # reapply original string label
                    X_test[col] = X_test[col].cat.add_categories(-1)
                    X_test.loc[no_activity_indexes, col] = -1
    return X_test


def find_explanations_for_completed_cases(X_test, test_cases, shapley_test, pred_column, column_type):
    """ new function for sending all explanations related to completed cases """

    # if column numeric if the predictions goes too far from the avg real data discard the prediction from the shap graph
    # true_mean = df['TEST'].mean()
    # prediction_out_of_range_high = df.loc[df[pred_column] > (df['TEST'] + true_mean), :].index
    # prediction_out_of_range_low = df.loc[df[pred_column] < (df['TEST'] - true_mean), :].index

    historical_explanations = []
    case_explanations = {}
    # before you must bin numerical variables into categories
    #X_test = bin_numerical_variables_for_explanation(X_test)
    X_test = bin_numerical_variables_with_trees_for_explanation(X_test, column_type)
    case = test_cases.iloc[0]
    print('G SUPERAT!')
    import ipdb; ipdb.set_trace()
    #calculate avg explanations for each case
    #TODO: is there a way to improve the performances here?
    for i in range(len(shapley_test)):
        # if the prediction is wrong don't consider the explanation
        # if column_type == 'Numeric':
        #     if i in prediction_out_of_range_high or i in prediction_out_of_range_low:
        #         continue
        if test_cases.iloc[i] != case:
            historical_explanations = update_completed_explanations(case, historical_explanations, case_explanations, pred_column)
            case = test_cases.iloc[i]
            case_explanations = {}

        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test,
                                                                                                     shapley_test, i)
        df_current_explanations = refine_explanations_name_and_aggregate_history(x_test_instance, shap_values_instance)
        case_explanations = update_case_explanations(df_current_explanations, case_explanations)

    #last iteration
    historical_explanations = update_completed_explanations(case, historical_explanations, case_explanations, pred_column)
    #plot shap only for paper
    df_shap = pd.DataFrame(historical_explanations)
    del df_shap[df_shap.columns[0]]
    df_shap["value"] = df_shap["value"].astype("float")
    df_shap_grouped = df_shap.groupby("explanation").mean().reset_index()
    df_explanations = df_shap_grouped.iloc[(-df_shap_grouped["value"].abs()).argsort()].iloc[:15]
    if "time" in pred_column:
        df_explanations["value"] = df_explanations["value"] / (3600 * 24 * 1000)
    plot_histogram_new(df_explanations, pred_column)
    write(historical_explanations, folders['results']['explanations_completed'])
    print("Generated explanations")

def calculate_histogram_for_shap_values(X_test, shapley_test, pred_column):
    # if column numeric if the predictions goes too far from the avg real data discard the prediction from the shap graph
    # true_mean = df['TEST'].mean()
    # prediction_out_of_range_high = df.loc[df[pred_column] > (df['TEST'] + true_mean), :].index
    # prediction_out_of_range_low = df.loc[df[pred_column] < (df['TEST'] - true_mean), :].index

    historical_explanations = {}
    # before you must bin numerical variables into categories
    X_test = bin_numerical_variables_for_explanation(X_test)

    for i in range(len(shapley_test)):
        # if the prediction is wrong don't consider the explanation
        # if column_type == 'Numeric':
        #     if i in prediction_out_of_range_high or i in prediction_out_of_range_low:
        #         continue

        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test, shapley_test, i)
        df_current_explanations = refine_explanations_name_and_aggregate_history(x_test_instance, shap_values_instance)
        historical_explanations = update_case_explanations(df_current_explanations, historical_explanations)
        # explanations = update_explanations(explanation_values, shap_values_instance, x_test_instance, feature_columns, explanations)

    # Order by mean influencing value (you discard discordant explanations)
    historical_explanations = {k: v for k, v in sorted(historical_explanations.items(), key=lambda
        item: abs(np.mean(item[1])), reverse=True)}
    essential_histogram = keep_most_relevant_explanations(historical_explanations, pred_column)

    write(essential_histogram, folders['results']['explanations_histogram'])
    #in local you can plot your explanations here
    #plot_histogram(essential_histogram, experiment_name)


def find_explanations_for_running_cases(shapley_test, X_test, df, pred_column):

    explanations_running = []

    # before you must bin numerical variables into categories
    X_test = bin_numerical_variables_for_explanation(X_test)

    for i in range(len(shapley_test)):
        x_test_instance, shap_values_instance, explanation_values = find_instance_explanation_values(X_test,
                                                                                                     shapley_test, i)
        df_current_explanations = refine_explanations_name_and_aggregate_history(x_test_instance, shap_values_instance)

        explanations_running = update_running_explanations(df, i, df_current_explanations, explanations_running, pred_column)

    return explanations_running


