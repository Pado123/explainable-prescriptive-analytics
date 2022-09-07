import numpy as np
import pandas as pd
import math, random
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score, \
    f1_score, confusion_matrix, plot_confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, mean_absolute_error
import matplotlib.pyplot as plt

import os
from IO import read, write, folders

def prepare_csv_results(predictions, test_case_ids, test_activities, target_column_name, pred_column,
                        mode, column_type, current, pred_attributes=None, y_test=None):
    # qui deve appendere o una serie o un dataframe
    predictions = pd.Series(predictions)
    if target_column_name == "lead_time":
        predictions.rename("lead_time", inplace=True)
    else:
        if pred_attributes is None:
            predictions.rename(pred_column, inplace=True)
        else:
            if target_column_name == "retained_activity":
                predictions.rename("churn prediction", inplace=True)
            else:
                predictions.rename(pred_attributes + "_prediction", inplace=True)
    if mode == "train":
        y_test.rename('TEST', inplace=True)

    # compute a df with predictions against the real values to be plotted later
    if mode == "train":
        df = pd.concat([test_case_ids.reset_index(drop=True), test_activities.reset_index(drop=True),
                        current.reset_index(drop=True),
                        predictions.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)
    else:
        #if you are in real running cases you have only one prediction per case
        df = pd.concat([pd.Series(test_case_ids.unique()), test_activities.reset_index(drop=True),
                        current.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)
    if pred_column == 'remaining_time':
        # convert to days only if you predict remaining time and if you're not returning results to MyInvenio
        if mode == "train":
            df.iloc[:, 2:] = df.iloc[:, 2:] / (24.0 * 3600)
        else:
            #convert to milliseconds if you have to return results to MyInvenio, otherwise predict in local in seconds
            df.iloc[:, 2:] = df.iloc[:, 2:] * 1000
    df.rename(columns={df.columns[0]: 'CASE ID', df.columns[1]: "Activity"}, inplace=True)
    return df


def write_and_plot_results(df, pred_attributes):
    df.to_csv(folders['results']['results_completed'], index=False)
    if pred_attributes is not None:
        if not type(pred_attributes) == np.str:
            plot_auroc_curve(df, ["churn prediction"], ["TEST"])
            plot_precision_recall_curve(df, ["churn prediction"], ["TEST"])
        else:
            plot_auroc_curve(df, [pred_attributes + "_prediction"], ["TEST"])
            plot_precision_recall_curve(df, [pred_attributes + "_prediction"], ["TEST"])

    # df["Error"] = abs(df["TEST"] - df.loc[:, df.columns[2]])

    # plot avg error per case duration (to see influence of outliers)
    # df = df.sort_values("TEST").groupby("TEST").mean().reset_index()
    # plt.scatter(df["TEST"], abs(df["lead_time"] - df["TEST"]), color="blue", lw=2, label="Case")
    # # sns.regplot(x=df["TEST"], y=abs(df["lead_time"] - df["TEST"]), lowess=True)
    # # plot also average case length and 95% quantile
    # mean_length = round(df["TEST"].mean())
    # quantile_95 = round(df.loc[round(len(df["TEST"]) / 100 * 95), "TEST"])
    # cases_percentage_below_error = round((len(df[abs(df["lead_time"] - df["TEST"]) < 50]) / len(df)) * 100, 2)
    # plt.axvline(x=mean_length, color='r', label=f'Avg case duration= {mean_length} days')
    # plt.axvline(x=quantile_95, color='k', label=f'95% quantile= {quantile_95} days')
    # plt.xticks(rotation=45)
    # plt.xlabel("Case duration")
    # plt.ylabel("Prediction error")
    # plt.legend()
    # plt.savefig("experiment_files/plots/error_on_case_duration.png", dpi=300, bbox_inches="tight")

    # plot frequency for each step (to understand in which points the error should be low)
    # frequency_step = df.groupby(["Events from end"]).count()["Error"]
    # df.groupby(["Events from end"]).count()["Error"].plot()
    # plt.xticks(np.arange(min(frequency_step.index), max(frequency_step.index) + 1, 1))
    # plt.gca().invert_xaxis()
    # plt.savefig(experiment_name + "/plots/frequency_step.png", dpi=300, bbox_inches="tight")
    # plt.clf()

    # # compute mean and median error per step
    # error_step = df.groupby(["Events from end"]).mean()["Error"].round(2)
    # df.groupby(["Events from end"]).mean()["Error"].round(2).plot()
    # plt.xticks(np.arange(min(error_step.index), max(error_step.index) + 1, 1))
    # plt.gca().invert_xaxis()
    # plt.savefig(experiment_name + "/plots/error_step_mean.png", dpi=300, bbox_inches="tight")
    # plt.clf()

    # error_step = df.groupby(["Events from end"]).median()["Error"].round(2)
    # df.groupby(["Events from end"]).median()["Error"].round(2).plot()
    # plt.xticks(np.arange(min(error_step.index), max(error_step.index) + 1, 1))
    # plt.gca().invert_xaxis()
    # plt.savefig(experiment_name + "/plots/error_step_median.png", dpi=300, bbox_inches="tight")
    # plt.clf()

    # # heatmap mean and median error per step-activity
    # df_mean = df.groupby(["Activity", "Events from end"]).mean()["Error"].round(2).unstack().sort_index(axis=1,
    #                                                                                                ascending=False)
    # fig, ax = plt.subplots(figsize=(20, 10))
    # sns.heatmap(df_mean, cbar=True, cmap="RdBu_r", center=0, annot=True, fmt='g', robust=True, ax=ax,
    #             annot_kws={"fontsize": 13})
    # plt.savefig(experiment_name + "/plots/error_heatmap_mean.png", dpi=300, bbox_inches="tight")

    # df_median = df.groupby(["Activity", "Events from end"]).median()["Error"].round(2).unstack().sort_index(axis=1,
    #                                                                                                  ascending=False)
    # fig, ax = plt.subplots(figsize=(20, 10))
    # sns.heatmap(df_median, cbar=True, cmap="RdBu_r", center=0, annot=True, fmt='g', robust=True, ax=ax,
    #             annot_kws={"fontsize": 13})
    # plt.savefig(experiment_name + "/plots/error_heatmap_median.png", dpi=300, bbox_inches="tight")
    # print("Plotted Errors")

def write_scores(y_test, y_pred, target_column_name, pred_attributes=None):
    if pred_attributes is None:
        mae = mean_absolute_error(y_test, y_pred)
        to_days = lambda x: x / (3600 * 24)
    if 'time' in target_column_name:
        days = round(to_days(mae), 3)
        print('Prediction MAE is:', days)
        scores = {'MAE': days}
        write(scores, folders['results']['scores'])
    elif pred_attributes is None:
        res = round(mae, 3)
        print('Prediction MAE is:', res)
        res = {'MAE': res}
        write(res, folders['results']['scores'])
    else:
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)
        auroc_score = roc_auc_score(y_test, y_pred)
        if not type(pred_attributes) == np.str:
            target = target_column_name
        else:
            target = pred_attributes
        # print(f"Accuracy score for {target}: {accuracy}\n")
        # print(f"Confusion matrix: {cm}")
        # print(f"Auroc score: {auroc_score}")
        # print(f"Average precision recall score: {average_precision}")
        # print(f"F1 score for {target}: {f1}\n")
        res = {'F1': f1, "Accuracy": accuracy}
        write(res, folders['results']['scores'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Will not be performed', 'Will be performed'])
        disp.plot(cmap=plt.cm.Blues, values_format='.5g')
        plt.savefig(f"{os.getcwd()}/experiment_files/plots/confusion_matrix_{target}.png", dpi=300, bbox_inches="tight")


def plot_auroc_curve(df, predictions_names, target_column_names, experiment_name=None):
    false_positive_rates = dict()
    true_positive_rates = dict()
    roc_auc = dict()
    for i, predictions, target in zip(range(len(predictions_names)), predictions_names, target_column_names):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(df[target], df[predictions])
        false_positive_rates[i] = false_positive_rate
        true_positive_rates[i] = true_positive_rate
        roc_auc[i] = auc(false_positive_rate, true_positive_rate)
    # First aggregate all false positive rates (not nan)
    n_classes = len(predictions_names)
    all_fpr = np.unique(np.concatenate([false_positive_rates[i] for i in range(n_classes) if
                                        not math.isnan(auc(false_positive_rates[i], true_positive_rates[i]))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    instances_for_mean = 0
    # from mean macro exclude examples that are nan
    for i in range(n_classes):
        if not math.isnan(auc(false_positive_rates[i], true_positive_rates[i])):
            mean_tpr += interp(all_fpr, false_positive_rates[i], true_positive_rates[i])
            instances_for_mean = instances_for_mean + 1

            # Finally average it and compute AUC
    mean_tpr /= instances_for_mean

    false_positive_rates["macro"] = all_fpr
    true_positive_rates["macro"] = mean_tpr
    roc_auc["macro"] = auc(false_positive_rates["macro"], true_positive_rates["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(false_positive_rates["macro"], true_positive_rates["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    color = []
    for i in range(len(predictions_names)):
        r = lambda: random.randint(0, 255)
        color.append('#%02X%02X%02X' % (r(), r(), r()))
    colors = cycle(color)
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(false_positive_rates[i], true_positive_rates[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(predictions_names[i].replace(" prediction", ""), roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=(1, -.10))
    plt.savefig(f"{os.getcwd()}/experiment_files/plots/auroc_{predictions_names[0]}.png", dpi=300, bbox_inches="tight")


def plot_precision_recall_curve(df, predictions_names, target_column_names):
    precisions = dict()
    recalls = dict()
    average_precisions = dict()
    for i, predictions, target in zip(range(len(predictions_names)), predictions_names, target_column_names):
        precision, recall, thresholds = precision_recall_curve(df[target], df[predictions])
        precisions[i] = precision
        recalls[i] = recall
        average_precisions[i] = average_precision_score(df[target], df[predictions])
        # print("Target {}:\n  Precision {}\n  Recall {}\n  Thresholds {}\n".format(predictions, precision, recall,
        #                                                                           thresholds))

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    color = []
    for i in range(len(predictions_names)):
        r = lambda: random.randint(0, 255)
        color.append('#%02X%02X%02X' % (r(), r(), r()))
    colors = cycle(color)

    for i, color in zip(range(len(predictions_names)), colors):
        l, = plt.plot(recalls[i], precisions[i], color="blue", lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(predictions_names[i], average_precisions[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(1.02,+.01), prop=dict(size=14))
    plt.savefig(f"{os.getcwd()}/experiment_files/plots/pr_curve_{predictions_names[0]}.png", dpi=300, bbox_inches="tight")

def cast_from_seconds_to_days_hours(seconds):
    from datetime import datetime, timedelta
    date_extended = datetime(1, 1, 1) + timedelta(seconds=int(seconds))
    date = ""
    if date_extended.day != 0:
        date += f"{date_extended.day - 1}d "
    if date_extended.hour != 0:
        date += f"{date_extended.hour}h "
    if date_extended.minute != 0:
        date += f"{date_extended.minute}m"

    return date

def write_grid_results(model_type, mean_events, model_score, model_best_params, column_type):
    if model_type != "no history":
        best_params = read(folders['model']['params'])
        overall_scores = read(folders['model']['scores'])
        early_stop_counter = best_params["early_counter"]
    else:
        best_params = {}
        overall_scores = {}
        early_stop_counter = 0

    # std_train_error = min(cv_results["cv_results"]["train-MAE-std"])
    # std_validation_error = min(cv_results["cv_results"]["test-MAE-std"])
    best_params[f"{model_type}_depth"] = model_best_params["depth"]
    best_params[f"{model_type}_iterations"] = model_best_params["iterations"]
    if column_type != "Categorical":
        print(f'Train error: {model_score["train"]}\n'
              f'Validation error: {model_score["validation"]}\n')
    else:
        print(f'Train F1 score: {model_score["train"]}\n'
              f'Validation F1 score: {model_score["validation"]}\n')
    print(f"Best params: {model_best_params}\n")

    overall_scores[f"{model_type}_train"] = model_score["train"]
    overall_scores[f"{model_type}_validation"] = model_score["validation"]
    # overall_scores[f"{model_type}_train_std"] = std_train_error
    # overall_scores[f"{model_type}_validation_std"] = std_validation_error
    if model_type == "no history":
        overall_scores["Best"] = model_score["validation"]
        best_params["early_counter"] = early_stop_counter
        write(best_params, folders['model']['params'])
        write(overall_scores, folders['model']['scores'])
    else:
        if column_type != "Categorical":
            if model_score["validation"] < overall_scores["Best"]:
                # if the increment in accuracy is lower than 1% then stop
                if (1 - model_score["validation"] / overall_scores["Best"]) < 0.01:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                overall_scores["Best"] = model_score["validation"]
            else:
                early_stop_counter += 1
        else:
            if model_score["validation"] > overall_scores["Best"]:
                # if the increment in accuracy is lower than 1% then stop
                if (model_score["validation"] / overall_scores["Best"] - 1) < 0.01:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                overall_scores["Best"] = model_score["validation"]
            else:
                early_stop_counter += 1
        best_params["early_counter"] = early_stop_counter
        write(best_params, folders['model']['params'])
        write(overall_scores, folders['model']['scores'])
    print(f'Saved scores for {model_type}')
    if model_type == mean_events:
        # if early_stop_counter == 2 or model_type == mean_events:
        # pick the model with the best result and save it
        best_saved_value = overall_scores.pop("Best")
        best_model = (list(overall_scores.keys())[list(overall_scores.values()).index(
            best_saved_value)]).replace('_validation', '')
        #this is needed just in case validation best score is equal to train best score
        if "_train" in best_model:
            best_model = best_model.replace('_train', '')
        best_params["history"] = best_model
        # overall_scores["Best"] = best_saved_value
        best_params["best_iterations"] = best_params[f"{best_model}_iterations"]
        best_params["best_depth"] = best_params[f"{best_model}_depth"]
        write(best_params, folders['model']['params'])
        # write(overall_scores, folders['model']['scores'])
    else:
        return

def compare_best_validation_curves(pred_column, mean_reference_target=None):
    x = []
    y_train_percentage = []
    y_validation_percentage = []
    y_validation = []
    y_train = []
    lw = 2
    overall_scores = read(folders['model']['scores'])
    for name, score in overall_scores.items():
        if name != "Best" and "_std" not in name:
            if "_validation" in name:
                x.append(name.replace("_validation", ""))
                if pred_column == "remaining_time":
                    y_validation.append(score / 3600)
                    y_validation_percentage.append(round((score / (mean_reference_target / 1000)) * 100, 2))
                else:
                    y_validation.append(score)
                    if mean_reference_target is not None:
                        y_validation_percentage.append(round((score / mean_reference_target)*100, 2))
            else:
                if pred_column == "remaining_time":
                    y_train.append(score / 3600)
                    y_train_percentage.append(round((score / (mean_reference_target / 1000)) * 100, 2))
                else:
                    y_train.append(score)
                    if mean_reference_target is not None:
                        y_train_percentage.append(round((score / mean_reference_target)*100, 2))

    if mean_reference_target is not None:
        plt.clf()
        plt.xlabel("History timesteps")
        plt.ylabel("Error %")
        plt.plot(x, y_train_percentage, color="red", lw=lw, label="Training score")
        plt.plot(x, y_validation_percentage, color="blue", lw=lw, label="Validation score")
        plt.xticks(rotation=30)
        plt.legend(loc="best")
        plt.savefig(f"{os.getcwd()}/experiment_files/plots/compare_best_error_percentage.png", dpi=300, bbox_inches="tight")

    plt.clf()
    plt.xlabel("History timesteps")
    if pred_column == "remaining_time":
        plt.ylabel("Error hours (MAE)")
    elif pred_column == "case_cost":
        plt.ylabel("Error Euros (MAE)")
    else:
        plt.ylabel("F1 score")
    plt.plot(x, y_train, color="red", lw=lw, label="Training score")
    plt.plot(x, y_validation, color="blue", lw=lw, label="Validation score")
    plt.xticks(rotation=30)
    plt.legend(loc="best")
    if pred_column == "remaining_time":
        plt.savefig(f"{os.getcwd()}/experiment_files/plots/compare_best_error_hours.png", dpi=300, bbox_inches="tight")
    else:
        plt.savefig(f"{os.getcwd()}/experiment_files/plots/compare_best_error.png", dpi=300, bbox_inches="tight")
    print("Plotted train and validation curves")

def histogram_median_events_per_dataset(df, case_id_name, activity_column_name, start_date_name, end_date_name=None):
    #the dataset has dates in milliseconds
    if end_date_name is not None:
        avg_duration_days = (df.groupby(case_id_name)[end_date_name].max() -
                             df.groupby(case_id_name)[start_date_name].min()).mean() / (1000*3600*24)
        median_duration_days = (df.groupby(case_id_name)[end_date_name].max() -
                             df.groupby(case_id_name)[start_date_name].min()).median() / (1000*3600*24)
        std_dev_duration_days = (df.groupby(case_id_name)[end_date_name].max() -
                                df.groupby(case_id_name)[start_date_name].min()).std() / (1000 * 3600 * 24)
    else:
        avg_duration_days = (df.groupby(case_id_name)[start_date_name].max() -
                             df.groupby(case_id_name)[start_date_name].min()).mean() / (1000*3600*24)
        median_duration_days = (df.groupby(case_id_name)[start_date_name].max() -
                             df.groupby(case_id_name)[start_date_name].min()).median() / (1000*3600*24)
        std_dev_duration_days = (df.groupby(case_id_name)[start_date_name].max() -
                                df.groupby(case_id_name)[start_date_name].min()).std() / (1000 * 3600 * 24)

    # la serie è gruppata per sè stessa per avere il count
    distribution_of_cases_length = df.groupby(case_id_name).count()[activity_column_name].groupby(
        df.groupby(case_id_name).count()[activity_column_name]).sum()
    ax = distribution_of_cases_length.plot(kind="bar", figsize=(20, 10), color="blue")
    ax.set_xlabel("Events")
    ax.set_ylabel("# Cases")

    median_events = np.median(df.groupby(case_id_name).count()[activity_column_name])
    mean_events = np.mean(df.groupby(case_id_name).count()[activity_column_name])
    text = f"Mean events / case: {round(mean_events, 2)}\nMedian events / case: {median_events}\n" \
           f"Mean process duration: {round(avg_duration_days, 2)} days\n" \
           f"Median process duration: {round(median_duration_days, 2)} days\n" \
           f"Standard deviation process duration: {round(std_dev_duration_days, 2)} days"
    print(text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.savefig(f"{os.getcwd()}/experiment_files/plots/distribution_of_cases_length.png", dpi=300, bbox_inches="tight")
    plt.clf()
    print("Plotted dataset statistics")