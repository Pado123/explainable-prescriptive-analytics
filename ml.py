from statistics import median

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, average_precision_score, roc_auc_score, make_scorer, \
    mean_absolute_error, log_loss
from sklearn.preprocessing import LabelEncoder
import utils

import shap
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from catboost.utils import select_threshold

from explainable import (find_explanations_for_completed_cases,
                         find_explanations_for_running_cases)
from write_results import (prepare_csv_results, write_and_plot_results,
                           write_scores, write_grid_results, compare_best_validation_curves)
from logme import log_it

from IO import read, write, folders
from os.path import join, exists

import pickle


@log_it
def save_column_information_for_real_predictions(dfTrain, dfTrain_without_valid, dfTest, dfValid, train_cases,
                                                 event_level, target_column_name, column_type, categorical_features,
                                                 activity_name):
    # save columns info to be later retrieved when predicting real data
    # you need this for the shape when you have to predict real data
    # save also case indexes you have randomly taken (for old replication purposes, now fixed seed)

    info = read(folders['model']['data_info'])

    if type(train_cases) == np.ndarray:
        info["train_cases"] = train_cases.tolist()
    else:
        info["train_cases"] = train_cases
    info['columns'] = dfTrain.iloc[:, 1:-1].columns.to_list()
    df_types = dfTrain.dtypes.to_frame('dtypes').reset_index()
    info["column_types"] = df_types.set_index('index')['dtypes'].astype(str).to_dict()
    if event_level == 0:
        info['test'] = 'case'
        info['y_columns'] = target_column_name
    else:
        info['test'] = 'event'
        info['y_columns'] = [target_column_name]
    info["column_type"] = column_type
    info["categorical_features"] = categorical_features.to_list()
    dfTrain = utils.change_history(dfTrain, activity_name)
    dfTrain_without_valid = utils.change_history(dfTrain_without_valid, activity_name)
    dfValid = utils.change_history(dfValid, activity_name)
    dfTest = utils.change_history(dfTest, activity_name)
    write(info, folders['model']['data_info'])
    write(dfTrain, folders['model']['dfTrain'])
    write(dfTrain_without_valid, folders['model']['dfTrain_without_valid'])
    write(dfValid, folders['model']['dfValid'])
    write(dfTest, folders['model']['dfTest'])


def custom_2cv(train_indexes, valid_indexes):
    yield train_indexes, valid_indexes


def grid_search(model_type, mean_events, column_type):
    info = read(folders['model']['data_info'])
    categorical_features = info["categorical_features"]
    X_train = read(folders['model']['dfTrain']).iloc[:, 1:-1]
    y_train = read(folders['model']['dfTrain']).iloc[:, -1]

    params = {"task_type": "CPU", "thread_count": 4,
              "learning_rate": 0.01, "early_stopping_rounds": 5, "logging_level": "Silent"}
    if column_type != "Categorical":
        params["loss_function"] = "MAE"
        model = CatBoostRegressor(**params)
    else:
        params["loss_function"] = "Logloss"
        # optimize custom metric on validation set
        params["eval_metric"] = "F1"
        model = CatBoostClassifier(**params)

    # try each grid on every model
    param_grid = {
        'iterations': [1500, 3000, 4000],
        'depth': [3, 6, 10]
    }

    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    # X_train_without_valid = read(folders['model']['dfTrain_without_valid']).iloc[:, 1:-1].reset_index(drop=True)
    # X_valid = read(folders['model']['dfValid']).iloc[:, 1:-1].reset_index(drop=True)
    # dfTrain = X_train_without_valid.append(X_valid, ignore_index=True)
    # train_indexes = dfTrain[:len(X_train_without_valid)].index
    # valid_indexes = dfTrain[len(X_train_without_valid):].index
    # custom_cv = custom_2cv(train_indexes, valid_indexes)
    # cv_results = model.grid_search(param_grid, train_pool, shuffle=False, refit=False, cv=custom_cv)
    cv_results = model.grid_search(param_grid, train_pool, shuffle=False, refit=False)

    model_score = {}
    model_best_params = cv_results["params"]

    # the min is related to the best iteration (number of trees) (try all parameters for this particular history)
    if column_type != "Categorical":
        model_score["train"] = cv_results["cv_results"]["train-MAE-mean"][-1]
        model_score["validation"] = cv_results["cv_results"]["test-MAE-mean"][-1]
    else:
        model_score["train"] = cv_results["cv_results"]["train-F1-mean"][-1]
        model_score["validation"] = cv_results["cv_results"]["test-F1-mean"][-1]

    write_grid_results(model_type, mean_events, model_score, model_best_params, column_type)


def grid_search_sklearn(X_train, y_train, model_type, mean_events, pred_column, activity_name):
    from sklearn.model_selection import GridSearchCV
    params = {"task_type": "CPU", "thread_count": 4,
              "learning_rate": 0.01, "early_stopping_rounds": 5, "logging_level": "Silent"}
    if pred_column != activity_name:
        params["loss_function"] = "MAE"
        model = CatBoostRegressor(**params)
    else:
        params["loss_function"] = "Logloss"
        model = CatBoostClassifier(**params)

    param_grid = {
        'iterations': [1500, 3000, 4000],
        'depth': [3, 6, 10]
    }

    # le griglie per il grid search hanno sempre bisogno di trasformare gli attributi categorici
    X_train_enc = X_train.apply(LabelEncoder().fit_transform)
    scorer = make_scorer(mean_absolute_error)
    fit_params = {"n_jobs": 4}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, refit=False,
                        return_train_score=True, cv=3, verbose=1, **fit_params)
    # results here are worse because categorical variables must be encoded
    grid.fit(X_train_enc, y_train, verbose=False)

    model_best_params = grid.best_params_
    model_score = {}
    model_score["validation"] = grid.best_score_
    model_score["train"] = grid.cv_results_["mean_train_score"][
        np.where(grid.cv_results_["rank_test_score"] == 1)[0][0]]

    write_grid_results(model_type, mean_events, model_score, model_best_params)


def catboost_cv(X_train, y_train, model_type, mean_events, pred_column, activity_name, categorical_features):
    from catboost import cv

    params = {"task_type": "CPU", "thread_count": 4, "learning_rate": 0.01, "early_stopping_rounds": 5,
              "logging_level": "Silent", "iterations": 3000, "depth": 10}
    if pred_column != activity_name:
        params["loss_function"] = "MAE"
    else:
        params["loss_function"] = "Logloss"

    cv_dataset = Pool(data=X_train, label=y_train, cat_features=categorical_features)
    print(f'Starting training for {model_type}...')
    scores = cv(cv_dataset, params, shuffle=False, fold_count=3)

    model_score = {}
    model_best_params = {"iterations": 3000, "depth": 10}

    if pred_column != activity_name:
        model_score["train"] = min(scores["train-MAE-mean"])
        model_score["validation"] = min(scores["test-MAE-mean"])
    else:
        model_score["train"] = min(scores["train-Logloss-mean"])
        model_score["validation"] = min(scores["cv_results"]["test-Logloss-mean"])

    write_grid_results(model_type, mean_events, model_score, model_best_params)


def balance_examples_target_column(df, case_id_name, train_cases_0, train_cases_1):
    scale_pos_weight = len(train_cases_0) / len(train_cases_1)
    # replicate unbalanced class (we balance case ids, not the single events)
    retained_cases = train_cases_1
    dfTrain = df.loc[df[df[case_id_name].isin(retained_cases)].index.repeat(scale_pos_weight)].reset_index(drop=True)
    dfTrain = dfTrain.append(df[df[case_id_name].isin(train_cases_0)].reset_index(drop=True), ignore_index=True)
    balanced_cases = dfTrain[case_id_name]
    # 20% of case ids are taken as validation
    valid_cases = np.random.choice(balanced_cases.unique(), size=int(len(balanced_cases.unique()) / 100 * 20),
                                   replace=False)
    dfValid = dfTrain[dfTrain[case_id_name].isin(valid_cases)]
    dfTrain_without_valid = dfTrain[~dfTrain[case_id_name].isin(valid_cases)]
    return dfTrain, dfTrain_without_valid, dfValid


def balance_weights(y_train, params):
    # balance weights if dataset unbalanced (we consider unbalanced when class one is max 10% of the other class)
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    if scale_pos_weight >= 10:
        params["scale_pos_weight"] = scale_pos_weight
    return params


@log_it
def generate_train_and_test_sets(df, target_column, target_column_name, event_level, column_type, override,
                                 case_id_name, df_completed_cases, activity_name):
    # reattach predict column before splitting
    df[target_column_name] = target_column
    df.columns = df.columns.str.replace('time_from_midnight', 'daytime')
    # around 2/3 cases are the training set, 1/3 is the test set
    second_quartile = len(np.unique(df.iloc[:, 0])) / 2
    third_quartile = len(np.unique(df.iloc[:, 0])) / 4 * 3

    categorical_features = df.iloc[:, 1:-1].select_dtypes(exclude=np.number).columns
    df[categorical_features] = df[categorical_features].astype(str)

    if (column_type == "Categorical") and ((len(df[df[target_column_name] == 0][case_id_name].unique()) /
                                            len(df[df[target_column_name] == 1][case_id_name].unique())) > 10):
        unbalanced = True
    else:
        unbalanced = False

    # DO NOT REMOVE CASE, IT IS NEEDED IF WE WANT TO COMPARE DIFFERENT ALGORITHMS ON THE TEST SET
    # if trained model exists just pick the previously chosen case indexes
    if ("train_cases" in read(folders['model']['data_info'])) or (exists(folders['model']['model'])
                                                                  and override is False):
        train_cases = read(folders['model']['data_info'])["train_cases"]
        print("Reloaded train cases")
        # dfTrain = df[df[case_id_name].isin(train_cases)]
        dfTest = df[~df[case_id_name].isin(train_cases)]
    else:
        # we consider the class unbalanced when the class is 1/10 or lower (equally distribute the 1 targets)
        if unbalanced:
            number_train_cases = round((third_quartile + second_quartile) / 2)
            # 50% of cases_1 go in train, 50% in test
            cases_0 = df[df[target_column_name] == 0][case_id_name].unique()
            cases_1 = df[df[target_column_name] == 1][case_id_name].unique()
            # if there is at least a 1 in the case (cases_1) then do not include it in cases_0
            cases_0 = cases_0[~np.isin(cases_0, cases_1)]
            number_train_cases_1 = round(len(cases_1) / 2)
            number_train_cases_0 = number_train_cases - number_train_cases_1
            train_cases_0 = np.random.choice(cases_0, size=number_train_cases_0, replace=False)
            train_cases_1 = np.random.choice(cases_1, size=number_train_cases_1, replace=False)
            train_cases = np.append(train_cases_0, train_cases_1)
            dfTest = df[~df[case_id_name].isin(train_cases)]

        else:
            # take cases for training in random order (the seed is fixed for replicability)
            cases = df[case_id_name].unique()
            number_train_cases = round((third_quartile + second_quartile) / 2)
            train_cases = np.random.choice(cases, size=number_train_cases, replace=False)
            # dfTrain = df[df[case_id_name].isin(train_cases)]
            dfTest = df[~df[case_id_name].isin(train_cases)]
    df_completed_cases = df_completed_cases.loc[df_completed_cases['CASE ID'].isin(dfTest[case_id_name].unique()), :]
    df_completed_cases.to_csv(folders['results']['completed'], index=False)

    # TODO: investigate reducing number of 0 examples. Investigate reducing 1 and using fraud detection algo
    # Investigate 3 models in parallel trained on balanced datasets
    # in that case how do we cope with the parameters and the grid_search?

    if unbalanced:
        # The 1 targets should be distributed proportionally also between validation and train
        if ("train_cases" in read(folders['model']['data_info'])):
            dfTrain = df[df[case_id_name].isin(train_cases)]
            train_cases_0 = dfTrain.loc[dfTrain[target_column_name] == 0, case_id_name].unique()
            train_cases_1 = dfTrain.loc[dfTrain[target_column_name] == 1, case_id_name].unique()
            train_cases_0 = train_cases_0[~np.isin(train_cases_0, train_cases_1)]
        valid_cases_0 = np.random.choice(train_cases_0, size=int(len(train_cases_0) / 100 * 20), replace=False)
        valid_cases_1 = np.random.choice(train_cases_1, size=int(len(train_cases_1) / 100 * 20), replace=False)
        valid_cases = np.append(valid_cases_0, valid_cases_1)
    else:
        valid_cases = np.random.choice(train_cases, size=int(len(train_cases) / 100 * 20), replace=False)
    dfValid = df[df[case_id_name].isin(valid_cases)]
    dfTrain_without_valid = df[df[case_id_name].isin(train_cases) & ~df[case_id_name].isin(valid_cases)]
    dfTrain = dfTrain_without_valid.append(dfValid, ignore_index=True)

    # if unbalanced: #leave this if you want to balance the num of the targets
    #     #at this point you can balance the number of the targets
    #     dfTrain, dfTrain_without_valid, dfValid = balance_examples_target_column(df, case_id_name, train_cases_0, train_cases_1)

    if not exists(folders['model']['model']) or override:
        save_column_information_for_real_predictions(dfTrain, dfTrain_without_valid, dfTest, dfValid, train_cases,
                                                     event_level, target_column_name, column_type, categorical_features,
                                                     activity_name)


def fit_model(column_type, history, case_id_name, activity_name, experiment_name, oversample=False):

    info = read(folders['model']['data_info'])
    categorical_features = info["categorical_features"]
    column_types = info["column_types"]

    X_train_without_valid = read(folders['model']['dfTrain_without_valid'], dtype=column_types).iloc[:, 1:-1]
    y_train_without_valid = read(folders['model']['dfTrain_without_valid'], dtype=column_types).iloc[:, -1]
    X_train = read(folders['model']['dfTrain'], dtype=column_types).iloc[:, 1:-1]
    y_train = read(folders['model']['dfTrain'], dtype=column_types).iloc[:, -1]
    X_valid = read(folders['model']['dfValid'], dtype=column_types).iloc[:, 1:-1]
    y_valid = read(folders['model']['dfValid'], dtype=column_types).iloc[:, -1]

    params = {
        'depth': 10,
        'learning_rate': 0.01,
        'iterations': 3000,
        'early_stopping_rounds': 5,
        'thread_count': 4,
        'logging_level': 'Silent',
        'task_type': "CPU"  # "GPU" if int(os.environ["USE_GPU"]) else "CPU"
    }

    if exists(folders['model']['params']):
        best_params = read(folders['model']['params'])
        # comment these 3 lines when you just reload the model
        print(
            f'Best params for model - Depth: {best_params["best_depth"]} Iterations: {best_params["best_iterations"]}')
        params["depth"] = best_params["best_depth"]
        params["iterations"] = best_params["best_iterations"]
    else:
        config = {"history": history[0]}
        write(config, folders['model']['params'])

    # pickle.dump(categorical_features, open('vars/act/categorical_features.pkl','wb'))
    # pickle.dump(params, open('vars/act/model_params.pkl','wb'))

    if not exists(folders['model']['model']):
        print('Starting training...')
        if column_type != "Categorical":
            params["loss_function"] = "MAE"
            train_data = Pool(X_train, y_train, cat_features=categorical_features)
            model = CatBoostRegressor(**params)
            model.fit(train_data)
        else:
            params["loss_function"] = "Logloss"
            params["eval_metric"] = "F1"
            # balance_weights(y_train, params) #leave this if you want to balance weights
            # we train without valid and then we find a good threshold on the valid, to be later applied on the test set

            if oversample:
                from collections import Counter
                counter = Counter(y_train_without_valid)
                print('Before was', counter)

                print('balancing')
                from imblearn.over_sampling import SMOTENC
                idxs = []
                c = 0
                a = list(X_train.loc[0])
                for i in a:
                    if isinstance(i, str):
                        print(c)
                        idxs.append(c)
                    c += 1

                sm = SMOTENC(random_state=1618, categorical_features=idxs)
                X_train_without_valid, y_train_without_valid = sm.fit_resample(X_train_without_valid,
                                                                               y_train_without_valid)

                counter = Counter(y_train_without_valid)
                print('After is', counter)

            train_data = Pool(X_train_without_valid, y_train_without_valid, cat_features=categorical_features)
            eval_pool = Pool(X_valid, y_valid, cat_features=categorical_features)
            model = CatBoostClassifier(**params)
            model.fit(train_data)

            fnrs = [0.10, 0.20, 0.30, 0.40, 0.50]
            decision_thresholds = []
            for fnr in fnrs:
                decision_threshold = select_threshold(model=model, data=eval_pool, FNR=fnr)
                print(f"Decision threshold for FNR {fnr}: {decision_threshold}")
                decision_thresholds.append(decision_threshold)
            info["decision_thresholds"] = decision_thresholds
            train_data = Pool(X_train, y_train, cat_features=categorical_features)
            model = CatBoostClassifier(**params)
            print('Re training on all train set...')
            model.fit(train_data)

        write(model, folders['model']['model'])
        write(info, folders['model']['data_info'])
        print('Saved model')


def predict(column_type, target_column_name, activity_name):

    info = read(folders['model']['data_info'])
    categorical_features = info["categorical_features"]
    column_types = info["column_types"]

    dfTest = read(folders['model']['dfTest'], dtype=column_types)
    X_test = dfTest.iloc[:, 1:-1]
    y_test = dfTest.iloc[:, -1]
    test_data = Pool(X_test, cat_features=categorical_features)
    model = read(folders['model']['model'])
    print('Reloaded model')
    print('Starting predicting...')
    if column_type != "Categorical":
        y_pred = predict_numerical_kpi(model, test_data)
    else:
        y_pred = predict_activity(model, test_data, y_test, target_column_name)
    return y_pred


def predict_numerical_kpi(model, test_data):
    y_pred = model.predict(test_data)
    return y_pred


def predict_activity(model, test_data, y_test, target_column_name):
    info = read(folders['model']['data_info'])
    decision_thresholds = info["decision_thresholds"]

    # we train without valid and then we find a good threshold on the valid, to be later applied on the test set
    scores = {}
    best = 0

    y_pred_proba = model.predict_proba(test_data)
    for decision_threshold in decision_thresholds:
        y_pred = [0 if x[1] < decision_threshold else 1 for x in y_pred_proba]
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        average_precision = average_precision_score(y_test, y_pred)
        auroc_score = roc_auc_score(y_test, y_pred)
        log_loss_score = log_loss(y_test, y_pred)
        print(f"\nMetrics for decision threshold {decision_threshold}")
        print(f"Confusion matrix: {cm}")
        print(f"Log_loss score: {log_loss_score}")
        print(f"Auroc score: {auroc_score}")
        print(f"Average precision recall score: {average_precision}")
        print(f"F1 score for {target_column_name}: {f1}\n")
        score = f1
        scores[f"{decision_threshold}"] = score
        if score > best:
            best = score

    best_decision_threshold = float(list(scores.keys())[list(scores.values()).index(best)])
    print(f"Best decision threshold: {best_decision_threshold}\n")
    y_pred = [0 if x[1] < best_decision_threshold else 1 for x in y_pred_proba]
    info["decision_threshold"] = best_decision_threshold
    write(info, folders['model']['data_info'])
    return y_pred


def write_results(y_pred, activity_column_name, target_column_name, pred_attributes, pred_column, mode, column_type, experiment_name, case_id_name):
    column_types = read(folders['model']['data_info'])["column_types"]

    dfTest = read(folders['model']['dfTest'], dtype=column_types)
    X_test = dfTest.iloc[:, 1:-1]
    y_test = dfTest.iloc[:, -1]

    test_case_ids = X_test.iloc[:, 0]
    test_activities = dfTest[activity_column_name] #TODO poi rimetti e togli riga sotto
    # test_activities = X_test[activity_name]

    # retrieve test case id to later show predictions
    test_case_ids = X_test.iloc[:, 0]
    if "activity_duration" in X_test.columns:
        current_times = X_test["time_from_start"] + X_test["activity_duration"]
    else:
        current_times = X_test["time_from_start"]

    df = prepare_csv_results(y_pred, test_case_ids, test_activities, target_column_name, pred_column, mode, column_type,
                             current_times, pred_attributes, y_test)
    # write_and_plot_results(df, pred_attributes)
    write_scores(y_test, y_pred, target_column_name, pred_attributes)


def explain(pred_column, column_type):
    column_types = read(folders['model']['data_info'])["column_types"]
    dfTest = read(folders['model']['dfTest'], dtype=column_types)
    categorical_features = read(folders["model"]["data_info"])["categorical_features"]
    test_data = Pool(dfTest.iloc[:, 1:-1], cat_features=categorical_features)
    model = read(folders['model']['model'])

    print("Calculating explanations...")
    explainer = shap.TreeExplainer(model)
    shapley_test = explainer.shap_values(test_data)
    test_cases = dfTest.iloc[:, 0]
    find_explanations_for_completed_cases(dfTest.iloc[:, 1:-1], test_cases, shapley_test, pred_column, column_type)


def prepare_data_for_ml_model_and_predict(df, target_column, target_column_name, event_level, column_type,
                                          experiment_name, mode, override, activity_column_name, pred_column,
                                          pred_attributes, model_type, mean_events, mean_reference_target, history,
                                          df_completed_cases, case_id_name, grid, shap):

    generate_train_and_test_sets(df, target_column, target_column_name, event_level, column_type, override,
                                 case_id_name, df_completed_cases, activity_name=activity_column_name)

    if grid is True:
        if not exists(folders['model']['params']):
            grid_search(model_type, mean_events, column_type)
            return
        elif "history" not in read(folders['model']['params']):
            grid_search(model_type, mean_events, column_type)
            if "history" in read(folders['model']['params']) or model_type == "end":
                compare_best_validation_curves(pred_column, mean_reference_target)
            return

    fit_model(column_type, history, case_id_name, activity_name=activity_column_name, experiment_name=experiment_name)
    y_pred = predict(column_type, target_column_name, activity_name=activity_column_name)
    write_results(y_pred, activity_column_name, target_column_name, pred_attributes, pred_column, mode, column_type, experiment_name, case_id_name)

    if shap is True:
        explain(pred_column, column_type)
