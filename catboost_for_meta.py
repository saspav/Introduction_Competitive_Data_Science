import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import seaborn as sns

import optuna
from optuna.integration import CatBoostPruningCallback
from catboost import CatBoostRegressor, Pool
from itertools import product

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from data_process import (RANDOM_SEED, WORK_PATH, MODELS_LOGS_REG, get_max_num, DataTransform,
                          set_all_seeds, make_predict_reg, add_info_to_log, merge_submits,
                          MODEL_PATH)

from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

set_all_seeds(seed=RANDOM_SEED)


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial: optuna.Trial) -> float:
    # params_reg = {
    #     "depth": trial.suggest_int("depth", 3, 9),
    #     "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
    #     "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30),
    #     "border_count": trial.suggest_int("border_count", 128, 384),
    #     "random_strength": trial.suggest_float("random_strength", 0.1, 10),
    #     "one_hot_max_size": trial.suggest_int("one_hot_max_size", 3, 9),
    #     "rsm": trial.suggest_float("rsm", 0.5, 1.0),
    #     "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
    #     "bootstrap_type": trial.suggest_categorical("bootstrap_type",
    #                                                 ["Bayesian", "Bernoulli", "MVS"]),
    #     "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method",
    #                                                         ["Newton", "Gradient"]),
    # }

    params = {
        "depth": trial.suggest_int("depth", 2, 5),
        "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.4),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30),
        "border_count": trial.suggest_int("border_count", 64, 384),
        "random_strength": trial.suggest_float("random_strength", 0.1, 10),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 3, 10),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type",
                                                    ["Bayesian", "Bernoulli", "MVS"]),
        "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method",
                                                            ["Newton", "Gradient"]),
    }

    # params = params_reg

    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 20.0)
    elif params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    loss_function = 'RMSE'  # Используем RMSE для регрессии
    eval_metric = 'RMSE'  # Метрическая оценка RMSE
    clf = CatBoostRegressor(
        loss_function=loss_function,
        eval_metric=eval_metric,
        cat_features=cat_columns,
        random_seed=RANDOM_SEED,
        # task_type="GPU",
        **params
    )

    pruning_callback = CatBoostPruningCallback(trial, eval_metric)

    clf.fit(pool_train, eval_set=pool_valid,
            verbose=0,
            early_stopping_rounds=50,
            callbacks=[pruning_callback],
            )

    # evoke pruning manually.
    pruning_callback.check_pruned()

    return clf.best_score_['validation'][eval_metric]  # Получаем RMSE с валидационного набора


def df_stats(df, stat_cols=None):
    if stat_cols is None:
        # Получение списка наименований числовых колонок
        stat_cols = df.select_dtypes(include='number').columns.tolist()
    for func in (np.min, np.max, np.mean, np.median, np.std, pd.Series.skew):
        df[f'{func.__name__}_row'] = df[stat_cols].apply(func, axis=1)
    return df


max_num = get_max_num(log_file=MODELS_LOGS_REG)
sub_pref = 'cbm_'
start_time = print_msg('Обучение Catboost классификатор...')

cat_columns = ['model', 'car_type', 'fuel_type', 'target_class']
features2drop = ['car_id']
numeric_columns = []

# Чтение и предобработка данных
data_cls = DataTransform(use_catboost=True,
                         category_columns=cat_columns,
                         features2drop=features2drop,
                         )

# train_df, test_df = data_cls.make_agg_data()
train, test = data_cls.make_agg_data(file_with_target_class='cb_submit_074_local.csv')

file_train = 'X_train_for_meta.xlsx'
file_test = 'X_test_for_meta.xlsx'
path_for_meta = WORK_PATH.joinpath('for_meta')
train_df = pd.read_excel(path_for_meta.joinpath(file_train), index_col='car_id')
test_df = pd.read_excel(path_for_meta.joinpath(file_test), index_col='car_id')

train_df = df_stats(train_df)
test_df = df_stats(test_df)

print(f'Размер train_df = {train_df.shape}, test = {test_df.shape}')

exclude_columns = [
    # 'CatBoost',
    # 'LightGBM',
    # 'XGBoost',

    # 'ExtraTrees',
    # 'RandomForest',

    'min_row',
    'max_row',
    'mean_row',
    'median_row',
    'std_row',
    'skew_row',
]

ft = pd.read_excel(WORK_PATH.joinpath('features_cb_248.xlsx'))

model_columns = [col for col in test_df.columns.to_list() if col not in exclude_columns]

add_features = ft[ft.Importance > 0.2].Feature.tolist()
if 'target_class' not in add_features:
    add_features.insert(0, 'target_class')

# Читаем список признаков для отдельно обученной модели регрессии:
# нужно взять от той модели которая будет в final_estimator - у нас: для CatBoost'а
ft = pd.read_excel(MODEL_PATH.joinpath('features_cb_248.xlsx'))
# Отфильтруем наиболее важные признаки - они будут добавлены к признакам для final_estimator
add_features = [col for col in ft[ft.Importance > 0.2].Feature if col in model_columns]
if 'target_class' not in add_features and 'target_class' in model_columns:
    add_features.insert(0, 'target_class')

final_cat_columns = [col for col in cat_columns if col in add_features]

# Попробовать разные комбинации

# ['CatBoost', 'XGBoost', 'RandomForest'] RMSE=5.61
# ['CatBoost', 'XGBoost', 'ExtraTrees'] RMSE=5.72
# ['CatBoost', 'LightGBM', 'RandomForest'] RMSE=5.65
# ['CatBoost', 'LightGBM', 'ExtraTrees'] RMSE=5.66
# ['CatBoost', 'LightGBM', 'XGBoost'] RMSE=5.56
# ['CatBoost', 'LightGBM', 'XGBoost', 'ExtraTrees'] RMSE=5.69
# ['CatBoost', 'LightGBM', 'XGBoost', 'RandomForest'] RMSE=5.69
# ['CatBoost', 'LightGBM', 'XGBoost', 'ExtraTrees', 'RandomForest'] RMSE=5.66
model_columns = ['CatBoost', 'LightGBM', 'XGBoost'] + add_features

final_cat_columns = [col for col in cat_columns if col in add_features]

cat_columns = final_cat_columns

train = train_df[model_columns].copy()
target = train_df['target_reg']
test_df = test_df[model_columns].copy()

for col in cat_columns:
    train[col] = train[col].astype('category')
    test_df[col] = test_df[col].astype('category')

print('Обучаюсь на колонках:', model_columns)
print('Категорийные колонки:', cat_columns)
print('Исключенные колонки:', exclude_columns)

print('train.shape', train.shape, 'пропусков:', train.isna().sum().sum())
print('test.shape', test_df.shape, 'пропусков:', test_df.isna().sum().sum())

test_df.reset_index(names='car_id', inplace=True)

test_sizes = (0.2,)

for test_size in test_sizes:

    max_num += 1

    # test_size = 0.3

    # num_iters = 7000
    # SEED = 17

    num_folds = 5

    test_size = round(test_size, 2)

    print(f'valid_size: {test_size} SEED={RANDOM_SEED}')

    stratified = None
    stratified = ['target_class']

    stratified_target = train_df[stratified] if stratified else None

    # Разделение на обучающую и валидационную выборки

    X_train, X_valid, y_train, y_valid = train_test_split(train, target,
                                                          test_size=test_size,
                                                          stratify=stratified_target,
                                                          random_state=RANDOM_SEED)

    splited = X_train, X_valid, y_train, y_valid

    print('X_train.shape', X_train.shape, 'пропусков:', X_train.isna().sum().sum())
    print('X_valid.shape', X_valid.shape, 'пропусков:', X_valid.isna().sum().sum())

    pool_train = Pool(data=X_train, label=y_train, cat_features=cat_columns)
    pool_valid = Pool(data=X_valid, label=y_valid, cat_features=cat_columns)

    skf = StratifiedKFold(n_splits=num_folds, random_state=RANDOM_SEED, shuffle=True)
    split_kf = KFold(n_splits=num_folds, random_state=RANDOM_SEED, shuffle=True)

    fit_on_full_train = False
    use_grid_search = True
    use_cv_folds = True
    build_model = True
    write_log = True

    models, models_scores, predicts = [], [], []

    loss_function = 'RMSE'  # Используем RMSE для регрессии
    eval_metric = 'RMSE'  # Метрическая оценка RMSE

    iterations = 1_000

    clf_params = dict(cat_features=cat_columns,
                      loss_function=loss_function,
                      eval_metric=eval_metric,
                      iterations=iterations,
                      # learning_rate=0.01,
                      early_stopping_rounds=iterations // (10, 20)[iterations > 5_000],
                      random_seed=RANDOM_SEED,
                      # task_type="GPU",
                      # border_count=254,
                      )

    best_params = {'boosting_type': 'Ordered',
                   'bootstrap_type': 'Bernoulli',
                   'border_count': 312,
                   'depth': 4,
                   'early_stopping_rounds': 100,
                   'eval_metric': 'RMSE',
                   'iterations': 1000,
                   'l2_leaf_reg': 5.660362834793042,
                   'leaf_estimation_method': 'Newton',
                   'learning_rate': 0.11198889224499226,
                   'loss_function': 'RMSE',
                   'one_hot_max_size': 3,
                   'random_seed': 127,
                   'random_strength': 5.5662738991994365,
                   'rsm': 0.7206973785465829,
                   'subsample': 0.6846905256786417,
                   'task_type': 'CPU'}

    # clf_params.update(best_params)

    clf = CatBoostRegressor(**clf_params)

    if use_grid_search:
        # Установить уровень логирования Optuna на WARNING
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Выполнить оптимизацию гиперпараметров
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        study.optimize(objective,
                       n_trials=600,
                       timeout=600,
                       show_progress_bar=True,
                       )

        print("Количество завершенных испытаний: {}".format(len(study.trials)))
        print("Лучшее испытание:")
        trial = study.best_trial
        print("  Значение: {}".format(trial.value))
        print("  best_params =", trial.params)

        best_params = trial.params

        clf_params.update(best_params)
        if (clf_params.get('boosting_type', '') == 'Ordered'
                or clf_params.get('bootstrap_type', '') == 'MVS'):
            clf_params['task_type'] = 'CPU'

        print('clf_params', clf_params)

        clf = CatBoostRegressor(**clf_params)

    info_cols = (model_columns, exclude_columns, cat_columns)

    comment = {}
    comment.update({'test_size': test_size,
                    'SEED': RANDOM_SEED,
                    'stratified': stratified,
                    })

    if use_cv_folds:
        comment['num_folds'] = num_folds

        if stratified:
            skf_folds = skf.split(train, stratified_target)
        else:
            skf_folds = split_kf.split(train)

        for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
            print(f'Фолд {idx} из {num_folds}')

            X_train, X_valid = train.iloc[train_idx], train.iloc[valid_idx]
            y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]

            splited = X_train, X_valid, y_train, y_valid

            train_data = Pool(data=X_train, label=y_train, cat_features=cat_columns)
            valid_data = Pool(data=X_valid, label=y_valid, cat_features=cat_columns)

            clf = CatBoostRegressor(**clf_params)
            clf.fit(train_data, eval_set=valid_data, use_best_model=True, verbose=200)

            models.append(clf)

            if build_model:
                DTS = (*splited, train, target, test_df, model_columns)
                valid_scores = make_predict_reg(idx, clf, DTS, max_num,
                                                submit_prefix=sub_pref)
                models_scores.append(valid_scores)

                comment['clf_iters'] = clf.best_iteration_

                add_info_to_log(sub_pref, max_num, idx, clf, valid_scores, info_cols, comment,
                                log_file=MODELS_LOGS_REG)

        if build_model:
            # объединение сабмитов
            merge_submits(max_num=max_num, submit_prefix=sub_pref, num_folds=num_folds,
                          post_fix='_reg')

    else:
        DTS = (*splited, train, target, test_df, model_columns)

        clf.fit(pool_train, eval_set=pool_valid, use_best_model=True, verbose=200)

        best_model = clf
        models.append(clf)

        if build_model:
            DTS = (*splited, train, target, test_df, model_columns)
            valid_scores = make_predict_reg(0, clf, DTS, max_num, submit_prefix=sub_pref)
            models_scores.append(valid_scores)

    print('best_params =', clf.get_params())

    if build_model:
        if len(models) > 1:
            valid_scores = [np.mean(arg) for arg in zip(*models_scores)]

            clf_iters = [clf.best_iteration_ for clf in models]
            clf_lr = [round(clf.get_all_params().get('learning_rate', 0), 8)
                      for clf in models]
        else:
            clf_iters = models[0].best_iteration_
            clf_lr = round(models[0].get_all_params().get('learning_rate', 0), 8)

        comment['clf_iters'] = clf_iters

        add_info_to_log(sub_pref, max_num, 0, models[0], valid_scores, info_cols, comment,
                        log_file=MODELS_LOGS_REG)

print_time(start_time)
