import re
import os
import random
import numpy as np
import pandas as pd
import joblib

import featuretools as ft
from woodwork.logical_types import Age, Categorical, Datetime

from copy import deepcopy
from pathlib import Path
from datetime import date, datetime, timedelta
from tqdm import tqdm
from calendar import monthrange
from itertools import product

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, roc_auc_score, classification_report

from df_addons import memory_compression
from print_time import print_time, print_msg

__import__("warnings").filterwarnings('ignore')

WORK_PATH = Path('Z:/python-datasets/competative-data-science-course-by-data-feeling')
if not WORK_PATH.is_dir():
    WORK_PATH = Path('D:/python-datasets/competative-data-science-course-by-data-feeling')

DATASET_PATH = WORK_PATH.joinpath('data')

if not WORK_PATH.is_dir():
    WORK_PATH = Path('.')
    DATASET_PATH = WORK_PATH

MODEL_PATH = WORK_PATH.joinpath('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR = WORK_PATH.joinpath('predictions')
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_LOGS = WORK_PATH.joinpath('scores_local.logs')

if not DATASET_PATH.exists():
    DATASET_PATH = Path('.')
    __file__ = Path('.')
    LOCAL_FILE = ''
else:
    LOCAL_FILE = '_local'

RANDOM_SEED = 127


def get_max_num(log_file=None):
    """Получение максимального номера итерации обучения моделей
    :param log_file: имя лог-файла с полным путем
    :return: максимальный номер
    """
    if log_file is None:
        log_file = MODELS_LOGS

    if not log_file.is_file():
        with open(log_file, mode='a') as log:
            log.write('num;mdl;fold;auc_macro;auc_micro;auc_wght;f1_macro;f1_micro;f1_wght;'
                      'tst_score;model_columns;exclude_columns;cat_columns;comment\n')
        max_num = 0
    else:
        df = pd.read_csv(log_file, sep=';')
        # df.num = df.index + 1
        max_num = df.num.max()
    return max_num if max_num else 0


def clean_column_name(col_name):
    col_name = col_name.lower()  # преобразование в нижний регистр
    col_name = col_name.replace('(', '_')  # замена скобок на _
    col_name = col_name.replace(')', '')  # удаление закрывающих скобок
    col_name = col_name.replace('.', '_')  # замена точек на _
    col_name = re.sub(r'(?<=\d)_(?=\d)', '', col_name)  # удаление подчеркивания между числами
    return col_name


class DataTransform:
    def __init__(self, use_catboost=True, numeric_columns=None, category_columns=None,
                 drop_first=False, scaler=None, args_scaler=None):
        """
        Преобразование данных
        :param use_catboost: данные готовятся для catboost
        :param numeric_columns: цифровые колонки
        :param category_columns: категориальные колонки
        :param drop_first: из dummy переменных удалить первую колонку
        :param scaler: какой скайлер будем использовать
        :param degree: аргументы для скайлера, например: степень для полином.преобразования
        """
        self.use_catboost = use_catboost
        self.category_columns = [] if category_columns is None else category_columns
        self.numeric_columns = [] if numeric_columns is None else numeric_columns
        self.drop_first = drop_first
        self.exclude_columns = []
        self.new_columns = []
        self.comment = {'drop_first': drop_first}
        self.transform_columns = None
        self.scaler = scaler
        self.args_scaler = args_scaler
        self.preprocess_path_file = 'reprocess_data.pkl'
        self.aggregate_path_file = 'aggregate_data.pkl'
        self.agg_df = pd.DataFrame()

    def cat_dummies(self, df):
        """
        Отметка категориальных колонок --> str для catboost
        OneHotEncoder для остальных
        :param df: ДФ
        :return: ДФ с фичами
        """
        # если нет цифровых колонок --> заполним их
        if self.category_columns and not self.numeric_columns:
            self.numeric_columns = [col_name for col_name in df.columns
                                    if col_name not in self.category_columns]
        # если нет категориальных колонок --> заполним их
        if self.numeric_columns and not self.category_columns:
            self.category_columns = [col_name for col_name in df.columns
                                     if col_name not in self.numeric_columns]

        for col_name in self.category_columns:
            if col_name in df.columns:
                if self.use_catboost:
                    df[col_name] = df[col_name].astype(str)
                else:
                    print(f'Трансформирую колонку: {col_name}')
                    # Create dummy variables
                    df = pd.get_dummies(df, columns=[col_name], drop_first=self.drop_first)

                    self.new_columns.extend([col for col in df.columns
                                             if col.startswith(col_name)])
        return df

    def apply_scaler(self, df):
        """
        Масштабирование цифровых колонок
        :param df: исходный ДФ
        :return: нормализованный ДФ
        """
        if not self.transform_columns:
            self.transform_columns = self.numeric_columns
        if self.scaler and self.transform_columns:
            print(f'Применяю scaler: {self.scaler.__name__} '
                  f'с аргументами: {self.args_scaler}')
            args = self.args_scaler if self.args_scaler else tuple()
            scaler = self.scaler(*args)
            scaled_data = scaler.fit_transform(df[self.transform_columns])
            if scaled_data.shape[1] != len(self.transform_columns):
                print(f'scaler породил: {scaled_data.shape[1]} колонок')
                new_columns = [f'pnf_{n:02}' for n in range(scaled_data.shape[1])]
                df = pd.concat([df, pd.DataFrame(scaled_data, columns=new_columns)], axis=1)
                self.exclude_columns.extend(self.transform_columns)
            else:
                df[self.transform_columns] = scaled_data

            self.comment.update(scaler=self.scaler.__name__, args_scaler=self.args_scaler)
        return df

    def fit(self, df):
        """
        Формирование фич
        :param df: исходный ФД
        :return: ДФ с агрегациями
        """
        start_time = print_msg('Группировка по целевому признаку...')

        self.agg_df = df.pivot_table(index='model',
                                     columns='target_class',
                                     values=['car_rating', 'year_to_work', 'riders'],
                                     aggfunc=[
                                         'median',
                                         'mean'
                                     ]).fillna(0)
        self.agg_df.columns = [f'{i[2]}_{i[1]}_{i[0]}_tenc' for i in self.agg_df.columns]
        self.agg_df.reset_index(inplace=True)

        grp = df.pivot_table(index='model',
                             columns='target_class',
                             values=['car_id'],
                             aggfunc=['count']).fillna(0).astype(int)
        grp.columns = [f'{i[2]}_{i[0]}_tenc' for i in grp.columns]
        grp.reset_index(inplace=True)

        self.agg_df = self.agg_df.merge(grp, on='model', how='left')

        # Удаляем константные колонки
        self.agg_df = self.drop_constant_columns(self.agg_df)

        print_time(start_time)

        return df

    def transform(self, df, model_columns=None):
        """
        Формирование остальных фич
        :param df: ДФ
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        print(self.agg_df.info(), df.info())
        print(self.agg_df.isna().sum().sum(), df.isna().sum().sum())

        if not self.agg_df.empty:
            df = df.merge(self.agg_df, on='model', how='left')

        df = self.cat_dummies(df)

        df = self.apply_scaler(df)

        if model_columns is None:
            model_columns = df.columns.tolist()

        model_columns.extend(self.new_columns)

        exclude_columns = [col for col in self.exclude_columns if col in df.columns]
        exclude_columns.extend(col for col in df.columns if col not in model_columns)

        if exclude_columns:
            df.drop(columns=exclude_columns, inplace=True, errors='ignore')

        self.exclude_columns = exclude_columns

        # Переводим типы данных в минимально допустимые - экономим ресурсы
        df = memory_compression(df)

        return df

    def fit_transform(self, df, model_columns=None):
        """
        Fit + transform data
        :param df: исходный ФД
        :param model_columns: список колонок, которые будут использованы в модели
        :return: ДФ с фичами
        """
        df = self.fit(df)
        df = self.transform(df, model_columns=model_columns)
        return df

    def preprocess_data(self, fill_nan=True, remake_file=False):
        """
        Предобработка данных
        :param fill_nan: заполняем пропуски в данных
        :param remake_file: формирование файлов заново / используем подготовленные ранее файлы
        :return:
        """
        preprocess_file = None

        if self.preprocess_path_file:
            preprocess_file = WORK_PATH.joinpath(self.preprocess_path_file)

        if self.preprocess_path_file and preprocess_file.is_file() and not remake_file:
            start_time = print_msg('Читаю подготовленные данные...')
            with open(preprocess_file, 'rb') as in_file:
                train, test, rides, drivers, fixes, df = joblib.load(in_file)
            print_time(start_time)
            return train, test, rides, drivers, fixes, df

        start_time = print_msg('Загрузка данных...')

        # Чтение данных из файлов
        train = pd.read_csv(DATASET_PATH.joinpath('car_train.csv'))
        test = pd.read_csv(DATASET_PATH.joinpath('car_test.csv'))
        rides = pd.read_csv(DATASET_PATH.joinpath('rides_info.csv'))
        drivers = pd.read_csv(DATASET_PATH.joinpath('driver_info.csv'))
        fixes = pd.read_csv(DATASET_PATH.joinpath('fix_info.csv'))

        rides['ride_date'] = rides['ride_date'].str.findall('\d+')
        rides['ride_date'] = rides['ride_date'].map(
            lambda z: '{:04}-{:02}-{:02}'.format(*map(int, z)))

        drivers['first_ride_date'] = drivers['first_ride_date'].str.findall('\d+')
        drivers['first_ride_date'] = drivers['first_ride_date'].map(
            lambda z: '{:04}-{:02}-{:02}'.format(*map(int, z)))

        fixes['fix_date'] = fixes['fix_date'].str.findall('\d+')
        fixes['fix_date'] = fixes['fix_date'].map(
            lambda z: '{:04}-{:02}-{:02} {:02}:{:02}'.format(*map(int, z)))

        fixes['fix_date'] = fixes['fix_date'].map(lambda z: z.split()[0])

        # Даты в кривом формате: во всех месяцах по 30 дней
        # drivers['frd'] = pd.to_datetime(drivers['first_ride_date'],
        #                                 format='%Y-%m-%d', errors='coerce')
        # map_bad_dates = {}
        # # Создаем словарь для замены из массива несуществующих дат
        # for bad_date in drivers[drivers.frd.isna()].first_ride_date.unique():
        #     year, month, _ = map(int, bad_date.split('-'))
        #     last_day = monthrange(year, month)[1]
        #     map_bad_dates[bad_date] = f'{year}-{month}-{last_day}'
        # # Замена кривых дат
        # drivers['first_ride_date'] = drivers['first_ride_date'].map(
        #     lambda z: map_bad_dates.get(z, z))
        #
        # drivers['first_ride_date'] = pd.to_datetime(drivers['first_ride_date'],
        #                                             format='%Y-%m-%d', errors='coerce')
        # drivers.drop(columns='frd', inplace=True)
        #
        # fixes['fd'] = pd.to_datetime(fixes['fix_date'],
        #                                    format='%Y-%m-%d %H:%M', errors='coerce')

        if fill_nan:
            # drivers - Заполнение пропусков user_time_accident, стратегии:
            # заполнить 0,
            # заполнить -1
            drivers['user_time_accident'].fillna(-1, inplace=True)

        all_data = pd.concat([train[test.columns], test], ignore_index=True)

        df = rides.merge(all_data, on='car_id', how='left')
        df = df.merge(drivers, on='user_id', how='left')

        # print(df.isna().sum())

        if fill_nan:
            fillna_col = 'user_ride_quality'
            # Заполнение пропусков user_ride_quality, стратегии:
            # взять среднее user_ride_quality по user_id, model
            # если пропуски в пред.пункте --> взять среднее user_ride_quality по user_id, car_type
            # если пропуски в пред.пункте --> взять среднее user_ride_quality по user_id.
            df['avg_urq'] = df.groupby(['user_id', 'model'])[fillna_col].transform('mean')
            df['avg_uct'] = df.groupby(['user_id', 'car_type'])[fillna_col].transform('mean')
            df['avg_urq_user'] = df.groupby('user_id')[fillna_col].transform('mean')
            # заполняем пропуски
            df['avg_urq'] = df['avg_urq'].fillna(df['avg_uct'])
            df['avg_urq'] = df['avg_urq'].fillna(df['avg_urq_user'])
            # Заполняем пропуски в user_ride_quality
            df[fillna_col] = df[fillna_col].fillna(df['avg_urq'])
            df.drop(columns=['avg_urq', 'avg_uct', 'avg_urq_user'], inplace=True)

            # print(df.isna().sum())

            fillna_col = 'speed_max'
            # Заполнение пропусков speed_max, стратегии:
            # взять среднее speed_max по model и user_id,
            # взять среднее speed_max / speed_avg по user_id, model и результат * на speed_avg,
            # если пропуски в пред.пункте --> взять среднее speed_max по user_id, car_type
            # если пропуски в пред.пункте --> взять среднее speed_max по user_id
            # если пропуски в пред.пункте --> взять среднее speed_max по model ?
            df['max_div_avg'] = df['speed_max'] / df['speed_avg']
            df['avg_um'] = df.groupby(['user_id', 'model']).max_div_avg.transform('mean')
            df['avg_uct'] = df.groupby(['user_id', 'car_type']).max_div_avg.transform('mean')
            df['avg_user_id'] = df.groupby('user_id').max_div_avg.transform('mean')
            df['avg_um'] = df['avg_um'].fillna(df['avg_uct'])
            df['avg_um'] = df['avg_um'].fillna(df['avg_user_id'])

            df['avg_mda'] = df['avg_um'] * df['speed_avg']

            df['avg_spm'] = df.groupby(['user_id', 'model'])[fillna_col].transform('mean')
            df['avg_uct'] = df.groupby(['user_id', 'car_type'])[fillna_col].transform('mean')
            df['avg_user_id'] = df.groupby('user_id')[fillna_col].transform('mean')
            # заполняем пропуски
            df['avg_spm'] = df['avg_spm'].fillna(df['avg_uct'])
            df['avg_spm'] = df['avg_spm'].fillna(df['avg_user_id'])
            # Заполняем пропуски в speed_max
            df[[fillna_col]] = df[[fillna_col]].fillna(df[['avg_mda', 'avg_spm']].max(axis=1))
            temps = ['max_div_avg', 'avg_um', 'avg_user_id', 'avg_uct', 'avg_mda', 'avg_spm']
            df.drop(columns=temps, inplace=True)

        # print(df.isna().sum())

        print_time(start_time)

        if self.preprocess_path_file:
            save_time = print_msg('Сохраняем предобработанные данные...')
            with open(preprocess_file, 'wb') as file:
                joblib.dump((train, test, fixes, df), file, compress=7)
            print_time(save_time)

        return train, test, rides, drivers, fixes, df

    @staticmethod
    def drop_constant_columns(df):
        # Ищем колонки с константным значением для удаления
        col_to_drop = []
        for col in df.columns:
            if df[col].nunique() == 1:
                col_to_drop.append(col)
        if col_to_drop:
            df.drop(columns=col_to_drop, inplace=True)
        return df

    def make_agg_data(self, remake_file=False, use_featuretools=False):
        """
        Подсчет разных агрегированных статистик
        :param remake_file:
        :param use_featuretools:
        :return:
        """
        aggregate_path_file = None

        if self.aggregate_path_file:
            aggregate_path_file = WORK_PATH.joinpath(self.aggregate_path_file)

        if self.aggregate_path_file and aggregate_path_file.is_file() and not remake_file:
            start_time = print_msg('Читаю подготовленные данные...')
            with open(aggregate_path_file, 'rb') as in_file:
                train_df, test_df = joblib.load(in_file)
            print_time(start_time)
            return train_df, test_df

        # Загрузка предобработанных данных
        train, test, rides, drivers, fixes, df = self.preprocess_data(remake_file=remake_file)

        start_time = print_msg('Агрегация данных...')

        if use_featuretools:
            all_data = pd.concat([train[test.columns], test], ignore_index=True)

            # Создаём отношения между источниками данных
            es = ft.EntitySet(id="car_data")

            es = es.add_dataframe(dataframe_name="cars",
                                  dataframe=all_data,
                                  index="car_id",
                                  logical_types={"car_type": Categorical,
                                                 "fuel_type": Categorical,
                                                 "model": Categorical,
                                                 },
                                  )

            es = es.add_dataframe(dataframe_name="rides",
                                  dataframe=rides.drop(columns=["ride_id"]),
                                  index="index",
                                  time_index="ride_date",
                                  )

            es = es.add_dataframe(dataframe_name="drivers",
                                  dataframe=drivers,
                                  index="user_id",
                                  logical_types={"sex": Categorical,
                                                 "first_ride_date": Datetime,
                                                 "age": Age,
                                                 },
                                  )

            es = es.add_dataframe(dataframe_name="fixes",
                                  dataframe=fixes,
                                  index="index",
                                  logical_types={"work_type": Categorical,
                                                 "worker_id": Categorical,
                                                 },
                                  )

            es = es.add_relationship("cars", "car_id", "rides", "car_id")
            es = es.add_relationship("drivers", "user_id", "rides", "user_id")
            es = es.add_relationship("cars", "car_id", "fixes", "car_id")

            # Генерируем новые признаки
            all_data, _ = ft.dfs(entityset=es,
                                 target_dataframe_name="cars",
                                 max_depth=2,
                                 )

            # Удаляем константные признаки
            all_data = ft.selection.remove_single_value_features(all_data)

            # Приведение наименований колонок в нормальный вид
            all_data.columns = all_data.columns.map(clean_column_name)

            test_df = all_data.loc[test.car_id].reset_index()
            train_df = all_data.loc[train.car_id].reset_index()
            # Добавим целевые признаки из трейна
            train_df = train_df.merge(train[['car_id', 'target_reg', 'target_class']],
                                      on=['car_id'], how='left')

        else:
            # Количество: уникальных водителей, использования машины
            grp_car = df.groupby(['car_id', 'model', 'car_type']).agg(
                nunique_user_id=('user_id', 'nunique'),
                mean_age=('age', 'mean'),
                mean_sex=('sex', 'mean'),
            ).reset_index()

            # По этим колонкам будем группировать
            group_columns = ['car_id',
                             # 'model',
                             # 'car_type',
                             ]
            # По этим колонкам будем агрегировать
            agg_columns = ['speed_max', 'speed_avg', 'ride_duration', 'distance', 'ride_cost',
                           'rating', 'stop_times', 'refueling', 'user_ride_quality',
                           'deviation_normal',
                           'age', 'user_rating', 'user_rides', 'user_time_accident',
                           ]
            agg_funcs = [
                np.max,
                # np.mean,
                np.median,
                np.min,
                np.std,
                pd.Series.skew,
                np.sum,
            ]
            # По этим колонкам будем считать сумму
            sum_columns = ['ride_duration', 'distance', 'ride_cost', 'stop_times',
                           'refueling']
            for grp_col, agg_col in product(group_columns, agg_columns):
                # Функции для подсчета статистик
                funcs = agg_funcs.copy()
                if agg_col in sum_columns and np.sum not in funcs:
                    funcs.append(np.sum)
                grp = df.groupby(grp_col)[agg_col].agg(funcs)
                grp.columns = [f'{grp_col}_{agg_col}_{col}' for col in grp.columns]
                grp.reset_index(inplace=True)
                # Добавляем группировку в общую кучу
                grp_car = grp_car.merge(grp, on=grp_col, how='left')

            # Количество: уникальных ремонтников и ремонтных работ. Самый частый тип ремонта.
            get_series_mode = lambda z: pd.Series.mode(z)[0]
            grp = fixes.groupby('car_id').agg(nunique_worker_id=('worker_id', 'nunique'),
                                              nunique_work_type=('work_type', 'nunique'),
                                              mode_work_type=('work_type', get_series_mode),
                                              ).reset_index()
            # Добавляем группировку в общую кучу
            grp_car = grp_car.merge(grp, on='car_id', how='left')

            grp = fixes.pivot_table(index='car_id',
                                    columns='work_type',
                                    values=['destroy_degree', 'work_duration'],
                                    aggfunc=['min', 'max', 'mean']).fillna(0)
            grp.columns = [f'{i[2]}_{i[1]}_{i[0]}' for i in grp.columns]
            grp.reset_index(inplace=True)
            # Добавляем группировку в общую кучу
            grp_car = grp_car.merge(grp, on='car_id', how='left')

            grp = fixes.pivot_table(index='car_id',
                                    columns='work_type',
                                    values='worker_id',
                                    aggfunc='count').fillna(0).astype(int)
            grp.columns = [f'{i}_count' for i in grp.columns]

            # Смысла возможно не имеет, т.к. 2 значения: 34 или 35
            grp['total_count'] = grp.sum(axis=1)

            grp.reset_index(inplace=True)
            # Добавляем группировку в общую кучу
            grp_car = grp_car.merge(grp, on='car_id', how='left')

            # Удаляем константные колонки
            grp_car = self.drop_constant_columns(grp_car)

            train_df = train.merge(grp_car, on=['car_id', 'model', 'car_type'], how='left')
            test_df = test.merge(grp_car, on=['car_id', 'model', 'car_type'], how='left')

        print_time(start_time)

        if self.aggregate_path_file:
            save_time = print_msg('Сохраняем агрегированные данные...')
            with open(aggregate_path_file, 'wb') as file:
                joblib.dump((train_df, test_df), file, compress=7)
            print_time(save_time)

        return train_df, test_df


def set_all_seeds(seed=RANDOM_SEED):
    # python's seeds
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_predict2(idx_fold, model, datasets, max_num=0, submit_prefix='cb_', label_enc=None,
                  text_features=None):
    """Предсказание для тестового датасета.
    Расчет метрик для модели: roc_auc и взвешенная F1-мера на валидации
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param max_num: максимальный порядковый номер обучения моделей
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :param label_enc: используемый label_encоder для target'а
    :return: разные roc_auc и F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df, model_columns = datasets

    features2drop = ['car_id']

    test = test_df[model_columns].drop(columns=features2drop, errors='ignore').copy()

    print('X_train.shape', X_train.shape)
    print('train.shape', train.shape)
    print('test.shape', test.shape)

    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''

    predict_valid = model.predict(X_valid)
    predict_test = model.predict(test)

    predict_proba_classes = model.classes_

    if label_enc:
        # преобразование обратно меток классов
        # predict_valid = label_enc.inverse_transform(predict_valid.reshape(-1, 1))
        predict_test = label_enc.inverse_transform(predict_test.reshape(-1, 1))
        predict_proba_classes = label_enc.inverse_transform(
            predict_proba_classes.reshape(-1, 1)).flatten()

    try:
        valid_proba = model.predict_proba(X_valid)
        predict_proba = model.predict_proba(test)
    except:
        valid_proba = predict_valid
        predict_proba = predict_test

    # Сохранение предсказаний в файл
    submit_csv = f'{submit_prefix}text_submit_{max_num:03}{nfld}{LOCAL_FILE}.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    submission = pd.DataFrame({'category': predict_test.flatten()})
    submission.to_csv(file_submit_csv, index=False)

    # Сохранение вероятностей в файл
    submit_proba = f'{submit_prefix}text_submit__proba_{max_num:03}{nfld}{LOCAL_FILE}.csv'
    file_submit_proba = PREDICTIONS_DIR.joinpath(submit_proba)
    submission_proba = pd.DataFrame(predict_proba, columns=predict_proba_classes)
    # Добавление идентификатора объекта
    submission_proba.to_csv(file_submit_proba, index=False)

    multi_class = model.get_params().get('objective', '')
    print('multi_class', multi_class)
    if multi_class == 'multiclassova':
        predict_proba = predict_proba / np.sum(predict_proba, axis=1, keepdims=True)
        valid_proba = valid_proba / np.sum(valid_proba, axis=1, keepdims=True)

    t_score = 0

    start_item = print_msg("Расчет ROC AUC...")
    # Для многоклассового ROC AUC, нужно указать multi_class
    auc_macro = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='macro')
    auc_micro = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='micro')
    auc_wght = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='weighted')
    print(f"auc_macro: {auc_macro}, auc_micro: {auc_micro}, auc_micro: {auc_wght}")
    print_time(start_item)

    start_item = print_msg("Расчет F1-score...")
    f1_macro = f1_micro = f1_wght = 0
    try:
        f1_macro = f1_score(y_valid, predict_valid, average='macro')
        f1_micro = f1_score(y_valid, predict_valid, average='micro')
        f1_wght = f1_score(y_valid, predict_valid, average='weighted')
    except:
        pass
    print(f'F1- f1_macro: {f1_wght:.6f}, f1_micro: {f1_wght:.6f}, f1_wght: {f1_wght:.6f}')
    print_time(start_item)

    try:
        if model.__class__.__name__ == 'CatBoostClassifier':
            eval_metric = model.get_params()['eval_metric']
            model_score = model.best_score_['validation'][eval_metric]
        elif model.__class__.__name__ == 'LGBMClassifier':
            model_score = model.best_score_['valid_0']['multi_logloss']
        elif model.__class__.__name__ == 'XGBClassifier':
            model_score = model.best_score
    except:
        model_score = 0

    return model_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, t_score


def make_predict(idx_fold, model, datasets, max_num=0, submit_prefix='cb_', label_enc=None):
    """Предсказание для тестового датасета.
    Расчет метрик для модели: roc_auc и взвешенная F1-мера на валидации
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param datasets: кортеж с тренировочной, валидационной и полной выборками
    :param max_num: максимальный порядковый номер обучения моделей
    :param submit_prefix: префикс для файла сабмита для каждой модели свой
    :param label_enc: используемый label_encоder для target'а
    :return: разные roc_auc и F1-мера
    """
    X_train, X_valid, y_train, y_valid, train, target, test_df, model_columns = datasets

    features2drop = ['car_id']

    test = test_df[model_columns].drop(columns=features2drop, errors='ignore').copy()

    print('X_train.shape', X_train.shape)
    print('train.shape', train.shape)
    print('test.shape', test.shape)

    # постфикс если было обучение на отдельных фолдах
    nfld = f'_{idx_fold}' if idx_fold else ''

    predict_valid = model.predict(X_valid)
    predict_test = model.predict(test)

    predict_proba_classes = model.classes_

    if label_enc:
        # преобразование обратно меток классов
        # predict_valid = label_enc.inverse_transform(predict_valid.reshape(-1, 1))
        predict_test = label_enc.inverse_transform(predict_test.reshape(-1, 1))
        predict_proba_classes = label_enc.inverse_transform(
            predict_proba_classes.reshape(-1, 1)).flatten()

    try:
        valid_proba = model.predict_proba(X_valid)
        predict_proba = model.predict_proba(test)
    except:
        valid_proba = predict_valid
        predict_proba = predict_test

    # Сохранение предсказаний в файл
    submit_csv = f'{submit_prefix}submit_{max_num:03}{nfld}{LOCAL_FILE}.csv'
    file_submit_csv = PREDICTIONS_DIR.joinpath(submit_csv)
    submission = pd.DataFrame({'car_id': test_df['car_id'],
                               'target_class': predict_test.flatten()})
    submission.to_csv(file_submit_csv, index=False)

    # Сохранение вероятностей в файл
    submit_proba = f'{submit_prefix}submit_proba_{max_num:03}{nfld}{LOCAL_FILE}.csv'
    file_submit_proba = PREDICTIONS_DIR.joinpath(submit_proba)
    submission_proba = pd.DataFrame(predict_proba, columns=predict_proba_classes)
    # Добавление идентификатора объекта
    submission_proba.insert(0, 'car_id', test_df['car_id'])
    submission_proba.to_csv(file_submit_proba, index=False)

    multi_class = model.get_params().get('objective', '')
    print('multi_class', multi_class)
    if multi_class == 'multiclassova':
        predict_proba = predict_proba / np.sum(predict_proba, axis=1, keepdims=True)
        valid_proba = valid_proba / np.sum(valid_proba, axis=1, keepdims=True)

    # Расчёт accuracy_score
    true_submit_csv = PREDICTIONS_DIR.joinpath('submission_true.csv')
    if true_submit_csv.is_file():
        true_submit = pd.read_csv(true_submit_csv)
        t_score = np.mean(true_submit['target_class'] == submission['target_class'])
    else:
        t_score = 0

    start_item = print_msg("Расчет ROC AUC...")
    # Для многоклассового ROC AUC, нужно указать multi_class
    auc_macro = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='macro')
    auc_micro = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='micro')
    auc_wght = roc_auc_score(y_valid, valid_proba, multi_class='ovr', average='weighted')
    print(f"auc_macro: {auc_macro}, auc_micro: {auc_micro}, auc_micro: {auc_wght}")
    print_time(start_item)

    start_item = print_msg("Расчет F1-score...")
    f1_macro = f1_micro = f1_wght = 0
    try:
        f1_macro = f1_score(y_valid, predict_valid, average='macro')
        f1_micro = f1_score(y_valid, predict_valid, average='micro')
        f1_wght = f1_score(y_valid, predict_valid, average='weighted')
    except:
        pass
    print(f'F1- f1_macro: {f1_wght:.6f}, f1_micro: {f1_wght:.6f}, f1_wght: {f1_wght:.6f}')
    print_time(start_item)

    try:
        if model.__class__.__name__ == 'CatBoostClassifier':
            eval_metric = model.get_params()['eval_metric']
            model_score = model.best_score_['validation'][eval_metric]
        elif model.__class__.__name__ == 'LGBMClassifier':
            model_score = model.best_score_['valid_0']['multi_logloss']
        elif model.__class__.__name__ == 'XGBClassifier':
            model_score = model.best_score
    except:
        model_score = 0

    return model_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, t_score


def add_info_to_log(prf, max_num, idx_fold, model, valid_scores, info_cols,
                    comment_dict=None, clf_lr=None, log_file=MODELS_LOGS):
    """
    Добавление информации об обучении модели
    :param prf: Префикс файла сабмита
    :param max_num: номер итерации обучения моделей
    :param idx_fold: номер фолда при обучении
    :param model: обученная модель
    :param valid_scores: скоры при обучении
    :param info_cols: информативные колонки
    :param comment_dict: комментарии
    :param clf_lr: список из learning_rate моделей
    :return:
    """
    m_score, auc_macro, auc_micro, auc_wght, f1_macro, f1_micro, f1_wght, score = valid_scores

    model_columns, exclude_columns, cat_columns = info_cols

    if comment_dict is None:
        comment = {}
    else:
        comment = deepcopy(comment_dict)

    model_clf_lr = feature_imp = None
    if model.__class__.__name__ == 'CatBoostClassifier':
        model_clf_lr = model.get_all_params().get('learning_rate', 0)
        feature_imp = model.feature_importances_

    elif model.__class__.__name__ == 'LGBMClassifier':
        model_clf_lr = model.get_params().get('learning_rate', 0)

    elif model.__class__.__name__ == 'XGBClassifier':
        model_clf_lr = model.get_params().get('learning_rate', 0)

    if feature_imp is not None:
        use_cols = [col for col in model_columns if col not in exclude_columns]
        features = pd.DataFrame({'Feature': use_cols,
                                 'Importance': feature_imp}).sort_values('Importance',
                                                                         ascending=False)
        features.to_excel(MODEL_PATH.joinpath(f'features_{prf}{max_num}.xlsx'), index=False)

    if model_clf_lr is not None:
        model_clf_lr = round(model_clf_lr, 8)

    if clf_lr is None:
        clf_lr = model_clf_lr

    comment['clf_lr'] = clf_lr

    comment.update(model.get_params())

    prf = prf.strip('_')

    with open(log_file, mode='a') as log:
        # log.write('num;mdl;fold;auc_macro;auc_micro;auc_wght;f1_macro;f1_micro;f1_wght;'
        #           'tst_score;model_columns;exclude_columns;cat_columns;comment\n')
        log.write(f'{max_num};{prf};{idx_fold};{m_score:.6f};{auc_macro:.6f};{auc_micro:.6f};'
                  f'{auc_wght:.6f};{f1_macro:.6f};{f1_micro:.6f};{f1_wght:.6f};{score:.6f};'
                  f'{model_columns};{exclude_columns};{cat_columns};{comment}\n')


def merge_submits(max_num=0, submit_prefix='cb_', num_folds=5, exclude_folds=None,
                  use_proba=False):
    """
    Объединение сабмитов
    :param max_num: номер итерации модели или список файлов, или список номеров сабмитов
    :param submit_prefix: префикс сабмита модели
    :param num_folds: количество фолдов модели для объединения
    :param exclude_folds: список списков для исключения фолдов из объединения:
                          длина списка exclude_folds должна быть равна длине списка max_num
    :param use_proba: использовать файлы с предсказаниями вероятностей
    :return: None
    """
    if use_proba:
        prob = '_proba'
    else:
        prob = ''
    # Читаем каждый файл и добавляем его содержимое в список датафреймов
    submits = pd.DataFrame()
    if isinstance(max_num, int):
        for nfld in range(1, num_folds + 1):
            submit_csv = f'{submit_prefix}submit{prob}_{max_num:03}_{nfld}{LOCAL_FILE}.csv'
            df = pd.read_csv(PREDICTIONS_DIR.joinpath(submit_csv), index_col='car_id')
            if use_proba:
                df.columns = [f'{col}_{nfld}' for col in df.columns]
            else:
                df.columns = [f'target_{nfld}']
            if nfld == 1:
                submits = df
            else:
                submits = submits.merge(df, on='car_id', suffixes=('', f'_{nfld}'))
        max_num = f'{max_num:03}'

    elif isinstance(max_num, (list, tuple)) and exclude_folds is None:
        for idx, file in enumerate(sorted(max_num)):
            df = pd.read_csv(PREDICTIONS_DIR.joinpath(file), index_col='car_id')
            if use_proba:
                df.columns = [f'{col}_{idx}' for col in df.columns]
            else:
                df.columns = [f'target_{idx}']
            if not idx:
                submits = df
            else:
                submits = submits.merge(df, on='car_id', suffixes=('', f'_{idx}'))
        max_num = '-'.join(sorted(re.findall(r'\d{3,}(?:_\d)?', ' '.join(max_num)), key=int))

    elif isinstance(max_num, (list, tuple)) and isinstance(exclude_folds, (list, tuple)):
        submits, str_nums = None, []
        for idx, (num, exc_folds) in enumerate(zip(max_num, exclude_folds), 1):
            str_num = str(num)
            for file in PREDICTIONS_DIR.glob(f'*submit{prob}_{num}_*.csv'):
                pool = re.findall(r'(?:(?<=\d{3}_)|(?<=\d{4}_))\d(?:(?=_local)|(?=\.csv))',
                                  file.name)
                if pool and int(pool[0]) not in exc_folds:
                    str_num += f'_{pool[0]}'
                    suffix = f'_{idx}_{pool[0]}'
                    df = pd.read_csv(file, index_col='car_id')
                    if use_proba:
                        df.columns = [f'{col}_{suffix}' for col in df.columns]
                    else:
                        df.columns = [f'target{suffix}']
                    if submits is None:
                        submits = df
                    else:
                        submits = submits.merge(df, on='car_id', suffixes=('', suffix))
            str_nums.append(str_num)
        max_num = '-'.join(sorted(str_nums))
        # print(df)
        print(max_num)

    # df.to_excel(WORK_PATH.joinpath(f'{submit_prefix}submit_{max_num}{LOCAL_FILE}.xlsx'))

    if use_proba:
        # Название классов поломок
        target_columns = sorted(set([col.rsplit('_', 1)[0] for col in submits.columns]))
        # Суммирование по классам поломок
        for col in target_columns:
            submits[col] = submits.filter(like=col, axis=1).sum(axis=1)
        # Получение имени класса поломки по максимуму из классов
        submits['target_class'] = submits.idxmax(axis=1)
    else:
        # Нахождение моды по строкам
        submits['target_class'] = submits.mode(axis=1)[0]

    submits_csv = f'{submit_prefix}submit_{max_num}{LOCAL_FILE}{prob}.csv'
    submits[['target_class']].to_csv(PREDICTIONS_DIR.joinpath(submits_csv))


if __name__ == "__main__":
    border_count = 254  # для кетбуста на ГПУ

    # тут разные опыты с классом...

    # dts = DataTransform()
    #
    # # train_data, test_data = dts.make_agg_data(remake_file=True, use_featuretools=True)
    # train_data, test_data = dts.make_agg_data(remake_file=True)
    #
    # print(train_data.columns)
    #
    # df = dts.fit(train_data)
    # print(df.columns)
    #
    # train_data = dts.fit_transform(train_data)
    # test_data = dts.transform(test_data)
    #
    # print(set(train_data.columns) - set(test_data.columns))

    # merge_submits(max_num=125, submit_prefix='xb_', use_proba=True)
