import os
from pathlib import Path
import numpy as np
import pandas as pd
import copy

import torch

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, normalize, OrdinalEncoder
# Вспомогательные блоки организации для пайплайна
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import mode

from set_all_seeds import set_all_seeds

__import__("warnings").filterwarnings('ignore')


def custom_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


class CustomStackingRegressor:
    def __init__(self, estimators, final_estimator=None, test_size=0.2, stratified=None,
                 cv_folds=None, cv_folds_meta=None, num_scaler=None, models_to_scale=None,
                 model_columns=None, cat_columns=None, num_columns=None, meta_columns=None,
                 features2drop=None, oof_pred_bag=True, metric=None,
                 n_jobs=-1, verbose=False, **kwargs):
        """
        Инициализация класса
        :param estimators: список моделей, состоящий из кортежей (имя, модель)
        :param final_estimator: модель для стекинга
        :param test_size: размер валидационной выборки
        :param stratified: колонка в ДФ для стратификации
        :param cv_folds: количество фолдов
        :param cv_folds_meta: количество фолдов для метамодели
        :param num_scaler: скейлер для числовых колонок
        :param models_to_scale: список моделей, которым необходим скейлер для числовых колонок
        :param model_columns: список колонок для обучения модели
        :param cat_columns: список категориальных колонок
        :param num_columns: список числовых колонок
        :param meta_columns: список колонок из исходного датасета, которые пойдут в стекинг
        :param features2drop: список для удаления
        :param pred_proba: использовать для предсказаний вероятности
        :param oof_pred_bag: предсказания теста усредняются по фолдам / или на всем трейне
        :param n_jobs:
        :param verbose:
        :param kwargs:
        """
        self.estimators = estimators
        self.final_estimator_name = final_estimator[0] if final_estimator else None
        self.final_estimator = final_estimator[1] if final_estimator else None
        self.stratified = stratified
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.cv_folds_meta = cv_folds_meta
        self.scaler = StandardScaler if num_scaler is None else num_scaler
        self.models_to_scale = ['LinReg',
                                'LogReg',
                                'TabNetWork',
                                ] if models_to_scale is None else models_to_scale
        self.model_columns = model_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.meta_columns = meta_columns
        self.features2drop = features2drop
        self.pred_proba = False
        self.use_voting = False
        self.oof_pred_bag = oof_pred_bag
        self.metric = metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = kwargs.get('random_state', 127)
        self.stratified_target = None
        self.preprocessor = None
        self.number_of_classes = 1
        self.X_train_for_meta = None
        self.y_train_for_meta = None
        self.X_test_for_meta = None
        self.fitted_models = {model_name: [] for model_name, _ in estimators}
        self.fitted_models_scores = copy.deepcopy(self.fitted_models)
        self.final_model = None
        self.classifier = False
        self.show_min_output = kwargs.get('show_min_output', False)
        self.work_path = kwargs.get('work_path', Path('.'))
        self.predictions_dir = kwargs.get('predictions_dir',
                                          self.work_path.joinpath('predictions'))
        self.label_encoder = LabelEncoder()
        set_all_seeds(seed=self.random_state)

    def fit_model(self, model_name, base_model, split, n_fold=0, cv_folds=None):
        """
        Функция обучения модели
        :param model_name: Имя модели
        :param base_model: Экземпляр класса модели
        :param split: Кортеж с данными (X_train, X_valid, y_train, y_valid)
        :param n_fold: номер фолда
        :param cv_folds: общее количество фолдов
        :return: обученная модель
        """

        X_train, X_valid, y_train, y_valid = split
        model = clone(base_model)
        if model_name == "CatBoost":
            model.fit(X_train, y_train,
                      eval_set=(X_valid, y_valid),
                      use_best_model=True, verbose=0,
                      )
        elif model_name == "LightGBM":
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      )
        elif model_name == "XGBoost":
            model.fit(X_train, y_train,
                      eval_set=[(X_valid, y_valid)],
                      verbose=0,
                      )
        elif model_name == "TabNetWork":
            model.fit(X_train.values, y_train,
                      eval_set=(X_valid.values, y_valid),
                      verbose=0,
                      )
        else:
            model.fit(X_train, y_train,
                      )

        score_ = None
        if self.show_min_output:
            if cv_folds is None:
                cv_folds = self.cv_folds
            metric = ''
            name_fold = f', фолд: {n_fold}/{cv_folds}' if n_fold else ''
            if all(obj is not None for obj in (self.metric, X_valid, y_valid)):
                if model_name == "TabNetWork":
                    y_pred = model.predict(X_valid.values)
                else:
                    y_pred = model.predict(X_valid)
                score_ = self.metric(y_valid, y_pred)
                metric = f', [metric={score_:.3f}]'
            print(f'Обучаю модель: {model_name}{name_fold}{metric}')
        if not hasattr(model, 'metric_score_'):
            setattr(model, 'metric_score_', score_)
        return model

    def process_data(self, model_name, X_train, X_valid=None, y_train=None, y_valid=None):
        """
        Функция подготовки данных для моделей: бустингам обрабатывать не требуется,
        для остальных моделей onehot-енкодинг и применение скейлера, если требуется
        :param model_name: Имя модели
        :param X_train: - формат пандас ДФ
        :param X_valid: - формат пандас Серия
        :param y_train: - формат пандас ДФ
        :param y_valid: - формат пандас Серия
        :return: кортеж предобработанных данных
        """
        if model_name in ("CatBoost", "LightGBM", "XGBoost"):
            return X_train, X_valid, y_train, y_valid

        # Трансформируем данные train_df
        X_train_prep = self.preprocessor.transform(X_train.copy())

        # Получаем имена новых колонок после трансформации
        num_columns_trn = self.num_columns
        if self.cat_columns:
            cat_cols_trn = self.preprocessor.named_transformers_["categorical"].named_steps[
                "onehot"].get_feature_names_out(self.cat_columns)
        else:
            cat_cols_trn = []

        model_columns_trn = num_columns_trn + list(cat_cols_trn)
        # Преобразуем в DataFrame
        train_ = pd.DataFrame(X_train_prep, columns=model_columns_trn, index=X_train.index)
        valid_ = None
        if X_valid is not None:
            X_valid_ = self.preprocessor.transform(X_valid.copy())
            valid_ = pd.DataFrame(X_valid_, columns=model_columns_trn, index=X_valid.index)
        if self.models_to_scale is None or model_name not in self.models_to_scale:
            # Признаки для этих моделей не нужно масштабировать
            train_[num_columns_trn] = X_train[num_columns_trn]
            if X_valid is not None:
                valid_[num_columns_trn] = X_valid[num_columns_trn]
        return train_, valid_, y_train, y_valid

    def get_split_folds(self, X, cv_folds=None):
        """
        Получение итератора с индексами разбиения теста на фолды
        :param X: Тренировочный датасет - формат пандас ДФ
        :param cv_folds: количество фолдов
        :return: Итератор
        """
        if cv_folds is None:
            cv_folds = self.cv_folds
        if self.stratified:
            skf = StratifiedKFold(n_splits=cv_folds, random_state=self.random_state,
                                  shuffle=True)
            skf_folds = skf.split(X, self.stratified_target)
        else:
            split_kf = KFold(n_splits=cv_folds, random_state=self.random_state, shuffle=True)
            skf_folds = split_kf.split(X)
        return skf_folds

    def get_new_columns(self, model_name: str, make_proba=False):
        """
        Формирование имени колонок с предсказаниями модели первого уровня
        :param model_name: Имя модели
        :param make_proba: Для вероятностей формируется список колонок по количеству классов
        :return: Имя одной колонки или список
        """
        if self.pred_proba or make_proba:
            return [f'{model_name}_{label}' for label in self.label_encoder.classes_]
        return [model_name]

    def fit(self, X, y, verbose=None, **kwargs):
        """
        Обучение моделей первого уровня
        :param X: Тренировочный датасет - формат пандас ДФ
        :param y: целевая переменная - формат пандас Серия
        :param verbose: уровень логгирования
        :param kwargs: словарь с параметрами для совместимости
        :return: self
        """
        if verbose:
            self.verbose = verbose

        if self.model_columns is None:
            self.model_columns = X.columns.tolist()

        # Выделение числовых колонок
        if self.num_columns is None:
            self.num_columns = X.select_dtypes(include=['number']).columns.tolist()

        # Выделение категориальных колонок
        if self.cat_columns is None:
            self.cat_columns = [col for col in self.model_columns
                                if col not in self.num_columns]

        categorical_transformer = Pipeline(steps=[
            ("onehot",
             OneHotEncoder(dtype=int, handle_unknown="ignore"))])

        numerical_transformer = Pipeline(steps=[
            ("scaler",
             FunctionTransformer(lambda Z: Z) if self.scaler is None else self.scaler())
        ])

        # соединим два предыдущих трансформера в один
        self.preprocessor = ColumnTransformer(transformers=[
            ("numerical", numerical_transformer, self.num_columns),
            ("categorical", categorical_transformer, self.cat_columns)])

        # Применяем препроцессор к данным: Обучаем препроцессор на данных train_df
        self.preprocessor.fit(X[self.model_columns])

        # Получаем массив для стратификации
        if isinstance(self.stratified, str) and self.stratified in X.columns:
            # Если это строка и название колонки есть в трейне
            self.stratified_target = X[self.stratified]
        elif hasattr(y, 'columns') and self.stratified in y.columns:
            # Если "y" - датафрейм и название колонки есть в таргете
            self.stratified_target = y[self.stratified]
        elif hasattr(y, self.stratified):
            # Если "y" - пандас серия и её имя - это название колонки
            self.stratified_target = y
        elif pd.api.types.is_list_like(self.stratified):
            # Если это спископодобный элемент - тогда его используем
            self.stratified_target = self.stratified

        if self.classifier:
            # Для классификатора сделаем таргет-енкодинг, т.к. не все модели понимают строки
            y_le = self.label_encoder.fit_transform(y.copy())
            # Завернем в серию, т.к. используются методы Pandas
            y = pd.Series(y_le, index=y.index)
            self.number_of_classes = y.nunique()

        self.X_train_for_meta = pd.DataFrame(index=X.index)
        self.y_train_for_meta = y.copy()

        name_models = []

        for model_name, base_model in self.estimators:
            name_models.append(model_name)
            new_columns = self.get_new_columns(model_name)
            df = pd.DataFrame(index=X.index)
            df[new_columns] = 0
            scores = []
            if self.cv_folds:
                skf_folds = self.get_split_folds(X)

                for idx, (train_id, valid_id) in enumerate(skf_folds, 1):
                    X_train, y_train = X[self.model_columns].iloc[train_id], y.iloc[train_id]
                    X_valid, y_valid = X[self.model_columns].iloc[valid_id], y.iloc[valid_id]

                    split = self.process_data(model_name, X_train, X_valid, y_train, y_valid)
                    model = self.fit_model(model_name, base_model, split, idx)
                    self.fitted_models[model_name].append(model)
                    scores.append(getattr(model, 'metric_score_', 0))

                    X_train_pd, X_valid_pd, y_train_pd, y_valid_pd = split
                    if model_name in ("TabNetWork",):
                        X_valid_pd_values = X_valid_pd.values
                    else:
                        X_valid_pd_values = X_valid_pd.copy()
                    if self.pred_proba:
                        y_pred_valid = model.predict_proba(X_valid_pd_values)
                        df.iloc[valid_id, :] = y_pred_valid
                    else:
                        y_pred_valid = model.predict(X_valid_pd_values)
                        df.iloc[valid_id, 0] = y_pred_valid

            else:
                split = train_test_split(X[self.model_columns], y,
                                         test_size=self.test_size,
                                         stratify=self.stratified_target,
                                         random_state=self.random_state)

                split = self.process_data(model_name, *split)
                full_train, *_ = self.process_data(model_name, X[self.model_columns])
                model = self.fit_model(model_name, base_model, split)
                self.fitted_models[model_name].append(model)
                scores.append(getattr(model, 'metric_score_', 0))

                if model_name in ("TabNetWork",):
                    full_train_values = full_train.values
                else:
                    full_train_values = full_train.copy()
                if self.pred_proba:
                    predict_full_train = model.predict_proba(full_train_values)
                else:
                    predict_full_train = model.predict(full_train_values).reshape(-1, 1)

                df[new_columns] = predict_full_train

            self.fitted_models_scores[model_name] = scores

            if self.show_min_output and all(scores) and len(scores) > 1:
                print(f'Модель: {model_name} --> [mean={np.mean(scores):.3f}]')

            self.X_train_for_meta = pd.concat([self.X_train_for_meta, df], axis=1)

        # Если используется голосование - тогда избавляемся от вероятностей
        if self.pred_proba and self.use_voting:
            # Преобразуем в 3D массив
            array = self.X_train_for_meta.values
            array = array.reshape((array.shape[0], len(name_models), -1))

            # Считаем средние значения по оси 1 и оформляем в датафрейм
            self.X_train_for_meta = pd.DataFrame(data=np.argmax(array, axis=2),
                                                 columns=name_models,
                                                 index=X.index)

        if self.meta_columns is not None and self.final_estimator is not None:
            X_raw, *_ = self.process_data(self.final_estimator_name, X[self.meta_columns])
            self.X_train_for_meta = pd.concat([self.X_train_for_meta, X_raw], axis=1)

        return self

    def fit_predict_meta_model(self, test, predict_columns):
        """
        Обучение метамодели и получение от неё предсказаний
        :param test: Тестовый датасет - формат пандас ДФ
        :param predict_columns: Список с именами колонок в которые будут записаны предсказания
        :return: Предсказания метамодели
        """

        result = pd.DataFrame(index=test.index)
        XM = self.X_train_for_meta.copy()
        yM = self.y_train_for_meta.copy()
        scores = []
        num_classes = len(predict_columns)

        if self.final_estimator is None:
            # Преобразуем в 3D массив
            result = self.X_test_for_meta.values
            result = result.reshape((result.shape[0], len(self.fitted_models), -1))
            # если у нас классификатор и нет вероятностей - тогда ставим метки классов
            if self.classifier and not self.pred_proba:
                # Вычисляем самые частые значения по строкам
                most_frequent = mode(result, axis=1, keepdims=True)
                # Извлекаем чаще встречающееся значение и убираем лишние измерения
                result = most_frequent.mode.squeeze()
                # вернем исходные метки целевой переменной
                result = self.label_encoder.inverse_transform(result)
            else:
                # Вычисление статистик по строкам: можно попробовать еще медиану !!!
                result = np.mean(result, axis=1)
                # Костыль от кривых рук: при использовании self.pred_proba = True
                if self.use_voting and result.shape[-1] < len(predict_columns):
                    # Объединяем result и колонку единиц горизонтально
                    result = np.hstack((1 - result, result))

            # Оформляем результат в датафрейм
            result = pd.DataFrame(data=result,
                                  columns=predict_columns,
                                  index=self.X_test_for_meta.index)
            return result

        if self.cv_folds_meta:
            # Инициализация массива для хранения предсказаний всех моделей
            predictions_array = np.zeros((len(self.X_test_for_meta), self.cv_folds_meta,
                                          num_classes))
            skf_folds = self.get_split_folds(XM, self.cv_folds_meta)
            for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
                X_train, y_train = XM.iloc[train_idx], yM.iloc[train_idx]
                X_valid, y_valid = XM.iloc[valid_idx], yM.iloc[valid_idx]

                split = X_train, X_valid, y_train, y_valid

                self.final_model = self.fit_model(self.final_estimator_name,
                                                  self.final_estimator,
                                                  split, idx, self.cv_folds_meta)
                if self.pred_proba:
                    pred_test_prep = self.final_model.predict_proba(self.X_test_for_meta)
                else:
                    pred_test_prep = self.final_model.predict(self.X_test_for_meta)
                # Проверяем размерность
                if pred_test_prep.ndim == 1:
                    # Добавляем новое измерение
                    pred_test_prep = np.expand_dims(pred_test_prep, axis=1)
                # Сохраняем предсказания в массив
                predictions_array[:, idx - 1, :] = pred_test_prep
                scores.append(getattr(self.final_model, 'metric_score_', 0))

            # Вычисление статистик по строкам
            result = pd.DataFrame(data=predictions_array.mean(axis=1),
                                  columns=[predict_columns],
                                  index=self.X_test_for_meta.index)
        else:
            split = train_test_split(XM, yM, test_size=self.test_size,
                                     stratify=self.stratified_target,
                                     random_state=self.random_state)

            self.final_model = self.fit_model(self.final_estimator_name,
                                              self.final_estimator,
                                              split)
            result[predict_columns] = self.final_model.predict(self.X_test_for_meta)
            scores.append(getattr(self.final_model, 'metric_score_', 0))

        if self.show_min_output and all(scores):
            print(f'Модель: {self.final_estimator_name}, [mean={np.mean(scores):.3f}]')

        return result

    def predict(self, test, save_to_excel=False, return_proba=False):
        """
        Получение предсказаний от моделей первого уровня,
        вызов функции обучения метамодели и получение от неё предсказаний
        :param test: Тестовый датасет
        :param save_to_excel: Сохранять результаты стекинга моделей первого уровня:
                              требуется для подбора гиперпараметров метамодели
        :param return_proba: True - возвращать вероятности для задачи классификации
        :return: предсказания метамодели
        """
        # Получаем предсказания на тестовой выборке
        name_models = []
        self.X_test_for_meta = pd.DataFrame(index=test.index)
        for model_name, models_list in self.fitted_models.items():
            test_prep, *_ = self.process_data(model_name, test[self.model_columns])

            new_columns = self.get_new_columns(model_name)

            name_models.append(model_name)

            # # Инициализация массива для хранения предсказаний всех моделей
            # predictions_array = np.zeros((len(test_prep), len(models_list)))
            # Инициализация массива для хранения предсказаний всех моделей
            num_classes = len(new_columns)
            predictions_array = np.zeros((len(test_prep), len(models_list), num_classes))

            # Сбор предсказаний от каждой модели, обученной по фолдам
            for idx, model in enumerate(models_list):
                if model_name in ("TabNetWork",):
                    test_prep_values = test_prep.values
                else:
                    test_prep_values = test_prep.copy()
                if self.pred_proba:
                    pred_test_prep = model.predict_proba(test_prep_values)
                else:
                    pred_test_prep = model.predict(test_prep_values)

                # Сохраняем предсказания в массив
                # predictions_array[:, idx] = pred_test_prep
                # Сохраняем предсказания в массив

                # Проверяем размерность
                if pred_test_prep.ndim == 1:
                    # Добавляем новое измерение
                    pred_test_prep = np.expand_dims(pred_test_prep, axis=1)

                predictions_array[:, idx, :] = pred_test_prep

            if self.oof_pred_bag:
                # если у нас классификатор и нет вероятностей - тогда ставим метки классов
                if self.classifier and not self.pred_proba:
                    # Вычисляем самые частые значения по строкам
                    most_frequent = mode(predictions_array, axis=1, keepdims=True)
                    # Извлекаем чаще встречающееся значение и убираем лишние измерения
                    preds = most_frequent.mode.squeeze()
                else:
                    # Вычисление статистик по строкам: можно попробовать еще медиану !!!
                    preds = predictions_array.mean(axis=1)
            else:
                # Берем предсказания одной лучшей модели <- нужно тестить для классификации
                scores = self.fitted_models_scores.get(model_name, [])
                idx_max = np.argmax(scores)
                preds = predictions_array[:, idx_max]

            # Создаем из него ДФ
            df = pd.DataFrame(data=preds, columns=new_columns, index=test.index)
            # Добавляем ДФ к результатам
            self.X_test_for_meta = pd.concat([self.X_test_for_meta, df], axis=1)

        # self.X_test_for_meta.to_excel(self.work_path.joinpath('X_test_for_meta_.xlsx'))

        # Если используется голосование - тогда избавляемся от вероятностей
        if self.pred_proba and self.use_voting:
            # Преобразуем в 3D массив
            array = self.X_test_for_meta.values
            array = array.reshape((array.shape[0], len(name_models), -1))

            # Считаем средние значения по оси 1 и оформляем в датафрейм
            self.X_test_for_meta = pd.DataFrame(data=np.argmax(array, axis=2),
                                                columns=name_models,
                                                index=test.index)

        if self.meta_columns is not None and self.final_estimator is not None:
            X_raw, *_ = self.process_data(self.final_estimator_name, test[self.meta_columns])
            self.X_test_for_meta = pd.concat([self.X_test_for_meta, X_raw], axis=1)

        if save_to_excel:
            train_for_meta = pd.concat([self.X_train_for_meta, self.y_train_for_meta], axis=1)
            train_for_meta.to_excel(self.work_path.joinpath('X_train_for_meta.xlsx'))
            self.X_test_for_meta.to_excel(self.work_path.joinpath('X_test_for_meta.xlsx'))

        # Обучение мета модели и получение предсказаний
        predict_columns = self.get_new_columns('predict')
        result = self.fit_predict_meta_model(test, predict_columns)

        self.X_test_for_meta[predict_columns] = result[predict_columns]

        if save_to_excel:
            self.X_test_for_meta.to_excel(
                self.work_path.joinpath('X_test_for_meta_pred.xlsx'))

        if self.pred_proba and not return_proba:
            result_ = np.argmax(result[predict_columns].values, axis=1)
            if self.classifier:
                # вернем исходные метки целевой переменной
                result_ = self.label_encoder.inverse_transform(result_)
            return result_

        return result[predict_columns].values

    def get_params(self):
        return self.final_model.get_params() if hasattr(self.final_model,
                                                        'get_params') else None


class CustomStackingClassifier(CustomStackingRegressor):
    def __init__(self, *args, pred_proba=True, use_voting=False, **kwargs):
        """
        Класс для стекинга классификаторов
        :param args:
        :param pred_proba: использовать вероятности
        :param use_voting: использовать голосование
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.pred_proba = pred_proba
        self.use_voting = use_voting
        self.classifier = True

    def predict_proba(self, test, save_to_excel=False):
        return self.predict(test, save_to_excel=save_to_excel, return_proba=True)


if __name__ == '__main__':
    device = torch.device('cpu')
    print('device:', device)
