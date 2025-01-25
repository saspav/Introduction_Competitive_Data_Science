import os
from pathlib import Path
import numpy as np
import pandas as pd
import copy

import torch

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# Вспомогательные блоки организации для пайплайна
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score

from data_process import PREDICTIONS_DIR, WORK_PATH

from set_all_seeds import set_all_seeds

__import__("warnings").filterwarnings('ignore')


class CustomStackingRegressor:
    def __init__(self, estimators, final_estimator=None, test_size=0.2, stratified=None,
                 cv_folds=None, cv_folds_meta=None, num_scaler=None, models_to_scale=None,
                 model_columns=None, cat_columns=None, num_columns=None, meta_columns=None,
                 features2drop=None, pred_proba=False, n_jobs=-1, verbose=False, **kwargs):
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
        self.num_scaler = StandardScaler if num_scaler is None else num_scaler
        self.models_to_scale = ['LinReg',
                                'LogReg',
                                'TabNetWork',
                                ] if models_to_scale is None else models_to_scale
        self.model_columns = model_columns
        self.cat_columns = cat_columns
        self.num_columns = num_columns
        self.meta_columns = meta_columns
        self.features2drop = features2drop
        self.pred_proba = pred_proba
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = kwargs.get('random_state', 127)
        self.stratified_target = None
        self.preprocessor = None
        self.X_train_for_meta = None
        self.y_train_for_meta = None
        self.X_test_for_meta = None
        self.fitted_models = {model_name: [] for model_name, _ in estimators}
        self.final_model = None
        set_all_seeds(seed=self.random_state)

    def fit_model(self, model_name, base_model, split, n_fold=0):
        name_fold = f', фолд: {n_fold}/{self.cv_folds}' if n_fold else ''
        print(f'Обучаю модель: {model_name}{name_fold}')
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

        return model

    def process_data(self, model_name, X_train, X_valid=None, y_train=None, y_valid=None):
        if model_name in ("CatBoost", "LightGBM", "XGBoost"):
            return X_train, X_valid, y_train, y_valid

        # Трансформируем данные train_df
        X_train_prep = self.preprocessor.transform(X_train)

        # Получаем имена новых колонок после трансформации
        num_columns_trn = self.num_columns
        cat_columns_trn = self.preprocessor.named_transformers_["categorical"].named_steps[
            "onehot"].get_feature_names_out(self.cat_columns)

        model_columns_trn = num_columns_trn + list(cat_columns_trn)
        # Преобразуем в DataFrame
        train_ = pd.DataFrame(X_train_prep, columns=model_columns_trn, index=X_train.index)
        valid_ = None
        if X_valid is not None:
            X_valid_ = self.preprocessor.transform(X_valid)
            valid_ = pd.DataFrame(X_valid_, columns=model_columns_trn, index=X_valid.index)
        if self.models_to_scale is None or model_name not in self.models_to_scale:
            # Признаки для этих моделей не нужно масштабировать
            train_[num_columns_trn] = X_train[num_columns_trn]
            if X_valid is not None:
                valid_[num_columns_trn] = X_valid[num_columns_trn]
        return train_, valid_, y_train, y_valid

    def get_split_folds(self, X):
        if self.stratified:
            skf = StratifiedKFold(n_splits=self.cv_folds, random_state=self.random_state,
                                  shuffle=True)
            skf_folds = skf.split(X, self.stratified_target)
        else:
            split_kf = KFold(n_splits=self.cv_folds, random_state=self.random_state,
                             shuffle=True)
            skf_folds = split_kf.split(X)
        return skf_folds

    def get_new_columns(self, model_name, make_proba=False):
        if self.pred_proba or make_proba:
            return [f'{model_name}_{label}' for label in self.y_train_for_meta.unique()]
        return model_name

    def fit(self, X, y, verbose=None, **kwargs):
        if verbose:
            self.verbose = verbose

        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
        numerical_transformer = Pipeline(steps=[
            ("scaler", self.num_scaler())
        ])
        # соединим два предыдущих трансформера в один
        self.preprocessor = ColumnTransformer(transformers=[
            ("numerical", numerical_transformer, self.num_columns),
            ("categorical", categorical_transformer, self.cat_columns)])

        # Применяем препроцессор к данным: Обучаем препроцессор на данных train_df
        self.preprocessor.fit(X[self.model_columns])

        self.stratified_target = X[self.stratified] if self.stratified else None

        self.X_train_for_meta = pd.DataFrame(index=X.index)
        self.y_train_for_meta = y.copy()

        for model_name, base_model in self.estimators:
            new_columns = self.get_new_columns(model_name)
            df = pd.DataFrame(index=X.index)
            df[new_columns] = 0

            if self.cv_folds:
                skf_folds = self.get_split_folds(X)

                for idx, (train_id, valid_id) in enumerate(skf_folds, 1):
                    X_train, y_train = X[self.model_columns].iloc[train_id], y.iloc[train_id]
                    X_valid, y_valid = X[self.model_columns].iloc[valid_id], y.iloc[valid_id]

                    split = self.process_data(model_name, X_train, X_valid, y_train, y_valid)
                    model = self.fit_model(model_name, base_model, split, idx)
                    self.fitted_models[model_name].append(model)

                    X_train_pd, X_valid_pd, y_train_pd, y_valid_pd = split
                    if model_name in ("TabNetWork",):
                        if self.pred_proba:
                            y_pred_valid = model.predict_proba(X_valid_pd.values)
                        else:
                            y_pred_valid = model.predict(X_valid_pd.values)
                    else:
                        if self.pred_proba:
                            y_pred_valid = model.predict_proba(X_valid_pd)
                        else:
                            y_pred_valid = model.predict(X_valid_pd)
                    if self.pred_proba:
                        df.iloc[valid_id, :] = y_pred_valid
                    else:
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
                if model_name in ("TabNetWork",):
                    if self.pred_proba:
                        predict_full_train = model.predict_proba(full_train.values)
                    else:
                        predict_full_train = model.predict(full_train.values)
                else:
                    if self.pred_proba:
                        predict_full_train = model.predict_proba(full_train)
                    else:
                        predict_full_train = model.predict(full_train)
                df[new_columns] = predict_full_train

            self.X_train_for_meta = pd.concat([self.X_train_for_meta, df], axis=1)

        if self.meta_columns is not None:
            X_raw, *_ = self.process_data(self.final_estimator_name, X[self.meta_columns])
            self.X_train_for_meta = pd.concat([self.X_train_for_meta, X_raw], axis=1)

        return self

    def predict(self, test, save_to_excel=False):
        # Получаем предсказания на тестовой выборке
        self.X_test_for_meta = pd.DataFrame(index=test.index)
        for model_name, models_list in self.fitted_models.items():
            test_prep, *_ = self.process_data(model_name, test[self.model_columns])

            # Инициализация массива для хранения предсказаний всех моделей
            predictions_array = np.zeros((len(test_prep), len(models_list)))

            # Сбор предсказаний от каждой модели
            for idx, model in enumerate(models_list):
                if model_name in ("TabNetWork",):
                    pred_test_prep = model.predict(test_prep.values)
                else:
                    pred_test_prep = model.predict(test_prep)

                # Сохраняем предсказания в массив
                predictions_array[:, idx] = pred_test_prep

            new_columns = self.get_new_columns(model_name)
            # Вычисление статистик по строкам: можно попробовать еще медиану !!!
            df = pd.DataFrame(data=predictions_array.mean(axis=1),
                              columns=[new_columns],
                              index=test.index)

            self.X_test_for_meta = pd.concat([self.X_test_for_meta, df], axis=1)

        if self.meta_columns is not None:
            X_raw, *_ = self.process_data(self.final_estimator_name, test[self.meta_columns])
            self.X_test_for_meta = pd.concat([self.X_test_for_meta, X_raw], axis=1)

        # Обучаем метамодель
        predict_columns = self.get_new_columns('predict')
        result = pd.DataFrame(index=test.index)

        if save_to_excel:
            train_for_meta = pd.concat([self.X_train_for_meta, self.y_train_for_meta], axis=1)
            train_for_meta.to_excel(WORK_PATH.joinpath('X_train_for_meta.xlsx'))
            self.X_test_for_meta.to_excel(WORK_PATH.joinpath('X_test_for_meta.xlsx'))

        XM = self.X_train_for_meta
        yM = self.y_train_for_meta

        if self.cv_folds_meta:
            # Инициализация массива для хранения предсказаний всех моделей
            predictions_array = np.zeros((len(self.X_test_for_meta), self.cv_folds_meta))

            skf_folds = self.get_split_folds(XM)
            for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
                X_train, y_train = XM.iloc[train_idx], yM.iloc[train_idx]
                X_valid, y_valid = XM.iloc[valid_idx], yM.iloc[valid_idx]

                split = X_train, X_valid, y_train, y_valid

                self.final_model = self.fit_model(self.final_estimator_name,
                                                  self.final_estimator,
                                                  split, idx)

                pred_test_prep = self.final_model.predict(self.X_test_for_meta)
                # Сохраняем предсказания в массив
                predictions_array[:, idx - 1] = pred_test_prep

            # Вычисление статистик по строкам
            result = pd.DataFrame(data=predictions_array.mean(axis=1),
                                  columns=[predict_columns],
                                  index=self.X_test_for_meta.index)
            # Преобразование в метку 0 / 1
            result[predict_columns] = result[predict_columns].map(lambda z: int(z >= 0.5))

        else:
            split = train_test_split(XM, yM, test_size=self.test_size,
                                     stratify=self.stratified_target,
                                     random_state=self.random_state)

            self.final_model = self.fit_model(self.final_estimator_name,
                                              self.final_estimator,
                                              split)
            result[predict_columns] = self.final_model.predict(self.X_test_for_meta)

        self.X_test_for_meta[predict_columns] = result[predict_columns]
        if save_to_excel:
            self.X_test_for_meta.to_excel(WORK_PATH.joinpath('X_test_for_meta_pred.xlsx'))

        return result[predict_columns].values

    def get_params(self):
        return self.final_model.get_params()


class CustomStackingClassifier(CustomStackingRegressor):
    def __init__(self, *args, pred_proba=True, **kwargs):
        super().__init__(*args, pred_proba=pred_proba, **kwargs)

    def predict_proba(self, test, save_to_excel=False):
        # Получаем предсказания на тестовой выборке
        self.X_test_for_meta = pd.DataFrame(index=test.index)
        for model_name, models_list in self.fitted_models.items():
            test_prep, *_ = self.process_data(model_name, test[self.model_columns])

            new_columns = self.get_new_columns(model_name)
            df = pd.DataFrame(index=test.index)
            df[new_columns] = 0
            # print(df[new_columns].shape)

            for model in models_list:
                if model_name in ("TabNetWork",):
                    pred_test_prep = model.predict_proba(test_prep.values)
                else:
                    pred_test_prep = model.predict_proba(test_prep)

                df[new_columns] = df[new_columns].values + pred_test_prep

            if self.cv_folds:
                df[new_columns] /= self.cv_folds

            self.X_test_for_meta = pd.concat([self.X_test_for_meta, df], axis=1)

        # Обучаем метамодель
        predict_columns = self.get_new_columns('predict')
        result = pd.DataFrame(index=test.index)
        result[predict_columns] = 0
        if save_to_excel:
            train_for_meta = pd.concat([self.X_train_for_meta, self.y_train_for_meta], axis=1)
            train_for_meta.to_excel(WORK_PATH.joinpath('X_train_for_meta_proba.xlsx'))
            self.X_test_for_meta.to_excel(WORK_PATH.joinpath('X_test_for_meta_proba.xlsx'))

        if self.y_train_for_meta.nunique() == 2:
            meta_cols = [col for col in self.X_train_for_meta.columns if col.endswith('_1')]
            meta_cols_new = ['cls_1_min', 'cls_1_max', 'cls_1_mean']
            for col, func in zip(meta_cols_new, (np.min, np.max, np.mean)):
                train_result = self.X_train_for_meta.filter(like='_0').apply(func, axis=1)
                test_result = self.X_test_for_meta.filter(like='_1').apply(func, axis=1)

            y_pred = (test_result >= 0.5).astype(int)
            true_submit_csv = PREDICTIONS_DIR.joinpath('titanic_true_submit.csv')
            if true_submit_csv.is_file():
                true_submit = pd.read_csv(true_submit_csv)
                true_submit = true_submit['Survived']
                # Рассчитываем accuracy_score
                print('ensemble score:', round(accuracy_score(true_submit, y_pred), 6))

        XM = self.X_train_for_meta
        yM = self.y_train_for_meta

        if self.cv_folds_meta:
            skf_folds = self.get_split_folds(XM)
            for idx, (train_idx, valid_idx) in enumerate(skf_folds, 1):
                X_train, y_train = XM.iloc[train_idx], yM.iloc[train_idx]
                X_valid, y_valid = XM.iloc[valid_idx], yM.iloc[valid_idx]

                split = X_train, X_valid, y_train, y_valid

                self.final_model = self.fit_model(self.final_estimator_name,
                                                  self.final_estimator,
                                                  split, idx)

                result[predict_columns] += self.final_model.predict_proba(
                    self.X_test_for_meta)

            result[predict_columns] /= self.cv_folds_meta
        else:
            split = train_test_split(XM, yM, test_size=self.test_size,
                                     stratify=self.stratified_target,
                                     random_state=self.random_state)
            X_train, X_valid, y_train, y_valid = split

            self.final_model = self.fit_model(self.final_estimator_name,
                                              self.final_estimator,
                                              split)
            result[predict_columns] = self.final_model.predict_proba(self.X_test_for_meta)

        self.X_test_for_meta[predict_columns] = result[predict_columns]
        if save_to_excel:
            self.X_test_for_meta.to_excel(
                WORK_PATH.joinpath('X_test_for_meta_pred_proba.xlsx'))

        return result[predict_columns].values


if __name__ == '__main__':
    device = torch.device('cpu')
    print('device:', device)
