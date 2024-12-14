# https://stepik.org/lesson/779915/step/3?thread=solutions&unit=782489

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from itertools import permutations
from tqdm import tqdm

df = pd.read_csv('https://stepik.org/media/attachments/lesson/779915/fs_task1_10f.csv')

X = df.drop(['target'], axis=1, errors='ignore')
y = df['target'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                    stratify=y)

important_features = ['feature_1', 'feature_2', 'feature_6', 'feature_8', 'feature_9']

features_columns = [X.columns.to_list(), ]

features_columns.extend([*map(list, permutations(important_features))])

result = pd.DataFrame(columns=['features', 'iters', 'train', 'valid'])
for features in tqdm(features_columns[:3]):
    model = CatBoostClassifier(random_state=42,
                               thread_count=-1,
                               # task_type="GPU",
                               )
    model.fit(X_train[features], y_train, eval_set=(X_test[features], y_test),
              verbose=False, plot=False, early_stopping_rounds=100)

    result.loc[len(result)] = [features, model.best_iteration_,
                               model.best_score_['learn']['MultiClass'],
                               model.best_score_['validation']['MultiClass'],
                               ]

task_type = model.get_all_params()['task_type']
result.to_excel(f'task_3_3_3_results_{task_type}.xlsx', index=False)
