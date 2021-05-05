import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

import pickle
from sklearn.decomposition import PCA

from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import random
from sklearn.linear_model import LogisticRegression

df_all = pd.read_csv('res/data_tab.csv', index_col=0)
df_1 = df_all[df_all['target'] == 1]
df_0 = df_all[df_all['target'] == 0]

paciente_1 = df_1.id_paciente.unique()
pac_test_1 = random.sample(list(paciente_1), len(paciente_1) // 3)

paciente_0 = df_0.id_paciente.unique()
pac_test_0 = random.sample(list(paciente_0), len(paciente_0) // 3)

train = pd.concat([df_0.loc[~df_0.id_paciente.isin(pac_test_0), :],
                   df_1.loc[~df_1.id_paciente.isin(pac_test_1), :]])

test = pd.concat([df_0.loc[df_0.id_paciente.isin(pac_test_0), :],
                  df_1.loc[df_1.id_paciente.isin(pac_test_1), :]])

patrones = train.patron_reg.unique()

resultado = []
for i, patron in enumerate(patrones):
    y_train = train.loc[train['patron_reg'] == patron, 'target']
    x_train = train.loc[train['patron_reg'] == patron, train.columns[-3:]]
    cli_train  = train.loc[train['patron_reg'] == patron, 'id_paciente']
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    pipe = Pipeline([('imputer', imp),
                     ('scaler', StandardScaler()),
                     ('SVR', SVR())])
    opt = GridSearchCV(
        estimator=pipe,
        param_grid={'SVR__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'SVR__degree': [2, 3],
                    'SVR__C': [0.01, 0.1, 0.5, 1, 5, 10, 50, 100], },
        cv=2
    )

    opt.fit(x_train, y_train)

    y_test = test.loc[test['patron_reg'] == patron, 'target']
    x_test = test.loc[test['patron_reg'] == patron, test.columns[-3:]]
    cli_test = test.loc[test['patron_reg'] == patron, 'id_paciente']
    prob_dev = opt.predict(x_test)
    prob_train = opt.predict(x_train)



    print("Regresion SVR train svr" + patron, roc_auc_score(y_train, prob_train))
    print(len(train))
    print("Regresion SVR dev svr" + patron, roc_auc_score(y_test, prob_dev))
    print(len(test))
    # save the model to disk
    filename = 'res/model_svr_' + patron + '.sav'
    pickle.dump(opt, open(filename, 'wb'))

    res_SVR_train = roc_auc_score(y_train, prob_train)
    res_SVR_test = roc_auc_score(y_test, prob_dev)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    pipe = Pipeline([('imputer', imp),
                     ('scaler', StandardScaler()),
                     ('LR', LogisticRegression())])
    opt = GridSearchCV(
        estimator=pipe,
        param_grid={'LR__C': [0.1, 0.5, 1, 5, 10]},
        cv=2
    )

    opt.fit(x_train, y_train)

    y_test = test.loc[test['patron_reg'] == patron, 'target']
    x_test = test.loc[test['patron_reg'] == patron, test.columns[-3:]]

    prob_dev_lr = [x[1] for x in opt.predict_proba(x_test)]
    prob_train_lr = [x[1] for x in opt.predict_proba(x_train)]
    print("Regresion train LR" + patron, roc_auc_score(y_train, prob_train_lr))
    print(len(train))
    print("Regresion dev LR" + patron, roc_auc_score(y_test, prob_dev_lr))
    print(len(test))
    if i == 0:
        df_cli_train = pd.concat([ train.loc[train['patron_reg'] == patron, ['id_paciente','target','patron_reg']].reset_index(drop=True),
                                   pd.Series(prob_train), pd.Series(prob_train_lr)], axis=1, ignore_index=True)
        df_cli_test = pd.concat([test.loc[test['patron_reg'] == patron, ['id_paciente','target','patron_reg']].reset_index(drop=True),
                                 pd.Series(prob_dev), pd.Series(prob_dev_lr)], axis=1, ignore_index=True)
    else:
        df_tmp_train = pd.concat([train.loc[train['patron_reg'] == patron, ['id_paciente','target', 'patron_reg']].reset_index(drop=True),
                                  pd.Series(prob_train), pd.Series(prob_train_lr)], axis=1, ignore_index=True)
        df_tmp_test = pd.concat([test.loc[test['patron_reg'] == patron, ['id_paciente','target','patron_reg']].reset_index(drop=True),
                                 pd.Series(prob_dev), pd.Series(prob_dev_lr)], axis=1, ignore_index=True)
        df_cli_train = pd.concat([df_cli_train, df_tmp_train], axis=0)
        df_cli_test = pd.concat([df_cli_test, df_tmp_test], axis=0)
    # save the model to disk
    filename = 'res/model_LR_' + patron + '.sav'
    pickle.dump(opt, open(filename, 'wb'))
    resultado.append(
        [patron, res_SVR_train, res_SVR_test, roc_auc_score(y_train, prob_train), roc_auc_score(y_test, prob_dev)])

df_resultado = pd.DataFrame(resultado,
                            columns=['patron', 'res_SVR_train', 'res_SVR_test', 'res_LR_train', 'res_LR_test'])

df_cli_train.columns = ['id_paciente', 'target', 'patron', 'svr',  'lr']
df_cli_train['media_score'] = (df_cli_train['svr'] + df_cli_train['lr'] ) / 2
df_res_train_final = df_cli_train.groupby(['id_paciente', 'target']).media_score.mean()
df_res_train_final = df_res_train_final.reset_index()
roc_auc_score(df_res_train_final['target'], df_res_train_final['media_score'])


df_cli_test.columns = ['id_paciente', 'target', 'patron', 'svr',  'lr']
df_cli_test['media_score'] = (df_cli_test['svr'] + df_cli_test['lr'] ) / 2
df_res_test_final = df_cli_test.groupby(['id_paciente', 'target']).media_score.mean()
df_res_test_final = df_res_test_final.reset_index()
roc_auc_score(df_res_test_final['target'], df_res_test_final['media_score'])