import pandas as pd
import numpy as np
import shap
import lightgbm as lgb


def continuous2interval(df, df_target, percent_interval=0.1):
    special_target = []
    interval_target = []
    begin = False
    temp_percent = 0
    target_distribution = df[df_target].value_counts(normalize=True).reset_index()
    target_distribution.columns = [df_target, 'proportion']
    for index, row in target_distribution.sort_values(by=df_target).iterrows():
        if row['proportion'] >= percent_interval:
            special_target.append(row[df_target])
        else:
            temp_percent += row['proportion']
            if not begin:
                begin = row[df_target]
            if temp_percent >= percent_interval:
                interval_target.append([begin, row[df_target]])
                begin = False
                temp_percent = 0
    if begin:
        interval_target.append([begin, np.inf])
    return interval_target, special_target


class Features:
    """
    Class for:
    - features selection based on the population stability index PSI;
    - features selection by their importance using the SHAP method.
    """

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, SHAP_threshold=None):
        self.top_important = None
        self.PSI = None
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        if SHAP_threshold is None:
            self.SHAP_threshold = 0.01
        else:
            self.SHAP_threshold = SHAP_threshold


    def features_selection(self):
        print(f'Shape of X: {self.X_train.shape}\n')

        print("\033[4mProcessing features selection based on the population stability index PSI...\033[0m")
        tmp1 = self.X_train.shape[1]
        self.selection_by_PSI(self.X_train, self.X_val)
        self.selection_by_PSI(self.X_train, self.X_test)
        tmp2 = self.X_train.shape[1]
        print(f'{tmp1 - tmp2} - feature was filtered\n')
        print(f'Shape of X: {self.X_train.shape}\n')

        print("\033[4mProcessing features selection by their importance...\033[0m")
        self.features_importance()


    def PSI_factor_analysis(self, dev, val, column):
        intervals = [-np.inf] + [i[0] for i in continuous2interval(dev, column)[0]] + [np.inf]
        dev_temp = pd.cut(dev[column], intervals).value_counts(sort=False, normalize=True)
        val_temp = pd.cut(val[column], intervals).value_counts(sort=False, normalize=True)
        PSI = sum(((dev_temp - val_temp) * np.log(dev_temp / val_temp)).replace([np.inf, -np.inf], 0))
        self.PSI = PSI


    def selection_by_PSI(self, dev, val):
        columns_PSI_normal = []
        for column in dev.columns:
            self.PSI_factor_analysis(dev, val, column)
            if self.PSI < 0.2:
                columns_PSI_normal.append(column)
        self.X_train = self.X_train[columns_PSI_normal]
        self.X_val = self.X_val[columns_PSI_normal]
        self.X_test = self.X_test[columns_PSI_normal]


    def features_importance(self):
        model = lgb.LGBMClassifier(random_state=42, verbose=1)

        self.X_train.columns = self.X_train.columns.str.replace(r'[^\w\s]', '', regex=True)
        self.X_val.columns = self.X_val.columns.str.replace(r'[^\w\s]', '', regex=True)
        self.X_test.columns = self.X_test.columns.str.replace(r'[^\w\s]', '', regex=True)

        model.fit(self.X_train, self.y_train)

        explainer = shap.Explainer(model, self.X_train)
        shap_values = explainer(self.X_train)
        mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)
        importance = pd.DataFrame({'feature': self.X_train.columns, 'importance': mean_abs_shap_values}) \
            .sort_values(by='importance', ascending=False)

        shap.initjs()
        shap.summary_plot(shap_values, self.X_train)

        SHAP_threshold = self.SHAP_threshold

        max_importance = importance['importance'].max()
        self.top_important = importance[importance['importance'] >= SHAP_threshold * max_importance][
            'feature']
        print(f'top_important: {list(self.top_important)}')

        self.X_train = self.X_train[self.top_important]
        self.X_val = self.X_val[self.top_important]
        self.X_test = self.X_test[self.top_important]
        print(f'Shape of X: {self.X_train.shape}\n')
