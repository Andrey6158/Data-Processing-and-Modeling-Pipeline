import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import pickle

model = lgb.LGBMClassifier(random_state=42, verbose=-1)


def show_metrics(dataframes, model):
    for name, (X, y) in dataframes.items():
        pred = model.predict_proba(X)[:, 1]
        roc_auc = round(roc_auc_score(y, pred), 2)
        print(f"{name}: {roc_auc}")


class Training:
    """
    Class for:
    - training model and hyperparameter selection.
    """
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, param_grid=None):
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        if param_grid is None:
            self.param_grid = {
                'n_estimators': [3, 5, 10, 50, 100, 200],
                'max_depth': [-1, 10, 20],
                'num_leaves': [3, 5, 10, 20, 30],
                'learning_rate': [0.0001, 0.005, 0.01, 0.1, 0.2]
            }
        else:
            self.param_grid = param_grid

    def training_model(self):
        dataframes = {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test)
        }

        print("\033[4mProcessing training model and hyperparameter selection...\033[0m")
        model.fit(self.X_train, self.y_train)
        print('starting ROC-AUC:')
        show_metrics(dataframes, model=model)

        grid_search = GridSearchCV(model, self.param_grid, cv=3, scoring='roc_auc')
        grid_search.fit(self.X_val, self.y_val)
        best_params = grid_search.best_params_
        print()
        print(f"Best Parameters: {best_params}\n")

        best_model = grid_search.best_estimator_
        best_model.fit(self.X_train, self.y_train)
        print('finishing ROC-AUC:')
        show_metrics(dataframes, model=best_model)

        print()
        print("\033[4mProcessing of saving the model...\033[0m")
        with open('model_LGBMClassifier.pkl', 'wb') as file:
            pickle.dump(best_model, file)
        print("path: model_LGBMClassifier.pkl\n")

        print("\033[4mProcessing of saving the features...\033[0m")
        features = self.X_test.columns
        df_features = pd.DataFrame(features, columns=['features'])
        df_features.to_csv('features_model_LGBMClassifier.csv', index=False)
        print("path: features_model_LGBMClassifier.csv")
