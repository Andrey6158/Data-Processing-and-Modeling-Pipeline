import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split


class Clean:
    def __init__(self, path, target_name, encoding_method='one-hot'):
        """
        Class for preprocessing a dataset, including:
        - removing duplicates;
        - separating features by type;
        - encoding of object features;
        - processing nan values;
        - removing features with high mutual correlation;
        - splitting dataset into training, testing and validation.
        """
        self.dataset = pd.read_csv(path)
        self.target_name = target_name
        self.encoding_method = encoding_method
        self.numeric_features = []
        self.object_features = []
        self.bool_features = []
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.y_val_test = None
        self.X_val_test = None


    def clean_dataset(self):
        """
        Method for preprocessing a dataset, including:
        - removing duplicates;
        - separating features by type;
        - encoding of object features;
        - processing nan values;
        - removing features with high mutual correlation;
        - splitting dataset into training, testing and validation.
        """
        print(f'Shape of dataset: {self.dataset.shape}\n')

        tmp1 = self.dataset.shape[0]
        print("\033[4mRemoving duplicates...\033[0m")
        self.dataset.drop_duplicates(inplace=True)
        tmp2 = self.dataset.shape[0]
        print(str(tmp2 - tmp1) + " - strings was filtered\n")

        print("\033[4mSeparating features by type...\033[0m")
        self.numeric_features = self.dataset.select_dtypes(include=['number']).columns.tolist()
        self.object_features = self.dataset.select_dtypes(include=['object']).columns.tolist()
        self.bool_features = self.dataset.select_dtypes(include=['bool']).columns.tolist()
        print(f'numeric_features: {len(self.numeric_features)}')
        print(f'object_features: {len(self.object_features)}')
        print(f'bool_features: {len(self.bool_features)}\n')

        print("\033[4mRemoving features with high mutual correlation...\033[0m")
        self.remove_high_correlation_features()

        if self.encoding_method == 'one-hot':
            self.dataset = pd.get_dummies(self.dataset, columns=self.object_features)
            print()
            print("\033[4mPreprocessing encoding object features...\033[0m")
            print(f'data types in the dataset: {self.dataset.dtypes.unique()}\n')

            print('\033[4mSplitting dataset into training, testing and validation...\033[0m')
            self.splitt_dataset_train_test_val()

        else:
            print()
            print('\033[4mSplitting dataset into training, testing and validation...\033[0m')
            self.splitt_dataset_train_test_val()

            print("\033[4mPreprocessing encoding object features...\033[0m")
            encoder = TargetEncoder(cols=self.object_features)
            self.X_train[self.object_features] = \
                encoder.fit_transform(self.X_train[self.object_features], self.y_train)
            self.X_val[self.object_features] = encoder.transform(self.X_val[self.object_features])
            self.X_test[self.object_features] = encoder.transform(self.X_test[self.object_features])
            print(f'data types in the dataset:')
            print(f'X_train: {self.X_train.dtypes.unique()}')
            print(f'X_val: {self.X_val.dtypes.unique()}')
            print(f'X_test: {self.X_test.dtypes.unique()}\n')

        print("\033[4mProcessing nan value by k-nearest neighbors method...\033[0m")
        imputer = KNNImputer(n_neighbors=2)
        self.X_train = pd.DataFrame(imputer.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_val = pd.DataFrame(imputer.fit_transform(self.X_val), columns=self.X_val.columns)
        self.X_test = pd.DataFrame(imputer.fit_transform(self.X_test), columns=self.X_test.columns)
        print(f'number of nan values in dataset: '
              f'{self.X_train.isna().sum().sum() + self.X_val.isna().sum().sum() + self.X_test.isna().sum().sum()}\n')


    def remove_high_correlation_features(self, threshold=0.9):
        """
        Removes features with a high correlation above the specified threshold.
        Keeps the features with higher correlation with the target.
        """
        corr_matrix = self.dataset[self.numeric_features + self.bool_features]. \
            drop(columns=self.target_name).corr(method='spearman').abs()

        target_corr = self.dataset[self.numeric_features + self.bool_features]. \
            corrwith(self.dataset[self.target_name], method='spearman').abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = set()

        for column in upper_triangle.columns:
            for row in upper_triangle.index:
                if upper_triangle.loc[row, column] > threshold:
                    if target_corr[row] > target_corr[column]:
                        to_drop.add(column)
                    else:
                        to_drop.add(row)

        self.dataset.drop(columns=list(to_drop), inplace=True)
        print(f'removed futures with high correlation: {to_drop}')
        print(f'Shape of dataset: {self.dataset.shape}')


    def splitt_dataset_train_test_val(self):
        """
        Splitt dataset into train, val, test
        """
        X = self.dataset.drop(columns=self.target_name, axis=1)
        y = self.dataset[self.target_name]

        self.X_train, self.X_val_test, self.y_train, self.y_val_test = \
            train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        self.X_val, self.X_test, self.y_val, self.y_test = \
            train_test_split(self.X_val_test, self.y_val_test,
                             test_size=0.3, random_state=42, stratify=self.y_val_test)

        print("Shape of:")
        print(f'X_train: {self.X_train.shape}, X_val_test: {self.X_val_test.shape}')
        print(f'X_val: {self.X_val.shape}, X_test: {self.X_test.shape}\n')
