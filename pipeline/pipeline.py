from pipeline.clean import Clean
from pipeline.features import Features
from pipeline.training import Training


class Pipeline(Clean, Features, Training):
    def __init__(self, path, target_name, encoding_method='one-hot', SHAP_threshold=None, param_grid=None):
        Clean.__init__(self,
                       path = path,
                       target_name = target_name,
                       encoding_method = encoding_method
                       )
        Features.__init__(self,
                          SHAP_threshold = SHAP_threshold,
                          X_train = self.X_train,
                          X_val = self.X_val,
                          X_test = self.X_test,
                          y_train = self.y_train,
                          y_val = self.y_val,
                          y_test = self.y_test
                          )
        Training.__init__(self,
                          X_train = self.X_train,
                          X_val = self.X_val,
                          X_test = self.X_test,
                          y_train = self.y_train,
                          y_val = self.y_val,
                          y_test = self.y_test,
                          param_grid = param_grid
                          )

    def clean(self):
        self.clean_dataset()

    def features(self):
        self.features_selection()

    def training(self):
        self.training_model()