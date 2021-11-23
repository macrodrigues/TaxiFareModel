from mlflow.mleap import save_model
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from TaxiFareModel.data import clean_data, get_data
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property
from ml_flow_test import *
import joblib


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X = X
        self.y = y
        self.pipeline = None

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipe, 'model2.joblib')


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', RobustScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")


        self.model = GradientBoostingRegressor(learning_rate=0.1,
                                            alpha=0.9,
                                            n_estimators=50)

        self.mlflow_log_param('model', self.model)

        # workflow
        self.pipe = Pipeline(
            steps=[('feat_eng_bloc',
                    preproc_pipe), ('regressor', self.model)])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipe.fit(self.X, self.y)

    def get_pipe_params(self):
        return self.pipe.get_params()

    def fine_tune(self, X_train, y_train):
        param_grid = {'regressor__learning_rate': [0.0001, 0.001, 0.01, 0.1 ],
                'regressor__alpha':[0.1, 0.2, 0.5, 0.9],
                'regressor__n_estimators': [50, 100, 200, 300],
        }
        search = GridSearchCV(self.pipe,
                                param_grid,
                                cv=5)
        search.fit(X_train, y_train)
        print(search.best_params_)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipe.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('metric', rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri("https://mlflow.lewagon.co/")
        return MlflowClient()


    @memoized_property
    def mlflow_experiment_id(self):
        experiment_name = "[PT] [Lisbon] [macrodrigues] " + \
        str(self.model).split('(')[0] + " Taxifare_768_0"

        try:
            return self.mlflow_client.create_experiment(experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":

    # get data
    df = get_data()
    # clean data
    df_cleaned = clean_data(df)
    # set X and y
    y = df_cleaned.pop("fare_amount")
    X = df_cleaned
    # hold out
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=42)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.run()
    # evaluate
    rmse = trainer.evaluate(X_test, y_test)
    #get params
    #print(trainer.get_pipe_params())
    #trainer.fine_tune(X_train, y_train)
    # save model
    trainer.save_model()
