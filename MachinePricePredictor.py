# Standard Library Imports
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lightgbm import LGBMRegressor


class MachinePricePredictor:
    """
    MachinePricePredictor class to preprocess the data, train the model, and make predictions.

    Attributes:
    model: trained model
    preprocessor: data preprocessor
    label_encoders: dictionary of label encoders for categorical columns
    scaler: StandardScaler object
    df: pd.DataFrame, input dataset
    MODELSAVEPATH: Path, path to save the model
    VISUALPATH: Path, path to save visualizations

    Methods:
    load_data: Load and prepare the dataset
    preprocess_data: Apply feature engineering to the dataset
    prepare_features: Prepare features for model training
    train: Train the model on the provided data
    predict: Make predictions using the trained model
    evaluate_model: Evaluate model performance
    plot_actual_vs_predicted: Plot actual vs predicted values
    plot_prediction_distribution: Plot prediction distribution with confidence intervals
    visualize: Visualize the data before or after preprocessing
    """

    def __init__(self, data_path="data/machines.csv"):
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.df = self.load_data(data_path)
        self.MODELSAVEPATH = Path("models")
        self.VISUALPATH = Path("visualizations")

    def load_data(self, file_path):
        """Load and prepare the dataset.

        Args:
        file_path: str, path to the dataset file

        Returns:
        pd.DataFrame, loaded dataset
        """

        # define dtype mapping for columns
        dtype_mapping = {
            "Model Descriptor": str,
            "Screen Size": str,
            "Stick Length": str,
            "Thumb": str,
            "Product Class Description": str,
        }

        # read the dataset
        df = pd.read_csv(
            file_path, parse_dates=["Sales date"], dtype=dtype_mapping, low_memory=False
        )

        # ensure correct data types
        df["Sales Price"] = pd.to_numeric(df["Sales Price"], errors="coerce")
        df["Sales date"] = pd.to_datetime(df["Sales date"], errors="coerce")

        # remove the first column if it is an index (the .csv came with an index column)
        df.drop(columns=df.filter(regex="Unnamed"), inplace=True)

        # return the loaded dataset
        return df

    def preprocess_data(self, df, drop_cols=["Sales ID"], save=True):
        """Apply feature engineering to the dataset.

        Args:
        df: pd.DataFrame, input dataset
        drop_cols: list, columns to drop

        Returns:
        pd.DataFrame, preprocessed dataset
        """
        # make sure to work with a copy of the data, not the original
        df = df.copy()

        # log transform the target variable because it is left-skewed
        df["Sales Price"] = np.log1p(df["Sales Price"])

        # handle date features, separate into year, month, quarter, and day of week
        # quarter is more informative than month for seasonal patterns and has greater financial significance
        df["Sales Year"] = df["Sales date"].dt.year
        df["Sales Month"] = df["Sales date"].dt.month
        df["Sales Quarter"] = df["Sales date"].dt.quarter
        df["Sales Dayofweek"] = df["Sales date"].dt.dayofweek

        # cyclical encoding for month (sin and cos since moths are cyclical across years)
        # this is like a continuous version of one-hot encoding, the same as its done in some LLM encoders
        df["Sales Month Sin"] = np.sin(2 * np.pi * df["Sales Month"] / 12)
        df["Sales Month Cos"] = np.cos(2 * np.pi * df["Sales Month"] / 12)

        # add a feature for machine age
        df["Machine Age"] = df["Sales Year"] - df["Year Made"]
        df["Machine Age"] = df["Machine Age"].clip(lower=0)

        # handle missing values before calculating some new features
        df["MachineHours CurrentMeter"] = df["MachineHours CurrentMeter"].fillna(
            df["MachineHours CurrentMeter"].median()
        )
        df.fillna(
            {"Usage Band": "Unknown", "Enclosure": "None or Unspecified"}, inplace=True
        )

        # add a usage intensity feature as it is more informative than just machine hours
        # more usage = less value (expected)
        # add a feature for whether the machine has an enclosure or not
        # add a feature for whether the machine has AC or not
        df["Usage Intensity"] = df["MachineHours CurrentMeter"] / (
            df["Machine Age"] + 1
        )
        df["Has Enclosure"] = df["Enclosure"].apply(
            lambda x: 1 if x != "None or Unspecified" else 0
        )
        df["Has AC"] = df["Enclosure"].apply(lambda x: 1 if "AC" in str(x) else 0)

        # ensure correct str format for product since we will need to apply a regex pattern
        df["Product Class Description"] = (
            df["Product Class Description"].astype(str).fillna("")
        )
        df["Product Class Description"] = df["Product Class Description"].str.strip()

        # regex pattern base on: Type A - 100 to 200 Horsepower
        pattern = r"Type\s+([A-F])\s+-\s*(?:(\d+(?:\.\d+)?)\s*to\s*(?:(\d+(?:\.\d+)?))?\s*(?:Horsepower|Lb Operating Capacity|Metric Tons|Ft Standard Digging Depth))?"

        # extract Product Type, Min Value, Max Value
        df[["Product Type", "Min Value", "MaxValue"]] = df[
            "Product Class Description"
        ].str.extract(pattern, expand=True)

        # convert to numeric
        df["Min Value"] = pd.to_numeric(df["Min Value"], errors="coerce")
        df["MaxValue"] = pd.to_numeric(df["MaxValue"], errors="coerce")

        # extract if we are dealing with Horsepower, Lb Operating Capacity, Metric Tons, or Ft Standard Digging Depth
        df["Units"] = df["Product Class Description"].str.extract(
            r"(Horsepower|Lb Operating Capacity|Metric Tons|Ft Standard Digging Depth)",
            expand=False,
        )

        # handle missing values for Min Value and Max Value (some might only have one value or none)
        # if one value is missing, fill it with the other value
        df["MaxValue"] = df["MaxValue"].fillna(df["Min Value"])

        # drop the original columns that we engineered new features from
        df.drop(columns=["Sales date", "Product Class Description"], inplace=True)

        # drop columns that are not needed
        df.drop(columns=drop_cols, errors="ignore", inplace=True)

        # remove rows with invalid years (less than 1800 as it makes no sense)
        df = df[df["Year Made"] > 1800]

        # label encode categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        # do we want to save the processed data in a file?
        if save:
            self.PROCESSEDDATAPATH = Path("data")
            self.PROCESSEDDATAPATH.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.PROCESSEDDATAPATH / "processed_data.csv", index=False)

        # return the preprocessed dataset
        return df

    def prepare_features(self, X):
        """Prepare features for model training.

        Args:
        X: pd.DataFrame, input features

        Returns:
        ColumnTransformer, preprocessor for the features
        """
        # select numerical and categorical features
        numerical_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

        numeric_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),  # median is best than mean specially for skewed data
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),  # most frequent is best for categorical data and we have a lot of categorical data
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # create pipeline transformation
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        # return the preprocessor
        return self.preprocessor

    def train(self, save=True):
        """Train the model on the provided data.

        Args:
        save: bool, whether to save the trained model

        Returns:
        y_test: np.array, true target values
        y_pred: np.array, predicted target values
        """

        # preprocess data
        print("Preprocessing data for training...")
        df = self.preprocess_data(self.df)

        # split features and target
        X = df.drop(columns=["Sales Price"])
        y = df["Sales Price"]

        # split data according to the 80/20 rule
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # prepare the features
        # actually LGBM does not need scaling but it is good practice to scale the data and apply the transformations
        # as before
        preprocessor = self.prepare_features(X_train)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # train the model
        # we are using a LightGBM Regressor, since it is a good model for tabular data
        # as it is fast and accurate and it is our case. It is also good for tabular data.
        # a neural network would be overkill for this problem and it might need more data
        # or more complex preprocessing. Also, it could require more compute power.
        self.model = LGBMRegressor(random_state=42, n_estimators=4000, num_leaves=100)
        print("Training the model...")
        self.model.fit(X_train_transformed, y_train)

        # save the model ? if so save it as a pickle file
        if save:
            print("Saving the model...")
            self.MODELSAVEPATH.mkdir(parents=True, exist_ok=True)
            model_path = self.MODELSAVEPATH / "model.pkl"
            with open(model_path, "wb") as file:
                pickle.dump(self.model, file)

        # predict on the test set
        y_pred = self.model.predict(X_test_transformed)
        y_pred = np.expm1(y_pred)
        y_test = np.expm1(y_test)

        # return the true and predicted values
        return y_test, y_pred

    def predict(self, X):
        """Make predictions using the trained model.

        Args:
        X: pd.DataFrame, input features

        Returns:
        np.array, predicted target values
        """

        # preprocess the data using the preprocessor that was fitted during training
        X_transformed = self.preprocessor.transform(X)

        # make predictions
        y_pred = self.model.predict(X_transformed)

        # return the predictions (we need to apply the inverse transformation to get the actual price, the model was
        # trained on log1p values since the target was left-skewed)
        return np.expm1(y_pred)

    def evaluate_model(self, y_test, y_pred):
        """Evaluate model performance.

        Args:
        y_test: np.array, true target values
        y_pred: np.array, predicted target values
        """

        print("Model Performance:")
        # the r2 score is the best metric for this problem since it is a regression problem and the range of values
        # is continuous and large. See more on the README.md
        print(f"R2 Score: {r2_score(y_test, y_pred)}")

        # include other metrics for evaluation
        print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
        print(
            f"Average % Error: {np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.2f}%"
        )

        # plot actual vs predicted values and prediction distribution for visual purposes only
        self.plot_actual_vs_predicted(y_test, y_pred)
        self.plot_prediction_distribution(y_test, y_pred)

    def plot_actual_vs_predicted(self, y_test, y_pred):
        """Plot actual vs predicted values."""

        # ensure the directory exists, otherwise create it
        self.VISUALPATH.mkdir(parents=True, exist_ok=True)

        # plot the actual vs predicted values
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=y_test, y=y_pred)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted Prices")
        plt.savefig(f"{self.VISUALPATH}/actual_vs_predicted.png")

    def plot_prediction_distribution(self, y_test, y_pred):
        """Plot prediction distribution with confidence intervals."""

        # ensure the directory exists, otherwise create it
        self.VISUALPATH.mkdir(parents=True, exist_ok=True)

        # we need to group them because we want to calculate the average prediction for each true value
        # as well as the standard deviation to plot the confidence intervals
        df = pd.DataFrame({"True": y_test, "Predicted": y_pred})
        grouped = (
            df.groupby("True")
            .agg(avg_pred=("Predicted", "mean"), std_pred=("Predicted", "std"))
            .reset_index()
        )

        # plot the average prediction with confidence intervals
        plt.figure(figsize=(8, 6))
        plt.plot(
            grouped["True"],
            grouped["avg_pred"],
            color="r",
            label="Average Prediction",
            linewidth=2,
        )
        plt.fill_between(
            grouped["True"],
            grouped["avg_pred"] - grouped["std_pred"],
            grouped["avg_pred"] + grouped["std_pred"],
            color="b",
            alpha=0.2,
            label="1 Standard Deviation",
        )

        plt.xlabel("True Price")
        plt.ylabel("Predicted Price")
        plt.title("Average Predicted Price for Each True Price")
        plt.legend()
        plt.savefig(f"{self.VISUALPATH}/prediction_distribution.png")

    def visualize(self, prefeature=True):
        df = self.df.copy()

        # are we visualizing before or after feature engineering?
        if prefeature:
            SAVEPATH = Path("visualizations/prefeature")
        else:
            SAVEPATH = Path("visualizations/postfeature")
            df = self.preprocess_data(df, save=False)

        # ensure the directory exists, otherwise create it
        SAVEPATH.mkdir(parents=True, exist_ok=True)

        # 1. distribution of the target variable ('Sales Price'). This is the most important plot
        # drop missing values and convert to float
        sales_price = df["Sales Price"].dropna().values.flatten().astype(float)

        # plot the distribution of the target variable
        plt.figure(figsize=(8, 6))
        bins = np.linspace(sales_price.min(), sales_price.max(), 30)  # 30 bins
        skewness = df["Sales Price"].dropna().skew()
        sns.histplot(
            data=sales_price, bins=bins, kde=True, label=f"Skewness: {skewness:.2f}"
        )
        plt.xlabel("Sales Price")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title("Distribution of Sales Price")
        plt.grid(True)
        plt.savefig(f"{SAVEPATH}/distribution_sales_price.png")

        target_variable = "Sales Price"
        # remove the target variable from the list of features
        features = [col for col in df.columns if col != target_variable]
        # i want to plot 8 features per plot, otherwise it is too crowded or too many plots
        features_per_plot = 8
        num_features = len(features)
        # calculate the total number of plots required
        num_plots = math.ceil(num_features / features_per_plot)

        for i in range(num_plots):
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
            axes = axes.flatten()

            for j in range(features_per_plot):
                feature_index = i * features_per_plot + j
                if feature_index >= num_features:
                    # stop if there are no more features
                    break

                feature = features[feature_index]

                # is feature is categorical or numerical ?
                if df[feature].dtype == "object":
                    sns.boxplot(x=df[feature], y=df[target_variable], ax=axes[j])
                    # rotate for readability
                    axes[j].set_xticklabels(
                        axes[j].get_xticklabels(), rotation=45, ha="right"
                    )
                else:
                    sns.scatterplot(x=df[feature], y=df[target_variable], ax=axes[j])

                axes[j].set_title(f"{target_variable} vs. {feature}")
                axes[j].set_xlabel(feature)
                axes[j].set_ylabel(target_variable)

            plt.tight_layout()
            plt.savefig(f"{SAVEPATH}/sales_price_vs_features_{i+1}.png")
            plt.close()

        # correlation matrix of numerical features
        # select numerical features
        numerical_data = df.select_dtypes(include=["number"])
        correlation_matrix = numerical_data.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix of Numerical Features")
        plt.savefig(f"{SAVEPATH}/correlation_matrix.png")
        plt.close()

        # plots of categorical features with their counts
        categorical_features = df.select_dtypes(include=["object"]).columns
        num_features = len(categorical_features)

        # 8 features per plot
        features_per_plot = 8

        # calculate the total number of plots required
        num_plots = math.ceil(num_features / features_per_plot)

        for i in range(num_plots):
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 10))
            axes = axes.flatten()

            for j in range(features_per_plot):
                feature_index = i * features_per_plot + j
                if feature_index >= num_features:
                    break

                feature = categorical_features[feature_index]
                sns.countplot(x=feature, data=df, ax=axes[j])
                axes[j].set_xticklabels(
                    axes[j].get_xticklabels(), rotation=45, ha="right"
                )
                axes[j].set_title(f"Count of {feature}")

            plt.tight_layout()
            plt.savefig(f"{SAVEPATH}/count_categorical_features_{i+1}.png")
            plt.close()
