import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from ydata_profiling import ProfileReport

matplotlib.use('Agg')

class Data_utils():

    def __init__(self, path):
        self.path = path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.path)
        return self.df

    def clean_data(self):

        df = self.df.copy()

        df = df.drop_duplicates()

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No"})
        
        self.df = df

        return self.df

    def describe_data(self):
        print(self.df.describe(include='all'))

    def generate_correlation_plot(self, save_path: str = "data/correlation.png"):
        plt.figure(figsize=(10, 8))
        self.df['SeniorCitizen'] = pd.to_numeric(self.df['SeniorCitizen'], errors="coerce")
        numeric_df = self.df.select_dtypes(include="number")
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def export_csv(self, path):
        self.df.to_csv(path, index=False)

    def generate_eda_report(self, path):
        profile = ProfileReport(self.df, title="EDA Report", explorative=True)
        profile.to_file(path)