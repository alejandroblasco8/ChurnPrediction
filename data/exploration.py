from utils import Data_utils
from dotenv import load_dotenv
import os

load_dotenv()

CLEAN_PATH = os.getenv("CLEAN_PATH")
REPORT_PATH = os.getenv("REPORT_PATH")

def main():
    data = Data_utils()
    data.load_data()
    data.clean_data()
    data.describe_data()
    data.generate_correlation_plot()
    #data.export_csv()
    data.generate_eda_report(REPORT_PATH)


if __name__ == "__main__":
    main()
