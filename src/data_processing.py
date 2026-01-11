import pandas as pd
import numpy as np

"""

i used this to get the dataset 

from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets
df = pd.concat([X, y], axis=1)
df.to_csv("heart_disease_ucl.csv",index=False)
"""

DF_CLEAN = pd.read_csv("../Data/raw/heart_disease_ucl.csv")

# firstly we need to rename the columns since their names are unintuitive
# i will also define some of this to make the dataset alot easier to understand.
rename_map = {
    "age": "age",
    "sex": "sex_male",  # 1 means male, 0 female
    "cp": "chest_pain_type",  # 1 = typical angina,2 = atypical angina,3 = non-anginal pain,4=asymptomatic
    "trestbps": "resting_blood_pressure",
    "chol": "serum_cholesterol",
    "fbs": "fasting_blood_sugar",  # this column contains 1 if fbs > 120 or 0 if not.
    "restecg": "resting_electrocardiogram_results",
    # 0 = normal,1 = having wave abnormality,2 = definite ventricular hypertrophy
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_angina",  # 1 = yes, 0 = no
    "oldpeak": "st_depression",
    "slope": "slope",
    "ca": "colored_major_vessels",  # number of major vessels (0-3) colored by fluoroscopy
    "thal": "thalamesia_status",  # 3 = normal,6 = fixed defect,7 = reversible defect
    "num": "target"  #
}


# HELPER FUNCTIONS

def quality_report(dataframe: pd.DataFrame, name: str):
    '''
    just a helper function to write a report for the dataframe to show the before and after results.
    Takes in a dataframe and its name as an input
    Outputs a dictionary of information about the dataframe.
    '''

    report = {}
    report['name'] = name
    report['shape'] = dataframe.shape
    report['dtypes'] = dataframe.dtypes.astype(str)
    report['missing_values'] = dataframe.isna().sum()
    report['duplicate_rows'] = dataframe.duplicated().sum()
    num_cols = dataframe.select_dtypes(include=[np.number]).columns
    return report


def print_report(report: dict):
    print("\n" + "=" * 30)
    print(report['name'])
    print('=' * 30)
    print(f"Shape: {report['shape']}")
    print(f"\nDtypes: {report['dtypes']}")
    print(f"\nMissing values :\n {report['missing_values']}")
    print(f"\nDuplicate rows :\n {report['duplicate_rows']}")


def iqr_outlier_bounds(series: pd.Series):
    '''
  Defines the outlier bounds for the series.
  Takes in a series, and computes the 25 and 75 percentiles(q1 and q3)
  Outputs the lower and upper bound of the outlier.
  '''

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


# The columns to be capped are the ones which are continious and not the ones like sex_male.So i define a list of
# columns to be capped.
capped_cols = ['resting_blood_pressure', 'serum_cholesterol', 'max_heart_rate', 'st_depression']

def detect_outliers(df: pd.DataFrame, columns=capped_cols):
    outlier_summary = {}
    # i only want to do this for numeric columns
    for col in columns:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        s = df[col]
        lower, upper = iqr_outlier_bounds(s.dropna())
        mask_higher = s > upper
        mask_lower = s < lower
        outliers = int((mask_lower | mask_higher).sum())

        if outliers > 0:
            df.loc[mask_lower, col] = np.nan
            df.loc[mask_higher, col] = np.nan

        outlier_summary[col] = {
            'lower': lower,
            'upper': upper,
            'outliers detected': int(outliers)
        }

    summary_dataframe = pd.DataFrame(outlier_summary).T
    return df, summary_dataframe


def apply_domain_rules(dataframe: pd.DataFrame):
    '''
    This function checks whether the value is even logically possible.
    Input: a pandas dataframe
    output: the same dataframe altered where the non-realistic values are changed to the nan value
    '''

    rules = {
        "resting_blood_pressure": (80, 250),
        "serum_cholesterol": (100, 600),
        "max_heart_rate": (60, 220),
        "st_depression": (0, 8),
    }
    for col, (lo, hi) in rules.items():
        bad = dataframe[col].notna() & ((dataframe[col] < lo) | (dataframe[col] > hi))
        dataframe.loc[bad, col] = np.nan

    return dataframe


# I define sets of numerical and categorical columns
numeric_cols = ["resting_blood_pressure", "serum_cholesterol", "max_heart_rate", "st_depression"]
categorical_cols = ["chest_pain_type", "fasting_blood_sugar", "resting_electrocardiogram_results",
                    "exercise_induced_angina", "slope", "colored_major_vessels", "thalamesia_status"]


def convert_numeric_columns(dataframe: pd.DataFrame):
    '''
    This function takes in a dataframe and converts the columns in numeric_cols defined above into numeric
    This is just a check to see whether everything is in order with the dtypes.
    '''
    for col in numeric_cols:
        dataframe[col] = pd.to_numeric(dataframe[col], errors='coerce')
    return dataframe


def handle_missing_values(dataframe: pd.DataFrame):
    '''
    This function is for handling the missing values the dataset has.
    For numerical columns the missing values are filled in with a median since its a typical case of the
    For categorical columns the missing values sare filled in with a mode since
    '''
    for col in numeric_cols:
        median = dataframe[col].median()
        dataframe[col] = dataframe[col].fillna(median)

    for col in categorical_cols:
        mode = dataframe[col].mode(dropna=True)[0]
        dataframe[col] = dataframe[col].fillna(mode)

    return dataframe


def create_derived_features(dataframe: pd.DataFrame):
    """
    Creates some derived features from the given dataframe which will help train the ml model more efficiently
    """
    dataframe['age_adjusted_heart_rate'] = dataframe['max_heart_rate'] / (220 - dataframe['age'])
    # With this feature i combine two major risk factors together, blood pressure and cholesterol
    dataframe['bp_cholesterol_index'] = (dataframe['resting_blood_pressure'] * dataframe['serum_cholesterol']) / 10000
    # This counts how many metabolic risk flags an individual has
    dataframe['metabolic_risk'] = (
            dataframe['fasting_blood_sugar'].astype(int) + (dataframe['serum_cholesterol'] > 240).astype(int) +
            (dataframe['resting_blood_pressure'] > 140).astype(int))

    return dataframe


# This function will log the changes between the first dirty dataframe and the dataframe after we clean it.
def log_changes(df_before: pd.DataFrame, df_after: pd.DataFrame):
    '''
    Takes in 2 dataframes, one initial dataframe we loaded and one which we produce after we clean it.
    outputs a dataframe of changes, the column changes, the missing values before and after and changes basically.
    '''
    rows = []
    for col in df_after.columns:
        if col not in df_before.columns:
            rows.append([col, "New column", " ", " "])
            continue

        missing_before = int(df_before[col].isna().sum())
        missing_after = int(df_after[col].isna().sum())

        b = df_before[col].astype('object')
        a = df_after[col].astype('object')
        equal = (b == a) | (a.isna() & b.isna())
        changed = int((~equal).sum())

        rows.append([col, missing_before, missing_after, changed])

    return pd.DataFrame(rows, columns=['column', 'missing_before', 'missing_after', 'changed'])


def cleaning_decisions():
    return {
        "column renaming":(
            "Renamed the columns since their names were unintuitive but preserved the original idea."
        ),
        "outlier handling":(
            "Outliers were handled using the IQR method.Basically every value which is smaller than the "
            "25 th percentile by the value of 1.5 * the IQR(difference of the Q3 and Q1) or bigger than the 75 th "
            "percentile with that same value is considered a statistical outlier and thus is handled explicitly."
        ),
        "missing value handling":(
            "Target column was handled the most strictly since it is the most important part of the machine learning model "
            "and we cant handle it in such a way to not alter the model "
            "greatly.The label should not be handled with filling it with something."
            "The numerical missing values were handled by replacing them with the median since this way the filled values"
            "maintain the statistical structure in my opinion."
            "The categorical missing values were filled with mode since statistically more likely for them to have the mode"
            "of the column as their value."
        ),
        "derived features": (
                'I derived 3 features from the given dataset.'
                'age_adjusted_heart_rate: measures individuals observed maximum heart rate in contrast with their '
                'age predicted max heart rate.The approximation is that predicted heart rate = 220 - age.This basically'
                'normalizes the heart rate by age and is much more suitable for a machine learning model.\n'
                "bp_cholesterol_index: combines the blood pressure and cholesterol together in one single indicator of "
                "a heart disease.These two are interacting with eachother in a human physiology, thus combining them"
                "together is much more reliable and right than just using each alone.\n"
                "metabolic_risk: This feature basically measures how many metabolic red flags a patient has.Which is "
                "done with clinically meaningful thresholds.It is a combination of Elevated blood sugar,"
                "high cholesterol and high blood pressure."

        )
    }


def data_processing_pipeline(input_path, output_path):
    df_raw_local = pd.read_csv(input_path)
    df = df_raw_local.copy(deep=True)

    df = df.rename(columns=rename_map)

    df = convert_numeric_columns(df)

    # Since in this dataset the target variable can have a value in the range of 0 to 4,0 meaning no heart disease
    # and 4 meaning the most severe heart disease, i map this to binary 1 and 0
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"])
    df['target'] = (df['target'] > 0).astype(int)

    df_before = df.copy()
    report_before = quality_report(df_before, "Before cleaning")

    # apply domain rules
    df = apply_domain_rules(df)

    # handle outliers
    df, outlier_summary = detect_outliers(df)

    # Then handle missing values, the pipeline works like this because i put nan values in the functions above nd
    # with the handle_missing_values i then fill them.
    df = handle_missing_values(df)

    df = create_derived_features(df)

    log_output = log_changes(df_before, df)
    decisions = cleaning_decisions()
    report_after = quality_report(df, "After cleaning and adding derived features")

    df.to_csv(output_path, index=False)
    output = {
        "report_before": report_before,
        "report_after": report_after,
        "outlier_summary": outlier_summary,
        "change_log": log_output,
        "decisions": decisions
    }

    return df, output


# Now we run our pipeline
if __name__ == "__main__":
    df_final, output = data_processing_pipeline(
        "../Data/raw/heart_disease_dirty.csv",
         "../Data/processed/heart_disease_clean.csv"

    )

    print_report(output["report_before"])
    print_report(output["report_after"])

    print("\nOutliers")
    print(output["outlier_summary"])

    print("\nChange logs")
    print(output["change_log"])

    print("\nDecisions")
    for k, v in output["decisions"].items():
        print(f"\n{k.upper()}\n {v}")


