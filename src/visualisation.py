import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

sns.set_theme(style='whitegrid')


def ensure_parent_directory(path:str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent,exist_ok = True)


def load_data(path="../Data/processed/heart_disease_clean.csv"):
    '''
    Loads the cleaned data from the data directory
    Input - path of the dataframe.
    '''

    if not os.path.exists(path):
        raise FileNotFoundError(f"File does not exist {path}")
    df = pd.read_csv(path)
    print(f"Dataset Loaded, shape : {df.shape}")
    return df


def describe_statistics(dataframe: pd.DataFrame):
    '''
    Input - A dataframe
    Ouptut - a dictionary containing analysis stats
    Gives a description of given dataframe
    '''
    numerical_cols = dataframe.select_dtypes(include=[np.number]).columns
    stats = dataframe[numerical_cols].describe().T
    stats["median"] = dataframe[numerical_cols].median()
    stats["IQR"] = dataframe[numerical_cols].quantile(0.75) - dataframe[numerical_cols].quantile(0.25)

    print("\nDescriptive Statistics:\n", stats.round(4))
    return stats


def correlation_analysis(df:pd.DataFrame,save_path = "../reports/figures/correlation_heatmap.png"):
    '''
    correlation heatmap for only numeric features.
    Input - a dataframe which we want to analyze correlation on and the path we want to save the figure to.
    Output - correlation matrix, and a plot saved to the given path.
    '''
    num_cols = df.select_dtypes(include=[np.number])
    correlation = num_cols.corr()

    plt.figure(figsize = (16,10))
    sns.heatmap(correlation,annot=True,fmt='.2f',center = 0,square = True)
    plt.title('correlation heatmap on numeric features')
    plt.tight_layout()

    ensure_parent_directory(save_path)
    plt.savefig(save_path,dpi=300,bbox_inches = 'tight')
    plt.close()
    print(f"Correlation heatmap saved at {save_path}")
    return correlation


def iqr_outlier_analysis(df:pd.DataFrame,columns = ["resting_blood_pressure", "serum_cholesterol", "max_heart_rate", "st_depression"]):
    rows = []
    for col in columns:
        s = df[col]
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        low,high = q1 - 1.5 * iqr,q3 + 1.5 * iqr
        amount = int(((df[col] < low) | (df[col] > high)).sum())
        rows.append([col,amount,round(amount / len(df) * 100,2),round(low,2),round(high,2)])

    output = pd.DataFrame(rows,columns = ['feature',"outliers","percentage of outliers","lower bound","upper bound"])
    print("\nOutlier report \n")
    print(output.sort_values("outliers",ascending=False))
    return output



def distribution_plots(df : pd.DataFrame,save_path ="../reports/figures/distributions.png"):
    '''
    Disribution plots: histogram and kde with mean and median
    Input : dataframe
    output : plots
    '''

    features =["age", "resting_blood_pressure", "serum_cholesterol", "max_heart_rate", "st_depression","bp_cholesterol_index"]
    fig,axes = plt.subplots(2,3,figsize = (16,9))
    axes = axes.ravel()

    for i,feature in enumerate(features):
        sns.histplot(df[feature],kde=True,ax = axes[i],bins=27)
        axes[i].set_title(f'Distribution of {feature}')

        mean_val = df[feature].mean()
        median_val = df[feature].median()
        axes[i].axvline(mean_val,linestyle='--',label=f"Mean: {mean_val:.2f}")
        axes[i].axvline(median_val,linestyle='--',label=f"Median: {median_val:.2f}")
        axes[i].legend()

    plt.suptitle("Key health indicators distributions",fontsize = 16)
    plt.tight_layout()

    ensure_parent_directory(save_path)


    plt.savefig(save_path,dpi=300,bbox_inches='tight')
    plt.close()
    print(f"Distribution plots saved to {save_path}")


def categorical_analysis(df: pd.DataFrame,save_path="../reports/figures/categorical_analysis.png"):
    fig,axes = plt.subplots(2,2,figsize=(16,10))

    sns.countplot(data=df,x='sex_male',hue='target',ax=axes[0,0])
    axes[0,0].set_title("sex (male = 1) by target")

    sns.countplot(data = df,x='chest_pain_type',hue='target',ax=axes[0,1])
    axes[0,1].set_title("Chest pain type by target")

    sns.countplot(data=df,x='exercise_induced_angina',hue='target',ax = axes[1,0])
    axes[1,0].set_title("Exercise induced angina")

    sns.countplot(data = df,x='metabolic_risk',hue='target',ax=axes[1,1])
    axes[1,1].set_title('Metabolic risk')

    plt.tight_layout()

    ensure_parent_directory(save_path)

    plt.savefig(save_path,dpi=300,bbox_inches = 'tight')
    plt.close()
    print(f"Categorical analysis plots saved at {save_path}")


def box_plots_analysis(df: pd.DataFrame,save_path = "../reports/figures/boxplots.png"):
    """
    Box plots for outlier inspection on key features
    """
    features = ["resting_blood_pressure", "serum_cholesterol", "max_heart_rate", "st_depression",
                "age_adjusted_heart_rate", "bp_cholesterol_index"]

    fig,axes = plt.subplots(2,3,figsize=(16,8))
    axes = axes.ravel()

    for i,feature in enumerate(features):
        sns.boxplot(y=df[feature],ax=axes[i])
        axes[i].set_title(feature)
        axes[i].set_ylabel("")

    plt.suptitle("Box Plots on numeric features")
    plt.tight_layout()
    ensure_parent_directory(save_path)

    plt.savefig(save_path,dpi = 300,bbox_inches='tight')
    plt.close()
    print(f"Box plots saved at {save_path}")


def scatter_plot(df:pd.DataFrame,save_path = "../reports/figures/scatter_plots.png"):
    fig,axes = plt.subplots(2,2,figsize = (16,10))

    sns.regplot(data = df,x = 'resting_blood_pressure',y = 'serum_cholesterol',
                scatter_kws ={"alpha":0.4,"s":15},ax = axes[0,0])
    axes[0,0].set_title("bp vs cholesterol")

    sns.regplot(data=df, x='age', y='max_heart_rate',
                scatter_kws={"alpha": 0.4, "s": 15}, ax=axes[0, 1])
    axes[0, 1].set_title("age vs max heart rate")

    sns.regplot(data=df, x='st_depression', y='max_heart_rate',
                scatter_kws={"alpha": 0.4, "s": 15}, ax=axes[1, 0])
    axes[1, 0].set_title("st depression vs max heart rate")

    sns.regplot(data=df, x="bp_cholesterol_index", y="age_adjusted_heart_rate",
                scatter_kws={"alpha": 0.4, "s": 15}, ax=axes[1, 1])
    axes[1, 1].set_title("blood pressure + cholesterol index vs age adjusted heart rate")

    plt.tight_layout()

    ensure_parent_directory(save_path)

    plt.savefig(save_path,dpi = 300,bbox_inches ="tight")
    plt.close()
    print(f"Scatter plot saved at {save_path}")


def violin_plots(df:pd.DataFrame,save_path = '../reports/figures/violin_plots.png'):
    fig,axes = plt.subplots(1,2,figsize = (14,6))

    sns.violinplot(data = df,x = 'target',y = 'st_depression',ax=axes[0])
    axes[0].set_title('st depression by target')
    sns.violinplot(data = df,x = 'target',y = 'max_heart_rate',ax=axes[1])
    axes[1].set_title('Max heart rate by target')

    plt.tight_layout()
    ensure_parent_directory(save_path)
    plt.savefig(save_path,dpi = 300,bbox_inches = 'tight')
    plt.close()
    print(f"Violin plots saved at {save_path}")



def run_visualisation_pipeline(data_path = "../Data/processed/heart_disease_clean.csv"):
    os.makedirs("../reports/figures",exist_ok=True)

    df = load_data(data_path)
    #analysis
    describe_statistics(df)
    iqr_outlier_analysis(df)
    # visualisation
    correlation_analysis(df)
    categorical_analysis(df)
    distribution_plots(df)
    box_plots_analysis(df)
    violin_plots(df)
    scatter_plot(df)

    print("\nVisualisation done.")


if __name__ == "__main__":
    run_visualisation_pipeline()


