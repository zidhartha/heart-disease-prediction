import os

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib
from visualisation import ensure_parent_directory
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42


def load_data(path="../Data/processed/heart_disease_clean.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")
    df = pd.read_csv(path)
    print(f"File loaded sucessfully from {path}")
    return df


def select_features_and_targets(df: pd.DataFrame):
    """
    Feature selection to choose relevant columns
    """

    target_column = 'target'
    features = [
        "age",
        "sex_male",
        "chest_pain_type",
        "resting_blood_pressure",
        "serum_cholesterol",
        "fasting_blood_sugar",
        "resting_electrocardiogram_results",
        "max_heart_rate",
        "exercise_induced_angina",
        "st_depression",
        "slope",
        "colored_major_vessels",
        "thalamesia_status",
        "age_adjusted_heart_rate",
        "bp_cholesterol_index",
        "metabolic_risk",
    ]
    x = df[features].copy()
    y = df[target_column].astype(int).copy()
    return x, y, features


def split_and_scale(x: pd.DataFrame, y: pd.Series, test_size=0.2):
    '''
    Splits the data into test/train and scales features for the logistic regression
    The tree models do not need scaled data so we output the raw data as well to feed them those.

    Input: our x and y data from the select_features_and_targets function.
    Output: the scaled x,y and their tests and train sets,the raw datasets as well and the scaler.
    '''
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, x_train.values, x_test.values, y_train, y_test, scaler


def logistic_regression(x_train_scaled: np.ndarray, y_train: pd.Series):
    '''
    Trains the logistic regression model
    Input: scaled x train set and y train set.
    Output: The logistic regression model trained on the input.
    '''
    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    model.fit(x_train_scaled, y_train)
    return model


def decision_tree(x_train: np.ndarray, y_train: pd.Series):
    '''
    Trains the decision tree classifier model.
    Input: scaled x train set and y train set.
    Output: The decision tree clasifier model trained on the input.
    '''
    model = DecisionTreeClassifier(max_depth=7, min_samples_split=8, min_samples_leaf=3, random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    return model


def random_forest(x_train: np.ndarray, y_train: pd.Series):
    """
    Trains the decision tree classifier model.
    Input: scaled x train set and y train set.
    Output: random forest clasifier model trained on the input.
    """
    model = RandomForestClassifier(n_estimators=250, max_depth=7, min_samples_split=6, min_samples_leaf=2,
                                   random_state=RANDOM_STATE, n_jobs=1)
    model.fit(x_train, y_train)
    return model


def evaluate_models(model, x_test: np.ndarray, y_test: pd.Series, model_name: str):
    """
    gives a dictionary of models and their performances to see them side by side.
    """

    y_prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction, zero_division=0)
    recall = recall_score(y_test, y_prediction, zero_division=0)
    f1 = f1_score(y_test, y_prediction)

    print(f"\n{model_name} evaluation:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 score: {f1}")

    evaluation = {
        'model_name': model_name,
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_prediction': y_prediction
    }

    return evaluation


def confusion_matrix_plots(y_test: pd.Series, y_pred: np.ndarray, model_name: str, save_path='../reports/model_plots/confusion_matrix.png'):
    '''
    generate and plot confusion matrices
    '''

    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "count"}
    )
    plt.title(f'Confusion matrix: {model_name}')
    plt.xlabel('predicted value')
    plt.ylabel('Real value')
    plt.tight_layout()
    ensure_parent_directory(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved at {save_path}")


def plot_model_scores(metrics: list, save_path="../reports/model_plots/model_comparison.png"):
    df = pd.DataFrame(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model performance comparison')
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    titles = ['Accuracy', 'Precision', 'Recall', 'F1 score']

    for i, (m, t) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[i // 2, i % 2]
        bars = ax.bar(df['model_name'], df[m])
        ax.set_title(t)
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha='center', va='bottom')

    plt.tight_layout()
    ensure_parent_directory(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plots were saved at {save_path}")


def run_ml_pipeline(data_path="../Data/processed/heart_disease_clean.csv"):
    os.makedirs("../reports/model_plots", exist_ok=True)
    df = load_data(data_path)
    x, y, feature_names = select_features_and_targets(df)

    x_train_scaled, x_test_scaled, x_train, x_test, y_train, y_test, scaled = split_and_scale(x, y)
    lr = logistic_regression(x_train_scaled, y_train)
    dt = decision_tree(x_train, y_train)
    rf = random_forest(x_train, y_train)

    # evaluation of these
    evaluation_logistic = evaluate_models(lr, x_test_scaled, y_test, "Logistic regression")
    evaluation_tree = evaluate_models(dt, x_test, y_test, "Decision tree")
    evaluation_forest = evaluate_models(rf, x_test, y_test, "Random forest")

    all_metrics = [evaluation_logistic, evaluation_tree, evaluation_forest]
    # confusion matrices
    confusion_matrix_plots(y_test, evaluation_logistic['y_prediction'], 'Logistic regression',
                           '../reports/model_plots/logistic_regression_confusion_matrix.png')
    confusion_matrix_plots(y_test, evaluation_tree['y_prediction'], 'Decision tree',
                           '../reports/model_plots/Decision_tree_confusion_matrix.png')
    confusion_matrix_plots(y_test, evaluation_forest['y_prediction'], 'Random forest',
                           '../reports/model_plots/random_forest_confusion_matrix.png')

    # model comparison
    plot_model_scores(all_metrics, "../reports/model_plots/model_comparison.png")

    # summary
    summary = pd.DataFrame(all_metrics)[['model_name', 'accuracy', 'precision', 'recall', 'f1_score']]
    summary = summary.sort_values('f1_score', ascending=False)

    print("\n ================Summary========================")
    print(summary.to_string())

    csv_path = '../reports/model_plots/model_metrics_summary.csv'
    ensure_parent_directory(csv_path)
    summary.to_csv(csv_path, index=False)
    print("\nMetrics summary was saved at ../reports/model_plots/model_metrics_summary.csv")

    best_row = summary.iloc[0]
    best_name = best_row['model_name']
    print(f"Best model based on F1 score: {best_name}")
    print(f"Accuracy :{best_row['accuracy']:.4f}")
    print(f"Precision:{best_row['precision']:.4f}")
    print(f"Recall   :{best_row['recall']:.4f}")
    print(f"F1 Score : {best_row['f1_score']:.4f}")
    best_model = next(m['model'] for m in all_metrics if m['model_name'] == best_name)

    if hasattr(best_model,'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)
        print("\nTop 10 feature importances")
        print(importances.head(10))
    elif hasattr(best_model,'coef_'):
        coefs = pd.Series(best_model.coef_[0],index=feature_names).sort_values(ascending=False)
        print(coefs.head(10))
    # The if statement is for the tree based models and the else is for the logistic regression


if __name__ == "__main__":
    run_ml_pipeline()
