
## Project Overview
This project is an end-to-end data science and machine learning pipeline for analyzing and predicting heart disease risk using the UCI Heart Disease dataset.

The goal is to demonstrate the complete data science workflow:
- data cleaning and preprocessing
- exploratory data analysis (EDA)
- feature engineering
- machine learning model training
- model evaluation and comparison

The project is implemented in Python with a focus on clarity, reproducibility, and learning.

---

## Objectives
- Clean and preprocess real-world medical data
- Identify key health indicators related to heart disease
- Visualize distributions, correlations, and relationships
- Train and compare multiple classification models
- Clearly communicate insights and results

---

## Dataset
- Source: UCI Machine Learning Repository — Heart Disease Dataset
- Records: Patient demographic and clinical measurements
- Target variable:
  - target = 1 → presence of heart disease
  - target = 0 → no heart disease

Raw data is kept separate from processed data to ensure reproducibility.

---

## Data Processing & Cleaning
The data cleaning pipeline includes:
- Renaming columns for readability
- Data type validation and conversion
- Domain-based rules to remove unrealistic values
- Outlier detection using the IQR method
- Missing value handling:
  - numerical features → median
  - categorical features → mode
- Target variable binarization
- Feature engineering:
  - age-adjusted heart rate
  - blood pressure–cholesterol index
  - metabolic risk score
- Data quality reports before and after cleaning
- Logged changes for transparency

The cleaned dataset is saved for downstream analysis and modeling.

---

## Exploratory Data Analysis (EDA)
EDA includes both statistical analysis and visual exploration.

### Visualizations
- Distribution plots (histograms + KDE)
- Box plots for outlier inspection
- Correlation heatmap
- Scatter plots with trend lines
- Count plots for categorical variables
- Violin plots by target class

### Statistical Analysis
- Descriptive statistics (mean, median, IQR)
- Correlation analysis for numeric features
- Outlier prevalence analysis

All figures are automatically saved to the reports/figures directory.

---

## Machine Learning Models
This is a binary classification problem.

### Models Implemented
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Pipeline
- Train/test split (80/20, stratified)
- Feature scaling for Logistic Regression
- Model training
- Evaluation using multiple metrics
- Model comparison

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The best-performing model is selected based on F1-score.

---

## Project Structure
heart-disease-risk-analysis/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── src/
│ ├── data_processing.py
│ ├── visualization.py
│ └── models.py
│
├── reports/
│ ├── figures/
│ └── model_plots/
│
├── README.md
├── requirements.txt
└── .gitignore

---

## How to Run
1. Install dependencies
pip install -r requirements.txt



2. Run data processing
python src/data_processing.py


3. Run EDA and visualizations
python src/visualization.py

4. Run machine learning pipeline
python src/models.py
---

## Key Takeaways
- Proper data preprocessing is critical for reliable ML results
- Feature engineering improves model performance
- Tree-based models capture non-linear relationships well
- Clinical variables such as heart rate, blood pressure, and metabolic risk are strong predictors

---

## Author
Dato Jincharadze  
Computer Science / Data Science Student  
Kutaisi International University

---

## Course
Data Science with Python — Final Project
