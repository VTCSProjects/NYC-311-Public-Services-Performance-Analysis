# NYC 311 Public Services Performance Analysis

## Overview

I created this project to investigate how efficiently New York City handles public service requests and whether resolution times differ across boroughs and complaint types. I analyzed 296,434 service requests across four sampled periods in 2025.

## Tools Used

Python, SQL, Pandas, NumPy, Scikit-learn, SciPy, Statsmodels, SQLite, Matplotlib, Plotly, and Streamlit.

## What I Did

- Collected, cleaned, and validated nearly 300,000 service requests.
- Investigated missing values, invalid resolution times, and unusual complaint patterns.
- Built a SQLite database to analyze borough performance and service delays.
- Conducted chi-square tests and logistic regression.
- Trained machine learning models to predict requests taking longer than seven days to resolve.
- Created an interactive dashboard and visualizations.

## Key Findings

- Median resolution time: 9.41 hours.
- Manhattan had the highest percentage of requests exceeding seven days at 19.0%, compared with 13.9% in the Bronx.
- Statistical testing identified significant differences in delay rates across boroughs.

## Machine Learning Results

| Metric | Logistic Regression |
|---|---:|
| Accuracy | 88.47% |
| Precision | 60.16% |
| Recall | 73.24% |
| F1 Score | 0.6606 |
| ROC-AUC | 0.9419 |

## Data

The data comes from NYC Open Data. Some large datasets are excluded from this repository due to GitHub's file size restrictions. Data collection and processing scripts are included.

## Author

Aris Brunovskis — Virginia Tech, B.S. Computer Science
