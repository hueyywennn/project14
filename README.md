# Zomato Restaurants Reviews in Bangalore
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

## Project Overview
This project aims to analyze and predict restaurant reviews in Bangalore using the Zomato dataset. Key insights are extracted to understand customer sentiments, ratings, and popular restaurants by leveraging SQL queries within SQLite. The analysis is conducted in Jupyter Notebook using Pandas and SQLite for database management.

## Dataset Description
**Dataset Source:** [Zomato Bangalore Dataset](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants/data)
| Feature                         | Description                                         |
|---------------------------------|-----------------------------------------------------|
| name                            | Name of the restaurant                              |
| online_order                    | Availability of online ordering (Yes/No)           |
| book_table                      | Availability of table booking (Yes/No)             |
| rate                            | Aggregate user rating                              |
| votes                           | Number of votes received                           |
| location                        | Location of the restaurant                         |
| rest_type                       | Type of restaurant                                 |
| dish_liked                      | Most liked dish at the restaurant                  |
| cuisines                        | Cuisines offered                                   |
| approx_cost(for two people)     | Estimated cost for two people                     |
| reviews_list                    | Customer reviews                                  |
| menu_item                       | Menu items available                              |
| listed_in(type)                 | Type of meal (e.g., Buffet, Cafes, Desserts, etc.) |
| listed_in(city)                 | City of the restaurant                            |

## Project Objectives
1. **Data Cleaning & Preprocessing**: Handling missing values, normalizing text data, and formatting numerical values.
2. **SQL Querying**: Extracting meaningful insights using SQL queries executed within SQLite.
3. **Exploratory Data Analysis (EDA)**: Understanding trends in ratings, customer reviews, and restaurant types.
4. **Prediction Model**: Using extracted insights to predict customer ratings and sentiments.

## Machine Learning Models Used
- **Linear Regression**: Predicts restaurant ratings based on numerical and categorical features.
- **Random Forest**: An ensemble learning method that improves prediction accuracy.
- **XGBoost**: A gradient boosting algorithm that enhances performance and efficiency.
- **Grid Search (Random Forest)**: Hyperparameter tuning to optimize the random forest model.
- **Randomized Search CV (XGBoost)**: Hyperparameter tuning to find the best model configuration for XGBoost.

## Technologies Used
- **Database**: SQLite (executed via Pandas in Jupyter Notebook)
- **Programming Language**: Python (Pandas, NumPy, SQLite3, Scikit-Learn)
- **Visualization**: Matplotlib, Seaborn

## Project Workflow
1. **Data Ingestion**: Load the Zomato dataset into an SQLite database.
2. **SQL Query Execution**: Perform SQL queries to extract restaurant trends, rating distributions, and customer preferences.
3. **Data Analysis**: Conduct statistical analysis and visualization.
4. **Prediction Model**: Build a machine learning model to predict restaurant ratings based on extracted features.
5. **Results Interpretation**: Derive business insights from the analysis.
