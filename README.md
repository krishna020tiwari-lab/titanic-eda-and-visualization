# titanic-eda-and-visualization
An exploratory data analysis (EDA) and feature engineering project on the Titanic dataset using Pandas, Seaborn, and Matplotlib.
# Titanic Exploratory Data Analysis (EDA)

An exploratory data analysis of the Titanic dataset investigating key demographics and features that influenced passenger survival rates.

## Key Insights
* **Gender Disparity:** Women had a significantly higher survival rate than men.
* **Socioeconomic Class:** First-class passengers survived at a much higher rate compared to third-class passengers.
* **Feature Engineering:** Extracted `Title` from passenger names to accurately impute missing `Age` medians by group.

## Tech Stack
* Python
* Pandas & NumPy (Data Manipulation)
* Matplotlib & Seaborn (Visualization)

## Visualizations
![EDA Plots](link_or_screenshot_to_your_plot.png)

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/titanic-eda-exploration.git](https://github.com/your-username/titanic-eda-exploration.git)
## 🤖 Machine Learning Model

- **Algorithm:** Random Forest Classifier (`scikit-learn`)
- **Accuracy:** **83.80%**
- **Key Features Used:** `Pclass`, `Sex`, `Age`, `FamilySize`, `Fare`, `Embarked`, `Title`
- **Evaluation:** Evaluated on an 20% unseen test split using `train_test_split`.
