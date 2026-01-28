# Salary-Range-Prediction-Multi-Output-Regression
# Overview
In the competitive NYC public sector market, thousands of jobs are posted annually, but manual salary estimation often leads to inconsistencies and budgeting delays. This project tackles a complex Multi-Output Regression problem: simultaneously predicting the Salary Range Low and Salary Range High for job postings based on features like job description, agency, and civil service title.

By automating this process, organizations can ensure equitable compensation planning, reduce bias in salary offers, and streamline the recruitment cycle.

# Technologies Used
- Core Stack: Python, Pandas, NumPy
- Machine Learning: Scikit-Learn (MultiOutputRegressor), XGBoost, Random Forest, Gradient Boosting, Decision Trees
- Feature Engineering: Category Encoders (Target Encoding), Scikit-Learn Pipelines
- Visualization: Seaborn, Matplotlib
- Deployment Ready: Joblib for model persistence

# Features
- Multi-Output Prediction: Unlike standard regression that predicts one value, this model predicts two correlated targets (Salary-Min, Salary-Max) simultaneously using MultiOutputRegressor.
- High-Cardinality Handling: Implemented Target Encoding to effectively handle categorical features with thousands of unique values (e.g., "Civil Service Title" or "Agency") without creating sparse, unmanageable datasets.
- Log-Transformation Pipeline: Built a robust pre-processing pipeline that log-transforms salary targets to normalize skewed distributions, ensuring the model isn't biased by high-salary outliers.
- Production-Ready Artifacts: The final model and preprocessors are serialized (.pkl files), making the system ready for immediate API deployment.

# The Process
1. Data Understanding:
- Analyzed NYC Job Postings data, identifying high skewness in salary distributions.
- Challenge: The dataset contained mixed salary frequencies (Hourly vs. Annual).
- Solution: Standardized all salaries to an "Annual" basis for consistent modeling.

2. Feature Engineering:
- Extracted text-based features (e.g., length of Job Description).
- Used Target Encoding for the "Agency" and "Title" columns, which had too many categories for standard One-Hot Encoding.

3. Model Selection:
- Benchmarked 6 different algorithms.
- While complex models like Gradient Boosting scored high (99%), we found the Decision Tree offered the best balance of high accuracy (Mean R²: 88%) and interpretability, with extremely fast inference times.

4. Evaluation:
- Evaluated using Mean Absolute Error (MAE) and R² Score.
- Performed Residual Analysis to ensure the model wasn't systematically under-predicting salaries for specific agencies.

# What I Learned
- Multi-Target Strategy: I learned that predicting a range (Min and Max) is often better handled as a joint probability problem rather than two separate models, as the Min and Max are highly correlated.
- The Power of Pipelines: Building a scikit-learn Pipeline that handles everything from missing value imputation to encoding ensures no "data leakage" occurs during cross-validation.
- Business Trade-offs: Sometimes a slightly simpler model (Decision Tree) is preferable to a "Black Box" (XGBoost) if it makes it easier to explain to HR stakeholders why a salary range was predicted.

# Overall Growth
- This project moved me beyond "toy datasets" into real-world administrative data issues.
- Technical: Mastered MultiOutputRegressor and advanced categorical encoding techniques.
- Analytical: Learned to detect and fix Heteroscedasticity (where error variance changes) in financial data by using log-transformations.

# How can it be improved?
- NLP Integration: Using TF-IDF or BERT embeddings on the "Job Description" text column to capture semantic nuance (e.g., "Leadership" keywords should predict higher pay).

Interactive Web App: Building a Streamlit tool where an HR manager pastes a job description and gets an instant salary range recommendation.

Market Adjustment: Integrating external "Cost of Living" indices to adjust predictions for future years automatically.
