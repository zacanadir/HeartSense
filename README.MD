🫀 HeartSense: Uncertainty-Aware Heart Disease Prediction & Explanation
HeartSense is an interactive Streamlit app that combines predictive modeling, fairness diagnostics, and model interpretability to help users assess and understand the risk of heart disease.
🔍 Key Features
	•	Model Comparison: Evaluate and compare Logistic Regression, Random Forest, and a custom Neural Network using ROC and Precision-Recall curves.
	•	Fairness Analysis: Analyze model performance separately for men and women, and apply reweighting techniques to mitigate bias.
	•	Uncertainty-Aware Prediction: Make personalized predictions using Monte Carlo Dropout to visualize risk as a probability distribution, not just a static number.
	•	Consequential Feature Identification: Use gradient-based analysis to uncover which patient features most influence disease risk — with 95% confidence intervals for controllable, continuous variables.
🧠 Why It Matters
Rather than treating model predictions as fixed, HeartSense embraces uncertainty and interpretability to offer a more nuanced, actionable view of health risk — empowering users and practitioners to focus on what matters most for each individual case.