# 🎗️ Breast Cancer Prediction with Logistic Regression

    - This project analyzes breast cancer data to predict whether a tumor is malignant or benign using **Logistic Regression**.  
    Key steps include **EDA**, **preprocessing**, **outlier removal**, **feature scaling**, **model training**, **evaluation**, and **ROC analysis**.

---

## 📂 Dataset

    - `data.csv` (Breast Cancer Wisconsin Dataset)
    - Key features: Mean radius, texture, perimeter, area, and others.
    - Target: `diagnosis` (`M` for malignant, `B` for benign).

---

## 📌 Steps

### 1️⃣ Load & Inspect Data

    - Load dataset with `pandas`.
    - View head, tail, shape, info, and summary stats.
    - Check for null values.
    - Drop duplicates.

```python
    - df = pd.read_csv("data.csv")
    - df.head(), df.tail(), df.info(), df.describe(), df.isnull().sum()

### 2️⃣ Encode Categorical Target

    - Encode diagnosis column to numeric (0 = Benign, 1 = Malignant).

     from sklearn.preprocessing import LabelEncoder
     le = LabelEncoder()
     df['diagnosis'] = le.fit_transform(df['diagnosis'])

###3️⃣ Correlation Analysis
    - Calculate and visualize the correlation matrix.
    - Heatmap to see relationships between features and target.

    sns.heatmap(df.corr(), annot=True)

###4️⃣ Pairplot Visualization
    - Visualize feature distributions by diagnosis using sns.pairplot.
    sns.pairplot(df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']], hue='diagnosis')

###5️⃣ Boxplots for Outlier Detection
    - Boxplots for selected numerical features.

    numerical_cols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    sns.boxplot(x=df[col])


###6️⃣ Outlier Removal
    - Use IQR method to filter rows.
    - Remove rows with outliers in numerical columns.

    for col in numerical_cols:
        Q1 = ...
        Q3 = ...
        IQR = ...

###7️⃣ Feature Scaling
    - Drop unnecessary columns (id, Unnamed: 32).
    - Apply StandardScaler to scale features.

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

###8️⃣ Train-Test Split
    - 80% training, 20% testing.
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

###9️⃣ Train Logistic Regression
    - Train LogisticRegression model.
    - Predict on test set.
    - Evaluate with accuracy, classification report, and confusion matrix.

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

###🔟 ROC-AUC Score & ROC Curve
    - Calculate ROC-AUC.
    - Plot ROC curve.

    from sklearn.metrics import roc_auc_score, roc_curve

###1️⃣1️⃣ Adjust Threshold, Precision & Recall
    - Predict with a custom threshold (0.3).
    - Show new precision & recall.

###1️⃣2️⃣ Sigmoid Function
    - Plot sigmoid function for logistic regression interpretation.

    def sigmoid(z): ...

### Key Results
    - Logistic Regression provides good accuracy for binary cancer prediction.

    - ROC curve & AUC score show strong separability.

    - Lowering the threshold increases recall (catching more positives) but may reduce precision.

    - Outlier removal and feature scaling improve model robustness.

### Libraries Used
    - pandas, numpy — Data handling

    - seaborn, matplotlib — Visualizations

    - scikit-learn — Preprocessing, splitting, scaling, model training, evaluation

###Outcome
    - Early and accurate prediction of breast cancer can help save lives by enabling faster medical intervention.
    - This workflow shows a full machine learning pipeline from raw data → cleaned data → visualized data → trained model → evaluated results → ROC & threshold tuning.