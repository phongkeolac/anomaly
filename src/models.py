import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def detect_outliers_iqr(df, column, multiplier=1.5):
    df_out = df.copy()
    Q1 = df_out[column].quantile(0.25)
    Q3 = df_out[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    df_out['is_outlier'] = ((df_out[column] < lower_bound) | (df_out[column] > upper_bound)).astype(int)

    df_out['reason'] = df_out.apply(
        lambda row: f"Giá {row[column]:.2f} rớt xuống dưới ngưỡng ({lower_bound:.2f})" if row[column] < lower_bound 
        else (f"Giá {row[column]:.2f} vượt lên trên ngưỡng ({upper_bound:.2f})" if row[column] > upper_bound else ""), 
        axis=1
    )
    return df_out

def detect_outliers_isolation_forest(df, column, contamination=0.05):
    df_out = df.copy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    df_out['is_outlier'] = iso.fit_predict(df_out[[column]])
    df_out['score'] = iso.decision_function(df_out[[column]]) # Điểm dị thường
    df_out['is_outlier'] = df_out['is_outlier'].apply(lambda x: 1 if x == -1 else 0)
    
    df_out['reason'] = df_out.apply(
        lambda row: f"Anomaly Score: {row['score']:.3f} (Ngưỡng âm)" if row['is_outlier'] == 1 else "",
        axis=1
    )
    return df_out

def detect_outliers_ocsvm(df, column, nu=0.05):
    df_out = df.copy()
    ocsvm = OneClassSVM(nu=nu)
    df_out['is_outlier'] = ocsvm.fit_predict(df_out[[column]])
    df_out['score'] = ocsvm.decision_function(df_out[[column]])
    df_out['is_outlier'] = df_out['is_outlier'].apply(lambda x: 1 if x == -1 else 0)

    df_out['reason'] = df_out.apply(
        lambda row: f"Distance Score: {row['score']:.3f} (Ngưỡng âm)" if row['is_outlier'] == 1 else "",
        axis=1
    )
    return df_out