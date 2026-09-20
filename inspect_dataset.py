import pandas as pd

path = r'C:\Users\HP\Searches\Internship\P_4_WA_Fn-UseC_-Telco-Customer-Churn.csv'
df = pd.read_csv(path)
print('shape:', df.shape)
print('columns:', list(df.columns))
print('\nhead:\n', df.head(5).to_string(index=False))
print('\nDtypes:\n', df.dtypes.to_string())
print('\nMissing count:\n', df.isnull().sum().to_string())
print('\nTarget counts:\n', df['Churn'].value_counts(dropna=False).to_string())
