#필요한 라이브러리 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

our_df = pd.read_csv('./data/our_df.csv')

our_df = our_df.drop(columns=['Unnamed: 0','GarageArea'])
y = our_df['SalePrice']

numeric = our_df.select_dtypes(include='number')
object = our_df.select_dtypes(include='object')

#범주형 변수에 관해서 boxplot 
for i in object.columns:
    sns.boxplot(x=our_df[i], y=y, data=our_df)
    plt.show()


# 수치형 변수 히스토그램
for col in numeric.columns:
    plt.figure()
    our_df[col].hist(bins=30)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()