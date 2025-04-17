############################################################
#필요한 라이브러리 부르기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False




food_and_entertain = pd.read_csv('./data/머지한유흥업소데이터.csv', encoding='utf-8')
seoul_safetybell = pd.read_excel('./data/Seoul_Safetybell.xlsx', engine='openpyxl')
cctv = pd.read_csv('./data/Seoul_CCTV_info.csv',encoding='cp949')
crime_rate = pd.read_csv('./data/crime_rate.csv',encoding='euc-kr',sep='\t')
one_housed = pd.read_excel('./data/seoul_one_person_housed_updated.xlsx')
SeoulSafetyCenter = pd.read_excel('./data/Seoul_SafetyCener_info.xlsx')

#안전벨 개수
seoul_safetybell_df = seoul_safetybell.groupby('자치구')['번호'].count()
#cctv 개수
cctv_df = cctv.groupby('자치구')['CCTV 수량'].sum()

#  1인가구 수
one_housed_clean = one_housed.rename(columns={'서울시 1인가구수': '자치구', '계': '1인가구수'})
one_housed_clean = one_housed_clean[['자치구', '1인가구수']]

#  파출소 개수
station_counts = SeoulSafetyCenter['자치구'].value_counts().reset_index()
station_counts.columns = ['자치구', '파출소수']


df = crime_rate.merge(seoul_safetybell_df, on='자치구') \
                       .merge(cctv_df, on='자치구') \
                       .merge(food_and_entertain, on='자치구') \
                       .merge(one_housed_clean, on='자치구') \
                       .merge(station_counts, on='자치구')
df = df.rename(columns={'번호': '안전벨 개수','총_개수': '총 음식점 수'})
df = df.drop(columns=['Unnamed: 8','일반음식점_개수','유흥업소_개수','자치구코드'])


########################################################################

df2 = df.copy()

scaler = StandardScaler()
x = ['총생활인구수','총범죄건수','구별 경찰수','안전벨 개수','CCTV 수량','총 음식점 수','1인가구수','파출소수']
X_scaled = scaler.fit_transform(df2[x])


X_scaled_df = pd.DataFrame(X_scaled, columns=x, index=df2.index)


#우리가 생각하기에 범죄율에 영향을 미친다고 생각하는 변수들
X3 = X_scaled_df[['총 음식점 수','CCTV 수량','1인가구수','구별 경찰수','총생활인구수']]
y = X_scaled_df['총범죄건수']
X3 = sm.add_constant(X3)
model3 = sm.OLS(y, X3).fit()
print(model3.summary())


import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. 총 음식점 수 vs 총범죄건수
sns.regplot(data=X_scaled_df, x='총 음식점 수', y='총범죄건수', ax=axes[0], line_kws={'color': 'red'})
for i, row in X_scaled_df.iterrows():
    if row['총범죄건수'] > 3:
        axes[0].text(
            row['총 음식점 수'], row['총범죄건수'],
            '강남구',  # 표시할 텍스트
            fontsize=12, color='red', ha='left', va='bottom'
        )
axes[0].grid(True)

# 2. 총생활인구수 vs 총범죄건수
sns.regplot(data=X_scaled_df, x='총생활인구수', y='총범죄건수', ax=axes[1], line_kws={'color': 'green'})
for i, row in X_scaled_df.iterrows():
    if row['총범죄건수'] > 3:
        axes[1].text(
            row['총생활인구수'], row['총범죄건수'],
            '강남구',  # 표시할 텍스트
            fontsize=12, color='red', ha='left', va='bottom'
        )
axes[1].grid(True)

plt.tight_layout()
plt.show()


#aic로 stepwise한거
X1 = X_scaled_df[['총생활인구수', '구별 경찰수','안전벨 개수' ,'CCTV 수량', '총 음식점 수', '1인가구수','파출소수']]
y = X_scaled_df['총범죄건수']

lr = LinearRegression()
names = X1.columns
def aic_score(estimator,X1, y):
    X1 = sm.add_constant(X1) 
    model = sm.OLS(y, X1).fit()
    print("Model AIC:", model.aic)
    return -model.aic

# Perform SFS
sfs = SFS(lr,
          k_features=(1,7),   
          forward=True,      
          scoring=aic_score,  
          cv=0,
          verbose = 0)
sfs.fit(X1, y)

print('Selected features:', np.array(names)[list(sfs.k_feature_idx_)])


x1 = X_scaled_df[['총생활인구수', '구별 경찰수', 'CCTV 수량', '총 음식점 수', '파출소수']]
x1 = sm.add_constant(x1)
model1 = sm.OLS(y, x1).fit()
print(model1.summary())



#adj-r2 기준 변수 선택
X2 = X_scaled_df[['총생활인구수', '구별 경찰수','안전벨 개수' ,'CCTV 수량', '총 음식점 수', '1인가구수','파출소수']]
y = X_scaled_df['총범죄건수']
# Adj R2 스코어 함수 정의
def adjusted_r2_score(estimator, X2, y):
    y_pred = estimator.predict(X2)
    n = X2.shape[0]
    p = X2.shape[1]
    r2 = r2_score(y, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2


sfs = SFS(lr,
          k_features=(1,7),
          forward=True,
          scoring=adjusted_r2_score,
          cv=0)

sfs.fit(X2, y)


selected_indices_r2 = list(sfs.k_feature_idx_)
names_r2 = np.array(X2.columns)[:-1]

x2 = X_scaled_df[['총생활인구수', '구별 경찰수', '안전벨 개수', 'CCTV 수량', '총 음식점 수', '1인가구수']]
x2 = sm.add_constant(x2)
model2 = sm.OLS(y, x2).fit()
print(model2.summary())



print(f"모델 1의 AIC: {model1.aic:.3f}, 모델 1의 R2: {model1.rsquared:.3f}")
print(f"모델 2의 AIC: {model2.aic:.3f}, 모델 2의 R2: {model2.rsquared:.3f}")
print(f"모델 3의 AIC: {model3.aic:.3f}, 모델 3의 R2: {model3.rsquared:.3f}")


#최종 선정 모델
print(model1.summary())

# 잔차정규성
residuals = model1.resid
fitted_values = model1.fittedvalues
plt.figure(figsize=(15,4))
plt.subplot(1,2,1)
plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color='black', linestyle='--')
plt.subplot(1,2,2)
stats.probplot(residuals, plot=plt)
plt.show()

resid_stats, resid_pvalue = stats.shapiro(residuals)
print(resid_pvalue)
#p_value가 0.07로 유의수준 5%하에서 귀무가설을 기각할 수 없다
#잔차는 정규성을 따른다고 할 수 있다.



#잔차 등분산성
bptest = het_breuschpagan(model1.resid, model1.model.exog)
print('BP-test statistics: ', bptest[0])
print('BP-test p_value: ', bptest[1]) #p-value가 0.3으로 귀무가설 기각하지 못함
#등분산성 만족한다고 볼 수 있음


#잔차 독립성
dw_stat = durbin_watson(model1.resid)
print(dw_stat)  
#2 정도로 잔차 독립성 만족한다고 볼 수 있음




