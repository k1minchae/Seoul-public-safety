import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)



#자치구, 범죄율, 총생활인구수(내,외), 총 범죄건수, 구별 경찰수
crime_rate_data = pd.read_csv('./data/crime_rate.csv', encoding='cp949')

#서울시 자치구별 1인가구수 정보
one_housed = pd.read_excel('./data/seoul_one_person_housed_updated.xlsx')
#서울시 자치구별 파출소 수
SeoulSafetyCenter = pd.read_excel('./data/Seoul_SafetyCener_info.xlsx')
#서울시 자치구별 안전벨 수
bell = pd.read_excel('./data/Seoul_Safetybell.xlsx', engine='openpyxl')
#서울시 자치구별 유흥업소 수
shop = pd.read_csv('./data/머지한유흥업소데이터.csv', encoding='utf-8')



import pandas as pd
raw_df = crime_rate_data.copy()
#  문자열 분리 (\t split)
split_data = raw_df.iloc[:, 0].str.split('\t', expand=True)
#  컬럼명
split_data.columns = ['자치구', '총범죄건수', '자치구코드', '총생활인구수(내)', '총생활인구수(외)', '총생활인구수', '범죄율', '구별 경찰수','빈칸']

#  숫자형 컬럼 float변환
cols_to_float = ['총범죄건수', '총생활인구수(내)', '총생활인구수(외)', '총생활인구수', '범죄율', '구별 경찰수']
for col in cols_to_float:
    split_data[col] = pd.to_numeric(split_data[col], errors='coerce')

split_data = split_data.drop(columns=['빈칸'])

split_data.head()

import pandas as pd
import statsmodels.api as sm


#  1인가구 데이터 전처리
one_housed_clean = one_housed.rename(columns={'서울시 1인가구수': '자치구', '계': '1인가구수'})
one_housed_clean = one_housed_clean[['자치구', '1인가구수']]

#  파출소 개수 세기
station_counts = SeoulSafetyCenter['자치구'].value_counts().reset_index()
station_counts.columns = ['자치구', '파출소수']


# cctv총 수량
cctv = pd.read_csv('./data/Seoul_CCTV_info.csv', encoding='cp949')
cctv_by_gu = cctv.groupby('자치구')['CCTV 수량'].sum().reset_index()
cctv_by_gu.columns = ['자치구', 'CCTV 총수량']
print(cctv_by_gu)

#세이프티 밸 수량
bell = bell.groupby('자치구')['번호'].count()

#유흥업소 수량
shop=shop.groupby('자치구')['총_개수'].sum()



#  기존 범죄율 데이터 

#  병합
merged_df = split_data.merge(one_housed_clean, on='자치구', how='left')
merged_df = merged_df.merge(station_counts, on='자치구', how='left')
merged_df = merged_df.merge(cctv_by_gu, on='자치구', how='left')
merged_df = merged_df.merge(bell, on='자치구', how='left')
merged_df = merged_df.merge(shop, on='자치구', how='left')



# 결측치 확인 후 처리 (예: 없는 경우 0으로 대체)
merged_df.fillna(0, inplace=True)


# 독립 변수(X)와 종속 변수(y) 지정
X = merged_df[['총생활인구수(내)', '총생활인구수(외)', '총범죄건수', '1인가구수', '파출소수','CCTV 총수량','번호','총_개수']]
y = merged_df['범죄율']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

print(model.summary())


import statsmodels.api as sm

# X에 유의미한 변수만 선택
X_sig = merged_df['CCTV 총수량']
y = merged_df['범죄율']
X_sig = sm.add_constant(X_sig)
model_sig = sm.OLS(y, X_sig).fit()
# 결과 요약 출력
print(model_sig.summary())


merged_df.columns
merged_df.head()
merged_df.info()


# 잔차와 예측값 구하기
fitted_vals = model.fittedvalues
residuals = model.resid

#  정규성 검정

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

# Q-Q plot
sm.qqplot(residuals, line='45', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.show()

# 히스토그램
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.show()

# Shapiro-Wilk Test

from scipy.stats import shapiro
stat, p = shapiro(residuals)
print(f'Shapiro-Wilk Test: stat={stat:.4f}, p-value={p:.4f}')
# p > 0.05면 정규성 가정 만족


# 등분산성 검정 (Homoscedasticity)

# Fitted vs Residuals Plot

plt.scatter(fitted_vals, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()


# Breusch-Pagan Test
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residuals, X)
bp_labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

print(dict(zip(bp_labels, bp_test)))
# p-value > 0.05 → 등분산성 만족

# 다중공선성 확인 (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_df = pd.DataFrame()
vif_df["feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_df)
# VIF < 10 → 다중공선성 문제 없음


#  Cook's Distance (이상치 영향 확인)
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(model)
cooks_d = influence.cooks_distance[0]


# 시각화
plt.figure(figsize=(10, 4))
plt.stem(cooks_d, markerfmt='ro', linefmt='b-', basefmt='k-')
plt.title("Cook's Distance")
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.show()
#0.5~1 이상이면 영향력 큰 관측치 (주의 필요)



print(shop.head(2))


# 시각화 시작

import plotly.express as px
import json
with open('./data/seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)




fig = px.choropleth_mapbox(
    merged_df,
    geojson=geojson_data,
    locations='자치구',
    featureidkey='properties.SIG_KOR_NM',
    color='총생활인구수(내)',                               
    color_continuous_scale='Reds',                # 범죄율은 연속값이라 연속 색상 사용
    hover_name='자치구',
    hover_data={'범죄율': True, '총범죄건수': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 범죄율 시각화'
)
fig.show()

merged_df['인구_범주'] = pd.qcut(merged_df['총생활인구수(내)'], q=3, labels=['낮음', '중간', '높음'])





#  사용자 정의 색상 목록
custom_colors = ['#636EFA', '#EF553B', '#00CC96']  # 파란색, 빨간색, 초록색

#  Choropleth Mapbox 시각화
fig = px.choropleth_mapbox(
    merged_df,
    geojson=geojson_data,
    locations='자치구',
    featureidkey='properties.SIG_KOR_NM',
    color='인구_범주',
    color_discrete_sequence=custom_colors,
    hover_name='자치구',
    hover_data={'범죄율': True, '총범죄건수': True, '총생활인구수(내)': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 총생활인구수(내) 범주 시각화'
)

fig.show()


