import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt


crim = pd.read_csv('./data/seoul_crime_rate_20231231.csv', encoding='cp949')


crime_rate_data = pd.read_csv('./data/crime_rate.csv', encoding='cp949')


hot_place = pd.read_excel('./data/hot-place.xlsx')
one_housed = pd.read_excel('./data/seoul_one_person_housed_updated.xlsx')
SeoulSafetyCenter = pd.read_excel('./data/Seoul_SafetyCener_info.xlsx')



cctv = pd.read_csv('./data/Seoul_CCTV_info.csv', encoding='cp949')
cctv_by_gu = cctv.groupby('자치구')['CCTV 수량'].sum().reset_index()
cctv_by_gu.columns = ['자치구', 'CCTV 총수량']
# 결과 출력
print(cctv_by_gu)



# [data/hot-place.xlsx]: 유동 인구 (서울시 주요 장소별)
# [data/seoul_one _person_housed_updated.xlsx]: 서울시 1인 가구 수 (구별)
# [data/Seoul_SafetyCener_info.xlsx]: 서울시 치안 센터 수 (구별)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
crim.columns

crime_rate_data.columns
crime_rate_data.info()
crime_rate_data.describe()
crime_rate_data.head()



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

#  기존 범죄율 데이터 

#  병합
merged_df = split_data.merge(one_housed_clean, on='자치구', how='left')
merged_df = merged_df.merge(station_counts, on='자치구', how='left')
merged_df = merged_df.merge(cctv_by_gu, on='자치구', how='left')


# 결측치 확인 후 처리 (예: 없는 경우 0으로 대체)
merged_df.fillna(0, inplace=True)

# 독립 변수(X)와 종속 변수(y) 지정
X = merged_df[['총생활인구수(내)', '총생활인구수(외)', '총범죄건수', '1인가구수', '파출소수','CCTV 총수량']]
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
