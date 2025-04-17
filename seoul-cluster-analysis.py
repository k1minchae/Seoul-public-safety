##################################
### k-means clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 데이터 불러오기
crime_df = pd.read_csv('./data/crime_rate.csv', encoding="cp949", sep='\t')
adult_df = pd.read_csv('./data/머지한유흥업소데이터.csv', encoding='utf-8')
cctv_df = pd.read_csv('./data/Seoul_CCTV_info.csv',encoding='cp949')
cctv_df = cctv_df['자치구'].value_counts().reset_index()
cctv_df = cctv_df.rename(columns={'count':'cctv'})

one_df = pd.read_excel('./data/seoul_one_person_housed_updated.xlsx')
one_df = one_df.rename(columns={'서울시 1인가구수':'자치구'})

# merged
merged_df = pd.merge(crime_df, adult_df, on='자치구', how='inner')
merged_df = pd.merge(merged_df, cctv_df, on='자치구', how='inner')
merged_df = pd.merge(merged_df, one_df, on='자치구', how='inner')
print(merged_df.head())


cluster_features = ['구별 경찰수', '유흥업소_개수', '총생활인구수', 'cctv', '계']

# 전처리 + 스케일링
X = merged_df[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 클러스터링 (KMeans)
kmeans = KMeans(n_clusters=3, random_state=42)
merged_df['클러스터'] = kmeans.fit_predict(X_scaled)


first = merged_df.loc[merged_df['클러스터'] == 0, :]
second = merged_df.loc[merged_df['클러스터'] == 1, :]
third = merged_df.loc[merged_df['클러스터'] == 2, :]

group1 = first['자치구'].unique()
print(group1)
# '강북구', '광진구', '구로구', '금천구', '노원구', '도봉구', 
# '동대문구', '동작구', '마포구', '서대문구', '성동구', '성북구'
# '양천구', '용산구', '종로구', '중구', '중랑구'

group2 = second['자치구'].unique()
# '강남구', '서초구'


group3 = third['자치구'].unique()
# '강동구', '강서구', '관악구', '송파구', '영등포구', '은평구'

merged_df.groupby('클러스터')['총범죄건수'].mean()

merged_df.groupby('클러스터')['총생활인구수'].mean()



def assign_group(gu):
    if gu in group1:
        return 0
    elif gu in group2:
        return 1
    elif gu in group3:
        return 2
    else:
        return np.nan  

merged_df['클러스터'] = merged_df['자치구'].apply(assign_group)
merged_df['클러스터'] = merged_df['클러스터'].astype(str)


# 정규성 검정: 시각화 (Q-Q Plot)
import statsmodels.api as sm
# 클러스터 0
sm.qqplot(first['총범죄건수'], line='s')
plt.title('Q-Q Plot: 그룹 0')
plt.grid(True)
plt.show()

# 클러스터 1
sm.qqplot(second['총범죄건수'], line='s')
plt.title('Q-Q Plot: 그룹 1')
plt.grid(True)
plt.show()

# 클러스터 2
sm.qqplot(third['총범죄건수'], line='s')
plt.title('Q-Q Plot: 그룹 2')
plt.grid(True)
plt.show()


# qqplot보고, 데이터 수가 적어서 비모수 검정을 진행했다.


# 비모수 3검정
# ks test
import scipy.stats as stats


# Kruskal-Wallis 검정
# H0: 세 그룹간 범죄건수에 유의한 차이가 없다.
# HA: 세 그룹간 범죄건수에 유의한 차이가 있다.
stat, p = stats.kruskal(first['총범죄건수'], second['총범죄건수'], third['총범죄건수'])

print("H-statistic:", stat)
print("p-value:", p)        
# p-value: 0.001 유의수준 5%하. 귀무가설 기각.
# HA: 세 그룹간 범죄건수에 유의한 차이가 있다.

if p < 0.05:
    print("세 그룹 간에 범죄건수에 유의한 차이 있다.")
else:
    print("세 그룹 간에 범죄건수에 유의한 차이가 없다.")


# Boxplot 시각화
plt.rc('font', family='Malgun Gothic')

plt.figure(figsize=(8, 6))
sns.boxplot(x='클러스터', y='총범죄건수',hue='클러스터', data=merged_df, palette="Set3")
plt.title('군집별 총 범죄건 수')
plt.xlabel('군집')
plt.grid()
plt.legend(title='클러스터')
plt.ylabel('총 범죄건 수')
# 군집별로 범죄건 수 차이 확인 


# 사후 검정 수행
import scikit_posthocs as sp

posthoc = sp.posthoc_dunn(merged_df,
                          val_col='총범죄건수',
                          group_col='클러스터', 
                          p_adjust='bonferroni')

print("\nDunn's test 사후검정 결과:")
print(posthoc)
# 
# 표본 수가 적어서 그룹 1과, 그룹 2가 같다고 나온 것 같다.



##### 군집별 시각화
import pandas as pd
import plotly.express as px
import json


# GeoJSON 파일 불러오기
with open('./data/seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)


custom_colors = px.colors.qualitative.Set3 


# Choropleth Mapbox 시각화
fig = px.choropleth_mapbox(
    merged_df,
    geojson=geojson_data,
    locations='자치구',                        # 지역 기준
    featureidkey='properties.SIG_KOR_NM',     # GeoJSON의 자치구 이름 키
    color='클러스터',                          # 클러스터별 색상 분리
    color_discrete_sequence=custom_colors,   # 색상 변경
    hover_name='자치구',
    hover_data={'총범죄건수': True, '클러스터': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 클러스터 및 총범죄건수 시각화'
)

# 지도 크기 및 여백 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()




# 정규성 검정 (Shapiro-Wilk Test)
from scipy.stats import shapiro
stat0, p0 = shapiro(first['총범죄건수'])
stat1, p1 = shapiro(second['총범죄건수'])
stat2, p2 = shapiro(third['총범죄건수'])
print(f'클러스터 0 정규성 p값: {p0:.4f}')
print(f'클러스터 1 정규성 p값: {p1:.4f}')
print(f'클러스터 2 정규성 p값: {p2:.4f}')


# 군집 3개로 설정했을 때는 검정 결과가 만족스럽지 않아.
# (강남구, 서초구만 한 클러스터로 분류 되었다.)
# 군집 2개로 설정하고 군집분석을 수행했다.



#####################
# 2그룹
# 클러스터링 (KMeans)

# merged
merged_df2 = pd.merge(crime_df, adult_df, on='자치구', how='inner')
merged_df2 = pd.merge(merged_df2, cctv_df, on='자치구', how='inner')
merged_df2 = pd.merge(merged_df2, one_df, on='자치구', how='inner')
print(merged_df2.head())


cluster_features = ['구별 경찰수', '유흥업소_개수', '총생활인구수', 'cctv', '계']

# 전처리 + 스케일링
X = merged_df2[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans2 = KMeans(n_clusters=2, random_state=42)
merged_df2['클러스터'] = kmeans2.fit_predict(X_scaled)

# 각 군집에는 어떤 구가 있나?
first_2 = merged_df2.loc[merged_df2['클러스터'] == 0, :]
second_2 = merged_df2.loc[merged_df2['클러스터'] == 1, :]

# 각 군집별 구
group1_2 = first_2['자치구'].unique()
print(group1_2)
# ['강동구' '강북구' '광진구' '구로구' '금천구' '노원구'
#  '도봉구' '동대문구' '동작구' '마포구' '서대문구' '성동구'
#  '성북구' '양천구' '용산구' '종로구' '중구' '중랑구']
group2_2 = second_2['자치구'].unique()
print(group2_2)
# ['강남구' '강서구' '관악구' '서초구' '송파구' '영등포구'
#  '은평구']


# 조금 보기 불편
plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_df, x='범죄율', y='유흥업소_개수', hue='클러스터', style='자치구')
plt.title('자치구 치안 클러스터링')
plt.show()


# 군집별 총범죄건수 평균 확인
merged_df2.groupby('클러스터')['총범죄건수'].mean()

# 군집별 총생활인구수 평균 확인
merged_df2.groupby('클러스터')['총생활인구수'].mean()


def assign_group(gu):
    if gu in group1_2:
        return 0
    elif gu in group2_2:
        return 1
    else:
        return np.nan  

merged_df2['클러스터'] = merged_df2['자치구'].apply(assign_group)
merged_df2['클러스터'] = merged_df2['클러스터'].astype(str)


# 정규성 검정: 시각화 (Q-Q Plot)
import statsmodels.api as sm
# 클러스터 0
sm.qqplot(first_2['총범죄건수'], line='s')
plt.title('Q-Q Plot: 그룹 0')
plt.grid(True)
plt.show()

# 클러스터 1
sm.qqplot(second_2['총범죄건수'], line='s')
plt.title('Q-Q Plot: 그룹 1')
plt.grid(True)
plt.show()


# 정규성 검정 (Shapiro-Wilk Test)
from scipy.stats import shapiro
# 각집단의 정규성을 검정하기 위해 shapiro-wilk test 

# H0: 각 군집의 총범죄 건수가 정규분포를 따른다. 
# HA: 각 군집의 총범죄 건수가 정규분포를 따르지 않는다. 

stat0, p0 = shapiro(first_2['총범죄건수'])
stat1, p1 = shapiro(second_2['총범죄건수'])

print(f'클러스터 0 정규성 p값: {p0:.4f}')
print(f'클러스터 1 정규성 p값: {p1:.4f}')
# 클러스터 0 정규성 p값: 0.5609
# 클러스터 1 정규성 p값: 0.2681

# 유의수준 0.05하 귀무가설 기각할 수 없다.
# 두 집단 모두 정규성을 따른다. 


# qqplot, shapiro 검정 결과 정규성을 따른다고 나왔지만
# 표본 수가 작아서. 비모수 검정 실시.

# 비모수 2검정
from scipy.stats import mannwhitneyu
# H0: 두 집단의 총범죄 건수의 중앙값이 같다.
# HA: 두 집단의 총범죄 건수의 중앙값이 다르다.
u_stat, p_val = mannwhitneyu(first_2['총범죄건수'], second_2['총범죄건수'], alternative='two-sided')
print(f'Mann-Whitney U 검정 통계량: {u_stat:.4f}')
print(f'p-value: {p_val:.4f}')
# 0.05보다 작으므로 귀무가설 기각
# HA: 두 집단의 총범죄 건수의 중앙값이 다르다.

# 결과: x 변수들로 군집을 나눴다. 
# 그런데 범죄 건수에 차이가 있다. 
# ['구별 경찰수', '유흥업소_개수', '총생활인구수', 'cctv', '계']




# boxplot
# 클러스터링 결과를 시각화
plt.figure(figsize=(7, 5))

# boxplot 그리기
sns.boxplot(data=merged_df2, x='클러스터', y='총범죄건수', palette='pastel')

# 그래프 설정
plt.title('클러스터별 총 범죄건수 분포 (Boxplot)')
plt.xticks([0, 1], ['그룹 1', '그룹 2'])
plt.ylabel('총범죄건수')
plt.text(0, 14500 , f'↑강남구', ha='center', va='bottom', fontsize=17, color='red')
plt.grid(True)
plt.tight_layout()

# 출력
plt.show()


# 이상치 학인
second.loc[second['총범죄건수'] >= 16000, :]    # 강남구: 이상치



##### 군집별 지도 시각화 (2군집)
import pandas as pd
import plotly.express as px
import json

# GeoJSON 파일 불러오기
with open('./data/seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)


custom_colors = px.colors.qualitative.Set3 


# Choropleth Mapbox 시각화
fig = px.choropleth_mapbox(
    merged_df2,
    geojson=geojson_data,
    locations='자치구',                        # 지역 기준
    featureidkey='properties.SIG_KOR_NM',     # GeoJSON의 자치구 이름 키
    color='클러스터',                          # 클러스터별 색상 분리
    color_discrete_sequence=custom_colors,   # 색상 변경
    hover_name='자치구',
    hover_data={'총범죄건수': True, '클러스터': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 클러스터 및 총범죄건수 시각화'
)

# 지도 크기 및 여백 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()



############################## 군집화 끝
###########################################################