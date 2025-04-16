import pandas as pd 
import requests
import os
import numpy as np

# 서울 치안센터 수
safety_center = pd.read_excel('./data/Seoul_SafetyCener_info.xlsx')

safety_center['자치구'] = safety_center['지역경찰관서 신주소'].str.extract(r'\s([가-힣]+구)\s')

safety_center.to_excel('./data/Seoul_SafetyCener_info.xlsx', index=False)


# CCTV
cctv = pd.read_csv('./data/Seoul_CCTV_info.csv', encoding='cp949')

cctv.info()

cctv['자치구'].value_counts()


# 서울 안전벨 수
bell = pd.read_excel('./data/Seoul_Safetybell.xlsx')
bell['자치구'] = bell['관리기관명'].str.extract(r'([가-힣]+구)')
bell['자치구'].value_counts()
bell['자치구'] = bell['자치구'].str.replace('서울시성북구','성북구')
bell.to_excel('./data/Seoul_Safetybell.xlsx', index=False)


bell.info()         # 21091
bell.head()

bell['소재지도로명주소'].isna().sum()   # 2589
bell['소재지지번주소'].isna().sum()     # 5365


bell['구1'] = bell['소재지도로명주소'].str.extract(r'\s([가-힣]+구)\s')
bell['구1'].isna().sum()    # 4758



bell['구2'] = bell['소재지지번주소'].str.extract(r'\s([가-힣]+구)\s')
bell['구2'].isna().sum()       # 8392


bell['자치구'] = bell['구1'].fillna(bell['구2'])
bell.to_excel('./data/Seoul_Safetybell.xlsx', index=False)

# 시각화 시작

import plotly.express as px

import json
with open('./data/seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)


agg_df = (bell.groupby("자치구",
                         as_index=False)
                         .count())

agg_df = agg_df.rename(columns={"자치구": "SIG_KOR_NM"})
print(agg_df.head(2))


fig = px.choropleth_mapbox(
    agg_df,
    geojson=geojson_data,
    locations="SIG_KOR_NM",
    featureidkey="properties.SIG_KOR_NM",
    color="LCD합계",
    color_continuous_scale="Blues",
    mapbox_style="carto-positron",
    center={"lat": 37.5665, "lon": 126.9780},
    zoom=10,
    opacity=0.7,
    title="서울시 자치구별 LCD 거치대 수"
    )

fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()





















##################################3
### k-means clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


crime_df = pd.read_csv('./data/crime_rate.csv', encoding="cp949", sep='\t')
adult_df = pd.read_csv('./data/머지한유흥업소데이터.csv', encoding='utf-8')
cctv_df = pd.read_csv('./data/Seoul_CCTV_info.csv',encoding='cp949')
cctv_df = cctv_df['자치구'].value_counts().reset_index()
cctv_df = cctv_df.rename(columns={'count':'cctv'})

one_df = pd.read_excel('./data/seoul_one _person_housed_updated.xlsx')
one_df = one_df.rename(columns={'서울시 1인가구수':'자치구'})


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

plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_df, x='범죄율', y='유흥업소_개수', hue='클러스터', style='자치구')
plt.title('자치구 치안 클러스터링')
plt.show()


merged_df.groupby('클러스터')['총범죄건수'].mean()
merged_df.groupby('클러스터')['총생활인구수'].mean()
first = merged_df.loc[merged_df['클러스터'] == 0, :]
second = merged_df.loc[merged_df['클러스터'] == 1, :]
third = merged_df.loc[merged_df['클러스터'] == 2, :]


group1 = first['자치구'].unique()
group2 = second['자치구'].unique()
group3 = third['자치구'].unique()

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

# 정규성 검정 (Shapiro-Wilk Test)
from scipy.stats import shapiro
stat0, p0 = shapiro(first['총범죄건수'])
stat1, p1 = shapiro(second['총범죄건수'])
stat2, p2 = shapiro(third['총범죄건수'])
print(f'클러스터 0 정규성 p값: {p0:.4f}')
print(f'클러스터 1 정규성 p값: {p1:.4f}')
print(f'클러스터 2 정규성 p값: {p2:.4f}')



# 비모수 2검정
from scipy.stats import mannwhitneyu

u_stat, p_val = mannwhitneyu(first['총범죄건수'], second['총범죄건수'], alternative='two-sided')
print(f'Mann-Whitney U 검정 통계량: {u_stat:.4f}')
print(f'p-value: {p_val:.4f}')
# 0.05보다 작으므로 귀무가설 기각
# 두 집단은 다르다.


# 클러스터링 결과를 시각화
plt.figure(figsize=(7, 5))

# boxplot 그리기
sns.boxplot(data=merged_df, x='클러스터', y='총범죄건수', palette='pastel')

# 그래프 설정
plt.title('클러스터별 총범죄건수 분포 (Boxplot)')
plt.xticks([0, 1], ['그룹 1', '그룹 2'])
plt.ylabel('총범죄건수')
plt.text(1, 14500 , f'↑강남구', ha='center', va='bottom', fontsize=17, color='red')
plt.grid(True)
plt.tight_layout()

# 출력
plt.show()
second.loc[second['총범죄건수'] >= 16000, :]    # 강남구: 이상치





# 비모수 3검정
# ks test
import scipy.stats as stats


# Kruskal-Wallis 검정
# H0: 세 그룹간 범죄건수에 유의한 차이가 없다.
# HA: 세 그룹간 범죄건수에 유의한 차이가 있다.
stat, p = stats.kruskal(first['총범죄건수'], second['총범죄건수'], third['총범죄건수'])

print("H-statistic:", stat)
print("p-value:", p)        # p-value: 0.001 유의수준 5%하. 귀무가설 기각.

if p < 0.05:
    print("세 그룹 간에 범죄건수에 유의한 차이 있다.")
else:
    print("세 그룹 간에 범죄건수에 유의한 차이가 없다.")


# 사후 검정 수행
import scikit_posthocs as sp

posthoc = sp.posthoc_dunn(merged_df,
                          val_col='총범죄건수',
                          group_col='클러스터', 
                          p_adjust='bonferroni')

print("\nDunn's test 사후검정 결과:")
print(posthoc)

# Boxplot
plt.rc('font', family='Malgun Gothic')

plt.figure(figsize=(8, 6))
sns.boxplot(x='클러스터', y='총범죄건수',hue='클러스터', data=merged_df, palette="Set3")
plt.title('군집별 총 범죄건 수')
plt.xlabel('군집')
plt.grid()
plt.legend(title='클러스터')
plt.ylabel('총 범죄건 수')

























#### API try
#########################33

bell['관리기관명'].value_counts()
((bell['소재지도로명주소'].isna()) & (bell['소재지지번주소'].isna())).sum()
((bell['구1'].isna()) & (bell['구2'].isna())).sum()

missing = bell[(bell['구1'].isna()) & (bell['구2'].isna())]



rest_api_key = os.getenv("KAKAO_API_KEY")


#

test = missing['소재지지번주소'].isna().sum()
test = missing['소재지도로명주소'].isna().sum()

for i in test:
    #i = df['주소'][1]
    i = test[1]
    print(i)
    url = "https://dapi.kakao.com/v2/local/search/keyword.json?query={}".format(i)
    headers = {"Authorization": "KakaoAK " + rest_api_key}
    places = requests.get(url, headers=headers)
    # print(places.json()['documents'][0]['address'])
    print(places.json()['meta']['same_name']['selected_region'])
    print('---------------------')





