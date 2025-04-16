import pandas as pd
import numpy as np
API_KEY = "7357784257616c7338316a654e7568"
import requests

# 장소 정보 들어있는 엑셀 파일
place = pd.read_excel('./data/hot-place.xlsx')

# 장소 코드만 떼오기
area_cd = place['AREA_CD'].unique()

# 데이터 불러오는 URL 생성
urls = np.array([f'http://openapi.seoul.go.kr:8088/{API_KEY}/json/citydata_ppltn/1/999/{name}' for name in area_cd])

# 한개만 불러와보기
response = [requests.get(urls[0]).json()]

# 전체 결과 불러오기
responses = [requests.get(url).json() for url in urls]

# 데이터프레임으로 변환하자
night_data = []

for r in responses:
    try:
        area_info = r['SeoulRtd.citydata_ppltn'][0]
        area_name = area_info['AREA_NM']
        forecast_list = area_info['FCST_PPLTN']
        
        for fcst in forecast_list:
            hour = int(fcst['FCST_TIME'][-5:-3])  # '2025-04-14 22:00' → 22
            if hour >= 18 or hour <= 5:  # 밤 9시~새벽 5시
                night_data.append({
                    '지역': area_name,
                    '예측시간': fcst['FCST_TIME'],
                    '혼잡도': fcst['FCST_CONGEST_LVL'],
                    '예측최소인구': int(fcst['FCST_PPLTN_MIN']),
                    '예측최대인구': int(fcst['FCST_PPLTN_MAX']),
                })
    except Exception as e:
        print("에러 발생:", e)
        continue

df_night = pd.DataFrame(night_data)


# 시간 컬럼 추가
df_night.astype({'예측시간': 'datetime64[ns]'})
df_night['시간'] = pd.to_datetime(df_night['예측시간']).dt.hour


# 시간대 컬럼 추가
def 분류_시간대(hour):
    if 18 <= hour < 21:
        return '저녁'
    elif 21 <= hour <= 23:
        return '밤'
    return '새벽'


df_night['시간대'] = df_night['시간'].apply(분류_시간대)
df_night['지역'].unique().size



############################################################
# 클러스터링 (?) 할수있을까
# 클러스터링 해보기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


crime_df = pd.read_csv('./data/crime_rate.csv', encoding="cp949", sep='\t')
adult_df = pd.read_csv('./data/머지한유흥업소데이터.csv', encoding='utf-8')
merged_df = pd.merge(crime_df, adult_df, on='자치구', how='inner')
print(merged_df.head())


cluster_features = ['구별 경찰수', '유흥업소_개수', '총생활인구수']

# 전처리 + 스케일링
X = merged_df[cluster_features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 클러스터링 (KMeans)
kmeans = KMeans(n_clusters=2, random_state=42)
merged_df['클러스터'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=merged_df, x='범죄율', y='유흥업소_개수', hue='클러스터', style='자치구')
plt.title('자치구 치안 클러스터링')
plt.show()


merged_df.groupby('클러스터')['총범죄건수'].mean()
merged_df.groupby('클러스터')['총생활인구수'].mean()
first = merged_df.loc[merged_df['클러스터'] == 0, :]
second = merged_df.loc[merged_df['클러스터'] == 1, :]
first['자치구'].unique()
second['자치구'].unique()


# 정규성 검정: 시각화 (Q-Q Plot)
import statsmodels.api as sm
# 클러스터 0
sm.qqplot(first['총범죄건수'], line='s')
plt.title('Q-Q Plot: 그룹 1')
plt.grid(True)
plt.show()

# 클러스터 1
sm.qqplot(second['총범죄건수'], line='s')
plt.title('Q-Q Plot: 그룹 2')
plt.grid(True)
plt.show()


# 정규성 검정 (Shapiro-Wilk Test)
from scipy.stats import shapiro
stat0, p0 = shapiro(first['총범죄건수'])
stat1, p1 = shapiro(second['총범죄건수'])
print(f'클러스터 0 정규성 p값: {p0:.4f}')
print(f'클러스터 1 정규성 p값: {p1:.4f}')



# 비모수검정
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


# 인구수 평균차이검정
# 정규성 검정: 시각화 (Q-Q Plot)
import statsmodels.api as sm
# 클러스터 0
sm.qqplot(first['총생활인구수'], line='s')
plt.title('Q-Q Plot: 그룹 1')
plt.grid(True)
plt.show()

# 클러스터 1
sm.qqplot(second['총생활인구수'], line='s')
plt.title('Q-Q Plot: 그룹 2')
plt.grid(True)
plt.show()


# 정규성 검정 (Shapiro-Wilk Test)
from scipy.stats import shapiro
stat0, p0 = shapiro(first['총생활인구수'])
stat1, p1 = shapiro(second['총생활인구수'])
print(f'클러스터 0 정규성 p값: {p0:.4f}')
print(f'클러스터 1 정규성 p값: {p1:.4f}')


# 비모수검정
from scipy.stats import mannwhitneyu

u_stat, p_val = mannwhitneyu(first['총생활인구수'], second['총생활인구수'], alternative='two-sided')
print(f'Mann-Whitney U 검정 통계량: {u_stat:.4f}')
print(f'p-value: {p_val:.4f}')
# 0.05보다 작으므로 귀무가설 기각
# 두 집단은 다르다.


# 클러스터링 결과를 시각화
plt.figure(figsize=(7, 5))

# boxplot 그리기
sns.boxplot(data=merged_df, x='클러스터', y='총생활인구수', palette='pastel')

# 그래프 설정
plt.title('클러스터별 총생활인구수 분포 (Boxplot)')
plt.xticks([0, 1], ['그룹 1', '그룹 2'])
plt.ylabel('총생활인구수')
plt.grid(True)
plt.tight_layout()

# 출력
plt.show()