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
# 잔차(오차)가 정규분포를 따르는지 확인
# 잔차가 정규분포를 따라야 회귀계수의
# 신뢰구간, t-검정, F-검정 등의 통계적 추론이 정확

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

# Q-Q plot
# 잔차의 분포가 정규분포와 얼마나 유사한지 시각적으로 확인
sm.qqplot(residuals, line='45', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.show()

# 히스토그램
# 잔차 분포의 전체적인 모양 확인
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.show()

# Shapiro-Wilk Test
# Shapiro-Wilk Test: 정규성을 수치적으로 검정

from scipy.stats import shapiro
stat, p = shapiro(residuals)
print(f'Shapiro-Wilk Test: stat={stat:.4f}, p-value={p:.4f}')
# p > 0.05면 정규성 가정 만족


# 등분산성 검정 (Homoscedasticity)
# 잔차의 분산이 예측값과 무관하게 일정한지 확인
# Fitted vs Residuals Plot

plt.scatter(fitted_vals, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.show()


# Breusch-Pagan Test
# 통계적으로 등분산 여부 검정 (p < 0.05면 이분산성 있음)
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residuals, X)
bp_labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']

print(dict(zip(bp_labels, bp_test)))
# p-value > 0.05 → 등분산성 만족

# 다중공선성 확인 (VIF)
# 목적: 독립 변수 간 상관관계가 너무 높은지 확인
# 독립 변수끼리 강한 상관이 있으면 회귀계수가 불안정해지고 해석이 어려워짐
# 일반적으로 VIF > 10 이면 다중공선성 의심
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

vif_df = pd.DataFrame()
vif_df["feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_df)
# VIF < 10 → 다중공선성 문제 없음


#  Cook's Distance (이상치 영향 확인)
# 목적: 일부 데이터가 회귀 결과에 지나치게 큰 영향을 주는지 확인
# 영향력 높은 이상치는 전체 모델을 왜곡시킬 수 있음
# 일반적으로 0.5 이상이면 주의, 1 이상이면 영향력 큰 이상치로 간주
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




import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px


# 클러스터링에 사용할 변수 선택
features = ['총생활인구수(내)', '총생활인구수(외)', '1인가구수', '파출소수', 'CCTV 총수량']

# 선택한 변수로부터 데이터 추출
X = merged_df[features]

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 군집 개수 설정
k_values = [2, 3]

for k in k_values:
    # KMeans 모델 생성 및 학습
    kmeans = KMeans(n_clusters=k, random_state=42)
    merged_df[f'클러스터_{k}'] = kmeans.fit_predict(X_scaled)


def plot_cluster_map(df, cluster_col, title):
    # 군집 개수에 따라 색상 목록 정의
    num_clusters = df[cluster_col].nunique()
    color_map = {
        2: ['#636EFA', '#EF553B'],
        3: ['#636EFA', '#EF553B', '#00CC96']
    }
    colors = color_map.get(num_clusters, px.colors.qualitative.Plotly)

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson_data,
        locations='자치구',
        featureidkey='properties.SIG_KOR_NM',
        color=cluster_col,
        color_discrete_sequence=colors,
        hover_name='자치구',
        hover_data={
            '범죄율': True,
            '총범죄건수': True,
            '총생활인구수(내)': True,
            '1인가구수': True,
            '파출소수': True,
            'CCTV 총수량': True
        },
        mapbox_style='carto-positron',
        center={'lat': 37.5665, 'lon': 126.9780},
        zoom=10,
        opacity=0.7,
        title=title
    )
    fig.show()


# 군집 개수 2개 시각화
plot_cluster_map(merged_df, '클러스터_2', '서울시 자치구별 K-평균 클러스터링 (k=2)')

# 군집 개수 3개 시각화
plot_cluster_map(merged_df, '클러스터_3', '서울시 자치구별 K-평균 클러스터링 (k=3)')





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


df2 = pd.DataFrame(X_scaled, columns=x, index=df2.index)


#우리가 생각하기에 범죄율에 영향을 미친다고 생각하는 변수들
X3 = df2[['총 음식점 수','CCTV 수량','1인가구수','구별 경찰수','총생활인구수']]
y = df2['총범죄건수']
X3 = sm.add_constant(X3)
model3 = sm.OLS(y, X3).fit()
print(model3.summary())


#aic로 stepwise한거
X1 = df2[['총생활인구수', '구별 경찰수','안전벨 개수' ,'CCTV 수량', '총 음식점 수', '1인가구수','파출소수']]
y = df2['총범죄건수']

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


x1 = df2[['총생활인구수', '구별 경찰수', 'CCTV 수량', '총 음식점 수', '파출소수']]
x1 = sm.add_constant(x1)
model1 = sm.OLS(y, x1).fit()
print(model1.summary())



#adj-r2 기준 변수 선택
X2 = df2[['총생활인구수', '구별 경찰수','안전벨 개수' ,'CCTV 수량', '총 음식점 수', '1인가구수','파출소수']]
y = df2['총범죄건수']
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
          cv=0,
          verbose = 2)

sfs.fit(X2, y)


selected_indices_r2 = list(sfs.k_feature_idx_)
names_r2 = np.array(X2.columns)[:-1]

x2 = df2[['총생활인구수', '구별 경찰수', '안전벨 개수', 'CCTV 수량', '총 음식점 수', '1인가구수']]
x2 = sm.add_constant(x2)
model2 = sm.OLS(y, x2).fit()
print(model2.summary())



model1.aic
model2.aic
model3.aic

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


# -----------------------------------------------------

# 클러스터링에 사용할 변수 선택
features = ['구별 경찰수', '총 음식점 수', '총생활인구수',  '파출소수']

# 선택한 변수로부터 데이터 추출
X = df2[features]

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 군집 개수 설정
k_values = [2]

for k in k_values:
    # KMeans 모델 생성 및 학습
    kmeans = KMeans(n_clusters=k, random_state=42)
    df2[f'클러스터_{k}'] = kmeans.fit_predict(X_scaled)






def plot_cluster_map(df, cluster_col, title):
    # 복사본 생성 및 군집 번호를 보기 좋게 변경
    df_viz = df.copy()
    unique_clusters = sorted(df_viz[cluster_col].unique())
    
    # 군집 번호를 "군집 1", "군집 2" 형태로 매핑
    cluster_name_map = {cid: f"군집 {i+1}" for i, cid in enumerate(unique_clusters)}
    df_viz['군집명'] = df_viz[cluster_col].map(cluster_name_map)

    # 자연스러운 색상 조합 (ex: 연한 파랑/연한 주황/연한 초록)
    natural_colors = {
        2: ['#A6CEE3', '#FDBF6F'],      # 연파랑, 연주황
        3: ['#A6CEE3', '#FDBF6F', '#B2DF8A']  # + 연녹색
    }
    colors = natural_colors.get(len(unique_clusters), px.colors.qualitative.Set2)

    fig = px.choropleth_mapbox(
        df_viz,
        geojson=geojson_data,
        locations='자치구',
        featureidkey='properties.SIG_KOR_NM',
        color='군집명',  # 범례와 hover에 "군집 n" 으로 표시
        color_discrete_sequence=colors,
        hover_name='자치구',
        hover_data={
            '군집명': True,
            '구별 경찰수': True,
            '총 음식점 수': True,
            '총생활인구수': True,
            '파출소수': True,
        },
        mapbox_style='carto-positron',
        center={'lat': 37.5665, 'lon': 126.9780},
        zoom=10,
        opacity=0.7,
        title=title
    )
    
    # 색상 바 제거 (연속형 아님)
    fig.update_layout(coloraxis_showscale=False)
    fig.show()

# 군집 개수 2개 시각화
plot_cluster_map(df2, '클러스터_2', '서울시 자치구별 K-평균 클러스터링 (k=2)')

# ---------------------------------------

def show_top_correlations_by_cluster(df, cluster_col, features, top_n=5, top_gu_n=3, gu_sort_key='총생활인구수'):
    for idx, cluster_label in enumerate(sorted(df[cluster_col].unique()), start=1):
        print(f"\n🔹 군집 {idx} 상위 {top_n} 상관 변수쌍")

        # 해당 군집 데이터만 추출
        cluster_df = df[df[cluster_col] == cluster_label]
        cluster_data = cluster_df[features]

        # 상관관계 계산
        corr_matrix = cluster_data.corr().abs()

        # 자기 자신과의 상관 제거 후, 상위 top_n 추출
        corr_pairs = (
            corr_matrix.where(~np.eye(corr_matrix.shape[0], dtype=bool))
            .stack()
            .sort_values(ascending=False)
            .drop_duplicates()
        )

        top_corrs = corr_pairs.head(top_n)

        for (var1, var2), corr_val in top_corrs.items():
            print(f"  📌 {var1} & {var2} → 상관계수: {corr_val:.2f}")

        # 대표 자치구 출력 (기준 변수로 정렬)
        if gu_sort_key in cluster_df.columns:
            sorted_cluster_df = cluster_df.sort_values(by=gu_sort_key, ascending=False)
            gu_list = sorted_cluster_df['자치구'].head(top_gu_n).tolist()
        else:
            gu_list = cluster_df['자치구'].head(top_gu_n).tolist()  # 대체: 정렬 기준 없을 경우

        print(f"\n    🏙️ 대표 자치구 (기준: {gu_sort_key}, 상위 {top_gu_n}개): {', '.join(gu_list)}")



show_top_correlations_by_cluster(df2, '클러스터_2', features, top_n=5, top_gu_n=3)


# 군집 1은 총생활인구수가 다른 변수들과 강한 양의 상관을 가지며, 
# 인구가 많은 지역일수록 음식점도 많고, 경찰 수와 파출소 수도 많음 → 상업·유동 인구 중심형 지역

# 군집 2는 총생활인구수와 다른 변수들 간의 상관이 낮고, 
# 대신 구별 경찰수와 파출소수 간에만 높은 상관 → 주거형/공공안전 중심형 지역일 가능성

