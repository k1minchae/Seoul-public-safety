import pandas as pd
import numpy as np
import plotly.express as px
import json
#################

with open('./data/seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)



# 인구 데이터
# 생활 인구수 지도 시각화
master = pd.read_excel('./data/sanggwan_df.xlsx')
master.columns

fig = px.choropleth_mapbox(
    master,
    geojson=geojson_data,
    locations='자치구',                        # 지역 이름
    featureidkey='properties.SIG_KOR_NM',     
    color='총생활인구수',                         # 시각화에 사용할 값
    color_continuous_scale='Blues',            # 색상 스케일
    hover_name='자치구',
    hover_data={'총생활인구수': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},  # 서울 중심
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 구별 총생활인구수 시각화'
)

# 레이아웃 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()

print(master.sort_values(ascending=False, by="총생활인구수").loc[:, ['자치구', '총생활인구수']].head(3))
print(master.sort_values(ascending=False, by="1인가구수").loc[:, ['자치구', '1인가구수']].head(3))
print(master.sort_values(ascending=False, by="1인가구수").head(3))['자치구']


# 1인 가구 수 지도 시각화
master = pd.read_excel('./data/sanggwan_df.xlsx')
master.columns

fig = px.choropleth_mapbox(
    master,
    geojson=geojson_data,
    locations='자치구',                        # 지역 이름
    featureidkey='properties.SIG_KOR_NM',     
    color='1인가구수',                         # 시각화에 사용할 값
    color_continuous_scale='Blues',            # 색상 스케일
    hover_name='자치구',
    hover_data={'1인가구수': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},  # 서울 중심
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 구별 1인가구수 시각화'
)

# 레이아웃 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()



# 정규성 검정 (Shapiro-Wilk Test)
from scipy.stats import shapiro
stat0, p0 = shapiro(master['총생활인구수'])
stat1, p1 = shapiro(master['1인가구수'])
print(f'클러스터 0 정규성 p값: {p0:.4f}')
print(f'클러스터 1 정규성 p값: {p1:.4f}')

# 클러스터 0 정규성 p값: 0.5738
# 클러스터 1 정규성 p값: 0.0046

# 비모수 검정 실시
# 
# 비모수 2검정
from scipy.stats import mannwhitneyu
# H0: 총 인구수와 1인 가구수의 중앙값이 같다.
# HA: 총 인구수와 1인 가구수의 중앙값이 다르다.

u_stat, p_val = mannwhitneyu(master['총생활인구수'], master['1인가구수'], alternative='two-sided')
print(f'Mann-Whitney U 검정 통계량: {u_stat:.4f}')
print(f'p-value: {p_val:.4f}')
# Mann-Whitney U 검정 통계량: 576.0000
# p-value: 0.0000
# 0.05보다 작으므로 귀무가설 기각
# HA: 총 인구수와 1인 가구수의 중앙값이 다르다.




# 치안 데이터
# CCTV 수 지도 시각화
cctv = pd.read_csv('./data/Seoul_CCTV_info.csv',encoding='cp949')


gu_counts = cctv['자치구'].value_counts().reset_index()
top_3 = cctv['자치구'].value_counts().sort_values(ascending=False).head(3)
gu_counts.columns = ['SIG_KOR_NM', '건수']

# 시각화
fig = px.choropleth_mapbox(
    gu_counts,
    geojson=geojson_data,
    locations="SIG_KOR_NM",
    featureidkey="properties.SIG_KOR_NM",
    color="건수",
    color_continuous_scale="Greens",
    mapbox_style="carto-positron",
    center={"lat": 37.5665, "lon": 126.9780},
    zoom=10,
    opacity=0.7,
    title="서울시 자치구별 CCTV 설치 건수"
)
fig.update_layout(
    margin={"r":0, "t":30, "l":0, "b":0},
    height=600,
    width=700
)

fig.show()

# 서울 안전벨 수 지도 시각화

bell = pd.read_excel('./data/Seoul_Safetybell.xlsx')

bell['자치구'].value_counts()


gu_counts = bell['자치구'].value_counts().reset_index()
gu_counts.columns = ['SIG_KOR_NM', '건수']

# 시각화
fig = px.choropleth_mapbox(
    gu_counts,
    geojson=geojson_data,
    locations="SIG_KOR_NM",
    featureidkey="properties.SIG_KOR_NM",
    color="건수",
    color_continuous_scale="Greens",
    mapbox_style="carto-positron",
    center={"lat": 37.5665, "lon": 126.9780},
    zoom=10,
    opacity=0.7,
    title="서울시 자치구별 안심벨 설치 건수"
)
fig.update_layout(
    margin={"r":0, "t":30, "l":0, "b":0},
    height=600,
    width=700
)

fig.show()


# CCTV, 안심벨 상관관계
master.select_dtypes('number').corr()['CCTV총수량']['안전벨 수']
# 0.303161
# 생각보다 강한 상관관게를 보이지 않는다. 


# 정규성 검정 (Shapiro-Wilk Test)
from scipy.stats import shapiro
stat0, p0 = shapiro(master['CCTV총수량'])
stat1, p1 = shapiro(master['안전벨 수'])
print(f'클러스터 0 정규성 p값: {p0:.4f}')
print(f'클러스터 1 정규성 p값: {p1:.4f}')

# 클러스터 0 정규성 p값: 0.1179
# 클러스터 1 정규성 p값: 0.1461
# 구별 CCTV 수는 정규성을 따른다. 
# 구별 안심벨 수는 정규성을 따른다. 

# 둘의 평균이 같은지 보기위해 
# 2표본 t검정 실시.

from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(master['CCTV총수량'], master['안전벨 수'], 
                            equal_var=False)
t_stat, p_value
# (np.float64(9.589582094604252), np.float64(1.328225513688499e-10))


# 비모수 2검정
from scipy.stats import mannwhitneyu
# H0: 총 인구수와 1인 가구수의 중앙값이 같다.
# HA: 총 인구수와 1인 가구수의 중앙값이 다르다.

u_stat, p_val = mannwhitneyu(master['CCTV총수량'], master['안전벨 수'], alternative='two-sided')
print(f'Mann-Whitney U 검정 통계량: {u_stat:.4f}')
print(f'p-value: {p_val:.4f}')
# Mann-Whitney U 검정 통계량: 567.0000
# p-value: 0.0000
# 0.05보다 작으므로 귀무가설 기각
# HA: 총 인구수와 1인 가구수의 중앙값이 다르다.




# 치안 센터 수 지도 시각화
master = pd.read_excel('./data/sanggwan_df.xlsx')
master.columns

fig = px.choropleth_mapbox(
    master,
    geojson=geojson_data,
    locations='자치구',                        # 지역 이름
    featureidkey='properties.SIG_KOR_NM',     
    color='치안센터수',                         # 시각화에 사용할 값
    color_continuous_scale='Greens',            # 색상 스케일
    hover_name='자치구',
    hover_data={'치안센터수': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},  # 서울 중심
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 구별 치안센터수 시각화'
)

# 레이아웃 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()
# 종로구가 20개로 가장 많다. 
# 다음으로 중구가 15개
# 다음으로 강남구가 14개


# 경찰관수 지도 시각화
master = pd.read_excel('./data/sanggwan_df.xlsx')
master.columns

fig = px.choropleth_mapbox(
    master,
    geojson=geojson_data,
    locations='자치구',                        # 지역 이름
    featureidkey='properties.SIG_KOR_NM',     
    color='구별 경찰수',                         # 시각화에 사용할 값
    color_continuous_scale='Greens',            # 색상 스케일
    hover_name='자치구',
    hover_data={'구별 경찰수': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},  # 서울 중심
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 구별 경찰수 시각화'
)

# 레이아웃 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()

# 강남구가 1542명으로 가장많다. 총생활인구수가 2위 여서 많은 인력이 있는것 같다.





# 치안센터 수, 경찰관 수 상관관계
master.select_dtypes('number').corr()['치안센터수']['구별 경찰수']
# 0.65004
# 역시나 강한 상관관게를 보인다. 다른 변수들과 비교했을 때 가장 높은 상관관계

# 치안센터 수, 술집 수 상관관계
master.select_dtypes('number').corr()['치안센터수']['술집 수']
# 0.54607
# 다른 변수들과 비교했을 때 두번째로 높은 상관관계
# 치안을 위해 술집 수가 많을 수록 치안센터를 늘리는 경향이 있음.

# 자치구별 집계
police_sum = master.groupby('자치구')['구별 경찰수'].sum()
center_sum = master.groupby('자치구')['치안센터수'].sum()

# 하나의 데이터프레임으로 병합
df_plot = pd.DataFrame({
    '구별 경찰수': police_sum,
    '치안센터수': center_sum
}).reset_index()

# 시각화
plt.figure(figsize=(8, 3))
plt.plot(df_plot['자치구'], df_plot['구별 경찰수'], marker='o', label='구별 경찰수')
plt.plot(df_plot['자치구'], df_plot['치안센터수'] * 100, marker='s', linestyle='--', label='치안센터수 (x100)')  # 스케일 맞춤

plt.title('자치구별 경찰수 vs 치안센터수')
plt.xlabel('자치구')
plt.ylabel('경찰/치안센터 수')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 비모수 2검정
from scipy.stats import mannwhitneyu
# H0: 구별 치안센터수와  경찰수 중앙값이 같다.
# HA: 구별 치안센터수와  경찰수 중앙값이 다르다.

u_stat, p_val = mannwhitneyu(master['치안센터수'], master['구별 경찰수'], alternative='two-sided')
print(f'Mann-Whitney U 검정 통계량: {u_stat:.4f}')
print(f'p-value: {p_val:.4f}')
# Mann-Whitney U 검정 통계량: np.float64(0.0)
# p-value: 0.0000
# 0.05보다 작으므로 귀무가설 기각
# HA: 구별 치안센터 수와 경찰관 수의 중앙값이 다르다.






# 상권 데이터
# 술집 수 지도 시각화
master = pd.read_excel('./data/sanggwan_df.xlsx')
master.columns

fig = px.choropleth_mapbox(
    master,
    geojson=geojson_data,
    locations='자치구',                        # 지역 이름
    featureidkey='properties.SIG_KOR_NM',     
    color='술집 수',                         # 시각화에 사용할 값
    color_continuous_scale='OrRd',            # 색상 스케일
    hover_name='자치구',
    hover_data={'술집 수': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},  # 서울 중심
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 구별 술집 수 시각화'
)

# 레이아웃 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()

top_3 = master.sort_values('술집 수',ascending=True).loc[:, ['자치구', '술집 수']].head(3)
print(top_3)
# 강남구: 12700, 마포구: 8258, 서초구: 5563개 순으로 술집 수가 많다.
# 양천구: 3094, 금천구: 3179, 동작구: 3276 (동작구에 국립현충원이 있다.  유흥시설을 지을 수 없다.)
# 



# 범죄 데이터
# 서울 범죄 수 지도 시각화
master = pd.read_excel('./data/sanggwan_df.xlsx')
master.columns

fig = px.choropleth_mapbox(
    master,
    geojson=geojson_data,
    locations='자치구',                        # 지역 이름
    featureidkey='properties.SIG_KOR_NM',     
    color='총범죄건수',                         # 시각화에 사용할 값
    color_continuous_scale='YlOrRd',            # 색상 스케일
    hover_name='자치구',
    hover_data={'총범죄건수': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},  # 서울 중심
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 총범죄건수 시각화'
)

# 레이아웃 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()

top_3 = master.sort_values('총범죄건수',ascending=True).loc[:, ['자치구', '술집 수']].head(3)
print(top_3)

# 술집 수, 총범죄건수 상관관계
master.select_dtypes('number').corr()['술집 수']['총범죄건수']
# 0.83537
# 역시나 아주 강한 상관관게를 보인다. 
# 다른 변수들과 비교했을 때 가장 높은 상관관계




# 비모수 2검정
from scipy.stats import mannwhitneyu
# H0: 구별 치안센터수와  경찰수 중앙값이 같다.
# HA: 구별 치안센터수와  경찰수 중앙값이 다르다.

u_stat, p_val = mannwhitneyu(master['술집 수'], master['총범죄건수'], alternative='two-sided')
print(f'Mann-Whitney U 검정 통계량: {u_stat:.4f}')
print(f'p-value: {p_val:.4f}')
# Mann-Whitney U 검정 통계량: 123.0000
# p-value: 0.0007
# 0.05보다 작으므로 귀무가설 기각
# HA: 구별 치안센터 수와 경찰관 수의 중앙값이 다르다.







# 구별 범죄율 지도 시각화
master = pd.read_excel('./data/sanggwan_df.xlsx')
master.columns

fig = px.choropleth_mapbox(
    master,
    geojson=geojson_data,
    locations='자치구',                        # 지역 이름
    featureidkey='properties.SIG_KOR_NM',     
    color='범죄율',                         # 시각화에 사용할 값
    color_continuous_scale='YlOrRd',            # 색상 스케일
    hover_name='자치구',
    hover_data={'범죄율': True},
    mapbox_style='carto-positron',
    center={'lat': 37.5665, 'lon': 126.9780},  # 서울 중심
    zoom=10,
    opacity=0.7,
    title='서울시 자치구별 구별 범죄율 시각화'
)

# 레이아웃 조정
fig.update_layout(
    margin={"r": 0, "t": 30, "l": 0, "b": 0},
    height=700,
    width=800
)

fig.show()