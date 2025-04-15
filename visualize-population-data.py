import folium
from folium.features import GeoJson, GeoJsonTooltip
import pandas as pd
import numpy as np
import json
import requests
from scipy.stats import anderson
from scipy import stats
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./data/peopledata.csv', encoding='utf-8')
# GeoJSON 불러오기 (행정구역 구분)
url = 'https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json'
geo_data = requests.get(url).json()


df.loc[:, ['예측최대인구', '구']].groupby('구').mean(numeric_only=True).sort_values('예측최대인구', ascending=False).head(10).plot(kind='barh', figsize=(10, 6), title='시간대별 유동인구 예측 최대값')
plt.hist(df["예측최대인구"], bins=50, color='blue', alpha=0.7)
plt.xlabel("유동인구")
plt.ylabel("빈도수")
plt.title("유동인구 분포 히스토그램")
plt.show()

# 로그변환
df['log_예측최대인구'] = np.log1p(df['예측최대인구'])  # log(값 + 1)


Q1 = df['log_예측최대인구'].quantile(0.25)
Q3 = df['log_예측최대인구'].quantile(0.75)
IQR = Q3 - Q1

# 이상치 기준
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_filtered = df[(df['log_예측최대인구'] >= lower_bound) & (df['log_예측최대인구'] <= upper_bound)]


plt.hist(df_filtered["log_예측최대인구"], bins=30, color='red', alpha=0.7)
plt.xlabel("유동인구")
plt.ylabel("빈도수")
plt.title("유동인구 분포 히스토그램 (로그변환)")
plt.show()

result = anderson(df_filtered["log_예측최대인구"], dist='norm')

# Q-Q Plot
plt.figure(figsize=(6, 6))
stats.probplot(df_filtered["log_예측최대인구"], dist="norm", plot=plt)
plt.title("Q-Q Plot: 로그 변환 유동인구")
plt.grid()
plt.show()

lower_quantile = df_filtered["log_예측최대인구"].quantile(0.01)
upper_quantile = df_filtered["log_예측최대인구"].quantile(0.99)

df_trimmed = df_filtered[
    (df_filtered["log_예측최대인구"] >= lower_quantile) &
    (df_filtered["log_예측최대인구"] <= upper_quantile)
]

plt.figure(figsize=(6, 6))
stats.probplot(df_trimmed["log_예측최대인구"], dist="norm", plot=plt)
plt.title("Q-Q Plot: 로그 변환 유동인구")
plt.grid()
plt.show()


from scipy.stats import boxcox

# 로그 변환 대신 Box-Cox 적용
# 유동인구 값이 0보다 커야 함 (0 이상이면 +1 해주고)
adjusted_data = df_filtered["예측최대인구"] + 1  # 0값 회피

boxcox_transformed, fitted_lambda = boxcox(adjusted_data)

print(f"Box-Cox 변환 최적 λ (lambda): {fitted_lambda:.4f}")

stats.probplot(boxcox_transformed, dist="norm", plot=plt)
plt.title("Box-Cox 변환 후 Q-Q Plot")
plt.grid()
plt.show()

# Anderson-Darling 정규성 검정
result = anderson(boxcox_transformed, dist='norm')

print("Anderson-Darling 정규성 검정 결과")
print(f"Statistic: {result.statistic:.4f}")
for sig, crit in zip(result.significance_level, result.critical_values):
    print(f"Significance level: {sig:.1f}% - Critical value: {crit:.4f}")
    if result.statistic < crit:
        print("  -> 정규성 가정 만족 (기각할 수 없음)")
    else:
        print("  -> 정규성 기각 (정규분포 아님)")
# Significance level: 1.0% - Critical value: 1.0830
#   -> 정규성 가정 만족 (기각할 수 없음)

df_filtered["boxcox_예측최대인구"] = boxcox_transformed



# ✅ 시간대 필터링 (예: 밤 시간대만)
filtered = df_filtered[df_filtered['시간대'] == '밤']  # 또는 df_filtered[df_filtered['시간'] == 21] 등
df_filtered['시간대'].unique()

# 구별 boxcox_예측최대인구 평균 계산
grouped = filtered.groupby('구')['boxcox_예측최대인구'].mean().reset_index()
grouped.columns = ['구', 'boxcox_예측최대인구_평균']

# 지도 생성
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles='CartoDB positron')

# Choropleth 추가 (Box-Cox 평균 기준)
folium.Choropleth(
    geo_data=geo_data,
    data=grouped,
    columns=['구', 'boxcox_예측최대인구_평균'],
    key_on='feature.properties.name',
    fill_color='PuBuGn',
    fill_opacity=0.7,
    line_opacity=0.3,
    nan_fill_color='white',
    legend_name='Box-Cox 변환된 예측 최대 인구 (구별 평균)'
).add_to(m)

# Tooltip (구 이름 표시)
folium.GeoJson(
    geo_data,
    name="구 이름",
    tooltip=GeoJsonTooltip(fields=["name"], aliases=["구:"], sticky=False)
).add_to(m)

# HTML로 저장
m.save("night_map_boxcox_avg.html")



# GeoJSON과 DataFrame의 구 이름 매칭 확인
# 1. DataFrame에서 사용하는 구 이름들
df_gu_names = set(df_filtered['구'].unique())

# 2. GeoJSON에서 나오는 구 이름들
geo_gu_names = set(
    feature['properties']['name'] for feature in geo_data['features']
)

# 3. 매칭 안 되는 구들 확인
not_matched = df_gu_names - geo_gu_names
extra_in_geo = geo_gu_names - df_gu_names

# 4. 출력
print("⚠️ GeoJSON과 매칭되지 않는 구 (df에만 있음):", not_matched)
print("✅ df에는 없고 GeoJSON에는만 있는 구:", extra_in_geo)
