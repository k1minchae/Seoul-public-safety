# 범죄율 데이터 처리해보자
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('./data/seoul_crime_rate_20231231.csv', encoding='cp949')
df.columns = df.columns.str.replace('^서울', '', regex=True)

df['범죄대분류'].unique().size # 15
df['범죄대분류'].unique()
df['범죄중분류'].unique().size # 38


# 치안과 무관한 범죄 제거
# 지능범죄 (사기, 횡령 등)
# 특별경제범죄 (주가조작, 금융범죄 등)
# 보건범죄 (불법의료 등)
# 환경범죄 (폐기물 불법처리 등)
# 노동범죄 (임금체불, 부당해고 등)
# 선거범죄 (선거법 위반, 금품 제공 등)
# 병역범죄 (병역기피, 허위 진단서 등)
df = df[~df['범죄대분류'].isin(['지능범죄', '특별경제범죄', '보건범죄', '환경범죄', '노동범죄', '선거범죄', '병역범죄'])]


# DataFrame을 Long Format으로 변환
df_melted = df.melt(
    id_vars=['범죄대분류', '범죄중분류'],   
    var_name='자치구',                   
    value_name='발생건수'                
)

# 총범죄수 컬럼 추가
df_melted['총범죄수'] = df_melted.groupby('자치구')['발생건수'].transform('sum')


# 자치구별 발생건수 합계
gu_counts = df_melted.groupby('자치구')['발생건수'].sum().sort_values(ascending=False)

# 막대그래프 그리기
plt.figure(figsize=(12, 6))
plt.bar(gu_counts.index, gu_counts.values, color='skyblue')
plt.xticks(rotation=45)
plt.title('자치구별 범죄 발생 건수')
plt.xlabel('자치구')
plt.ylabel('발생건수')
plt.tight_layout()
plt.show()


# 범죄율 계산
# 범죄율 = (구별 총 범죄건수 / 구별 인구수) × 10,000
# 인구수 데이터 불러오기

# 서울시 내국인 인구 데이터
population_kor = pd.read_csv('./data/LOCAL_PEOPLE_GU_2024.csv', encoding='cp949')
population_kor = population_kor.rename(columns={'adstrd_code_se': '자치구코드', 'tot_lvpop_co': '총생활인구수(내)'})
population_kor = population_kor.loc[(population_kor['stdr_de_id'] == 20240101) & (population_kor['tmzon_pd_se'] == 0), :]

# 서울시 외국인 인구 데이터
population_for = pd.read_csv('./data/LONG_FOREIGNER_GU_2023.csv', encoding='cp949')
population_for = population_for.rename(columns={'총생활인구수': '총생활인구수(외)'})
population_for = population_for.loc[(population_for['기준일ID'] == 20231231) & (population_for['시간대구분'] == 0), :]

# 총 인구수만 추출
kor = population_kor.loc[:, ['자치구코드', '총생활인구수(내)']]
for_ = population_for.loc[:, ['자치구코드', '총생활인구수(외)']]


# 자치구코드 기준으로 병합
population = pd.merge(kor, for_, on='자치구코드', how='inner')
population['총생활인구수'] = population['총생활인구수(내)'] + population['총생활인구수(외)']

# 자치구코드 -> 자치구명 변환
area_cd = {
    11110: '종로구',
    11140: '중구',
    11170: '용산구',
    11200: '성동구',
    11215: '광진구',
    11230: '동대문구',
    11260: '중랑구',
    11290: '성북구',
    11305: '강북구',
    11320: '도봉구',
    11350: '노원구',
    11380: '은평구',
    11410: '서대문구',
    11440: '마포구',
    11470: '양천구',
    11500: '강서구',
    11530: '구로구',
    11545: '금천구',
    11560: '영등포구',
    11590: '동작구',
    11620: '관악구',
    11650: '서초구',
    11680: '강남구',
    11710: '송파구',
    11740: '강동구'
}
population['자치구'] = population['자치구코드'].map(area_cd)

# 범죄율 계산
population['범죄율'] = (df_melted.groupby('자치구')['발생건수'].sum() / population['총생활인구수']) * 10000

# 필요한 데이터만 추출
crime_df = df_melted.groupby('자치구')['발생건수'].sum().reset_index()
crime_df.columns = ['자치구', '총범죄건수']

# 범죄율 데이터와 인구수 데이터 병합
crime_df = pd.merge(crime_df, population, on='자치구', how='inner')

# 범죄율 계산
crime_df['범죄율'] = (crime_df['총범죄건수'] / crime_df['총생활인구수']) * 10000

# 범죄율 시각화
plt.figure(figsize=(12, 6))
plt.bar(crime_df['자치구'], crime_df['범죄율'], color='salmon')
plt.xticks(rotation=45)
plt.ylabel('범죄율 (10,000명당)')
plt.title('자치구별 범죄율')

# 파일로 변환
crime_df.to_csv('./data/crime_rate.csv', index=False, encoding='utf-8-sig')
