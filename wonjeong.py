import numpy as np
import pandas as pd
import re
pd.set_option('display.max_rows',None)

#편의점 데이터터
con_df = pd.read_csv('./data/seoul_con_store.csv',encoding='euc-kr')

con_df['dong_full'] = con_df['지번주소'].str.extract(r'([가-힣0-9.]+(?:동|가|읍|면))')
con_df['dong'] = con_df['dong_full'].str.extract(r'^([가-힣]+)')
con_df['gu'] = con_df['지번주소'].str.extract(r'\s([가-힣]+구)\s')

con_count = con_df.groupby('dong').size().sort_values(ascending=False)


#면적 데이터

region_df_total = pd.read_csv('./data/Administrative_districts_by_dong.csv')
region_df_total.columns = region_df_total.iloc[0]     # 첫 번째 행을 columns로
region_df_total = region_df_total.drop(index=0)       # 첫 번째 행은 이제 필요 없으니 삭제
region_df_total = region_df_total.reset_index(drop=True)
region_df = region_df_total[1:]
region_df_total.columns.values[3] = '면적(km2)' #면적 이름 바꿔주기(2개여서)
region_df['면적(km2)'] = region_df['면적(km2)'].astype(float) #object ->float

def normalize_dong_name(name):
    match = re.match(r'^([가-힣]+)', name)  #입력받은 name의 맨 앞에 나오는 한글 부분만 추출
    if match:
        return match.group(1) + '동'
    else:
        return name


# 동 이름 정제 적용
region_df.loc[:,'표준동이름'] = region_df['동별(3)'].apply(normalize_dong_name)

def fix_double_dong(name):
    if name.endswith("동동"):
        return name[:-1]  # 마지막 '동' 하나 제거
    return name

region_df.loc[:,'표준동이름'] = region_df['표준동이름'].apply(fix_double_dong)

region_df.loc[region_df['표준동이름'] == '소계동', '표준동이름'] = None

region_km2 = region_df.groupby('표준동이름')['면적(km2)'].sum()   #동별 단위면적

len(con_count) #충무로 이런거 다름ㅋㅋ
len(region_km2)




#######################################################
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',None)
entertain_df = pd.read_csv('./data/adult_entertainment2.csv', encoding='euc-kr')


entertain_df['구정보']
entertain_df['업태구분명'].unique()

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






##################################################
#cctv vs 범죄율 (회귀분석 수행행)
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
cctv_df = pd.read_csv('./data/Seoul_CCTV_info.csv',encoding='cp949')
cctv_gu = cctv_df.groupby('자치구')['CCTV 수량'].sum()
crime_gu = crime_df[['자치구','범죄율']]
df = pd.merge(cctv_gu,crime_gu,on='자치구')
df.sort_values('CCTV 수량',ascending=False)










import pandas as pd
import plotly.express as px
import json

# 1. geojson 파일 불러오기
geojson_path = './data/seoul_districts.geojson'
with open(geojson_path, 'r', encoding='utf-8') as f:
    seoul_geo = json.load(f)
seoul_geo['features'][0]
fig = px.choropleth(
    df,
    geojson=seoul_geo,
    locations='자치구',            # df에 있는 자치구명 컬럼
    featureidkey='properties.SIG_KOR_NM',  # geojson 속성 중 자치구명 키 (※ 꼭 확인 필요)
    color='CCTV 수량',
    color_continuous_scale='YlOrRd',
    hover_name='자치구',
    title='서울시 자치구별 CCTV 수량 분포'
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
fig.show()


















X = df[['CCTV 수량']]
y = df['범죄율']

X = sm.add_constant(X)

# 회귀모델 적합
model = sm.OLS(y, X).fit()

# 요약 결과 출력
print(model.summary())


import matplotlib.pyplot as plt
import seaborn as sns
# 예측값 계산
df['예측_범죄율'] = model.predict(X)
# 시각화
plt.figure(figsize=(10, 6))
sns.scatterplot(x='CCTV 수량', y='범죄율', data=df, s=80, label='실제 값')
# 회귀선 그리기
sns.lineplot(x='CCTV 수량', y='예측_범죄율', data=df, color='red', label='회귀선')
plt.title('CCTV 수량 vs 범죄율')
plt.xlabel('CCTV 수량')
plt.ylabel('범죄율')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#####################################################

#필요한 라이브러리 부르기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

food_and_entertain = pd.read_csv('./data/머지한유흥업소데이터.csv', encoding='utf-8')
seoul_safetybell = pd.read_excel('./data/Seoul_Safetybell.xlsx', engine='openpyxl')
cctv = pd.read_csv('./data/Seoul_CCTV_info.csv',encoding='cp949')
crime_rate = pd.read_csv('./data/crime_rate.csv',encoding='euc-kr',sep='\t')

seoul_safetybell_df = seoul_safetybell.groupby('자치구')['번호'].count()
cctv_df = cctv.groupby('자치구')['CCTV 수량'].sum()

df = food_and_entertain.merge(seoul_safetybell_df, on='자치구') \
                       .merge(cctv_df, on='자치구') \
                       .merge(crime_rate, on='자치구')
df = df.rename(columns={'번호': '안전벨 개수'})
df = df.drop(columns=['Unnamed: 8'])


#자치구별 음식점 수(일반음식점 + 유흥업소 포함)가 범죄율에 미치는 영향
X = df[['총_개수']]
y = df['범죄율']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

#자치구별 cctv수가 범죄율에 미치는 영향
X = df[['CCTV 수량']]
y = df['범죄율']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

#자치구별 안전벨수가가 범죄율에 미치는 영향
X = df[['안전벨 개수']]
y = df['범죄율']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())