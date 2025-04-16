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





