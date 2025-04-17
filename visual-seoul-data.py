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


# 치안 데이터
# CCTV 수 지도 시각화
cctv = pd.read_csv('./data/Seoul_CCTV_info.csv',encoding='cp949')


gu_counts = cctv['자치구'].value_counts().reset_index()
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