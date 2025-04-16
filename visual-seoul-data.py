import pandas as pd
import numpy as np
import plotly.express as px
import json
#################

with open('./data/seoul_districts.geojson', encoding='utf-8') as f:
    geojson_data = json.load(f)


# 서울 안전벨 수
bell = pd.read_excel('./data/Seoul_Safetybell.xlsx')
# bell.head()
# bell.info()

# bell['자치구'] = bell['관리기관명'].str.extract(r'([가-힣]+구)')
# bell['자치구'] = bell['자치구'].str.replace('서울시성북구','성북구')
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
    color_continuous_scale="OrRd",
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