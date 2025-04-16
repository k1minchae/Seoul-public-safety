import pandas as pd 
import requests
import os
import numpy as np

# 서울 치안센터 수
safety_center = pd.read_excel('./data/Seoul_SafetyCener_info.xlsx')

safety_center['구'] = safety_center['지역경찰관서 신주소'].str.extract(r'\s([가-힣]+구)\s')

safety_center['구'].value_counts()

# CCTV
cctv = pd.read_csv('./data/Seoul_CCTV_info.csv', encoding='cp949')

cctv.info()

cctv['자치구'].value_counts()


# 서울 안전벨 수
bell = pd.read_excel('./data/Seoul_Safetybell.xlsx')

bell.info()         # 21091
bell.head()

bell['소재지도로명주소'].isna().sum()   # 2589
bell['소재지지번주소'].isna().sum()     # 5365


bell['구1'] = bell['소재지도로명주소'].str.extract(r'\s([가-힣]+구)\s')
bell['구1'].isna().sum()    # 4758



bell['구2'] = bell['소재지지번주소'].str.extract(r'\s([가-힣]+구)\s')
bell['구2'].isna().sum()       # 8392


bell['구'] = bell['구1'].fillna(bell['구2'])
bell['구']


















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





