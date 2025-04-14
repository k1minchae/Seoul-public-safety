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
