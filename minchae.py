import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import os

# env 파일 로드
load_dotenv()
API_KEY = os.getenv("API_KEY")

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

df = pd.DataFrame(night_data)


# 시간 컬럼 추가
df.astype({'예측시간': 'datetime64[ns]'})
df['시간'] = pd.to_datetime(df['예측시간']).dt.hour


# 시간대 컬럼 추가
def 분류_시간대(hour):
    if 18 <= hour < 21:
        return '저녁'
    elif 21 <= hour <= 23:
        return '밤'
    return '새벽'

df['시간대'] = df['시간'].apply(분류_시간대)
df.to_csv("./data/peopledata.csv", encoding='utf-8')


# 카카오 API KEY
import time
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")

headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# API 테스트
url = "https://dapi.kakao.com/v2/local/search/keyword.json"
params = {"query": "강남역"}
response = requests.get(url, headers=headers, params=params)

print("상태 코드:", response.status_code)
print("응답 내용:", response.json())


# 주소 가져오는 API
# 주소 가져오는 함수 (재검색 포함)
def get_address(keyword):
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    
    for attempt in [keyword, keyword.split()[0]]:  # 재검색 포함
        print(f"[{attempt}] 주소 검색 시도 중...")
        params = {"query": attempt}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            documents = response.json().get('documents')
            if documents:
                address_name = documents[0].get('address_name')
                print(f"→ address_name: {address_name}")
                if address_name:
                    parts = address_name.split()
                    gu = parts[1] if len(parts) > 1 else None
                    dong = parts[2] if len(parts) > 2 else None
                    return gu, dong
        else:
            print(f"응답 실패: {response.status_code}")
    
    return None, None

# 중복 요청 방지용 캐시
place_to_address = {}

# 지역 이름 목록
place_list = df['지역'].unique()

# 주소 요청 및 저장
for place in place_list:
    gu, dong = get_address(place)
    place_to_address[place] = {'구': gu, '동': dong}
    time.sleep(0.3)

# 최종 맵핑 적용
df['구'] = df['지역'].map(lambda x: place_to_address[x]['구'])
df['동'] = df['지역'].map(lambda x: place_to_address[x]['동'])

# 그래도 결과가 없는 1개는 수동으로 지정
df.loc[df['구'].isna(), '구'] = '중구'
df.loc[df['동'].isna(), '동'] = '정동'

# csv 파일로 저장
df.to_csv("./data/peopledata.csv", encoding='utf-8')
