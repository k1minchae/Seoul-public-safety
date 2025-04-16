import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
data = pd.read_csv(r"C:\Users\USER\Documents\lsbigdata-gen4\2조 프로젝트\yasik\data\adult_entertainment2.csv", encoding='cp949')
data.head(2)
data.info()
# 유흥업소 
# 유흥업소 구
len(data['구정보'].unique())  ## 25

# 업태구분명
len(data['업태구분명'].unique()) 
## array(['룸살롱', '기타', '고고(디스코)클럽', '비어(바)살롱', '스텐드바', '요정', '간이주점', '노래클럽',
#         '카바레', '극장식당', '관광호텔나이트(디스코)', '관광호텔나이트(카바레)'], dtype=object)

# 구별 유흥업소 개수 
you_counts = data.groupby('구정보').size().reset_index(name='유흥업소_개수')

# 구별 업태구분 개수
gu_type_counts = data.groupby(['구정보', '업태구분명']).size().reset_index(name='업소_개수')



# 일반음식점 (술 팔 수 있는 음식점)
file_path = r"C:\Users\USER\Documents\lsbigdata-gen4\2조 프로젝트\yasik\data\sulzip.xlsx"
data2 = pd.read_excel(file_path)
data2.info()

# 폐점 삭제
data2_filter = data2[data2['영업상태명'] == '영업/정상'].reset_index(drop=True)

# 주소 열 생성
data2_filter['주소'] = data2_filter['지번주소'].fillna(data2_filter['도로명주소'])

# 경기도 삭제
data2_filter = data2_filter[~data2_filter['주소'].astype(str).str.contains('경기도')]

# 구 뽑는 정규표현식
data2_filter['구정보'] = data2_filter['주소'].astype(str).str.extract(r'([가-힣]+구)')
data2_filter['구정보'].unique()

# 구별 일반음식점 개수 
ilban_counts = data2_filter.groupby('구정보').size().reset_index(name='일반음식점_개수')



# 유흥주점 일반음식점 합침
merged_data = pd.merge(ilban_counts, you_counts, on='구정보')
# 합계열 추가
merged_data['총_개수'] = merged_data['일반음식점_개수'] + merged_data['유흥업소_개수']

merged_data

