import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
data = pd.read_csv(r"C:\Users\USER\Documents\lsbigdata-gen4\2조 프로젝트\yasik\data\adult_entertainment.csv", encoding='cp949')
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



#############################################################################################################
# 데이터
# [머지한유흥업소데이터.csv]: 서울시 유흥업소 + 서울시 술판매 일반음식점 (구별)
# [data/crime_rate.csv]: 서울시 범죄수, 범죄율, 총생활인구수 (구별)
# [data/seoul_crime_rate_20231231.csv]: 서울시 범죄수 (구별)
# [data/Seoul_CCTV_info.csv]: 서울시 CCTV 수 (구별)
# [data/Seoul_Safetybell.xlsx]: 서울시 안전벨 수 (구별)
# [data/Seoul_SafetyCener_info.xlsx]: 서울시 치안 센터 수 (구별)
# [data/seoul_one _person_housed_updated.xlsx]: 서울시 1인 가구 수 (구별)

# [data/hot-place.xlsx]: 유동 인구 (서울시 주요 장소별)
# [data/LOCAL_PEOPLE_GU_2024.csv]: 서울시 인구 (구별)
# [data/LONG_FOREIGNER_GU_2023.csv]: 서울시 외국인 인구 (구별)
# [data/peopledata.csv]: 서울시 유동 인구 (구별)


# 각 데이터
sulzip_df = pd.read_csv('./data/머지한유흥업소데이터.csv')
crime_rate_df = pd.read_csv('./data/crime_rate.csv', encoding='euc-kr', sep='\t')
Seoul_CCTV_df = pd.read_csv('./data/Seoul_CCTV_info.csv',encoding='cp949')
Seoul_bell_df = pd.read_excel('./data/Seoul_Safetybell.xlsx')
Seoul_SafetyCener_df = pd.read_excel('./data/Seoul_SafetyCener_info.xlsx')
seoul_one_people_df = pd.read_excel('./data/seoul_one _person_housed_updated.xlsx')

# 술집 정리 
sulzip_df = sulzip_df.iloc[:, [0, 3]]
sulzip_df = sulzip_df.rename(columns={'총_개수' : '술집 수'})

# cctv 정리
cctv_by_gu = Seoul_CCTV_df.groupby("자치구")["CCTV 수량"].sum().reset_index()
cctv_by_gu.columns = ["자치구", "CCTV총수량"]
cctv_by_gu

# 범죄 데이터 정리 - 총범죄건수, 총생활인구수, 범죄율, 구별 경찰수
crime_rate_df = crime_rate_df.iloc[:, [0,1,5,6,7]]

# 치안센터 정리
Seoul_SafetyCener_df = Seoul_SafetyCener_df.groupby("자치구")["관서명"].count().reset_index()
Seoul_SafetyCener_df = Seoul_SafetyCener_df.rename(columns={'관서명' : '치안센터수'})

# 1인 가구 정리
seoul_one_people_df = seoul_one_people_df.rename(columns={'서울시 1인가구수' : '자치구', '계':"1인가구수"})
seoul_one_df = seoul_one_people_df.iloc[:, [0,1]]

# 안전벨 
Seoul_bell_df = Seoul_bell_df.groupby("자치구")["설치목적"].count().reset_index()
Seoul_bell_df = Seoul_bell_df.rename(columns={'설치목적' : '안전벨 수'})

# 데이터 결합
merged_df = sulzip_df.merge(crime_rate_df, on="자치구") \
               .merge(Seoul_SafetyCener_df, on="자치구") \
               .merge(cctv_by_gu, on="자치구") \
               .merge(seoul_one_df, on="자치구")\
               .merge(Seoul_bell_df, on="자치구")

merged_df.to_excel('./data/sanggwan_df.xlsx', index=False)

merged_df= pd.read_excel('./data/sanggwan_df.xlsx')
merged_df = merged_df.drop(columns=['범죄율'])

# 상관계수 
corr_df = merged_df.drop(columns=["자치구"]).corr()
corr_df 

# 히트맵
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.figure(figsize=(12, 10))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("서울시 자치구별 변수 간 상관관계")
plt.tight_layout()
plt.show()



# 결과 
# 범죄율과의 상관관계
# 술집 수와 범죄율 간에는 양의 상관관계(0.61)가 있습니다. 
# 이는 술집이 많은 지역에서 범죄율이 높을 가능성이 있다는 것을 나타냅니다. 


#############################################################################################################




