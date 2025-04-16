import pandas as pd

# 서울 치안센터 수
safety_center = pd.read_excel('./data/Seoul_SafetyCener_info.xlsx')


safety_center['구'] = safety_center['지역경찰관서 신주소'].str.extract(r'\s([가-힣]+구)\s')
safety_center['구'].value_counts()


# 서울 안전벨 수
bell = pd.read_excel('./data/Seoul_Safetybell2.xlsx')

bell.info()         # 21091
bell.head()

bell['소재지도로명주소'].isna().sum()   # 2589
bell['소재지지번주소'].isna().sum() # 5365


bell['구1'] = bell['소재지도로명주소'].str.extract(r'\s([가-힣]+구)\s')
bell['구1'].isna().sum()    # 4758

safety_center['구'].value_counts()

bell['구2'] = bell['소재지지번주소'].str.extract(r'\s([가-힣]+구)\s')
bell['구2'].isna().sum()       # 8392


# CCTV
cctv = pd.read_csv('./data/Seoul_CCTV_info.csv', encoding='cp949')
cctv.info()

cctv['자치구'].value_counts()