> 이 레포는 2025-1 데이터마이닝 3조 팀 프로젝트([dongwon0002/DataMining_term](https://github.com/dongwon0002/DataMining_term))를 fork한 것입니다.
> 원본 결과는 아래에 그대로 두었고, 이후 개인 확장 작업은 [`v2/`](v2/README.md)에 정리했습니다.
> v2에서는 최종 데이터의 중복 행(같은 거래가 평균 3.6행)과 거래, 좌표 짝이 어긋난 문제를 바로잡고, 검증 방식(건물 단위)과 전환율을 다시 정해서 적정 월세 확인 화면(Streamlit, HTML 데모)까지 만들었습니다. 아래 v1 수치(R² 0.93 등)는 바로잡기 전 결과입니다.

## 서울시 월세예측&결정요인 분석
![image](https://github.com/user-attachments/assets/0126b02e-6635-4df4-a5cc-69273b7ac8c6)


### 배경 및 필요성
<pre>

 전세사기 증가로 월세 수요는 증가하는 가운데 월세에 대한 판단 기준이 부족하다고 판단하여
 분석 및 예측을 진행
 <기사출처>
 https://n.news.naver.com/article/022/0004031475?sid=101
 </pre>
  

#### 파일 구성
```
/
├── Analysis/            # 전처리 이후 분석 관련 코드   
├── Preprocessing/       # 전처리 과정 코드와 추가 데이터 전처리 과정 reamdme
├── data                 # 전처리 과정 데이터와 전처리 이후의 완성 output 데이터
├── streamlit            # 모델을 이용한 요인 분석 방식 프로토타입 
└── README.md            # repository 전체 소개
```  

### 모델링
------------
#### 데이터 전처리 과정
원본데이터 전처리
```python
원본데이터 컬럼 >>>
Index(['시군구', '번지', '본번', '부번', '전월세구분', '전용면적(㎡)', '계약년월', '계약일',
       '보증금(만원)', '월세금(만원)', '층', '건축년도', '도로명', '계약기간', '계약구분',
       '갱신요구권 사용', '주택유형', '건물명'],
      dtype='object')
```

1.아파트, 연립다세대 데이터 통합
  - 같은 컬럼을 가리고 있어 두 데이터를 합친다
```python
combined = pd.concat([df_apt, df_vil], ignore_index=True)
```
2. 데이터 타입변환, 세부컬럼 생성 
  - ```시군구```컬럼 ```시```,```군```,```구```로 나눔
  - ```보증금(만원)```,```월세금(만원)```컬럼의  컴마(,)제거하고 숫자형 컬럼으로 변경
  - ```계약년월```컬럼에서 ```계약년```, ```계약월```컬럼으로 세분화후 숫자형 변수로
  - ```계약기간```컬럼에서 ```시작연,시작월,종료연,종료월```컬럼 추출후 ```계약개월수```컬럼 생성
```python
# 1. '시군구' 분리
com['시'] = com['시군구'].str.split(expand=True)[0]
com['구'] = com['시군구'].str.split(expand=True)[1]
com['동'] = com['시군구'].str.split(expand=True)[2]

# 2. 보증금, 월세금에서 ',' 제거 후 int로 변환
com['보증금(만원)'] = com['보증금(만원)'].astype(str).str.replace(',', '', regex=False).astype(int)
com['월세금(만원)'] = com['월세금(만원)'].astype(str).str.replace(',', '', regex=False).astype(int)

# 3. 계약년월 → 계약년, 계약월 분리
com['계약년'] = com['계약년월'].astype(str).str[:4].astype(int)
com['계약월'] = com['계약년월'].astype(str).str[4:].astype(int)

# 계약기간을 시작과 종료로 나누기
com[['계약시작', '계약종료']] = com['계약기간'].str.split('~', expand=True)

# 연도와 월로 분리
com['시작연'] = com['계약시작'].str[:4].astype(int)
com['시작월'] = com['계약시작'].str[4:].astype(int)
com['종료연'] = com['계약종료'].str[:4].astype(int)
com['종료월'] = com['계약종료'].str[4:].astype(int)

# 개월 수 계산
com['계약개월수'] = (com['종료연'] - com['시작연']) * 12 + (com['종료월'] - com['시작월'])

# 사용한 컬럼 제거하고 계약기간을 계약개월수로 대체하려면:
# com.drop(columns=['계약기간', '계약시작', '계약종료', '시작연', '시작월', '종료연', '종료월'], inplace=True)
```
3. 불필요 컬럼 제거
```python
com.drop(['계약구분','갱신요구권 사용','시군구','계약년월','번지','본번','부번'],axis=1, inplace=True)
```
4. 모델 타겟 후보 피쳐등 생성
 - 테스트를 위한 타겟 컬럼을 여럿 생성    
 - 자기자본으로 보증금을 마련하는 경우 (기회비용)
$$[
\text{월 부담액} = \left( \frac{\text{보증금} \times \text{연 이자율}}{12} \right) + \text{월세}
]$$    
- 대출로 보증금을 마련하는 경우 (대출이자만) => 원금상환은 제외(전세금 대출 유형)
$$[
\text{월 부담액} = \left( \frac{\text{보증금} \times 0.7 \times \text{연 대출금리}}{12} \right) + \text{월세}
]$$
- 대출 방법과 종류가 많아서 0.04를 이자율로 선정하여 기회비용 측면에서 적용
```python
com['보증금/월세금'] = com['보증금(만원)']/com['월세금(만원)']
com['보증금/월세금'].round(2)

com['면적/월세금'] = com['전용면적(㎡)']/com['월세금(만원)']
com['면적/월세금'] = com['면적/월세금'].round(2)

com['월부담액'] = com['보증금(만원)']*0.04/12+com['월세금(만원)']
com['월부담액'] = com['월부담액'].round(2)

com['월세금/면적'] = com['월세금(만원)']/df00['전용면적(㎡)']
com['월세금/면적'] = com['면적/월세금'].round(2)
```
5. ```도로명``` 컬럼을 이용해 ```위도```, ```경도```컬럼 생성
  -  카카오맵 API를 사용하여 위경도 추출
```python
import requests
import pandas as pd
import time

def get_lat_lng(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    params = {"query": address}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        result = response.json()
        if result['documents']:
            location = result['documents'][0]['address']
            return float(location['y']), float(location['x'])  # 위도, 경도
    return None, None

# 1. 고유한 도로명 주소 리스트 추출
unique_addresses = com['도로명'].dropna().unique()

# 2. 주소별 위도/경도 저장 딕셔너리
address_to_latlng = {}

for addr in unique_addresses:
    lat, lng = get_lat_lng(addr)
    address_to_latlng[addr] = (lat, lng)
    time.sleep(0.2)  # API 과다 호출 방지

# 3. 원래 데이터프레임에 위도/경도 추가
com['위도'] = com['도로명'].map(lambda x: address_to_latlng.get(x, (None, None))[0])
com['경도'] = com['도로명'].map(lambda x: address_to_latlng.get(x, (None, None))[1])
````
6. ```위도```, ```경도```컬럼을 이용해 법정동, 행정도 데이터 생성
  - 법정동, 행정동 폴리곤 .shp파일을 이용하여 각각의 매물이 어느 동에 위치하는지를 나타내는 컬럼생성
  - 이후 분석과 추가 데이터 통합에 사용
  - 통합할 추가 데이터와의 동 이름 통일
```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# # 1. 행정동 GeoDataFrame 불러오기
F1 = '/content/drive/MyDrive/DataMining/term project/data/BND_ADM_DONG_PG.shp'
gdf = gpd.read_file(F1, encoding='cp949')
# 2. EPSG:5179 좌표계로 가정하고 명시적으로 설정 (중부원점 기준)
# gdf.set_crs(epsg=5179, inplace=True)

# 3. 위경도 좌표계 (WGS84: EPSG:4326)로 변환
gdf = gdf.to_crs(epsg=4326)

# 2. com: 위경도 기반 매물 데이터 (예: '경도', '위도' 컬럼 있음)
# 매물 포인트로 변환
df00['geometry'] = com.apply(lambda row: Point(row['경도'], row['위도']), axis=1)
gdf_points = gpd.GeoDataFrame(com, geometry='geometry', crs='EPSG:4326')  # 위경도는 EPSG:4326

# 3. 좌표계 맞추기 (행정동 데이터에 맞추어 변환)
# gdf_points = gdf_points.to_crs(epsg=5181)

# 4. 공간 조인: 어떤 동의 폴리곤에 포함되는지 계산
joined = gpd.sjoin(gdf_points, gdf[['ADM_NM', 'geometry']], how='left', predicate='within')

# 5. 결과: joined['ADM_NM'] 이 해당 매물의 행정동 이름
# 원래 com에 컬럼 추가
com['행정동'] = joined['ADM_NM']

# 추가할 데이터와 다른 동 이름 통일
com.loc[com['행정동']=='금호2·3가동','행정동']='금호2.3가동'
com.loc[com['행정동']=='면목3·8동','행정동']='면목3.8동'
com.loc[com['행정동']=='상계3·4동','행정동']='상계3.4동'
com.loc[com['행정동']=='상계6·7동','행정동']='상계6.7동'
com.loc[com['행정동']=='종로1·2·3·4가동','행정동']='종로1.2.3.4가동'
com.loc[com['행정동']=='종로5·6가동','행정동']='종로5.6가동'
com.loc[com['행정동']=='중계2·3동','행정동']='중계2.3동'
```
7. 서울시 병원위치 정보 데이터를 이용해 기준거리 이내 병원개수 컬럼 생성
  - (0.2, 0.5, 1)km 거리 이내의 병원개수 컬럼을 추가
  - BallTree알고리즘에서 Haversine metric을 이용하여 계산
```python
import numpy as np
from sklearn.neighbors import BallTree

# 위경도 → 라디안 변환 (Haversine 거리 계산을 위해)
def to_radians(df, lat_col='위도', lon_col='경도'):
    return np.radians(df[[lat_col, lon_col]].values)

# 1km, 3km, 10km를 라디안으로 변환 (지구 반지름 약 6371km 기준)
radius_km = [0.2, 0.5, 1]
radii = [r / 6371.0 for r in radius_km]

# 매물 위치, 병원 위치를 라디안으로 변환
df00_rad = to_radians(df00, '위도', '경도')
hos00_rad = to_radians(hos_loc, '병원위도', '병원경도')

# BallTree 생성 (Haversine metric 사용)
tree = BallTree(hos00_rad, metric='haversine')

# 매물별 거리 내 병원 개수 계산
for r, km in zip(radii, radius_km):
    count = tree.query_radius(df00_rad, r=r, count_only=True)
    df00[f'병원_{km}km내_개수'] = count
```
(법정동추가 방법은 유사 내용 생략)

8. 전국 음식점 정보 데이터를 이용해 기준거리 이내 음식점개수 컬럼 생성
  - [음식점 전처리_readme.md](https://github.com/dongwon0002/DataMining_term/tree/main/Preprocessing)
  - 추가 방법은 병원과 동일
9. 서울시 지구대/파출소 데이터를 이용해 기준거리 이내 지구대/파출소 개수 컬럼 생성
  - [지구대/파출소_readme.md](https://github.com/dongwon0002/DataMining_term/tree/main/Preprocessing)
10. 공원데이터를 이용한 기준거리 이내 공원 개수 컬럼 생성
  - [공원 전처리_readme.md](https://github.com/dongwon0002/DataMining_term/tree/main/Preprocessing)
11. 지하철역 위치 데이터를 이용한 기준거리 이내 지하철 역 개수 컬럼 생성
  - [지하철역 전처리_readme.md](https://github.com/dongwon0002/DataMining_term/tree/main/Preprocessing)
12. 대학/대학원 데이터를 이용한 기준거리 이내 공원 개수 컬럼 생성
  - [대학/대학원 전처리_readme.md](https://github.com/dongwon0002/DataMining_term/tree/main/Preprocessing)
13. 서울시 등록인구 통계 데이터와 원본데이터를 행정동을 기준으로 통합
```python
# 1. 필요한 컬럼만 추출
df_pop = pop[['동별', '연령별', '항목', '2024. 4/4']].copy()

# 2. 새로운 컬럼명 생성: "연령별_항목" 형태
df_pop['col_name'] = df_pop['연령별'] + '_' + df_pop['항목']

# 3. 피벗: 동별을 인덱스로, 새로운 컬럼명(col_name)을 컬럼으로
df_pivot = df_pop.pivot_table(index='동별', columns='col_name', values='2024. 4/4')

# 4. 컬럼 이름 초기화
df_pivot.reset_index(inplace=True)

# 5. df01에 병합 (동별 기준)
com = com.merge(df_pivot, left_on='행정동', right_on='동별', how='left')
```
14. 통합된 등록인구 데이터 이용해 ```합계_등록외국인```,```65세이상_인구비율```,```0-19대인구```,```20-34대인구```,```35-64대인구비```컬럼을 생성
```python
# 2. 65세 이상 인구 총합 계산 (계 컬럼 기준)
age_65_plus_cols = [
    '65~69세_계', '70~74세_계', '75~79세_계', '80~84세_계', '85~89세_계',
    '90~94세_계', '95~99세_계', '100세 이상_계'
]
foreigner_col = ['합계_등록외국인']
com['외국인_비율']= com[foreigner_col].sum(axis=1)/com['합계_계']
com['65세이상_인구비율'] = com[age_65_plus_cols].sum(axis=1)/com['합계_계']

# 연령대별 합 계산
com['0-19대인구'] = com[['0~4세_계', '5~9세_계', '10~14세_계', '15~19세_계']].sum(axis=1)
com['20-34대인구'] = com[['20~24세_계', '25~29세_계', '30~34세_계']].sum(axis=1)
com['35-64대인구'] = com[['35~39세_계', '40~44세_계', '45~49세_계', '50~54세_계', '55~59세_계', '60~64세_계']].sum(axis=1)

# 비율 계산
com['0-19대인구비'] = com['0-19대인구'] / com['합계_계']
com['20-34대인구비'] = com['20-34대인구'] / com['합계_계']
com['35-64대인구비'] = com['35-64대인구'] / com['합계_계']

# 필요 없으면 중간 컬럼 삭제
com.drop(columns=['0-19대인구', '20-34대인구', '35-64대인구'], inplace=True)
```
15. ```주택유형```컬럼으로 encoding
  - 0: 아파트, 1: 연립다세대의 값을 가지는 ```주택유형_encoded```컬럼생성
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
com['주택유형_encoded'] = com.fit_transform(df['주택유형'])
com = com.drop(['주택유형'],axis=1)
```
------------ 
### XGBoost 모델링

#### 알고리즘 선정 이유
 **XGBoost(eXtreme Gradient Boosting)**: 결정 트리 기반의 앙상블 회귀/분류 모델이며 Gradient Boosting 방식을 빠르고 정밀하게 구현한 알고리즘이다. 
초기에는 Random Forest를 고려했으나 40만 개가 넘는 객체의 방대한 규모를 다루기에는 모델 학습 시간과 효율성을 고려할 필요가 있었다. 
Random Forest의 경우 병렬성이 낮고 대규모일수록 학습 시간이 증가한다는 점과 다수의 feature를 고려하였을 때, 사용이 힘들 것이라 판단하였다. 
따라서, 더 빠르고 최적화된 학습이 가능하면서도 복잡한 변수 간 상호작용을 잘 학습할 수 있는 XGBoost를 선택하게 되었다.
(잔차(오차)를 예측하는 회귀 트리를 반복적으로 학습해서 Loss 기준으로 트리를 업데이트하며, L1,L2기반 정규화 및 2차 미분 기반으로 Loss Function 최적화하는 방식)


### 결과

#### 1. 데이터 불러오기
```python
```final_season_added.csv``` 파일의 저장위치를 파악하고 불러온다

import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/DataMining/term project/data/output/final_season_added.csv")
```


#### 2. ```object type```의 변수는 encoding 진행
#### ```주택유형_encoded``` => 0은 아파트 1은 연립다세대
#### ```계절``` => ```계절_겨울```, ```계절_봄```, ```계절_여름``` 컬럼 생성

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['주택유형_encoded'] = le.fit_transform(df['주택유형'])
df = pd.get_dummies(df, columns=['계절'], prefix='계절', drop_first=True)
df = df.drop(['주택유형'],axis=1)
```

#### (선택) 타겟으로 설정할 ```['보증금(만원)', '월세금(만원)', '월부담액', '보증금/월세금', '월세금/면적']``` 컬럼들의 값을 standard scaler를 사용할 것인지

```python
# 1. 원본 데이터에서 해당 컬럼들 제거 후 스케일된 컬럼으로 대체
df.drop(columns=target_cols, inplace=True)
df = pd.concat([df, df_scaled], axis=1)

# 2. 테스트 파이프라인 함수 생성
# train_eval_xgb함수
```

#### ```train_eval_xgb``` 함수가 하는 것
데이터를 train_test_split으로 나누고 train 셋에 대해 XGBRegressor모델을 생성하고, test 셋에 대해 RMSE, MAE, R^2값을 반환하고, 모델의 feature importance 그래프와 shap 그래프를 반환 가능하고 그래프는 True, False로 선택가능하다

#### 파라미터
- ```df```
- ``` target_col```
- ```test_size```
- ```random_state```
- ``` plot_feature_importance```(bool)
- ```plot_shap```(bool)
#### return
- ```model, (X_train, X_test, y_train, y_test)```      
---------

(ex)
```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df.drop(['월부담액','보증금/월세금','월세금/면적','보증금(만원)'],axis=1),'월세금(만원)', plot_feature_importance=False, plot_shap=False)
```

#### 모델링
\>>> df.drop(['월부담액','보증금/월세금','월세금/면적','보증금(만원)'],axis=1)데이터를 바탕으로 모델을 구성하고 ```'월세금(만원)'```을 타겟으로 선정, 두개 그래프 모두 그리지 않는다

    # 1. train/test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. XGBoost 회귀 모델 학습
    model = xgb.XGBRegressor(random_state=random_state, verbosity=0)
    model.fit(X_train, y_train)

    # 3. 예측 및 평가
    y_pred = model.predict(X_test)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R^2: {r2:.4f}")

#### 타겟변수 선택
```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df.drop(['월부담액','보증금/월세금','월세금/면적','보증금(만원)'],axis=1),'월세금(만원)',plot_feature_importance=False,plot_shap=False)
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df.drop(['월부담액','보증금/월세금','월세금/면적','월세금(만원)'],axis=1),'보증금(만원)',plot_feature_importance=False,plot_shap=False)
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df.drop(['월부담액','월세금/면적','월세금(만원)','보증금(만원)'],axis=1),'보증금/월세금',plot_feature_importance=False,plot_shap=False)
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df.drop(['월부담액','보증금/월세금','월세금(만원)','보증금(만원)'],axis=1),'월세금/면적',plot_feature_importance=False,plot_shap=False)
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)'],axis=1),'월부담액',plot_feature_importance=False,plot_shap=False)
```
![image](https://github.com/user-attachments/assets/92ad0f4b-cba2-4f82-becd-05908ba22db8)

#### -> 월부담액을 Target으로 설정!!
    

#### 성능 평가
```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)'],axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
![image](https://github.com/user-attachments/assets/06bc34e6-369d-4625-8c4c-7e3c0e56521c)


#### 주택유형의 Feature importance 높아 아파트와 연립다세대 분리하여 다시 실행


#### 아파트 연립다세대 분리해서 데이터 만들기
```python
# 4. 아파트 or 연립다세대로 구분하여 df_type0와 df_type1데이터 생성
df_type0 = df[df['주택유형_encoded']==0] #아파트
df_type1 = df[df['주택유형_encoded']!=0] #연립다세대
```

#### 모델 개선(아파트/다세대 분리> 각각 평가 > Feature importance로 거리기반 각각 중요만남긴다)

## 주의사항
아파트와 연립 다세대를 구분하여 테스트할경우 df_type0, df_type1데이터를 만든 이후 ```주택유형_encoded```컬럼도 지운다, 계절 컬럼은 계약월을 이용해 만든 컬럼이고, 계약월이 더 중요하다!
## 아파트 모델 
### ** 유형별(역, 공원...등) 최적의 거리 선택 과정 **

###   1. 유형별 모든 거리를 다 넣었을 때
```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type0.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'],axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
![image](https://github.com/user-attachments/assets/f5b042fd-627b-43e0-adb4-65e0b6e40183)

####  성능 확인

<pre>
Test RMSE: 0.3346
Test MAE: 0.1997
Test R^2: 0.9296
</pre>

### 2. 거리관련 features 다 뺀 모델 성능(ver. 아파트)

최적의 거리를 찾아가는 과정

```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type0.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'] + group_features,axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
####  성능 확인
<pre>
Test RMSE: 0.3266
Test MAE: 0.1947
Test R^2: 0.9329
</pre>

### 모델 설명

**아래 모델은 각 유형별 한개씩**
즉 hospital의 5개* station*2개 * 3 * 3 * 3 * 4  = 1080개의 조합 중 최고의 성능을 보여주는 상위의 5개 조합을 보여줌.

itertools.product()는 **두 개 이상의 iterable의 모든 가능한 데카르트 곱(Cartesian Product)**을 구할 때 사용해. 쉽게 말하면, 모든 가능한 조합(순서 중요)을 구해주는 함수를 이용하겠다!

```python
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# (예시) 각 그룹 변수 리스트 정의
hospital_group = ['병원_10km내_개수', '병원_3km내_개수','병원_0.5km내_개수','병원_1km내_개수','병원_0.2km내_개수']
station_group = ['500m_이내_역_개수', '1km_이내_역_개수']
restaurant_group = ['식당_0.2km내_개수','식당_0.5km내_개수','식당_1km내_개수']
park_group = ['공원_300m_이내_개수','공원_500m_이내_개수','공원_800m_이내_개수']
police_group = ['파출소/지구대_0.5km내_개수','파출소/지구대_1km내_개수','파출소/지구대_3km내_개수']
university_group = ['대학_0.2km내_개수','대학_0.5km내_개수','대학_1km내_개수','대학_2km내_개수']
기본변수리스트 = ['전용면적(㎡)','자치구코드','건축년도','층','법정동코드','아파트_거래수',
'계약개월수','계약월','35-64대인구비','연립다세대_거래수','0-19대인구비','20-34대인구비','65세이상_인구비율','외국인_비율']  # 입지 변수 제외한 주요 변수 리스트

group_lists = [
    hospital_group,
    station_group,
    restaurant_group,
    park_group,
    police_group,
    university_group
]

all_combinations = list(itertools.product(*group_lists))

results = []

for combo in all_combinations:
    feature_list = list(combo)
    cols_to_use = 기본변수리스트 + feature_list
    df_sub = df_type0[cols_to_use].copy()
    df_sub['월부담액'] = df_type0['월부담액']

    # 모델 학습 및 평가 (함수 이용)
    model, (X_train, X_test, y_train, y_test) = train_eval_xgb(
        df_sub,
        target_col='월부담액',
        plot_feature_importance=False,
        plot_shap=False
    )
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({'features': feature_list, 'r2': r2, 'mse': mse})

# 결과 DataFrame 생성
results_df = pd.DataFrame(results)

# r2(성능) 기준 내림차순 정렬, 상위 5개 조합 확인
top_results = results_df.sort_values('r2', ascending=False).head()
print(top_results)
```

### 결과_나온 최고의 조합
features: ['병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수']

```python
cols_to_add = ['병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수']
X_input = df_type0.drop(['보증금/월세금', '월세금/면적', '월세금(만원)', '보증금(만원)', '주택유형_encoded'] + group_features, axis=1)

X_input[cols_to_add] = df_type0[cols_to_add] 

model, (X_train, X_test, y_train, y_test) = train_eval_xgb(X_input, '월부담액', plot_feature_importance=True, plot_shap=False)
```
####  성능 확인
<pre>
Test RMSE: 0.3211
Test MAE: 0.1940
Test R^2: 0.9352
</pre>

![image](https://github.com/user-attachments/assets/60ea49b4-b147-4baa-839f-7b0d5eae146b)

## 결론

features: ['병원_1km내_개수', '500m_이내_역_개수', '식당_0.5km내_개수', '공원_800m_이내_개수', '파출소/지구대_1km내_개수', '대학_2km내_개수'] 

사용했을 때 
**Test R^2: 0.9329  -> Test R^2: 0.9352**  
성능이 유의미하게 증가함

## 연립다세대 모델 
아파트와 동일한 과정 반복
### 1. 유형별 모든 거리를 다 넣었을 때(ver.연립다세대)
```python
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type1.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'],axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
####  성능 확인
<pre>
  Test RMSE: 0.1673
  Test MAE: 0.1058
  Test R^2: 0.8388
</pre>
![image](https://github.com/user-attachments/assets/8c561ea8-0f64-4832-9824-08097bd288bd)

### 2. 거리관련 features 다 뺀 모델 성능(ver. 연립다세대)
```python
# 연립 다세대 basci model
model, (X_train, X_test, y_train, y_test) = train_eval_xgb(df_type1.drop(['보증금/월세금','월세금/면적','월세금(만원)','보증금(만원)','주택유형_encoded'] + group_features,axis=1),'월부담액',plot_feature_importance=True,plot_shap=False)
```
####  성능 확인
<pre>
Test RMSE: 0.1663
Test MAE: 0.1057
Test R^2: 0.8406
</pre>

### 아까와 같은 모델에 df_type1(연립 다세대)로만 바꿈
```python
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# (예시) 각 그룹 변수 리스트 정의
hospital_group = ['병원_10km내_개수', '병원_3km내_개수','병원_0.5km내_개수','병원_1km내_개수','병원_0.2km내_개수']
station_group = ['500m_이내_역_개수', '1km_이내_역_개수']
restaurant_group = ['식당_0.2km내_개수','식당_0.5km내_개수','식당_1km내_개수']
park_group = ['공원_300m_이내_개수','공원_500m_이내_개수','공원_800m_이내_개수']
police_group = ['파출소/지구대_0.5km내_개수','파출소/지구대_1km내_개수','파출소/지구대_3km내_개수']
university_group = ['대학_0.2km내_개수','대학_0.5km내_개수','대학_1km내_개수','대학_2km내_개수']
기본변수리스트 = ['전용면적(㎡)','자치구코드','건축년도','층','법정동코드','아파트_거래수',
'계약개월수','계약월','35-64대인구비','연립다세대_거래수','0-19대인구비','20-34대인구비','65세이상_인구비율','외국인_비율']  # 입지 변수 제외한 주요 변수 리스트

group_lists = [
    hospital_group,
    station_group,
    restaurant_group,
    park_group,
    police_group,
    university_group
]

all_combinations = list(itertools.product(*group_lists))

results = []

for combo in all_combinations:
    feature_list = list(combo)
    cols_to_use = 기본변수리스트 + feature_list
    df_sub = df_type1[cols_to_use].copy()
    df_sub['월부담액'] = df_type1['월부담액']

    # 모델 학습 및 평가 (함수 이용)
    model, (X_train, X_test, y_train, y_test) = train_eval_xgb(
        df_sub,
        target_col='월부담액',
        plot_feature_importance=False,
        plot_shap=False
    )
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    results.append({'features': feature_list, 'r2': r2, 'rmse': rmse})

# 결과 DataFrame 생성
results_df = pd.DataFrame(results)

# r2(성능) 기준 내림차순 정렬, 상위 5개 조합 확인
top_results = results_df.sort_values('r2', ascending=False).head()
print(top_results)
```

### 결과_나온 최고의 조합
features: ['병원_1km내_개수', '1km_이내_역_개수', '식당_0.5km내_개수', '공원_500m_이내_개수', '파출소/지구대_0.5km내_개수', '대학_1km내_개수']

```python
cols_to_add = ['병원_1km내_개수', '1km_이내_역_개수', '식당_0.5km내_개수', '공원_500m_이내_개수', '파출소/지구대_0.5km내_개수', '대학_1km내_개수']
X_input = df_type1.drop(['보증금/월세금', '월세금/면적', '월세금(만원)', '보증금(만원)', '주택유형_encoded'] + group_features, axis=1)

X_input[cols_to_add] = df_type1[cols_to_add] 

model, (X_train, X_test, y_train, y_test) = train_eval_xgb(X_input, '월부담액', plot_feature_importance=True, plot_shap=False)
```
####  성능 확인
<pre>
Test RMSE: 0.1599
Test MAE: 0.1050
Test R^2: 0.8528
</pre>

![image](https://github.com/user-attachments/assets/1aa7c968-c508-4655-92bd-f57b3e76ee64)

#### 결론

features: ['병원_1km내_개수', '1km_이내_역_개수', '식당_0.5km내_개수', '공원_500m_이내_개수', '파출소/지구대_0.5km내_개수', '대학_1km내_개수']

사용했을 때 
**Test R^2: 0.8406  -> Test R^2: 0.8528**  
성능이 유의미하게 증가함

### 최종 모델 해석 (아파트/다세대) 

#### < feature importance 순위표 ver.아파트>
| 순위 | 아파트           |  
| -- | ------------ | 
| 1  | 전용면적         |
| 2  | 자치구 코드       | 
| 3  | 건축년도         | 
| 4  | 층            | 
| 5  | 법정동 코드       |
| 6  | 800m 이내 공원 수 | 
| 7  | 500m 이내 역 개수 | 

<해석>
1. 전용면적  -> 아파트는 면적에 따른 가격 체계가 명확(큰 면적- 비쌈)
2. 자치구 코드 -> 예를 들어 월세: 강남 > 노원 
3. 건축년도 -> 신축 아파트는 브랜드,편의성이 좋고 이에 따른 프리미엄이 붙음
4. 층 -> 같은 단지라도 층수에 따라 월세도 차이 남(전망, 소음...)
5. 법정동 코드 -> 자치구 코드와 동일
6.  주변 환경(공원,역) -> 고급 아파트 수요층은 쾌적한 환경에 더 민감함(역세권/ 숲세권)

#### < feature importance 순위표 ver.연립다세대>  
| 순위 | 연립다세대           |  
| -- | ------------ | 
| 1  | 건축년도             |
| 2  | 아파트_거래수        |
| 3  | 전용면적             |
| 4  | 자치구 코드          |
| 5  | 계약개월수           |
| 6  | 65세 이상 인구 비율  |
| 7  | 계약월              |

<해석>
1. 건축년도  -> 연립다세대는 아파트에 비해 노후 비율이 높아서 가격에 더 민감하게 반응
2. 아파트_거래수 -> 아파트와 대체재 관계이거나 해당 지역 부동산 시장의 열기가 반영된 거라 볼 수도 있겠음
3. 전용면적 -> 당연히 면적은 중요하지만 아파트보단 덜함
4. 자치구 코드 -> 전용면적과 비슷한 이유
5. 계약개월수 -> 개별 임대인에 따라 계약기간도 유동적이고 계약기간이 짧으면 월세가 비싸질 수 있다
6. 65세 이상 인구 비율 -> 연립은 고령층/저속득층 주거지로 많이 분포
7. 계약월 -> 학기 시작등 이사철에 가격이 더 비싸진다.
   
#### < 정리 : 아파트 VS 연립 다세대 비교>
| 항목       | 아파트         | 연립 다세대              |
| -------- | ----------- | ------------------- |
| 주요 영향    | 입지 + 물리적 조건 | 시점 + 사회 환경 + 시장 분위기 |
| 거래량 영향   | 없음 (독립적)         | 아파트 거래수 영향 있음       |
| 공원/역 중요도 | 중요          | 중요도 낮음              |
| 고령 인구 영향 | 없음          | 있음                  |

### 시각화 (서울구별로 평귤 월부담액(아파트/다세대) 노원구만 법정동 따서 (아파트/다세대))
[mapping code](https://github.com/dongwon0002/DataMining_term/blob/main/data/mapping.ipynb)   
서울시 자치구를 기준으로 각각의 평균 월부담액을 choropleth mapping   
![image](https://github.com/user-attachments/assets/c3f6aa06-ef39-4793-a226-04c34c255395)   
![image](https://github.com/user-attachments/assets/60884edc-e906-4ccb-ae81-ae040316fee9)   
노원구 법정동을 기준으로 각각의 평균 월부담액을 choropleth mapping   
![image](https://github.com/user-attachments/assets/f6d9c805-dab8-4eab-91fb-b0bc6f15a415)   





###  SHAP 기여도 분석
어떠한 요인이 모델이 아닌 실제 월 부담액에 얼마나 정량적 영향을 주었는가?<br> 
이 질문에 답하기 위해 SHapley Additive exPlanations을 사용<br>
예를 들어, 법정동과 주택유형을 설정하였을 때<br>
월 부담액에 따른 각 요인들이 어떤 수치만큼 월 부담금에 기여했는지<br>
'정량적인 시각화'가 가능.<br>
[mapping code](https://github.com/dongwon0002/DataMining_term/blob/main/IMAGES/streamlit_%EC%9B%94%EC%84%B8%EC%9A%94%EC%9D%B8.gif)   
다른 추가적인 요인들로 설정할 수도 있으며,<br> 
feature을 더하거나 빼며 요소별 기여도에 대한 명확한 해석을 제공함
![image](https://github.com/user-attachments/assets/1f997a0b-c0e0-43c3-8a3c-44fcbd5fb284)   



### 시사점
##### 1. 세입자 관점 시사점
 - 세입자의 합리적 주거 판단 지원
 - 정보 비대칭 완화
 - 주거 조건 간 트레이드 오프 구조 시각화(위 예시에선 월 부담액으로만 진행)
##### 2. 정책/행정적 활용 시사점
 - 지리적 비용 불균형 확인을 시각적으로 설명 가능
 - 공공임대 우선 지역 선정 근거 제공 등의 정책 뒷받침
 - 특정 자치구, 법정동 월 부담액 왜곡이 심하다면 정책 개입 타이밍 포착 가능
##### 3. 시장 및 데이터 관점 시사점
 - 단순 지역 뿐이 아닌 면적, 층수, 건축년도, 역세권 등 다면적 요인 분석
 - 정량 모델 기반 협상력 확보
 - SHAP 기반 개별 분석도 가능


