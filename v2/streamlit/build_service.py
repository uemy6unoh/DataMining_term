"""
v2 서비스용 모델이랑 화면 데이터 만들기 (v1 streamlit/train_model.py 역할)

- 전체 데이터로 아파트, 연립다세대 모델 학습
- 무작위 5-fold 오차로 "적정 범위" 만들 분포 저장
- 화면에서 쓸 건물 목록, 실거래 목록 저장

실행 (v2/streamlit 폴더에서): python build_service.py
"""
import json
import os
import re

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

DATA_PATH = '../data/rent_v2.csv.gz'
CONFIG_PATH = '../data/v2_model_config.json'
OUT = 'model'
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA_PATH)
with open(CONFIG_PATH, encoding='utf-8') as f:
    config = json.load(f)
rates, params, features = config['전환율'], config['params'], config['features']

# 건물 묶기 (02, 03 노트북이랑 같은 기준)
parent = {}


def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    parent[find(a)] = find(b)


for 주소, 구, 동, 도로명주소, 건물명, 주택유형 in df[['주소', '구', '동', '도로명주소', '건물명', '주택유형']].drop_duplicates().itertuples(index=False):
    union('지번:' + 주소, '도로명:' + 구 + ' ' + str(도로명주소))
    if 주택유형 == '아파트' and isinstance(건물명, str):
        union('지번:' + 주소, '단지:' + 구 + ' ' + 동 + ' ' + 건물명)
df['건물'] = pd.factorize(pd.Series([find('지번:' + a) for a in df['주소']]))[0]

df['월부담액'] = df['보증금(만원)'] * df['주택유형'].map(rates) / 12 + df['월세금(만원)']

# 같은 건물의 다른 거래가 학습에 몇 건 있었는지에 따라 오차가 달라서 구간을 나눔
BINS = [0, 1, 5, 20, np.inf]
BIN_LABELS = ['0건', '1~4건', '5~19건', '20건 이상']

calib, base_value = {}, {}
for t, key in [('아파트', 'apt'), ('연립다세대', 'vil')]:
    d = df[df['주택유형'] == t].reset_index(drop=True)
    cols = features[t]

    # 1. 무작위 5-fold: 거래가 있던 건물의 새 계약을 맞힐 때 실제/예측 비율
    ratio = np.zeros(len(d))
    others = np.zeros(len(d))
    for tr, te in KFold(5, shuffle=True, random_state=42).split(d):
        m = xgb.XGBRegressor(random_state=42, verbosity=0, **params).fit(d[cols].iloc[tr], d['월부담액'].iloc[tr])
        ratio[te] = d['월부담액'].iloc[te].values / np.maximum(m.predict(d[cols].iloc[te]), 1)
        cnt_tr = d.iloc[tr]['건물'].value_counts()
        others[te] = d.iloc[te]['건물'].map(cnt_tr).fillna(0).values
    b = pd.cut(others, BINS, right=False, labels=BIN_LABELS)
    calib[t] = {}
    print(t)
    for lab in BIN_LABELS:
        r = ratio[b == lab]
        calib[t][lab] = np.percentile(r, np.arange(1, 100)).round(4).tolist()
        p10, p50, p90 = np.percentile(r, [10, 50, 90])
        print(f'   같은 건물 다른 거래 {lab:6s} : {len(r):6d}건, 80% 범위 = 예측 x {p10:.2f} ~ {p90:.2f} (중앙 {p50:.2f})')

    # 2. 전체 데이터로 최종 모델
    model = xgb.XGBRegressor(random_state=42, verbosity=0, **params).fit(d[cols], d['월부담액'])
    model.save_model(f'{OUT}/model_{key}.ubj')
    contrib = model.get_booster().predict(xgb.DMatrix(d[cols].head(1)), pred_contribs=True)
    base_value[t] = float(contrib[0, -1])

# 3. 화면용 건물 목록 (주소 단위)
loc_cols = sorted(set(sum(features.values(), [])) - {'전용면적(㎡)', '층', '계약개월수', '계약월', '건축년도'})
mode = lambda s: s.mode().iloc[0]
agg = {c: 'first' for c in ['도로명주소', '건물명', '구', '동', '위도', '경도', '건물'] + loc_cols}
agg['건축년도'] = mode
buildings = df.groupby(['주소', '주택유형']).agg(agg).reset_index()
buildings['건물 거래 수'] = buildings['건물'].map(df.groupby('건물').size())
buildings.to_csv(f'{OUT}/buildings.csv.gz', index=False, compression='gzip')

# 4. 화면용 실거래 목록
trades = df[['건물', '주소', '주택유형', '전용면적(㎡)', '층', '보증금(만원)', '월세금(만원)', '계약년', '계약월', '계약개월수']]
trades.to_csv(f'{OUT}/trades.csv.gz', index=False, compression='gzip')


# 5. 설명용 이름이랑 묶음
def nice_name(c):
    m = re.match(r'(.+?)_(\d+(?:\.\d+)?)km내_개수', c)
    if m:
        return f'{m.group(2)}km 안 {m.group(1)} 수'
    m = re.match(r'(\d+)(m|km)_이내_역_개수', c)
    if m:
        return f'{m.group(1)}{m.group(2)} 안 지하철역 수'
    m = re.match(r'공원_(\d+)m_이내_개수', c)
    if m:
        return f'{m.group(1)}m 안 공원 수'
    return {'전용면적(㎡)': '전용면적', '층': '층', '건축년도': '건축년도', '계약개월수': '계약기간', '계약월': '계약하는 달',
            '자치구코드': '자치구', '법정동코드': '동네', '아파트_거래수': '동네 아파트 월세 거래량',
            '연립다세대_거래수': '동네 연립다세대 월세 거래량', '0-19대인구비': '동네 19세 이하 비율',
            '20-34대인구비': '동네 20~34세 비율', '35-64대인구비': '동네 35~64세 비율',
            '65세이상_인구비율': '동네 65세 이상 비율', '외국인_비율': '동네 외국인 비율'}.get(c, c)


all_cols = sorted(set(sum(features.values(), [])))
group = {}
for c in all_cols:
    if c in ['전용면적(㎡)', '층', '건축년도']:
        group[c] = '집 조건'
    elif c in ['계약개월수', '계약월']:
        group[c] = '계약 조건'
    elif c in ['자치구코드', '법정동코드']:
        group[c] = '위치 (구, 동)'
    elif c in ['아파트_거래수', '연립다세대_거래수']:
        group[c] = '동네 월세 시장'
    elif '인구' in c or '비율' in c:
        group[c] = '동네 인구 구성'
    else:
        group[c] = '주변 시설'

meta = {'전환율': rates, 'features': features, 'base_value': base_value, 'bins': BINS[:-1], 'bin_labels': BIN_LABELS,
        'calib': calib, 'names': {c: nice_name(c) for c in all_cols}, 'groups': group}
with open(f'{OUT}/meta.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)

for fn in sorted(os.listdir(OUT)):
    print(fn, round(os.path.getsize(f'{OUT}/{fn}') / 1e6, 2), 'MB')
