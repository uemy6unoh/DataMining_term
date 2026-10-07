"""
streamlit 화면을 설치 없이 열어볼 수 있게 HTML 파일 하나로 만들기

- ../streamlit/model (build_service.py 결과)의 모델, 건물, 실거래를 HTML 안에 넣음
- 숫자 배열은 바이너리로 묶어서 gzip + base64 (그냥 JSON으로 넣으면 15MB가 넘음)
- 모델 예측이랑 SHAP은 브라우저(자바스크립트)에서 계산

실행 (v2/demo 폴더에서): python build_demo.py
"""
import base64
import gzip
import json

import numpy as np
import pandas as pd
import xgboost as xgb

M = '../streamlit/model'
TYPES = ['아파트', '연립다세대']

with open(f'{M}/meta.json', encoding='utf-8') as f:
    meta = json.load(f)
buildings = pd.read_csv(f'{M}/buildings.csv.gz')
trades = pd.read_csv(f'{M}/trades.csv.gz')

blob = bytearray()
arrays = {}


def put(name, values, dtype):
    """배열을 blob 뒤에 붙이고 위치를 기록 (자바스크립트 TypedArray로 바로 읽게 4바이트 정렬)"""
    global blob
    a = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    while len(blob) % 4:
        blob.append(0)
    arrays[name] = {'dtype': {np.float32: 'f4', np.int32: 'i4', np.int16: 'i2', np.uint16: 'u2', np.int8: 'i1', np.uint8: 'u1'}[dtype],
                    'offset': len(blob), 'length': int(a.size)}
    blob += a.tobytes()


# 1. 모델: 트리를 전위 순회 순서로 펴서 저장 (왼쪽 자식은 바로 다음 칸, 오른쪽 자식은 위치를 따로 저장)
models = []
for k, (t, key) in enumerate(zip(TYPES, ['apt', 'vil'])):
    booster = xgb.Booster()
    booster.load_model(f'{M}/model_{key}.ubj')
    features = meta['features'][t]
    F, V, R, C, roots, tree_mean = [], [], [], [], [], []
    for s in booster.get_dump(with_stats=True, dump_format='json'):
        roots.append(len(F))
        leaves = []

        def walk(node):
            i = len(F)
            F.append(-1); V.append(0.0); R.append(0); C.append(node['cover'])
            if 'leaf' in node:
                V[i] = node['leaf']
                leaves.append((node['leaf'], node['cover']))
                return
            F[i] = features.index(node['split'])
            V[i] = node['split_condition']
            child = {c['nodeid']: c for c in node['children']}
            walk(child[node['yes']])      # x < 기준값 이면 왼쪽
            R[i] = len(F)
            walk(child[node['no']])

        walk(json.loads(s))
        cov = sum(c for _, c in leaves)
        tree_mean.append(sum(v * c for v, c in leaves) / cov)
    base = float(json.loads(booster.save_config())['learner']['learner_model_param']['base_score'].strip('[]'))
    bias = base + sum(tree_mean)          # SHAP 기준값 = 트리별 평균 예측의 합
    print(t, f'노드 {len(F):,}개, SHAP 기준값 {bias:.3f} (build_service 값 {meta["base_value"][t]:.3f})')
    put(f'm{k}_F', F, np.int8); put(f'm{k}_V', V, np.float32); put(f'm{k}_R', R, np.int32)
    put(f'm{k}_C', C, np.float32); put(f'm{k}_roots', roots, np.int32)
    models.append({'features': features, 'base': base, 'bias': bias})

# 2. 건물
gu_list = sorted(buildings['구'].unique())
dong_list = sorted(buildings['동'].unique())
feat_cols = sorted(set(sum(meta['features'].values(), [])) - {'전용면적(㎡)', '층', '계약개월수', '계약월', '건축년도'})
put('b_type', buildings['주택유형'].map({t: i for i, t in enumerate(TYPES)}), np.uint8)
put('b_gu', buildings['구'].map({g: i for i, g in enumerate(gu_list)}), np.uint8)
put('b_dong', buildings['동'].map({d: i for i, d in enumerate(dong_list)}), np.uint16)
put('b_bid', buildings['건물'], np.int32)
put('b_ntx', buildings['건물 거래 수'], np.int32)
put('b_year', buildings['건축년도'], np.int16)
put('b_lat', buildings['위도'], np.float32)
put('b_lon', buildings['경도'], np.float32)
for c in feat_cols:
    put('bf_' + c, buildings[c], np.float32)   # XGBoost도 입력을 float32로 바꿔서 씀

# 3. 실거래
put('t_bid', trades['건물'], np.int32)
put('t_type', trades['주택유형'].map({t: i for i, t in enumerate(TYPES)}), np.uint8)
put('t_area', trades['전용면적(㎡)'], np.float32)
put('t_floor', trades['층'], np.int8)
put('t_dep', trades['보증금(만원)'], np.int32)
put('t_rent', trades['월세금(만원)'], np.int16)
put('t_month', trades['계약월'], np.uint8)
put('t_months', trades['계약개월수'], np.uint8)

info = {
    'arrays': arrays, 'models': models, 'b_feat_cols': feat_cols,
    'gu': gu_list, 'dong': dong_list,
    'name': buildings['건물명'].fillna('').tolist(), 'road': buildings['도로명주소'].fillna('').tolist(),
    'jibun': buildings['주소'].str.split().str[-1].tolist(),
    'rate': meta['전환율'], 'bins': meta['bins'], 'bin_labels': meta['bin_labels'], 'calib': meta['calib'],
    'names': meta['names'], 'groups': meta['groups'],
}


def pack(b):
    return base64.b64encode(gzip.compress(bytes(b), compresslevel=9, mtime=0)).decode()


html = open('template.html', encoding='utf-8').read()
html = html.replace('__META__', pack(json.dumps(info, ensure_ascii=False, separators=(',', ':')).encode('utf-8')))
html = html.replace('__NUM__', pack(blob))
with open('rent_check_demo.html', 'w', encoding='utf-8') as f:
    f.write(html)
print(f'숫자 배열 {len(blob) / 1e6:.1f}MB -> rent_check_demo.html {len(html.encode()) / 1e6:.1f}MB')
