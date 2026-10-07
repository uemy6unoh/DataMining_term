import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

BASE = Path(__file__).parent / 'model'

st.set_page_config(page_title="이 월세 괜찮을까?", layout="wide")


@st.cache_resource
def load():
    with open(BASE / 'meta.json', encoding='utf-8') as f:
        meta = json.load(f)
    models = {}
    for t, key in [('아파트', 'apt'), ('연립다세대', 'vil')]:
        m = xgb.Booster()
        m.load_model(str(BASE / f'model_{key}.ubj'))
        models[t] = m
    buildings = pd.read_csv(BASE / 'buildings.csv.gz')
    trades = pd.read_csv(BASE / 'trades.csv.gz')
    return meta, models, buildings, trades


meta, models, buildings, trades = load()

st.title("🏠 이 월세, 괜찮은 가격일까?")
st.caption("2024년 서울 아파트, 연립다세대 월세 실거래 12만 건으로 만든 적정 월세 확인 (데이터마이닝 3조 프로젝트 v2)")

# ---------------- 입력 ----------------
st.sidebar.header("매물 정보 입력")
house_type = st.sidebar.radio("주택유형", ['아파트', '연립다세대'], horizontal=True)
b = buildings[buildings['주택유형'] == house_type]
gu = st.sidebar.selectbox("구", sorted(b['구'].unique()))
dong = st.sidebar.selectbox("동", sorted(b.loc[b['구'] == gu, '동'].unique()))

cand = b[(b['구'] == gu) & (b['동'] == dong)].copy()
cand['label'] = (cand['건물명'].fillna('(이름 없음)') + '  |  ' + cand['도로명주소'].fillna('-')
                 + '  (' + cand['주소'].str.split().str[-1] + ')')
pick = st.sidebar.selectbox("건물 (이름, 도로명으로 검색 가능)", cand['label'].sort_values().tolist())
bld = cand[cand['label'] == pick].iloc[0]

# 기본값은 그 건물의 가장 최근 거래로
tg = trades[(trades['건물'] == bld['건물']) & (trades['주택유형'] == house_type)]
last = tg.sort_values(['계약년', '계약월']).iloc[-1]
k = bld['주소']
area = st.sidebar.number_input("전용면적 (㎡)", 5.0, 350.0, float(last['전용면적(㎡)']), 0.5, key=f'area_{k}')
floor = st.sidebar.number_input("층", -2, 70, int(last['층']), 1, key=f'floor_{k}')
months = st.sidebar.select_slider("계약기간 (개월)", options=[6, 12, 24, 36, 48], value=24, key=f'months_{k}')
month = st.sidebar.selectbox("계약하는 달", list(range(1, 13)), index=int(last['계약월']) - 1, key=f'month_{k}')
deposit = st.sidebar.number_input("보증금 (만원)", 0, 300000, int(last['보증금(만원)']), 100, key=f'dep_{k}')
rent = st.sidebar.number_input("월세 (만원)", 1, 5000, int(last['월세금(만원)']), 1, key=f'rent_{k}')

# ---------------- 예측 ----------------
r = meta['전환율'][house_type]
cols = meta['features'][house_type]
row = {c: bld[c] for c in cols if c in bld.index}
row.update({'전용면적(㎡)': area, '층': floor, '계약개월수': months, '계약월': month, '건축년도': bld['건축년도']})
X = pd.DataFrame([row])[cols].astype(float)
dm = xgb.DMatrix(X)
pred = max(float(models[house_type].predict(dm)[0]), 1.0)
contrib = models[house_type].predict(dm, pred_contribs=True)[0]

n_other = int(bld['건물 거래 수'])
lab = meta['bin_labels'][int(np.searchsorted(meta['bins'], n_other, side='right')) - 1]
pct = np.array(meta['calib'][house_type][lab])          # 실제/예측 비율의 1~99 백분위
lo, hi = pred * pct[9], pred * pct[89]
mine = deposit * r / 12 + rent
position = float(np.interp(mine / pred, pct, np.arange(1, 100)))

if position < 10:
    verdict, box = "시세보다 많이 낮음", st.warning
    note = "너무 싸면 이유가 있을 수 있음. 등기부등본(근저당, 압류), 건축물대장(위반건축물) 꼭 확인하기"
elif position < 35:
    verdict, box, note = "저렴한 편", st.success, "비슷한 조건 거래보다 낮은 편"
elif position <= 65:
    verdict, box, note = "적정 수준", st.success, "비슷한 조건 거래의 가운데쯤"
elif position <= 90:
    verdict, box, note = "비싼 편", st.info, "아래 '보증금 바꿔보기'랑 같은 건물 실거래를 근거로 조정해볼 만함"
else:
    verdict, box = "시세보다 많이 높음", st.error
    note = "비슷한 조건 거래 10건 중 9건보다 비쌈. 같은 건물 실거래부터 확인하기"

# ---------------- 결과 ----------------
st.subheader(f"{bld['건물명'] if isinstance(bld['건물명'], str) else '(이름 없음)'}  |  {gu} {dong}")
c1, c2, c3 = st.columns(3)
c1.metric("이 매물 월부담액", f"{mine:,.0f}만원", help=f"보증금 x {r * 100:.2f}% / 12 + 월세 (전환율은 2024년 실거래에서 찾은 값)")
c2.metric("모델 적정가", f"{pred:,.0f}만원")
c3.metric("적정 범위 (80%)", f"{lo:,.0f} ~ {hi:,.0f}만원", help=f"같은 건물 다른 거래 {lab} 구간에서 모델이 틀린 정도로 잡은 범위")
box(f"**{verdict}** : 비슷한 조건 거래 100건을 싼 것부터 줄 세우면 대략 **{position:.0f}번째**  \n{note}")

# 범위 그림
band = pd.DataFrame({'x0': [pred * pct[0]], 'x1': [pred * pct[98]]})
iq = pd.DataFrame({'x0': [lo], 'x1': [hi]})
pts = pd.DataFrame({'값': [pred, mine], '구분': ['모델 적정가', '이 매물']})
base_chart = alt.Chart(band).mark_rule(strokeWidth=2, color='#cccccc').encode(x=alt.X('x0:Q', title='월부담액 (만원)'), x2='x1:Q')
iq_chart = alt.Chart(iq).mark_rule(strokeWidth=14, color='#9ecae1').encode(x='x0:Q', x2='x1:Q')
pt_chart = alt.Chart(pts).mark_point(size=180, filled=True).encode(
    x='값:Q', color=alt.Color('구분:N', scale=alt.Scale(domain=['모델 적정가', '이 매물'], range=['#2f6db5', '#d62728'])),
    tooltip=['구분', alt.Tooltip('값:Q', format=',.0f')])
st.altair_chart((base_chart + iq_chart + pt_chart).properties(height=90), use_container_width=True)
st.caption("파란 띠 = 적정 범위 80%, 회색 선 = 98% 범위")

left, right = st.columns([1.1, 1])

# ---------------- 왜 이 가격? ----------------
with left:
    st.markdown("#### 📊 왜 이 가격이 나왔나 (SHAP)")
    base = float(contrib[-1])
    ser = pd.Series(contrib[:-1], index=cols)
    by_group = ser.groupby(lambda c: meta['groups'][c]).sum()
    gdf = by_group.reset_index()
    gdf.columns = ['요인', '만원']
    st.write(f"서울 {house_type} 평균 수준 **{base:,.0f}만원**에서 시작해서 아래 요인들을 더하고 빼면 **{pred:,.0f}만원**")
    bar = alt.Chart(gdf).mark_bar().encode(
        x=alt.X('만원:Q', title='월부담액에 더해진 금액 (만원)'),
        y=alt.Y('요인:N', sort='-x', title=None),
        color=alt.condition('datum.만원 > 0', alt.value('#d62728'), alt.value('#2f6db5')),
        tooltip=['요인', alt.Tooltip('만원:Q', format='+,.1f')])
    st.altair_chart(bar.properties(height=230), use_container_width=True)

    def show_value(c):
        v = row[c]
        if c == '자치구코드':
            return gu
        if c == '법정동코드':
            return dong
        if c == '전용면적(㎡)':
            return f'{v:.1f}㎡'
        if c == '층':
            return f'{int(v)}층'
        if c == '건축년도':
            return f'{int(v)}년'
        if c == '계약개월수':
            return f'{int(v)}개월'
        if c == '계약월':
            return f'{int(v)}월'
        if '비율' in c or '인구비' in c:
            return f'{v * 100:.1f}%'
        if '거래수' in c:
            return f'{int(v):,}건'
        return f'{int(v)}개'

    top = ser.reindex(ser.abs().sort_values(ascending=False).index).head(6)
    st.markdown("**영향이 큰 항목**")
    for c, v in top.items():
        st.write(f"- {meta['names'][c]} ({show_value(c)}) : {v:+,.1f}만원")

# ---------------- 보증금 바꿔보기 ----------------
with right:
    st.markdown("#### 💰 보증금 바꿔보기")
    st.write(f"2024년 실거래 기준 보증금 1,000만원 = 월세 약 **{1000 * r / 12:.1f}만원** (전환율 {r * 100:.2f}%)")
    deps = sorted({0, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, deposit})
    deps = [d for d in deps if d * r / 12 < min(mine, pred)]
    eq = pd.DataFrame({'보증금 (만원)': deps,
                       '이 매물과 같은 부담의 월세': [round(mine - d * r / 12, 1) for d in deps],
                       '적정가 기준 월세': [round(pred - d * r / 12, 1) for d in deps]})
    st.dataframe(eq, hide_index=True, use_container_width=True)

    st.markdown("#### 🗺️ 위치")
    st.map(pd.DataFrame({'lat': [bld['위도']], 'lon': [bld['경도']]}), zoom=14, height=220)

# ---------------- 같은 건물 실거래 ----------------
st.markdown("#### 🧾 같은 건물 2024년 실거래")
near = tg[(tg['전용면적(㎡)'] - area).abs() <= 5]
show = (near if len(near) else tg).copy()
show['월부담액'] = (show['보증금(만원)'] * r / 12 + show['월세금(만원)']).round(1)
show['계약'] = show['계약년'].astype(str) + '-' + show['계약월'].astype(str).str.zfill(2)
show = show.sort_values(['계약년', '계약월'], ascending=False)
st.write(f"전체 {len(tg)}건 중 면적 ±5㎡ 거래 {len(near)}건" + ("" if len(near) else " (없어서 전체 표시)"))
st.dataframe(show[['계약', '전용면적(㎡)', '층', '보증금(만원)', '월세금(만원)', '계약개월수', '월부담액']].head(30),
             hide_index=True, use_container_width=True)

with st.expander("이 화면은 어떻게 계산하나"):
    st.markdown(f"""
- **월부담액** = 보증금 x 전환율 / 12 + 월세. 전환율은 같은 건물, 같은 면적, 같은 달 계약끼리 비교해서 찾은 값
  (아파트 {meta['전환율']['아파트'] * 100:.2f}%, 연립다세대 {meta['전환율']['연립다세대'] * 100:.2f}%)
- **모델 적정가** : XGBoost가 집 조건(면적, 층, 건축년도), 계약 조건, 위치, 동네 인구, 주변 시설로 예측한 월부담액
- **적정 범위** : 2024년에 거래가 있던 건물의 새 계약을 모델이 맞혔을 때 오차 분포로 잡은 80% 범위.
  같은 건물 거래가 적을수록 범위가 넓어짐 (이 건물은 {n_other}건)
- **SHAP** : 모델 예측을 요인별 금액으로 나눈 값. 다 더하면 모델 적정가가 됨
- 2024년 거래 기준이라 이후 시세 변화는 반영 안 됨
""")
