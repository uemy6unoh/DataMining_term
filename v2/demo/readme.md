## HTML 데모

`rent_check_demo.html` 을 받아서(Download raw file) 브라우저로 열면 설치 없이 적정 월세 확인 화면을 써볼 수 있음

| 파일 | 내용 |
| --- | --- |
| rent_check_demo.html | 데모 (모델, 건물, 실거래 데이터가 파일 안에 들어있음, 약 6MB) |
| template.html | 화면 틀 |
| build_demo.py | ../streamlit/model 의 모델과 데이터를 넣어서 데모 파일을 만드는 코드 |

- Streamlit 화면과 같은 모델이고, 예측과 SHAP은 브라우저에서 계산함 (파이썬 XGBoost와 차이 0.001만원 미만)
- 최신 크롬, 엣지, 사파리, 파이어폭스에서 열림
