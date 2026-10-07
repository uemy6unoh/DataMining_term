## 적정 월세 확인 화면 (Streamlit)

![화면](../images/04_app.png)

| 파일 | 내용 |
| --- | --- |
| app.py | 화면 |
| build_service.py | 전체 데이터로 모델 학습, 적정 범위용 오차 분포, 화면용 건물과 실거래 목록 저장 (v1 train_model.py 역할) |
| model/ | build_service.py 결과물 (이미 들어있어서 바로 실행 가능) |

#### 실행
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
model 폴더를 다시 만들 때만 `python build_service.py` (03 노트북에서 만든 `data/v2_model_config.json`을 읽음)

설치 없이 보려면 [HTML 데모](../demo/rent_check_demo.html)
