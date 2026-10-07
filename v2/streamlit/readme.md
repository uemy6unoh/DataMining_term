## v2 적정 월세 확인 화면
![image](../images/04_app.png)

1. 주택유형, 구, 동, 건물을 고르고 면적, 층, 보증금, 월세를 넣음 (처음엔 그 건물의 가장 최근 거래가 채워져 있음)
2. 이 매물 월부담액이랑 모델 적정 범위를 비교해서, 비슷한 조건 거래 100건을 싼 것부터 줄 세우면 몇 번째쯤인지 보여줌
3. SHAP으로 적정가가 왜 그렇게 나왔는지 요인별로 만원 단위로 보여줌
4. 보증금을 바꾸면 월세가 얼마여야 같은 부담인지 (실거래에서 찾은 전환율로 계산)
5. 같은 건물의 2024년 실거래 (면적 비슷한 것만)

#### 파일
- ```build_service.py``` : 전체 데이터로 모델 학습, 적정 범위용 오차 분포, 화면용 건물/실거래 목록 저장 (v1 ```train_model.py``` 역할)
- ```app.py``` : streamlit 화면
- ```model/``` : build_service.py 결과물 (이미 들어있어서 바로 실행 가능)

#### 실행
```
cd v2/streamlit
pip install -r requirements.txt
streamlit run app.py
```
model 폴더를 다시 만들 때만 ```python build_service.py``` (03 노트북에서 만든 ```data/v2_model_config.json```을 읽음)
