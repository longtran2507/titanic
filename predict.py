import streamlit as st             # 웹 UI 생성
import pandas as pd               # 데이터 프레임 처리
import numpy as np                # 수치 계산
import seaborn as sns             # 시각화
import matplotlib.pyplot as plt   # 그래프 출력
import matplotlib as mpl          # 폰트 설정
import re                         # 정규식 사용 가능 (예: 이름 파싱 등)

# 머신러닝 관련 라이브러리
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

def show_predict():
    #  한글 폰트 설정
    mpl.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    #  다국어 설정: 세션에 저장
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    if st.button("🇰🇷 / 🇺🇸 Change Language"):
        st.session_state.language = "en" if st.session_state.language == "ko" else "ko"
    lang = st.session_state.language

    #  다국어 문구 딕셔너리
    text = {
        "title": {"ko": "🚢 타이타닉 생존 예측", "en": "🚢 Titanic Survival Prediction"},
        "accuracy": {"ko": "✅ 모델 정확도", "en": "✅ Model Accuracy"},
        "classification": {"ko": "📊 평가 지표", "en": "📊 Classification Report"},
        "importance": {"ko": "🔍 특성 중요도", "en": "🔍 Feature Importance"},
        "realtime": {"ko": "🚀 실시간 생존 예측", "en": "🚀 Real-time Survival Prediction"},
        "predict_btn": {"ko": "예측 실행", "en": "Predict"},
        "survival_prob": {"ko": "생존 확률", "en": "Survival Probability"},
    }

    #  피처명 번역 (컬럼명 UI 출력용)
    feature_trans = {
        "Pclass": {"ko": "좌석 등급", "en": "Pclass"},
        "Sex": {"ko": "성별", "en": "Sex"},
        "Age": {"ko": "나이", "en": "Age"},
        "SibSp": {"ko": "형제/배우자 수", "en": "Siblings/Spouse"},
        "Parch": {"ko": "부모/자녀 수", "en": "Parents/Children"},
        "Fare": {"ko": "운임", "en": "Fare"},
        "Embarked": {"ko": "탑승 항구", "en": "Embarked"},
        "FamilySize": {"ko": "가족 인원수", "en": "Family Size"},
        "IsAlone": {"ko": "혼자인가 여부", "en": "Is Alone"},
        "FarePerPerson": {"ko": "인당 요금", "en": "Fare per Person"},
    }

    #  평가 지표 행/열 번역
    row_trans = {
        "accuracy": {"ko": "정확도", "en": "accuracy"},
        "macro avg": {"ko": "매크로 평균", "en": "macro avg"},
        "weighted avg": {"ko": "가중 평균", "en": "weighted avg"}
    }

    col_trans = {
        "precision": {"ko": "정밀도", "en": "precision"},
        "recall": {"ko": "재현율", "en": "recall"},
        "f1-score": {"ko": "F1 점수", "en": "f1-score"},
        "support": {"ko": "샘플 수", "en": "support"}
    }

    #  데이터셋 로드 및 전처리 (캐시 사용)
    @st.cache_data
    def load_data():
        train = pd.read_csv("Titanic_datasets/train.csv")
        test = pd.read_csv("Titanic_datasets/test.csv")
        gender = pd.read_csv("Titanic_datasets/gender_submission.csv")
        test = test.merge(gender, on='PassengerId')  # test에 정답 병합

        train['Source'] = 'train'
        test['Source'] = 'test'
        full = pd.concat([train, test]).reset_index(drop=True)

        # 누락 데이터 처리
        full['Age'].fillna(full['Age'].median(), inplace=True)
        full['Embarked'].fillna(full['Embarked'].mode()[0], inplace=True)
        full['Fare'].fillna(full['Fare'].median(), inplace=True)

        # 파생 변수 생성
        full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
        full['IsAlone'] = np.where(full['FamilySize'] > 1, 0, 1)
        full['FarePerPerson'] = full['Fare'] / full['FamilySize']

        # 범주형 데이터 인코딩
        le_sex = LabelEncoder()
        le_emb = LabelEncoder()
        full['Sex'] = le_sex.fit_transform(full['Sex'])
        full['Embarked'] = le_emb.fit_transform(full['Embarked'])

        return full, le_sex, le_emb

    #  데이터 준비
    full_df, le_sex, le_embarked = load_data()
    features = list(feature_trans.keys())  # 전체 사용 feature
    train_df = full_df[full_df['Source'] == 'train']
    test_df = full_df[full_df['Source'] == 'test']
    X_train, y_train = train_df[features], train_df['Survived']
    X_test, y_test = test_df[features], test_df['Survived']

    #  모델 학습 (랜덤 포레스트 사용)
    model = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    #  테스트 데이터 예측 및 정확도 계산
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['사망자 / Dead', '생존자 / Survived'], output_dict=True)

    #  정확도 출력
    st.title(text["title"][lang])
    st.header(text["accuracy"][lang])
    st.success(f"{acc*100:.2f}%")

    #  평가 지표 표로 표시
    st.subheader(text["classification"][lang])
    rep_df = pd.DataFrame(report).transpose()
    rep_df.index = [row_trans.get(idx, {lang: idx}).get(lang, idx) for idx in rep_df.index]
    rep_df.columns = [col_trans.get(col, {lang: col}).get(lang, col) for col in rep_df.columns]
    rep_df = rep_df.round(2)
    rep_df = rep_df.drop("정확도" if lang=="ko" else "accuracy")
    st.dataframe(rep_df)

    #  피처 중요도 시각화
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': features, 'Importance': np.round(importances, 3)})
    imp_df["Feature"] = imp_df["Feature"].map(lambda x: feature_trans[x][lang])
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    st.subheader(text["importance"][lang])
    st.bar_chart(imp_df.set_index('Feature'))

    #  실시간 생존 예측 폼
    st.header(text["realtime"][lang])
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            Pclass = st.selectbox(feature_trans["Pclass"][lang], [1, 2, 3], index=2)
            Sex = st.selectbox(feature_trans["Sex"][lang], ["male", "female"])
            min_age = max(1, int(full_df['Age'].min()))
            max_age = int(full_df['Age'].max())
            age_range = st.slider(feature_trans["Age"][lang], min_value=min_age, max_value=max_age, value=(20, 40))
            Age = sum(age_range) / 2
            SibSp = st.number_input(feature_trans["SibSp"][lang], 0, 10, 0)
            Parch = st.number_input(feature_trans["Parch"][lang], 0, 10, 0)
        with col2:
            Fare = st.number_input(feature_trans["Fare"][lang], 0.0, 600.0, 32.0)
            Embarked = st.selectbox(feature_trans["Embarked"][lang], ['C', 'Q', 'S'], index=2)

        submit_btn = st.form_submit_button(text["predict_btn"][lang])

        #  예측 실행
        if submit_btn:
            FamilySize = SibSp + Parch + 1
            IsAlone = 0 if FamilySize > 1 else 1
            FarePerPerson = Fare / FamilySize
            sex_enc = le_sex.transform([Sex])[0]
            emb_enc = le_embarked.transform([Embarked])[0]

            input_data = np.array([[Pclass, sex_enc, Age, SibSp, Parch, Fare, emb_enc, FamilySize, IsAlone, FarePerPerson]])
            survival_proba = model.predict_proba(input_data)[0][1]

            st.write(f"{text['survival_prob'][lang]}: **{survival_proba*100:.2f}%**")
