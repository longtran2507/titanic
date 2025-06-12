import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Thiết lập font tiếng Hàn
import matplotlib as mpl
mpl.rc('font', family='AppleGothic')
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

def show_search():

    # Song ngữ toggle
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    lang_button = st.button("🇰🇷 / 🇺🇸 Change Language")
    if lang_button:
        st.session_state.language = "en" if st.session_state.language == "ko" else "ko"
    lang = st.session_state.language

    # Từ điển song ngữ
    text = {
        "title": {"ko": "🔎 승객 검색 및 예측", "en": "🔎 Passenger Search & Prediction"},
        "select_passenger": {"ko": "승객 이름을 선택하세요", "en": "Choose Passenger"},
        "passenger_info": {"ko": "📋 승객 정보", "en": "📋 Passenger Info"},
        "result": {"ko": "🚀 예측 결과", "en": "🚀 Prediction Result"},
        "survival_prob": {"ko": "생존 확률", "en": "Survival Probability"},
        "survived": {"ko": "✅ 예측: 생존 가능성이 높습니다!", "en": "✅ Prediction: Likely Survived!"},
        "dead": {"ko": "⚠️ 예측: 사망 가능성이 높습니다.", "en": "⚠️ Prediction: Likely Did Not Survive."},
    }

    st.title(text["title"][lang])

    # Load dữ liệu
    train = pd.read_csv("Titanic_datasets/train.csv")
    test = pd.read_csv("Titanic_datasets/test.csv")
    gender = pd.read_csv("Titanic_datasets/gender_submission.csv")
    test = test.merge(gender, on='PassengerId')

    train['Source'] = 'train'
    test['Source'] = 'test'
    full = pd.concat([train, test]).reset_index(drop=True)

    # Tiền xử lý dữ liệu giống như predict.py
    full['Age'].fillna(full['Age'].median(), inplace=True)
    full['Embarked'].fillna(full['Embarked'].mode()[0], inplace=True)
    full['Fare'].fillna(full['Fare'].median(), inplace=True)
    full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
    full['IsAlone'] = np.where(full['FamilySize'] > 1, 0, 1)
    full['FarePerPerson'] = full['Fare'] / full['FamilySize']

    le_sex = LabelEncoder()
    le_emb = LabelEncoder()
    full['Sex'] = le_sex.fit_transform(full['Sex'])
    full['Embarked'] = le_emb.fit_transform(full['Embarked'])

    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','FamilySize','IsAlone','FarePerPerson']

    # Train mô hình
    train_df = full[full['Source']=='train']
    test_df = full[full['Source']=='test']

    X_train, y_train = train_df[features], train_df['Survived']
    model = RandomForestClassifier(n_estimators=300, max_depth=7, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # Dropdown chọn tên
    passenger_names = train['Name'].tolist()
    selected_name = st.selectbox(text["select_passenger"][lang], passenger_names)

    # Hiển thị thông tin hành khách
    field_dict = {
        "Pclass": {"ko": "좌석 등급", "en": "Pclass"},
        "Sex": {"ko": "성별", "en": "Sex"},
        "Age": {"ko": "나이", "en": "Age"},
        "SibSp": {"ko": "형제/배우자 수", "en": "Siblings/Spouse"},
        "Parch": {"ko": "부모/자녀 수", "en": "Parents/Children"},
        "Fare": {"ko": "운임", "en": "Fare"},
        "Embarked": {"ko": "탑승 항구", "en": "Embarked"}
    }
    
    passenger_row = full[full['Name'] == selected_name].iloc[0]
    st.subheader(text["passenger_info"][lang])

    passenger_info = {
        field_dict["Pclass"][lang]: int(passenger_row['Pclass']),
        field_dict["Sex"][lang]: "남성" if passenger_row['Sex'] == 1 and lang=="ko" else ("여성" if lang=="ko" else ("male" if passenger_row['Sex']==1 else "female")),
        field_dict["Age"][lang]: int(passenger_row['Age']),
        field_dict["SibSp"][lang]: int(passenger_row['SibSp']),
        field_dict["Parch"][lang]: int(passenger_row['Parch']),
        field_dict["Fare"][lang]: round(float(passenger_row['Fare']), 2),
        field_dict["Embarked"][lang]: le_emb.inverse_transform([passenger_row['Embarked']])[0]
    }

    st.write(passenger_info)

    # Dự đoán
    input_data = np.array([[
        passenger_row['Pclass'], passenger_row['Sex'], passenger_row['Age'],
        passenger_row['SibSp'], passenger_row['Parch'], passenger_row['Fare'],
        passenger_row['Embarked'], passenger_row['FamilySize'],
        passenger_row['IsAlone'], passenger_row['FarePerPerson']
    ]])

    survival_proba = model.predict_proba(input_data)[0][1]
    survival_pred = model.predict(input_data)[0]

    st.subheader(text["result"][lang])
    st.write(f"**{text['survival_prob'][lang]}: {survival_proba*100:.2f}%**")
    if survival_pred == 1:
        st.success(text["survived"][lang])
    else:
        st.error(text["dead"][lang])
