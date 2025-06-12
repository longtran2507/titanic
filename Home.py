import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
def show_Home():
    # Font Hàn ngữ
    mpl.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    # ✅ Language Toggle (song ngữ)
    if "language" not in st.session_state:
        st.session_state.language = "ko"
    if st.button("🇰🇷 / 🇺🇸 Change Language"):
        st.session_state.language = "en" if st.session_state.language == "ko" else "ko"
    lang = st.session_state.language

    # ✅ Song ngữ dictionary
    text = {
        "header": {"ko": "🚢 타이타닉 데이터 분석 대시보드", "en": "🚢 Titanic Data Analysis Dashboard"},
        "sub_intro": {"ko": "대화형 시각화와 타이타닉 데이터셋 종합 분석", "en": "Interactive visualization and Titanic dataset analysis"},
        "dataset_intro": {"ko": "📁 데이터셋 소개", "en": "📁 Dataset Introduction"},
        "dataset_detail": {
            "ko": """
    타이타닉 데이터셋에는 승객에 대한 다음과 같은 정보가 포함되어 있습니다:
    - 👤 성별
    - 🎟️ 좌석 등급 (Pclass)
    - 🎂 나이
    - 👨‍👩‍👧‍👦 동승 가족 수 (SibSp, Parch)
    - 🛳️ 탑승 항구, 티켓 번호, 선실 등

    일반적으로 이 데이터를 기반으로 생존 여부를 분석하는 것이 목표입니다.
            """,
            "en": """
    The Titanic dataset includes information about passengers such as:
    - 👤 Gender (Sex)
    - 🎟️ Ticket Class (Pclass)
    - 🎂 Age
    - 👨‍👩‍👧‍👦 Family members onboard (SibSp, Parch)
    - 🛳️ Embarkation port, ticket number, cabin, etc.

    The goal is usually to analyze survival based on these features.
            """
        },
        "preview": {"ko": "🔍 데이터셋 미리 보기", "en": "🔍 Dataset Preview"},
        "filter_summary": {"ko": "📌 필터된 데이터 요약", "en": "📌 Filtered Data Summary"},
        "visual_analysis": {"ko": "📊 시각화 분석", "en": "📊 Visual Analysis"},
        "sex_plot": {"ko": "성별에 따른 생존자 수", "en": "Survival Count by Gender"},
        "pclass_plot": {"ko": "좌석 등급에 따른 생존자 수", "en": "Survival Count by Ticket Class"},
        "age_plot": {"ko": "생존 여부에 따른 나이 분포", "en": "Age Distribution by Survival"},
        "no_data": {"ko": "❗선택한 조건에 맞는 데이터가 없습니다.", "en": "❗No data matches selected filters."},
        "filter_info": {"ko": "👉 사이드바에서 조건을 선택하고 필터 적용 버튼을 눌러주세요.", "en": "👉 Please select filters in sidebar and click Apply."}
    }

    # 데이터 불러오기
    train = pd.read_csv("Titanic_datasets/train.csv")

    # 사이드바 필터
    with st.sidebar:
        st.header("⚙️ 데이터 필터" if lang == "ko" else "⚙️ Data Filter")

        sex_options = train['Sex'].unique().tolist()
        sex_filter = st.multiselect("성별" if lang=="ko" else "Sex", options=sex_options, default=sex_options)

        pclass_options = sorted(train['Pclass'].unique())
        pclass_filter = st.multiselect("좌석 등급 (Pclass)" if lang=="ko" else "Ticket Class (Pclass)", options=pclass_options, default=pclass_options)

        apply_button = st.button("📊 필터 적용" if lang=="ko" else "📊 Apply Filter")

    # 제목 및 소개
    st.markdown(f"<h1 style='text-align: center;'>{text['header'][lang]}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; font-size: 16px;'>{text['sub_intro'][lang]}</div>", unsafe_allow_html=True)

    st.markdown(f"### {text['dataset_intro'][lang]}")
    st.markdown(text["dataset_detail"][lang])

    # 데이터 미리보기
    with st.expander(text["preview"][lang]):
        st.dataframe(train.head(20), use_container_width=True)

    st.markdown("---")

    # 필터 적용 후 분석
    if apply_button:
        filtered_data = train[(train['Sex'].isin(sex_filter)) & (train['Pclass'].isin(pclass_filter))]

        if filtered_data.empty:
            st.warning(text["no_data"][lang])
        else:
            st.subheader(text["filter_summary"][lang])
            st.dataframe(filtered_data.head(), use_container_width=True)

            st.markdown(f"### {text['visual_analysis'][lang]}")

            palette = {0: "tomato", 1: "royalblue"}

            col1, col2 = st.columns(2)

            # ▶️ 성별 생존자 수
            # 성별 생존자 수
            with col1:
                st.markdown(f"#### {text['sex_plot'][lang]}")
                fig1, ax1 = plt.subplots()
                sns.countplot(data=filtered_data, x='Sex', hue='Survived', ax=ax1, palette=palette)
                for container in ax1.containers:
                    ax1.bar_label(container, fmt="%d", padding=3)
                plt.legend(title="생존 여부" if lang=="ko" else "Survival", 
                        labels=['사망자', '생존자'] if lang=="ko" else ['Dead', 'Survived'])
                ax1.set_xlabel("성별" if lang=="ko" else "Sex")
                ax1.set_ylabel("인원수" if lang=="ko" else "Count")
                st.pyplot(fig1)


            # ▶️ 좌석 등급 생존자 수
            # 좌석 등급 생존자 수
            with col2:
                st.markdown(f"#### {text['pclass_plot'][lang]}")
                fig2, ax2 = plt.subplots()
                sns.countplot(data=filtered_data, x='Pclass', hue='Survived', ax=ax2, palette=palette)
                for container in ax2.containers:
                    ax2.bar_label(container, fmt="%d", padding=3)
                plt.legend(title="생존 여부" if lang=="ko" else "Survival", 
                        labels=['사망자', '생존자'] if lang=="ko" else ['Dead', 'Survived'])
                ax2.set_xlabel("좌석 등급" if lang=="ko" else "Ticket Class")
                ax2.set_ylabel("인원수" if lang=="ko" else "Count")
                st.pyplot(fig2)

            # 나이 분포
            st.markdown(f"#### {text['age_plot'][lang]}")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.histplot(data=filtered_data, x='Age', hue='Survived', kde=True, bins=30, ax=ax3, palette=palette, multiple="stack")
            plt.legend(title="생존 여부" if lang=="ko" else "Survival", 
                    labels=['사망자', '생존자'] if lang=="ko" else ['Dead', 'Survived'])
            ax3.set_xlabel("나이" if lang=="ko" else "Age")
            ax3.set_ylabel("인원수" if lang=="ko" else "Count")
            st.pyplot(fig3)


            st.markdown("---")

    else:
        st.info(text["filter_info"][lang])
