import streamlit as st             # 웹 앱 UI 생성을 위한 라이브러리
import pandas as pd               # 데이터프레임 처리
import seaborn as sns             # 고급 시각화용 라이브러리
import matplotlib.pyplot as plt   # 기본 시각화 라이브러리
import numpy as np                # 수치 계산용
import matplotlib as mpl          # 폰트 등 matplotlib 전역 설정에 사용
import platform  
def set_font():
    os_name = platform.system()
    if os_name == "Darwin":  # macOS
        mpl.rc('font', family='AppleGothic')
    elif os_name == "Windows":
        mpl.rc('font', family='Malgun Gothic')
    else:  # Linux hoặc hệ điều hành khác
        mpl.rc('font', family='DejaVu Sans')
    plt.rcParams['axes.unicode_minus'] = False
def show_Home():
    #  한글 폰트 설정 (Mac: AppleGothic)
    mpl.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False  # 음수 깨짐 방지

    #  언어 상태 초기화 (세션에 언어 저장)
    if "language" not in st.session_state:
        st.session_state.language = "ko"  # 기본값: 한국어

    #  언어 전환 버튼 클릭 시 한국어 <-> 영어 변경
    if st.button("🇰🇷 / 🇺🇸 Change Language"):
        st.session_state.language = "en" if st.session_state.language == "ko" else "ko"
    lang = st.session_state.language  # 현재 언어 저장

    #  다국어 문구 정의 (한국어/영어)
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
            """,
            "en": """
The Titanic dataset includes information about passengers such as:
- 👤 Gender (Sex)
- 🎟️ Ticket Class (Pclass)
- 🎂 Age
- 👨‍👩‍👧‍👦 Family members onboard (SibSp, Parch)
- 🛳️ Embarkation port, ticket number, cabin, etc.
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

    #  데이터셋 로딩 (train.csv: 학습용 데이터)
    train = pd.read_csv("Titanic_datasets/train.csv")  # 상대경로 기준

    #  사이드바: 조건 선택 (성별, Pclass)
    with st.sidebar:
        st.header("⚙️ 데이터 필터" if lang == "ko" else "⚙️ Data Filter")
        sex_options = train['Sex'].unique().tolist()  # 성별 값 추출
        sex_filter = st.multiselect("성별" if lang=="ko" else "Sex", options=sex_options, default=sex_options)
        pclass_options = sorted(train['Pclass'].unique())
        pclass_filter = st.multiselect("좌석 등급 (Pclass)" if lang=="ko" else "Ticket Class (Pclass)",
                                       options=pclass_options, default=pclass_options)
        apply_button = st.button("📊 필터 적용" if lang=="ko" else "📊 Apply Filter")

    #  화면 상단: 제목 및 소개 텍스트
    st.markdown(f"<h1 style='text-align: center;'>{text['header'][lang]}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; font-size: 16px;'>{text['sub_intro'][lang]}</div>", unsafe_allow_html=True)
    st.markdown(f"### {text['dataset_intro'][lang]}")
    st.markdown(text["dataset_detail"][lang])

    #  데이터 미리보기 (최대 20행)
    with st.expander(text["preview"][lang]):
        st.dataframe(train.head(20), use_container_width=True)

    st.markdown("---")

    #  필터 적용 시 조건에 맞는 데이터 추출
    if apply_button:
        filtered_data = train[(train['Sex'].isin(sex_filter)) & (train['Pclass'].isin(pclass_filter))]

        if filtered_data.empty:
            st.warning(text["no_data"][lang])
        else:
            st.subheader(text["filter_summary"][lang])
            st.dataframe(filtered_data.head(), use_container_width=True)

            st.markdown(f"### {text['visual_analysis'][lang]}")

            #  색상 정의 (0: 사망, 1: 생존)
            palette_countplot = {0: "tomato", 1: "royalblue"}

            col1, col2 = st.columns(2)

            #  성별별 생존자 수 시각화 (막대그래프)
            with col1:
                st.markdown(f"#### {text['sex_plot'][lang]}")
                fig1, ax1 = plt.subplots()
                sns.countplot(data=filtered_data, x='Sex', hue='Survived', ax=ax1, palette=palette_countplot)
                for container in ax1.containers:
                    ax1.bar_label(container, fmt="%d", padding=3)
                ax1.set_xlabel("성별" if lang=="ko" else "Sex")
                ax1.set_ylabel("인원수" if lang=="ko" else "Count")
                ax1.legend(title="생존 여부" if lang=="ko" else "Survival",
                           labels=['사망자', '생존자'] if lang=="ko" else ['Dead', 'Survived'])
                st.pyplot(fig1)

            #  Pclass별 생존자 수 시각화 (막대그래프)
            with col2:
                st.markdown(f"#### {text['pclass_plot'][lang]}")
                fig2, ax2 = plt.subplots()
                sns.countplot(data=filtered_data, x='Pclass', hue='Survived', ax=ax2, palette=palette_countplot)
                for container in ax2.containers:
                    ax2.bar_label(container, fmt="%d", padding=3)
                ax2.set_xlabel("좌석 등급" if lang=="ko" else "Ticket Class")
                ax2.set_ylabel("인원수" if lang=="ko" else "Count")
                ax2.legend(title="생존 여부" if lang=="ko" else "Survival",
                           labels=['사망자', '생존자'] if lang=="ko" else ['Dead', 'Survived'])
                st.pyplot(fig2)

            #  나이 분포 시각화 (히스토그램 + KDE 곡선)
            st.markdown(f"#### {text['age_plot'][lang]}")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            palette_histplot = ["tomato", "royalblue"]
            sns.histplot(data=filtered_data, x='Age', hue='Survived', hue_order=[0, 1],
                         kde=True, bins=30, ax=ax3, palette=palette_histplot, multiple="stack")
            ax3.set_xlabel("나이" if lang=="ko" else "Age")
            ax3.set_ylabel("인원수" if lang=="ko" else "Count")
            ax3.legend(title="생존 여부" if lang=="ko" else "Survival",
                       labels=['사망자', '생존자'] if lang=="ko" else ['Dead', 'Survived'])
            st.pyplot(fig3)

            st.markdown("---")
    else:
        #  필터 적용 안 했을 때 안내 메시지
        st.info(text["filter_info"][lang])
