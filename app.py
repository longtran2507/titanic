import streamlit as st
from streamlit_option_menu import option_menu
import Home
import predict
import search
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Đường dẫn tới file font
font_path = 'fonts/NanumGothic-Regular.ttf'
font_prop = fm.FontProperties(fname=font_path)

plt.rc('font', family=font_prop.get_name())
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="Titanic Survival Analysis", layout="wide", page_icon="🚢")

# 메인 페이지에 위치한 네비게이션 메뉴
selected = option_menu(
    menu_title=None,
    options=["Home", "Predict", "search"],
    icons=["house", "bar-chart-line", "search"],
    orientation="horizontal"
)

if selected == "Home":
    Home.show_Home()  # 원본 home 파일 호출
elif selected == "Predict":
    predict.show_predict()  # 원본 predict 파일 호출
elif selected == "search":
    search.show_search()
