# 🚢 Titanic Data Analysis Dashboard

This is a Streamlit web application that provides interactive visualizations and insights based on the Titanic dataset from Kaggle. Users can explore survival trends by gender, ticket class, and age using filterable, bilingual charts (Korean 🇰🇷 / English 🇺🇸).

---

## 📁 Dataset

The dataset used is the classic [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) dataset, which contains information about passengers, such as:

- Gender (Sex)
- Ticket Class (Pclass)
- Age
- Number of siblings/spouses aboard (SibSp)
- Number of parents/children aboard (Parch)
- Embarkation port, cabin number, and more

Make sure the file `train.csv` is located in the folder `Titanic_datasets/`.

---

## 🖥️ Features

- Multilingual UI (Korean / English toggle)
- Interactive sidebar filters:
  - Filter by gender
  - Filter by ticket class
- Visualization charts:
  - Survival by gender
  - Survival by ticket class
  - Age distribution by survival status
- Stylish and accessible dashboard with emoji-enhanced UI

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/longtran2507/titanic.git
cd titanic

2. Create and activate virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not available, install manually:

bash
Copy
Edit
pip install streamlit pandas seaborn matplotlib numpy
4. Run the app
bash
Copy
Edit
streamlit run titanic_app.py
