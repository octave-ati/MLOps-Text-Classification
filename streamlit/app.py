from pathlib import Path

import pandas as pd

import streamlit as st
from classifyops import main, utils
from config import config


@st.cache()
def load_data():
    projects_fp = Path(config.DATA_DIR, "labeled_projects.csv")
    df = pd.read_csv(projects_fp)
    return df


# Title
st.title("Classifyops - MLOps ML Project Classifier")

# Sections
st.header("🔢 Data")
df = load_data()
st.text(f"Number of Projects: {len(df)}")
st.write(df)


st.header("📊 Performance")
performance_fp = Path(config.MODEL_DIR, "performance.json")
performance = utils.load_dict(filepath=performance_fp)
st.text("Overall performance:")
st.write(performance["overall"])
tag = st.selectbox("Choose a specific tag: ", list(performance["class"].keys()))
st.write(performance["class"][tag])
tag = st.selectbox("Choose a slice of the dataset: ", list(performance["slices"].keys()))
st.write(performance["slices"][tag])

st.header("🚀 Inference")
text = st.text_input("Enter text:", "Best text classifier in the whole world")
run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
prediction = main.predict_tag(text=text, run_id=run_id)
st.write(prediction)
