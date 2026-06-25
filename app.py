import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.set_page_config(page_title="Internship Domain Recommendation System")

st.title("🎯 Internship Domain Recommendation System")

name = st.text_input("Enter Your Name")

education = st.selectbox(
    "Education",
    ["B.Tech", "M.Tech", "BCA", "MCA", "B.Sc", "M.Sc"]
)

skills = st.text_input(
    "Enter Skills (comma separated)",
    placeholder="python,pandas,machine learning"
)

interest = st.text_input(
    "Enter Interests",
    placeholder="AI, Data Analysis"
)

if st.button("Get Recommendations"):

    df = pd.read_csv("data/domains.csv")

    user_profile = skills + "," + interest

    all_data = list(df["skills"])
    all_data.append(user_profile)

    cv = CountVectorizer()
    vectors = cv.fit_transform(all_data)

    similarity = cosine_similarity(vectors)

    user_index = len(all_data) - 1

    scores = list(enumerate(similarity[user_index][:-1]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    st.success("Recommendations Generated!")

    st.subheader("Top Recommended Domains")

    top_domain = None

    for idx, score in scores[:3]:

        percentage = round(score * 100, 2)

        row = df.iloc[idx]

        st.write(f"## {row['domain']}")
        st.write(f"**Match Score:** {percentage}%")
        st.write(f"**Description:** {row['description']}")
        st.write(f"**Roadmap:** {row['roadmap']}")
        st.write("---")

        if top_domain is None:
            top_domain = row["domain"]

    history = pd.DataFrame({
        "Name": [name],
        "Education": [education],
        "Skills": [skills],
        "Interest": [interest],
        "Recommended Domain": [top_domain]
    })

    file = "user_history.csv"

    if os.path.exists(file):
        history.to_csv(file, mode="a", header=False, index=False)
    else:
        history.to_csv(file, index=False)

    st.success("Profile Saved Successfully!")