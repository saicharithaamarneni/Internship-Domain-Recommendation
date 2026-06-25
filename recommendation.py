import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("data/domains.csv")

user_skills = input("Enter your skills: ")
interest = input("Enter your interest: ")

profile = user_skills + "," + interest

data = list(df["skills"])
data.append(user_skills)

vectorizer = CountVectorizer()
vectors = vectorizer.fit_transform(data)

similarity = cosine_similarity(vectors)

user_index = len(data) - 1

scores = list(enumerate(similarity[user_index][:-1]))

scores = sorted(scores, key=lambda x: x[1], reverse=True)

print("\nRecommended Internship Domains:\n")

for i in scores[:3]:
    print(df.iloc[i[0]]["domain"])