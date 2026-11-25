"""
app.py - website for our Fake Review Detector

How to run:  streamlit run app.py
Needs:       utils.py and models/model.pkl in the same folder
"""

import joblib
import pandas as pd
import streamlit as st

import utils

st.title("Fake Review Detector")
st.write("Python Lab Project - paste a product review to check if it looks fake.")
st.caption("The model was trained on Amazon reviews to spot computer-generated (fake) reviews.")

model = joblib.load("models/model.pkl")

# ---------------- Part 1: check one review ----------------
st.header("Check a review")

review = st.text_area("Review text")
rating = st.slider("Star rating", 1, 5, 5)

if st.button("Check"):
    if review.strip() == "":
        st.warning("Please type a review first.")
    else:
        prob = utils.predict_fake_probability(model, [review], [rating])[0]

        if prob >= 0.5:
            st.error(f"This review looks FAKE ({prob:.0%} chance)")
        else:
            st.success(f"This review looks GENUINE ({1 - prob:.0%} chance)")

        st.write(f"Fake probability: {prob:.0%}")
        st.progress(float(prob))

        fake_words, genuine_words = utils.suspicious_words(model, review)
        st.write("Words that look fake:", ", ".join(fake_words) if fake_words else "none")
        st.write("Words that look genuine:", ", ".join(genuine_words) if genuine_words else "none")

        st.write("Features of this review:")
        st.table(utils.features_table([review], [rating]))

# ---------------- Part 2: check many reviews from a CSV ----------------
st.header("Check many reviews (CSV file)")
st.write("Upload a CSV file with a column named 'text' and optionally 'rating'.")

file = st.file_uploader("Choose CSV file", type="csv")

if file is not None:
    data = pd.read_csv(file)
    if "rating" not in data.columns:
        data["rating"] = 3
    data["rating"] = data["rating"].fillna(3)

    data["fake_probability"] = utils.predict_fake_probability(model, data["text"], data["rating"]).round(2)
    data["result"] = ["Fake" if p >= 0.5 else "Genuine" for p in data["fake_probability"]]

    fake_percent = (data["result"] == "Fake").mean() * 100
    st.write(f"Total reviews: {len(data)}")
    st.write(f"Fake reviews: {fake_percent:.1f}%")
    st.write(f"Average rating (all reviews): {data['rating'].mean():.2f}")
    st.write(f"Average rating (genuine only): {data.loc[data['result'] == 'Genuine', 'rating'].mean():.2f}")

    st.dataframe(data[["text", "rating", "fake_probability", "result"]])
