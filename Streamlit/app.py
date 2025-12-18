import streamlit as st
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# Load files saved in preprocessing.ipynb
# ---------------------------
df = pd.read_csv("../Dataset/services_cleaned.csv")
cat_enc = joblib.load("../Model files/cat_encoder.joblib")
tfidf = joblib.load("../Model files/tfidf_encoder.joblib")
service_features = joblib.load("../Model files/service_features.joblib")

cc = ["Target_Business_Type", "Price_Category", "Language_Support", "Location_Area"]
tc = "Description"


# ---------------------------
# Helper functions
# ---------------------------
def quality_from_score(score):
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium"
    return "Low"


def recommend(user, top_k=3):
    # Encode user categoricals
    user_df = pd.DataFrame([[user[c] for c in cc]], columns=cc)
    user_cat = cat_enc.transform(user_df)

    # Encode user keywords
    user_text = tfidf.transform([user["Keywords"]])

    # Combine
    user_vec = sp.hstack([user_cat, user_text])

    # Similarity
    scores = cosine_similarity(user_vec, service_features)[0]

    # Attach scores to df
    out = df.copy()
    out["Match_Score"] = scores
    out["Match_Quality"] = out["Match_Score"].apply(quality_from_score)

    # Sort top k
    return out.sort_values("Match_Score", ascending=False).head(top_k)


# ---------------------------
# STREAMLIT UI
# ---------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffd1df;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown(
    "<h1 style='text-align:center;font-size:42px; font-weight:700;color:#02066f;'>Service Recommendation System</h1>",
 

    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#000;'>ML-Powered Recommendation Engine</p>",
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #63e5ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)
 

 

 


with st.sidebar:
    st.header("Your Preferences")

    user = {
        "Target_Business_Type": st.selectbox(
            "Business Type", sorted(df["Target_Business_Type"].unique())
        ),
        "Price_Category": st.selectbox(
            "Budget", sorted(df["Price_Category"].unique())
        ),
        "Language_Support": st.selectbox(
            "Language", sorted(df["Language_Support"].unique())
        ),
        "Location_Area": st.selectbox(
            "Location", sorted(df["Location_Area"].unique())
        ),
        "Keywords": st.text_area("Keywords (optional)"),
    }

    k = st.slider("Top K", 1, 10, 3)
    run = st.button("Recommend")


if run:
    results = recommend(user, k)

    for i, row in results.iterrows():
        st.markdown(f"### {i+1}. {row['Service_Name']}")
        st.write("**Business Type:**", row["Target_Business_Type"])
        st.write("**Budget:**", row["Price_Category"])
        st.write("**Language:**", row["Language_Support"])
        st.write("**Location:**", row["Location_Area"])
        st.write("**Match Score:**", round(row["Match_Score"], 3))
        st.write("**Quality:**", row["Match_Quality"])
        st.write("**Description:**", row["Description"])
        st.markdown("---")
