
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE SETTINGS ----------
st.set_page_config(
    page_title="InternMatch AI",
    page_icon="🎯",
    layout="wide"
)

# ---------- FILE PATHS ----------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "internships.csv"
SAVED_PATH = BASE_DIR / "saved_internships.json"


# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)

    required_columns = [
        "title", "company", "domain", "skills",
        "location", "work_mode", "stipend"
    ]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(
                f"Missing column in CSV: {column}"
            )

    if "description" not in data.columns:
        data["description"] = ""

    if "apply_url" not in data.columns:
        data["apply_url"] = ""

    data["stipend"] = pd.to_numeric(
        data["stipend"], errors="coerce"
    )

    data = data.fillna({
        "title": "",
        "company": "",
        "domain": "",
        "skills": "",
        "location": "",
        "work_mode": "",
        "description": "",
        "apply_url": ""
    })

    return data


def get_key(row):
    return f"{row['title']}|||{row['company']}"


def get_skills(skill_text):
    return [
        skill.strip().lower()
        for skill in str(skill_text).split(";")
        if skill.strip()
    ]


# ---------- SAVE DATA PERMANENTLY ----------
def load_saved_internships():
    if not SAVED_PATH.exists():
        return []

    try:
        with open(SAVED_PATH, "r", encoding="utf-8") as file:
            saved = json.load(file)

        if isinstance(saved, list):
            return saved

        return []

    except (json.JSONDecodeError, OSError):
        return []


def save_internships_to_file(saved):
    with open(SAVED_PATH, "w", encoding="utf-8") as file:
        json.dump(saved, file, ensure_ascii=False, indent=4)


# ---------- LOAD CSV SAFELY ----------
try:
    df = load_data()

except Exception as error:
    st.error(f"Unable to load internship data: {error}")
    st.info(
        "Check that internships.csv is inside the data folder."
    )
    st.stop()


# ---------- SESSION STATE ----------
if "saved_internships" not in st.session_state:
    st.session_state.saved_internships = (
        load_saved_internships()
    )

if "search_results" not in st.session_state:
    st.session_state.search_results = None

if "save_message" not in st.session_state:
    st.session_state.save_message = ""


# ---------- SIDEBAR ----------
st.sidebar.title("🎯 InternMatch AI")
st.sidebar.write("AI-powered internship discovery")

page = st.sidebar.radio(
    "Navigation",
    [
        "Find Internships",
        "Dashboard",
        "Saved Internships"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Smart Internship Discovery")


# ==================================================
# FIND INTERNSHIPS
# ==================================================
if page == "Find Internships":

    st.title("🔍 Find Your Best Internship")
    st.write(
        "Enter your skills and discover internships "
        "that match your profile."
    )

    st.info(
        "Tip: Enter skills separated by commas, "
        "for example Python, Pandas, SQL."
    )

    # Show save message after a rerun
    if st.session_state.save_message:
        st.success(st.session_state.save_message)
        st.session_state.save_message = ""

    with st.form("internship_search_form"):

        name = st.text_input(
            "Your Name",
            placeholder="Enter your name"
        )

        year = st.selectbox(
            "Your Year",
            ["1st Year", "2nd Year", "3rd Year", "4th Year"]
        )

        user_skills = st.text_input(
            "Your Skills",
            placeholder="Python, Pandas, SQL, Excel"
        )

        domains = [
            "All Domains"
        ] + sorted(
            df["domain"].astype(str).unique().tolist()
        )

        selected_domain = st.selectbox(
            "Preferred Domain",
            domains
        )

        search_button = st.form_submit_button(
            "✨ Find Matching Internships"
        )

    # Search only when the form is submitted
    if search_button:

        if not name.strip():
            st.warning("Please enter your name.")

            st.session_state.search_results = None

        elif not user_skills.strip():
            st.warning("Please enter at least one skill.")

            st.session_state.search_results = None

        else:
            skills_list = [
                skill.strip().lower()
                for skill in user_skills.split(",")
                if skill.strip()
            ]

            results = df.copy()

            if selected_domain != "All Domains":
                results = results[
                    results["domain"] == selected_domain
                ].copy()

            if results.empty:
                st.session_state.search_results = None
                st.warning(
                    "No internships found in this domain."
                )

            else:
                user_profile = " ".join(skills_list)

                internship_text = (
                    results["skills"].astype(str)
                    + " "
                    + results["domain"].astype(str)
                    + " "
                    + results["description"].astype(str)
                )

                documents = (
                    [user_profile]
                    + internship_text.tolist()
                )

                try:
                    vectorizer = TfidfVectorizer()
                    vectors = vectorizer.fit_transform(
                        documents
                    )

                    scores = cosine_similarity(
                        vectors[0:1],
                        vectors[1:]
                    )[0]

                    results["Match Score"] = scores * 100

                except ValueError:
                    results["Match Score"] = 0.0

                results["Matched Skills"] = results[
                    "skills"
                ].apply(
                    lambda text: [
                        skill for skill in skills_list
                        if skill in get_skills(text)
                    ]
                )

                results["Missing Skills"] = results[
                    "skills"
                ].apply(
                    lambda text: [
                        skill for skill in get_skills(text)
                        if skill not in skills_list
                    ]
                )

                results = results.sort_values(
                    by="Match Score",
                    ascending=False
                )

                # IMPORTANT: Preserve search results after button clicks
                st.session_state.search_results = (
                    results.to_dict(orient="records")
                )

    # Display stored results on every rerun
    saved_results = st.session_state.search_results

    if saved_results is not None:

        st.subheader(
            f"🎉 Found {len(saved_results)} Internships"
        )

        for i, row in enumerate(saved_results):

            with st.container(border=True):

                st.subheader(row["title"])

                st.write(
                    "**Company:**",
                    row["company"]
                )

                st.write(
                    "**Domain:**",
                    row["domain"]
                )

                st.write(
                    "**Work Mode:**",
                    row["work_mode"]
                )

                st.write(
                    "**Location:**",
                    row["location"]
                )

                stipend = row["stipend"]

                if pd.notna(stipend):
                    st.write(
                        f"**Stipend:** ₹{stipend:,.0f}"
                    )
                else:
                    st.write(
                        "**Stipend:** Not specified"
                    )

                score = float(row["Match Score"])

                st.progress(
                    min(max(score / 100, 0), 1)
                )

                st.write(
                    f"**AI Match Score:** {score:.1f}%"
                )

                matched = row["Matched Skills"]
                missing = row["Missing Skills"]

                st.write(
                    "**Matched Skills:**",
                    ", ".join(matched) if matched else "None"
                )

                st.write(
                    "**Skills to Learn:**",
                    ", ".join(missing)
                    if missing else "No listed skill gaps"
                )

                if str(row["description"]).strip():
                    st.write("**Description:**")
                    st.write(row["description"])

                # ---------- FIXED SAVE INTERNSHIP ----------
                internship_key = get_key(row)

                if internship_key in st.session_state.saved_internships:

                    st.success("⭐ Internship Saved")

                else:
                    if st.button(
                        "⭐ Save Internship",
                        key=f"save_internship_{i}"
                    ):
                        st.session_state.saved_internships.append(
                            internship_key
                        )

                        try:
                            save_internships_to_file(
                                st.session_state.saved_internships
                            )

                            st.session_state.save_message = (
                                f"Saved successfully: {row['title']}"
                            )

                        except OSError as error:
                            # Roll back if the file cannot be written
                            st.session_state.saved_internships.remove(
                                internship_key
                            )

                            st.error(
                                f"Could not save internship: {error}"
                            )

                        st.rerun()

                apply_url = str(
                    row.get("apply_url", "")
                ).strip()

                if apply_url.startswith(("https://", "http://")):
                    st.link_button(
                        "Apply Now",
                        apply_url
                    )

    st.divider()

    st.caption(
        "InternMatch AI | Smart Internship Discovery | "
        "Demo dataset"
    )


# ==================================================
# DASHBOARD - ORIGINAL CHARTS PRESERVED
# ==================================================
elif page == "Dashboard":

    st.title("📊 Internship Analytics Dashboard")
    st.write("Explore insights from our internship dataset.")
    st.caption("Statistics below use sample demo data.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Internships", len(df))
    c2.metric("Domains", df["domain"].nunique())
    c3.metric("Companies", df["company"].nunique())

    avg = df["stipend"].mean()

    c4.metric(
        "Average Stipend",
        f"₹{avg:,.0f}" if pd.notna(avg) else "N/A"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Internships by Domain")

        counts = df["domain"].value_counts().reset_index()
        counts.columns = ["Domain", "Count"]

        fig = px.pie(
            counts,
            names="Domain",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Work Mode Distribution")

        counts = df["work_mode"].value_counts().reset_index()
        counts.columns = ["Work Mode", "Count"]

        fig = px.bar(
            counts,
            x="Work Mode",
            y="Count",
            text="Count"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🔥 Most Demanded Skills")

    skill_counts = (
        df["skills"].fillna("")
        .str.split(";")
        .explode()
        .str.strip()
    )

    skill_counts = skill_counts[
        skill_counts != ""
    ].value_counts()

    skill_data = skill_counts.head(10).reset_index()
    skill_data.columns = ["Skill", "Count"]

    fig = px.bar(
        skill_data,
        x="Count",
        y="Skill",
        orientation="h",
        text="Count"
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("💰 Stipend Comparison")

    stipend_df = df.dropna(subset=["stipend"])

    if not stipend_df.empty:
        fig = px.bar(
            stipend_df,
            x="title",
            y="stipend",
            color="domain",
            labels={
                "title": "Internship",
                "stipend": "Stipend (₹)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No stipend information available.")


# ==================================================
# SAVED INTERNSHIPS
# ==================================================
elif page == "Saved Internships":

    st.title("⭐ Saved Internships")

    st.write(
        "Your bookmarked internships appear here."
    )

    if not st.session_state.saved_internships:

        st.info(
            "You have not saved any internships yet. "
            "Go to Find Internships to save some."
        )

    else:

        saved_df = df[
            df.apply(
                lambda row: get_key(row)
                in st.session_state.saved_internships,
                axis=1
            )
        ]

        if saved_df.empty:
            st.warning(
                "Saved internships were not found in the current CSV."
            )

        else:
            st.subheader(
                f"You have {len(saved_df)} saved internships"
            )

            for i, (_, row) in enumerate(saved_df.iterrows()):

                with st.container(border=True):

                    st.subheader(row["title"])

                    st.write(
                        "**Company:**",
                        row["company"]
                    )

                    st.write(
                        "**Domain:**",
                        row["domain"]
                    )

                    st.write(
                        "**Location:**",
                        row["location"]
                    )

                    if pd.notna(row["stipend"]):
                        st.write(
                            f"**Stipend:** ₹{row['stipend']:,.0f}"
                        )

                    if str(row["description"]).strip():
                        st.write(row["description"])

                    apply_url = str(
                        row["apply_url"]
                    ).strip()

                    if apply_url.startswith(("https://", "http://")):
                        st.link_button(
                            "Apply Now",
                            apply_url
                        )

                    if st.button(
                        "🗑️ Remove Saved Internship",
                        key=f"remove_saved_{i}"
                    ):
                        internship_key = get_key(row)

                        st.session_state.saved_internships.remove(
                            internship_key
                        )

                        try:
                            save_internships_to_file(
                                st.session_state.saved_internships
                            )

                            st.session_state.save_message = (
                                f"Removed: {row['title']}"
                            )

                        except OSError as error:
                            st.error(
                                f"Could not update saved file: {error}"
                            )

                        st.rerun()

    st.divider()

    st.caption(
        "InternMatch AI | Your Saved Internships"
    )