
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="InternMatch AI",
    page_icon="💼",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "internships.csv"
SAVED_PATH = BASE_DIR / "saved_internships.json"

COLUMNS = [
    "title", "company", "domain", "skills", "location",
    "work_mode", "stipend", "description", "apply_url"
]

# ================= DESIGN =================

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
[data-testid="stMetric"] {
    background: rgba(100, 120, 220, 0.10);
    border: 1px solid rgba(120,120,150,0.25);
    padding: 16px;
    border-radius: 14px;
}
.hero {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(120deg, #173b72, #6546a5);
    color: white;
    margin-bottom: 20px;
}
.hero h1 {color: white;}
</style>
""", unsafe_allow_html=True)


# ================= LOAD CSV =================

def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as error:
        st.error(f"CSV file not read: {error}")
        return pd.DataFrame(columns=COLUMNS)

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""

    text_columns = [
        "title", "company", "domain", "skills",
        "location", "work_mode", "description", "apply_url"
    ]

    for column in text_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["stipend"] = pd.to_numeric(
        df["stipend"], errors="coerce"
    ).fillna(0)

    df = df[
        (df["title"] != "") | (df["company"] != "")
    ]

    return df[COLUMNS].reset_index(drop=True)


def load_saved():
    if not SAVED_PATH.exists():
        return []
    try:
        with open(SAVED_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return []


def save_saved(items):
    with open(SAVED_PATH, "w", encoding="utf-8") as file:
        json.dump(items, file, indent=2)


def get_key(row):
    return str(row["title"]) + "|" + str(row["company"])


# ================= MATCHING =================

def recommend_internships(df, skills):
    result = df.copy()

    if result.empty:
        result["match_score"] = pd.Series(dtype=float)
        return result

    text = skills.strip().lower()

    if not text:
        result["match_score"] = 0.0
        return result

    descriptions = (
        result["title"] + " " +
        result["domain"] + " " +
        result["skills"] + " " +
        result["description"]
    ).tolist()

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform([text] + descriptions)

        scores = cosine_similarity(
            vectors[0:1], vectors[1:]
        ).flatten()

        # Give an additional boost for matching skill keywords.
        query_words = {
            word.strip(".,;:/()[]{}").lower()
            for word in text.replace(",", " ").replace(";", " ").split()
            if len(word.strip(".,;:/()[]{}")) > 1
        }

        final_scores = []

        for score, (_, row) in zip(scores, result.iterrows()):
            row_text = (
                row["title"] + " " + row["domain"] + " " +
                row["skills"] + " " + row["description"]
            ).lower()

            overlap = sum(
                1 for word in query_words if word in row_text
            )

            ratio = overlap / max(len(query_words), 1)
            final_scores.append(
                round(min(100, (0.75 * score + 0.25 * ratio) * 100), 1)
            )

        result["match_score"] = final_scores

    except ValueError:
        result["match_score"] = 0.0

    return result.sort_values(
        "match_score", ascending=False
    ).reset_index(drop=True)


# ================= INTERNSHIP CARD =================

def show_internship(row, card_id):
    st.subheader(row["title"] or "Internship")

    c1, c2, c3 = st.columns(3)
    c1.write("**Company:** " + str(row["company"]))
    c2.write("**Domain:** " + str(row["domain"]))
    c3.write("**Work mode:** " + str(row["work_mode"]))

    c1.write("**Location:** " + str(row["location"]))
    c2.write(f'**Stipend:** ₹{row["stipend"]:,.0f}')

    required_skills = set()
    skills_text = str(row.get("skills", ""))

    for skill in skills_text.replace(";", ",").split(","):
        skill = skill.strip().lower()
        if skill:
            required_skills.add(skill)

    student_skills = st.session_state.get("student_skills", [])

    if isinstance(student_skills, str):
        student_skills = student_skills.split(",")

    student_skills = {
        str(skill).strip().lower()
        for skill in student_skills
        if str(skill).strip()
    }

    matched_skills = sorted(required_skills & student_skills)
    missing_skills = sorted(required_skills - student_skills)

    if "match_score" in row.index:
        score = float(row["match_score"])
        st.progress(
            min(max(score / 100, 0), 1),
            text=f"Estimated Skill Match: {score:.1f}%"
        )

    st.markdown("**Matched Skills**")
    if matched_skills:
        st.write(", ".join(s.title() for s in matched_skills))
    else:
        st.write("No matching skills found.")

    st.markdown("**Missing Skills to Learn**")
    if missing_skills:
        st.write(", ".join(s.title() for s in missing_skills))
    else:
        st.success("You have all the listed required skills!")

    if row["description"]:
        st.write(row["description"])

    url = str(row.get("apply_url", "")).strip()

    if url.lower().startswith(("https://", "http://")):
        st.link_button("Apply Now ↗", url)
    else:

        st.button(
            "Apply Now (Link Coming Soon)",
            key=f"apply_{card_id}",
            disabled=True
    )
    saved = load_saved()
    key = get_key(row)

    if st.button("Save Internship", key=f"save_{card_id}"):
        if key not in saved:
            saved.append(key)
            save_saved(saved)
            st.success("Internship saved!")
        else:
            st.info("This internship is already saved.")

# ================= SIDEBAR =================

st.sidebar.title("💼 InternMatch AI")
st.sidebar.caption("AI-Based Internship Finder")

st.sidebar.subheader("Student Details")

student_name = st.sidebar.text_input(
    "Student Name",
    placeholder="Enter your name"
)

student_year = st.sidebar.selectbox(
    "Academic Year",
    ["1st Year", "2nd Year", "3rd Year", "4th Year"]
)

st.sidebar.divider()
st.sidebar.subheader("Your Skills")

student_skills = st.sidebar.text_area(
    "Enter skills separated by commas",
    placeholder="Python, Java, SQL, HTML, Pandas, Excel"
)

search_clicked = st.sidebar.button(
    "🔍 Find Matching Internships",
    use_container_width=True,
    type="primary"
)

if search_clicked:
    if student_skills.strip():
        st.session_state["student_skills"] = [
            skill.strip().lower()
            for skill in student_skills.split(",")
            if skill.strip()
        ]
        st.session_state["skills_submitted"] = True
    else:
        st.session_state["student_skills"] = []
        st.session_state["skills_submitted"] = False
        st.sidebar.warning("Please enter your skills first!")

page = st.sidebar.radio(
    "Navigate",
    [
        "Find Internships",
        "Dashboard",
        "Saved Internships",
        "Skill Gap Analysis"
    ]
)

if "search_done" not in st.session_state:
    st.session_state.search_done = False

if "search_skills" not in st.session_state:
    st.session_state.search_skills = ""

if search_clicked:
    if student_skills.strip():
        st.session_state.search_done = True
        st.session_state.search_skills = student_skills.strip()
    else:
        st.session_state.search_done = False
        st.sidebar.warning("Enter your skills first!.")

df = load_data()

st.sidebar.divider()
st.sidebar.caption(f"CSV records: {len(df)}")


# ================= FIND INTERNSHIPS =================


# ================= FIND INTERNSHIPS =================

if page == "Find Internships":

    st.markdown("""
    <div class="hero">
        <h1>Find Your Next Opportunity 🚀</h1>
        <p>Enter your skills and discover relevant internships.</p>
    </div>
    """, unsafe_allow_html=True)

    # ================= AI RESUME SCANNER =================

    st.divider()
    st.header("📄 AI Resume Scanner")
    st.write("Upload your resume PDF to detect your technical skills.")

    uploaded_resume = st.file_uploader(
        "Upload your resume (PDF)",
        type=["pdf"],
        key="resume_uploader"
    )

    if uploaded_resume is not None:
        try:
            import re

            pdf_reader = PdfReader(uploaded_resume)
            resume_text = ""

            for pdf_page in pdf_reader.pages:
                resume_text += (pdf_page.extract_text() or "") + " "

            if not resume_text.strip():
                st.warning(
                    "Text could not be extracted. Please upload a text-based PDF."
                )
            else:
                skill_keywords = [
                    "Python", "Java", "C++", "C", "SQL",
                    "Pandas", "NumPy", "Excel", "HTML",
                    "CSS", "JavaScript", "React", "Git",
                    "Machine Learning", "Data Analysis",
                    "Streamlit", "Django", "Flask",
                    "OOP", "Communication"
                ]

                detected_skills = []

                for skill in skill_keywords:
                    pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"

                    if re.search(pattern, resume_text, re.IGNORECASE):
                        detected_skills.append(skill.lower())
                
                # ============ ATS RESUME SCORE ============

                # 1. Skills score: maximum 50 marks
                skills_score = min(len(detected_skills) / 8, 1) * 50

                # 2. Resume sections score: maximum 30 marks
                section_keywords = [
                    "education",
                    "projects",
                    "experience",
                    "certifications",
                    "summary"
                ]

                sections_found = 0

                for section in section_keywords:
                    if section in resume_text.lower():
                        sections_found += 1

                sections_score = (sections_found / 5) * 30

                # 3. Resume content score: maximum 20 marks
                word_count = len(resume_text.split())

                if word_count >= 300:
                    content_score = 20
                elif word_count >= 150:
                    content_score = 15
                elif word_count >= 75:
                    content_score = 10
                else:
                    content_score = 5

                ats_score = round(
                    skills_score + sections_score + content_score
                )

                st.subheader("📊 ATS Resume Score")
                st.progress(ats_score / 100)
                st.metric("ATS Score", f"{ats_score}/100")

                st.caption(
                    "Demo score based on skills, resume sections "
                    "and content length. Not an official ATS score."
                )        

                st.subheader("✅ Resume Scan Results")

                if detected_skills:
                    st.success(
                        f"Detected {len(detected_skills)} skills in your resume."
                    )

                    st.write(", ".join(
                        skill.title() for skill in detected_skills
                    ))

                    if st.button("Use Resume Skills for Matching"):
                        st.session_state["student_skills"] = detected_skills
                        st.session_state["skills_submitted"] = True
                        st.session_state["search_skills"] = ", ".join(
                            detected_skills
                        )
                        st.session_state["search_done"] = True

                        st.success(
                            "Resume skills saved! Internship results are below."
                        )
                        

                else:
                    st.warning(
                        "No listed skills detected. Try another text-based PDF."
                    )

        except Exception as error:
            st.error(f"Could not read this PDF: {error}")

    st.divider()

    if student_name.strip():
        st.caption(
            f"Welcome, {student_name} | Academic Year: {student_year}"
        )

    if not st.session_state.search_done:
        st.info(
            "Enter skills in the sidebar or upload your resume above, "
            "then use the matching button."
        )

    elif df.empty:
        st.warning(
            "There are no records in the internships CSV. "
            "Add records to data/internships.csv."
        )

    else:
        f1, f2, f3 = st.columns(3)

        domains = sorted(x for x in df["domain"].unique() if str(x).strip())
        modes = sorted(x for x in df["work_mode"].unique() if str(x).strip())
        locations = sorted(x for x in df["location"].unique() if str(x).strip())

        with f1:
            domain = st.selectbox("Domain", ["All"] + domains)

        with f2:
            mode = st.selectbox("Work Mode", ["All"] + modes)

        with f3:
            location = st.selectbox("Location", ["All"] + locations)

        filtered = df.copy()

        if domain != "All":
            filtered = filtered[filtered["domain"] == domain]

        if mode != "All":
            filtered = filtered[filtered["work_mode"] == mode]

        if location != "All":
            filtered = filtered[filtered["location"] == location]

        results = recommend_internships(
            filtered, st.session_state.search_skills
        )

        results = results[
            results["match_score"] > 0
        ].reset_index(drop=True)

        st.session_state["matched_internships_df"] = results.drop(
            columns=["match_score"], errors="ignore"
        ).copy()

        total_companies = df.loc[
            df["company"].str.strip() != "", "company"
        ].nunique()

        a, b, c = st.columns(3)
        a.metric("Matching internships", len(results))
        b.metric("Total companies in CSV", total_companies)
        c.metric("All internships in CSV", len(df))

        if results.empty:
            st.warning(
                "No matching internships found. Try changing your skills or filters."
            )
        else:
            st.success(f"{len(results)} matching internship(s) found!")

            for i, (_, row) in enumerate(results.iterrows()):
                with st.container(border=True):
                    show_internship(row, f"result_{i}")



# ================= ENHANCED DASHBOARD =================

elif page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>📊 Internship Insights Dashboard</h1>
        <p>Explore internship, company, domain and stipend data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    dashboard_df = df.copy()

    if st.session_state.get("skills_submitted", False):
        if "matched_internships_df" in st.session_state:
            dashboard_df = st.session_state["matched_internships_df"].copy()

    if student_name.strip():
        st.caption(f"{student_name} | {student_year}")

    if dashboard_df.empty:
        st.warning("No matchinig internship data is available for the dashboard.")

    else:
        companies = dashboard_df.loc[
            dashboard_df["company"].str.strip() != "", "company"
        ].nunique()

        domains_total = dashboard_df.loc[
            dashboard_df["domain"].str.strip() != "", "domain"
        ].nunique()

        avg_stipend = dashboard_df["stipend"].mean()
        paid_count = int((dashboard_df["stipend"] > 0).sum())

        st.subheader("Overview")

        a, b, c, d, e = st.columns(5)
        a.metric("Total Internships", len(dashboard_df))
        b.metric("Unique Companies", companies)
        c.metric("Domains", domains_total)
        d.metric("Average Stipend", f"₹{avg_stipend:,.0f}")
        e.metric("Paid Records", paid_count)

        st.caption(
            "Dashboard counts the full CSV. Find Internships counts only results matching your skills and filters."
        )

        st.divider()
        left, right = st.columns(2)

        
        with left:
            st.subheader("Internships by Domain")

            domain_data = df.copy()

            domain_data["domain"] = (
                domain_data["domain"]
                .fillna("Not specified")
                .replace("", "Not specified")
            )

            counts = (
                domain_data.groupby("domain")
                .size()
                .reset_index(name="Internships")
            )

            fig = px.bar(
                counts,
                x="domain",
                y="Internships",
                text="Internships",
                color="domain",
                title="Internships by Domain"
            )

            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis_title="Domain",
                yaxis_title="Number of Internships",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.subheader("Work Mode Distribution")
            mode_data = dashboard_df.copy()
            mode_data["work_mode"] = mode_data["work_mode"].replace(
                "", "Not specified"
            )
            counts = (
                mode_data["work_mode"].value_counts()
                .rename_axis("Work Mode")
                .reset_index(name="Count")
            )

            fig = px.pie(
                counts,
                names="Work Mode",
                values="Count",
                hole=0.45,
                title="Remote, Hybrid and On-site"
            )
            st.plotly_chart(fig, use_container_width=True)

        left2, right2 = st.columns(2)

        with left2:
            st.subheader("Top Companies")
            company_data = dashboard_df[dashboard_df["company"].str.strip() != ""]

            if company_data.empty:
                st.info("Company names are not available.")
            else:
                counts = (
                    company_data["company"].value_counts()
                    .head(10)
                    .rename_axis("Company")
                    .reset_index(name="Internships")
                )

                fig = px.bar(
                    counts.sort_values("Internships"),
                    x="Internships",
                    y="Company",
                    orientation="h",
                    text="Internships",
                    title="Companies with Most Listings"
                )
                st.plotly_chart(fig, use_container_width=True)

        with right2:
            st.subheader("Stipend Distribution")
            paid_data = dashboard_df[dashboard_df["stipend"] > 0]

            if paid_data.empty:
                st.info("No positive stipend values listed in CSV.")
            else:
                fig = px.histogram(
                    paid_data,
                    x="stipend",
                    nbins=10,
                    title="Listed Stipends",
                    labels={"stipend": "Stipend (₹)"}
                )
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Internship Records")

        search_text = st.text_input(
            "Search by company or internship title",
            placeholder="Enter a name..."
        )

        display_df = dashboard_df.copy()

        if search_text.strip():
            mask = (
                display_df["title"].str.contains(
                    search_text, case=False, regex=False, na=False
                )
                | display_df["company"].str.contains(
                    search_text, case=False, regex=False, na=False
                )
            )
            display_df = display_df[mask]

        st.write(f"Showing {len(display_df)} of {len(dashboard_df)} records.")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="internmatch_internships.csv",
            mime="text/csv"
        )


# ================= SAVED INTERNSHIPS =================

elif page == "Saved Internships":

    st.title("⭐ Saved Internships")
    saved_items = load_saved()

    if not saved_items:
        st.info("No internships have been saved yet.")

    elif df.empty:
        st.info("Saved records are not available in the current CSV.")

    else:
        current_df = df.copy()
        current_df["internship_id"] = current_df.apply(get_key, axis=1)

        saved_df = current_df[
            current_df["internship_id"].isin(saved_items)
        ]

        if saved_df.empty:
            st.info("No saved internships found in the current CSV.")
        else:
            for i, (_, row) in enumerate(saved_df.iterrows()):
                with st.container(border=True):
                    show_internship(row, f"saved_{i}")

                    if st.button("Remove Saved Internship", key=f"remove_{i}"):
                        remaining = [
                            item for item in saved_items
                            if item != get_key(row)
                        ]
                        save_saved(remaining)
                        st.rerun()


# ================= SKILL GAP ANALYSIS =================
elif page == "Skill Gap Analysis":
    st.title("Skill Gap Analysis")
    st.write("Compare your skills with internship requirements.")

    student_skills = st.session_state.get("student_skills", [])

    if isinstance(student_skills, str):
        student_skills = [
            s.strip().lower()
            for s in student_skills.split(",")
            if s.strip()
        ]
    else:
        student_skills = [
            str(s).strip().lower()
            for s in student_skills
            if str(s).strip()
        ]

    if not student_skills:
        st.warning("Please enter your skills first.")
        st.info("Go to Find Internships and submit your skills.")

    elif "matched_internships_df" not in st.session_state:
        st.info("First search for internships on the Find Internships page.")

    else:
        gap_df = st.session_state["matched_internships_df"].copy()

        st.subheader("Your Skills")
        st.write(", ".join(s.title() for s in student_skills))

        if gap_df.empty or "skills" not in gap_df.columns:
            st.warning("No matching internships or skills found.")
        else:
            required_skills = set()

            for skills_text in gap_df["skills"].dropna():
                for skill in str(skills_text).replace(";", ",").split(","):
                    skill = skill.strip().lower()
                    if skill:
                        required_skills.add(skill)

            missing_skills = sorted(required_skills - set(student_skills))

            st.subheader("Your Skill Gap")
            st.metric("Matching Internships", len(gap_df))
            st.metric("Required Skills", len(required_skills))
            st.metric("Skills to Learn", len(missing_skills))

            if not required_skills:
                st.info("No required skills found.")
            elif not missing_skills:
                st.success("You have all the listed required skills!")
            else:
                st.write("Skills you may need to learn:")
                for skill in missing_skills:
                    st.write("- " + skill.title())

            st.caption("Search again after changing your skills.")

# AI Career Assistant Chatbot
st.divider()
st.header("🤖 AI Career Assistant")
st.write("Ask me about internships, skills, resumes, and interview preparation.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("Ask your career question...")

if user_question:
    st.session_state.chat_messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    try:
        client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"]
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.models.generate_content(
                    
                    model="gemini-3.8-flash",
                    contents=(
                        "You are InternMatch AI, a concise career assistant "
                        "for college students. "
                        "Answer in simple language in 3-5 bullet points. "
                        "Do not claim to check live internship listings.\n\n"
                        "Question: " + user_question
                    )
                )
                

                answer = response.text or "Please try asking your question again."
                st.markdown(answer)

        st.session_state.chat_messages.append(
            {"role": "assistant", "content": answer}
        )

    except Exception as e:
        st.error("Gemini AI Error: " + str(e))
       