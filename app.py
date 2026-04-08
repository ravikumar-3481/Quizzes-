import streamlit as st
import pandas as pd
import json
import re
import io
import random

# ==========================================
# World-Class ML & DL Integration
# ==========================================
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

class MLEngine:
    """Handles advanced ML and DL tasks for the MCQ Platform."""
    
    @staticmethod
    def remove_duplicates(questions, threshold=0.85):
        """Uses TF-IDF and Cosine Similarity to remove semantically duplicate questions."""
        if not SKLEARN_AVAILABLE or len(questions) < 2:
            return questions
            
        texts = [q['question'] for q in questions]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        unique_questions = []
        to_skip = set()
        
        for i in range(len(texts)):
            if i in to_skip:
                continue
            unique_questions.append(questions[i])
            for j in range(i + 1, len(texts)):
                if cosine_sim[i][j] >= threshold:
                    to_skip.add(j) # Mark as duplicate
                    
        return unique_questions

    @staticmethod
    @st.cache_resource
    def load_dl_model():
        """Loads a Deep Learning Zero-Shot classifier for difficulty prediction."""
        if TRANSFORMERS_AVAILABLE:
            # Using a lightweight model for speed in Streamlit
            return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        return None

    @staticmethod
    def predict_difficulty(question_text, dl_model):
        """Predicts question difficulty using a DL model or NLP heuristics."""
        if TRANSFORMERS_AVAILABLE and dl_model:
            try:
                result = dl_model(
                    question_text, 
                    candidate_labels=["easy", "medium", "hard"],
                )
                return result['labels'][0].capitalize()
            except:
                pass
        
        # Fallback NLP Heuristic based on text length and complex words
        words = question_text.split()
        if len(words) > 25 or any(len(w) > 10 for w in words):
            return "Hard"
        elif len(words) > 12:
            return "Medium"
        return "Easy"


# ==========================================
# Parsing Engine
# ==========================================
class FileParser:
    """Robust parsing engine supporting CSV, JSON, TXT, and PDF."""
    
    @staticmethod
    def parse_csv(file_bytes):
        df = pd.read_csv(io.BytesIO(file_bytes))
        questions = []
        df.columns = [col.strip().lower() for col in df.columns]
        
        for _, row in df.iterrows():
            q_col = next((col for col in df.columns if 'question' in col), None)
            ans_col = next((col for col in df.columns if 'answer' in col or 'correct' in col), None)
            opt_cols = [col for col in df.columns if 'option' in col or col in ['a', 'b', 'c', 'd']]
            
            if q_col and ans_col and len(opt_cols) >= 2:
                options = [str(row[opt]) for opt in opt_cols if pd.notna(row[opt])]
                answer = str(row[ans_col]).strip()
                if len(answer) == 1 and answer.lower() in ['a', 'b', 'c', 'd']:
                    idx = ord(answer.lower()) - 97
                    if idx < len(options):
                        answer = options[idx]
                questions.append({"question": str(row[q_col]), "options": options, "answer": answer})
        return questions

    @staticmethod
    def parse_json(file_bytes):
        data = json.loads(file_bytes.decode('utf-8'))
        questions = []
        for item in data:
            if "question" in item and "options" in item and "answer" in item:
                questions.append({"question": item["question"], "options": item["options"], "answer": item["answer"]})
        return questions

    @staticmethod
    def parse_txt_heuristics(text):
        questions = []
        q_pattern = re.compile(r'(?:Q\d*:?|Question\d*:?)\s*(.*?)(?=(?:Q\d*:?|Question\d*:?)|\Z)', re.IGNORECASE | re.DOTALL)
        blocks = q_pattern.findall(text)
        
        for block in blocks:
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            if not lines: continue
            
            question_text = lines[0]
            options = []
            answer = ""
            
            for line in lines[1:]:
                opt_match = re.match(r'^[a-d][\.\)]\s*(.*)', line, re.IGNORECASE)
                if opt_match:
                    options.append(opt_match.group(1).strip())
                
                ans_match = re.match(r'^(?:Answer|Ans|Correct)[\:\s]*([a-d])', line, re.IGNORECASE)
                if ans_match:
                    ans_letter = ans_match.group(1).lower()
                    idx = ord(ans_letter) - 97
                    if idx < len(options):
                        answer = options[idx]
                else:
                    ans_text_match = re.match(r'^(?:Answer|Ans|Correct)[\:\s]*(.*)', line, re.IGNORECASE)
                    if ans_text_match:
                        answer = ans_text_match.group(1).strip()
            
            if question_text and len(options) >= 2 and answer:
                questions.append({"question": question_text, "options": options, "answer": answer})
        return questions
        
    @staticmethod
    def parse_pdf(file_bytes):
        if not PDF_AVAILABLE:
            st.warning("PyPDF2 is not installed. Skipping PDF processing.")
            return []
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return FileParser.parse_txt_heuristics(text)

# ==========================================
# Streamlit UI Configuration & State Management
# ==========================================
st.set_page_config(page_title="AI MCQ Platform", page_icon="🤖", layout="centered")

# Custom CSS for better button aesthetics
st.markdown("""
    <style>
    div.stButton > button:first-child {
        height: 3em;
        font-size: 16px;
        font-weight: 500;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session states
state_vars = ['questions', 'current_q_index', 'score', 'answered', 'selected_option', 'quiz_started']
for var in state_vars:
    if var not in st.session_state:
        st.session_state[var] = [] if var == 'questions' else (0 if var in ['current_q_index', 'score'] else (None if var == 'selected_option' else False))

# ==========================================
# Main UI Layout (No Sidebar)
# ==========================================
st.title("🤖 World-Class AI Assessment Platform")
st.markdown("Upload documents and let our **Deep Learning & ML engines** extract, deduplicate, and analyze questions.")

if not st.session_state.quiz_started:
    st.info("💡 **Pro Tip:** Install `scikit-learn`, `transformers`, `torch`, and `PyPDF2` to unlock full AI capabilities.")
    
    uploaded_files = st.file_uploader(
        "Upload Assessment Files (PDF, CSV, JSON, TXT)", 
        type=['csv', 'json', 'txt', 'pdf'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🚀 Process Files & Initialize AI", type="primary", use_container_width=True):
            all_questions = []
            
            # Show progress during heavy ML tasks
            with st.status("Engaging AI Core...", expanded=True) as status:
                st.write("📄 Parsing documents...")
                for file in uploaded_files:
                    bytes_data = file.getvalue()
                    if file.name.endswith('.csv'):
                        all_questions.extend(FileParser.parse_csv(bytes_data))
                    elif file.name.endswith('.json'):
                        all_questions.extend(FileParser.parse_json(bytes_data))
                    elif file.name.endswith('.txt'):
                        all_questions.extend(FileParser.parse_txt_heuristics(bytes_data.decode('utf-8')))
                    elif file.name.endswith('.pdf'):
                        all_questions.extend(FileParser.parse_pdf(bytes_data))
                
                initial_count = len(all_questions)
                st.write(f"🔍 Found {initial_count} raw questions. Running ML deduplication...")
                all_questions = MLEngine.remove_duplicates(all_questions)
                final_count = len(all_questions)
                
                st.write("🧠 Loading Deep Learning models to analyze difficulty...")
                dl_model = MLEngine.load_dl_model()
                
                for q in all_questions:
                    q['difficulty'] = MLEngine.predict_difficulty(q['question'], dl_model)
                
                status.update(label=f"Analysis Complete! Removed {initial_count - final_count} duplicates.", state="complete", expanded=False)
            
            if all_questions:
                random.shuffle(all_questions)
                st.session_state.questions = all_questions
                st.session_state.quiz_started = True
                st.rerun()
            else:
                st.error("No valid questions could be extracted. Please check file formats.")

# --- QUIZ EXECUTION SECTION ---
else:
    total_q = len(st.session_state.questions)
    
    if st.session_state.current_q_index < total_q:
        q_num = st.session_state.current_q_index + 1
        
        # Top Dashboard
        st.progress(q_num / total_q)
        dash_col1, dash_col2, dash_col3 = st.columns(3)
        dash_col1.metric("Progress", f"Q {q_num} of {total_q}")
        dash_col2.metric("Score", f"{st.session_state.score}")
        
        current_q = st.session_state.questions[st.session_state.current_q_index]
        difficulty_color = {"Easy": "green", "Medium": "orange", "Hard": "red"}.get(current_q.get('difficulty', 'Medium'), "gray")
        dash_col3.markdown(f"**Difficulty:** <span style='color:{difficulty_color}'>{current_q.get('difficulty', 'Unknown')}</span>", unsafe_allow_html=True)
            
        st.divider()
        st.subheader(current_q['question'])
        st.write("") # Spacer
        
        # Display Options
        if not st.session_state.answered:
            st.markdown("### Choose your answer:")
            for idx, option in enumerate(current_q['options']):
                if st.button(f"{chr(65+idx)}. {option}", key=f"opt_{q_num}_{idx}", use_container_width=True):
                    st.session_state.selected_option = option
                    st.session_state.answered = True
                    
                    if option.strip().lower() == current_q['answer'].strip().lower():
                        st.session_state.score += 1
                    st.rerun()
                    
        # Check Answer Results & Next Button
        else:
            correct_ans = current_q['answer']
            user_ans = st.session_state.selected_option
            
            st.markdown("### Results:")
            if user_ans.strip().lower() == correct_ans.strip().lower():
                st.success(f"🎉 **Correct!** You selected: {user_ans}")
            else:
                st.error(f"❌ **Incorrect.** You selected: {user_ans}")
                st.info(f"💡 **The correct answer is:** {correct_ans}")
            
            st.write("")
            if st.button("Continue to Next Question ➡️", type="primary", use_container_width=True):
                st.session_state.current_q_index += 1
                st.session_state.answered = False
                st.session_state.selected_option = None
                st.rerun()
                
    # --- END OF QUIZ SECTION ---
    else:
        st.balloons()
        st.header("🏆 Assessment Completed!")
        
        percentage = (st.session_state.score / total_q) * 100
        
        col1, col2 = st.columns(2)
        col1.metric(label="Final Score", value=f"{st.session_state.score} / {total_q}")
        col2.metric(label="Accuracy", value=f"{percentage:.1f}%")
        
        if percentage >= 80:
            st.success("Exceptional! Your knowledge is solid.")
        elif percentage >= 50:
            st.warning("Good effort! Review the areas you missed.")
        else:
            st.error("Keep studying! You'll get it next time.")
            
        st.divider()
        if st.button("🔄 Start New Assessment", use_container_width=True, type="primary"):
            for var in state_vars:
                st.session_state[var] = [] if var == 'questions' else (0 if var in ['current_q_index', 'score'] else (None if var == 'selected_option' else False))
            st.rerun()
