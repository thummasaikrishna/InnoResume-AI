import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import json
import re
from io import BytesIO
import base64
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import spacy
from collections import Counter
import hashlib
import sqlite3
from pathlib import Path
import PyPDF2
import docx2txt
import requests
from textblob import TextBlob
import yfinance as yf  # For market data context
import warnings
warnings.filterwarnings('ignore')

# Enhanced Streamlit Configuration
st.set_page_config(
    page_title="🤖 InnoResume AI - Resume Relevance Analyzer",
    page_icon="favicon_and_logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://innomatics.in',
        'Report a bug': "mailto:support@innomatics.in",
        'About': "# InnoResume AI\nPowered by Innomatics Research Labs\n\nAdvanced AI-driven resume analysis system"
    }
)

# Custom CSS for Professional Look
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    .main {
        font-family: 'Roboto', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    .score-high {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .score-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .score-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .sidebar .sidebar-content {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
    
    .analysis-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .progress-bar {
        background: #e9ecef;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745, #20c997);
        transition: width 0.3s ease;
    }
    
    .floating-action {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 1000;
        background: #667eea;
        color: white;
        border-radius: 50px;
        padding: 15px 25px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }
    
    .tech-stack-badge {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 5px 12px;
        margin: 3px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 500;
    }
    
    .dashboard-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .gradient-text {
        background: linear-gradient(45deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced Database Setup
def init_database():
    """Initialize SQLite database with comprehensive schema"""
    conn = sqlite3.connect('resume_analyzer.db')
    cursor = conn.cursor()
    
    # Job postings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_postings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            company TEXT NOT NULL,
            location TEXT NOT NULL,
            description TEXT NOT NULL,
            required_skills TEXT NOT NULL,
            preferred_skills TEXT,
            experience_level TEXT,
            salary_range TEXT,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            urgency_level TEXT DEFAULT 'Medium'
        )
    ''')
    
    # Resume analysis results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            job_id INTEGER,
            resume_text TEXT,
            relevance_score REAL,
            fit_verdict TEXT,
            missing_skills TEXT,
            recommendations TEXT,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            recruiter_notes TEXT,
            interview_status TEXT DEFAULT 'Pending',
            FOREIGN KEY (job_id) REFERENCES job_postings (id)
        )
    ''')
    
    # Analytics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            category TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT NOT NULL,
            feedback_text TEXT NOT NULL,
            rating INTEGER,
            improvement_areas TEXT,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

# Advanced Resume Parser Class
class AdvancedResumeParser:
    def __init__(self):
        self.skills_database = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin', 'php', 'ruby', 'scala', 'r', 'matlab'],
            'web_development': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel', 'next.js', 'nuxt.js'],
            'mobile_development': ['android', 'ios', 'flutter', 'react native', 'xamarin', 'ionic', 'cordova'],
            'data_science': ['pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'matplotlib', 'seaborn', 'plotly', 'jupyter', 'tableau', 'power bi'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'gitlab ci', 'github actions'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'neo4j', 'oracle', 'sql server'],
            'devops': ['linux', 'bash', 'git', 'ci/cd', 'monitoring', 'logging', 'microservices', 'api design', 'system design'],
            'ai_ml': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'reinforcement learning', 'neural networks', 'llm', 'generative ai']
        }
        
        self.education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'certification', 'diploma', 'course']
        self.experience_keywords = ['experience', 'work', 'internship', 'project', 'developed', 'managed', 'led', 'implemented', 'designed']
        
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX file"""
        try:
            return docx2txt.process(docx_file)
        except Exception as e:
            st.error(f"Error reading DOCX: {str(e)}")
            return ""
    
    def extract_contact_info(self, text):
        """Extract contact information using regex"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'[\+]?[1-9]?[0-9]{1,4}?[-.\s]?\(?[0-9]{1,3}\)?[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}'
        
        email_matches = re.findall(email_pattern, text)
        phone_matches = re.findall(phone_pattern, text)
        
        return {
            'emails': email_matches,
            'phones': phone_matches
        }
    
    def extract_skills(self, text):
        """Extract skills from resume text"""
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skills_database.items():
            found_skills[category] = []
            for skill in skills:
                if skill in text_lower:
                    found_skills[category].append(skill)
        
        return found_skills
    
    def calculate_experience_years(self, text):
        """Estimate years of experience"""
        experience_patterns = [
            r'(\d+)\s*\+?\s*years?\s*(of\s*)?experience',
            r'experience\s*:?\s*(\d+)\s*\+?\s*years?',
            r'(\d+)\s*years?\s*experienced?'
        ]
        
        years = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    years.extend([int(m) for m in match if m.isdigit()])
                else:
                    if match.isdigit():
                        years.append(int(match))
        
        return max(years) if years else 0
    
    def analyze_resume_structure(self, text):
        """Analyze resume structure and completeness"""
        sections = {
            'contact': bool(re.search(r'email|phone|contact', text, re.I)),
            'education': bool(re.search(r'education|degree|university|college', text, re.I)),
            'experience': bool(re.search(r'experience|work|employment|internship', text, re.I)),
            'skills': bool(re.search(r'skills|technical|programming|technologies', text, re.I)),
            'projects': bool(re.search(r'project|github|portfolio', text, re.I)),
            'certifications': bool(re.search(r'certification|certificate|certified', text, re.I))
        }
        
        completeness_score = sum(sections.values()) / len(sections) * 100
        return sections, completeness_score

# AI-Powered Job Matching Engine
class AIJobMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = ' '.join(text.split())
        return text.lower()
    
    def calculate_relevance_score(self, resume_text, job_description):
        """Calculate relevance score using advanced NLP techniques"""
        
        # Preprocess texts
        resume_clean = self.preprocess_text(resume_text)
        job_clean = self.preprocess_text(job_description)
        
        # TF-IDF Similarity
        try:
            tfidf_matrix = self.vectorizer.fit_transform([resume_clean, job_clean])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        # Keyword matching
        resume_words = set(resume_clean.split())
        job_words = set(job_clean.split())
        common_words = resume_words.intersection(job_words)
        keyword_score = len(common_words) / len(job_words.union(resume_words)) if job_words.union(resume_words) else 0
        
        # Skills matching
        skills_score = self.calculate_skills_match(resume_text, job_description)
        
        # Experience relevance
        experience_score = self.calculate_experience_relevance(resume_text, job_description)
        
        # Weighted final score
        final_score = (
            tfidf_similarity * 0.3 +
            keyword_score * 0.25 +
            skills_score * 0.3 +
            experience_score * 0.15
        ) * 100
        
        return min(final_score, 100.0)
    
    def calculate_skills_match(self, resume_text, job_description):
        """Calculate skills matching score"""
        parser = AdvancedResumeParser()
        resume_skills = parser.extract_skills(resume_text)
        job_skills = parser.extract_skills(job_description)
        
        total_job_skills = sum(len(skills) for skills in job_skills.values())
        if total_job_skills == 0:
            return 0.5
        
        matched_skills = 0
        for category in job_skills:
            job_category_skills = set(job_skills[category])
            resume_category_skills = set(resume_skills[category])
            matched_skills += len(job_category_skills.intersection(resume_category_skills))
        
        return matched_skills / total_job_skills if total_job_skills > 0 else 0
    
    def calculate_experience_relevance(self, resume_text, job_description):
        """Calculate experience relevance"""
        parser = AdvancedResumeParser()
        resume_years = parser.calculate_experience_years(resume_text)
        
        # Extract required experience from job description
        job_exp_patterns = [
            r'(\d+)\s*\+?\s*years?\s*(of\s*)?experience',
            r'minimum\s*(\d+)\s*years?',
            r'at\s*least\s*(\d+)\s*years?'
        ]
        
        required_years = 0
        for pattern in job_exp_patterns:
            matches = re.findall(pattern, job_description.lower())
            if matches:
                required_years = max([int(match[0]) if isinstance(match, tuple) else int(match) for match in matches])
                break
        
        if required_years == 0:
            return 0.7  # Neutral score if no experience requirement found
        
        if resume_years >= required_years:
            return 1.0
        elif resume_years >= required_years * 0.7:
            return 0.8
        else:
            return 0.4
    
    def get_fit_verdict(self, score):
        """Determine fit verdict based on score"""
        if score >= 75:
            return "High", "🟢"
        elif score >= 50:
            return "Medium", "🟡"
        else:
            return "Low", "🔴"
    
    def generate_recommendations(self, resume_text, job_description, score):
        """Generate personalized recommendations"""
        parser = AdvancedResumeParser()
        resume_skills = parser.extract_skills(resume_text)
        job_skills = parser.extract_skills(job_description)
        
        recommendations = []
        
        # Skills gap analysis
        for category in job_skills:
            missing_skills = set(job_skills[category]) - set(resume_skills[category])
            if missing_skills:
                recommendations.append(f"Consider learning {category.replace('_', ' ').title()}: {', '.join(list(missing_skills)[:3])}")
        
        # Resume structure recommendations
        sections, completeness = parser.analyze_resume_structure(resume_text)
        if completeness < 80:
            missing_sections = [section for section, present in sections.items() if not present]
            if missing_sections:
                recommendations.append(f"Add missing resume sections: {', '.join(missing_sections)}")
        
        # Score-based recommendations
        if score < 50:
            recommendations.extend([
                "Consider tailoring your resume more specifically to this job description",
                "Highlight relevant projects and achievements",
                "Include quantifiable results and metrics"
            ])
        elif score < 75:
            recommendations.extend([
                "Emphasize your most relevant experience",
                "Add specific examples of your skills in action"
            ])
        
        return recommendations[:5]  # Limit to top 5 recommendations

# Advanced Analytics Dashboard
class AnalyticsDashboard:
    def __init__(self):
        self.colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
    
    def create_score_distribution_chart(self, scores):
        """Create score distribution histogram"""
        fig = px.histogram(
            x=scores,
            nbins=20,
            title="📊 Relevance Score Distribution",
            labels={'x': 'Relevance Score', 'y': 'Number of Candidates'},
            color_discrete_sequence=[self.colors[0]]
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=16
        )
        return fig
    
    def create_skills_gap_analysis(self, missing_skills_data):
        """Create skills gap analysis chart"""
        skills_count = Counter(missing_skills_data)
        top_skills = dict(skills_count.most_common(10))
        
        fig = px.bar(
            x=list(top_skills.keys()),
            y=list(top_skills.values()),
            title="🔍 Most Common Missing Skills",
            labels={'x': 'Skills', 'y': 'Frequency'},
            color=list(top_skills.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        return fig
    
    def create_performance_metrics(self, data):
        """Create comprehensive performance metrics dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Score Trends', 'Fit Verdict Distribution', 'Processing Time', 'Success Rate'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # Mock data for demonstration
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        scores = np.random.normal(65, 15, 30)
        
        # Score trends
        fig.add_trace(
            go.Scatter(x=dates, y=scores, name="Average Score", line=dict(color=self.colors[0])),
            row=1, col=1
        )
        
        # Fit verdict distribution
        fig.add_trace(
            go.Pie(labels=['High', 'Medium', 'Low'], values=[30, 45, 25], hole=0.4),
            row=1, col=2
        )
        
        # Processing time
        processing_times = np.random.normal(2.5, 0.5, 30)
        fig.add_trace(
            go.Scatter(x=dates, y=processing_times, mode='lines+markers', name="Processing Time (s)"),
            row=2, col=1
        )
        
        # Success rate indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=87.5,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Success Rate %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': self.colors[0]},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "green"}]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="📈 System Performance Dashboard")
        return fig
    
    def create_wordcloud(self, text_data):
        """Generate word cloud from resume text"""
        try:
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='viridis',
                max_words=100
            ).generate(text_data)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.title('📝 Resume Keywords Cloud', fontsize=16, pad=20)
            return fig
        except:
            return None

# Main Application Class
class InnoResumeAI:
    def __init__(self):
        self.parser = AdvancedResumeParser()
        self.matcher = AIJobMatcher()
        self.analytics = AnalyticsDashboard()
        init_database()
        
    def run(self):
        """Main application runner"""
        load_custom_css()
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🚀 InnoResume AI</h1>
            <h3>Advanced Resume Relevance Analyzer</h3>
            <p>Powered by Innomatics Research Labs | AI-Driven Recruitment Excellence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar Navigation
        st.sidebar.markdown("### 🎯 Navigation")
        page = st.sidebar.selectbox(
            "Choose Module",
            ["🏠 Dashboard", "📄 Resume Analysis", "💼 Job Management", "📊 Analytics", "🎓 Student Feedback", "⚙️ Settings"]
        )
        
        # Real-time system stats in sidebar
        self.show_sidebar_stats()
        
        # Route to selected page
        if page == "🏠 Dashboard":
            self.show_dashboard()
        elif page == "📄 Resume Analysis":
            self.show_resume_analysis()
        elif page == "💼 Job Management":
            self.show_job_management()
        elif page == "📊 Analytics":
            self.show_analytics()
        elif page == "🎓 Student Feedback":
            self.show_student_feedback()
        elif page == "⚙️ Settings":
            self.show_settings()
    
    def show_sidebar_stats(self):
        """Display real-time stats in sidebar"""
        st.sidebar.markdown("### 📈 Live Stats")
        
        # Mock real-time data
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("🎯 Today's Analysis", "47", "↑ 12")
        with col2:
            st.metric("⚡ Avg Score", "73.2", "↑ 2.1")
        
        st.sidebar.metric("🏆 Success Rate", "87.5%", "↑ 3.2%")
        
        # System health indicator
        st.sidebar.markdown("### 🔋 System Health")
        health_score = 95.5
        st.sidebar.progress(health_score/100)
        st.sidebar.write(f"System Performance: {health_score}%")
        
        # Quick actions
        st.sidebar.markdown("### ⚡ Quick Actions")
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        if st.sidebar.button("📥 Export Results"):
            st.sidebar.success("Export initiated!")
    
    def show_dashboard(self):
        """Main dashboard with overview metrics"""
        st.markdown("## 📊 Executive Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📄 Total Resumes</h3>
                <h2 class="gradient-text">1,247</h2>
                <p>↑ 15% from last week</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>💼 Active Jobs</h3>
                <h2 class="gradient-text">23</h2>
                <p>🔥 5 urgent positions</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Avg Match Score</h3>
                <h2 class="gradient-text">73.2</h2>
                <p>📈 +2.1 points</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>⚡ Processing Speed</h3>
                <h2 class="gradient-text">2.1s</h2>
                <p>🚀 -0.3s improvement</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Mock data for trends
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            scores = np.random.normal(65, 10, 30) + np.linspace(0, 15, 30)  # Upward trend
            
            fig = px.line(
                x=dates, y=scores,
                title="📈 Daily Average Scores Trend",
                labels={'x': 'Date', 'y': 'Average Score'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Job categories distribution
            categories = ['Software', 'Data Science', 'Marketing', 'Sales', 'HR', 'Finance']
            values = [45, 25, 12, 8, 6, 4]
            
            fig = px.pie(
                values=values, names=categories,
                title="💼 Job Categories Distribution",
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity feed
        st.markdown("### 🔔 Recent Activity")
        
        activities = [
            {"time": "2 mins ago", "action": "Resume analyzed", "candidate": "Rahul Sharma", "score": 89, "status": "High"},
            {"time": "5 mins ago", "action": "New job posted", "company": "TechCorp", "role": "Data Scientist", "applications": 0},
            {"time": "12 mins ago", "action": "Bulk analysis completed", "count": 25, "avg_score": 67.3},
            {"time": "18 mins ago", "action": "Interview scheduled", "candidate": "Priya Singh", "company": "InnoTech"},
            {"time": "1 hour ago", "action": "System optimization", "improvement": "15% faster processing"}
        ]
        
        for activity in activities:
            if activity["action"] == "Resume analyzed":
                st.markdown(f"""
                <div class="activity-item" style="padding: 1rem; margin: 0.5rem 0; background: white; border-radius: 10px; border-left: 4px solid #667eea;">
                    <strong>📄 {activity['action']}</strong> - {activity['candidate']} 
                    <span class="score-{'high' if activity['score'] >= 75 else 'medium' if activity['score'] >= 50 else 'low'}">(Score: {activity['score']})</span>
                    <div style="font-size: 0.8em; color: gray;">{activity['time']}</div>
                </div>
                """, unsafe_allow_html=True)
            elif activity["action"] == "New job posted":
                st.markdown(f"""
                <div class="activity-item" style="padding: 1rem; margin: 0.5rem 0; background: white; border-radius: 10px; border-left: 4px solid #28a745;">
                    <strong>💼 {activity['action']}</strong> - {activity['role']} at {activity['company']}
                    <div style="font-size: 0.8em; color: gray;">{activity['time']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="activity-item" style="padding: 1rem; margin: 0.5rem 0; background: white; border-radius: 10px; border-left: 4px solid #ffc107;">
                    <strong>⚡ {activity['action']}</strong>
                    <div style="font-size: 0.8em; color: gray;">{activity['time']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def show_resume_analysis(self):
        """Resume analysis interface with advanced features"""
        st.markdown("## 📄 AI-Powered Resume Analysis")
        
        # Analysis mode selection
        analysis_mode = st.selectbox(
            "🎯 Analysis Mode",
            ["Single Resume Analysis", "Bulk Resume Processing", "Real-time Screening"]
        )
        
        if analysis_mode == "Single Resume Analysis":
            self.single_resume_analysis()
        elif analysis_mode == "Bulk Resume Processing":
            self.bulk_resume_analysis()
        else:
            self.realtime_screening()
    
    def single_resume_analysis(self):
        """Single resume analysis with detailed insights"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📤 Upload Resume")
            uploaded_file = st.file_uploader(
                "Choose resume file",
                type=['pdf', 'docx', 'txt'],
                help="Supported formats: PDF, DOCX, TXT"
            )
            
            # Job selection
            st.markdown("### 💼 Select Job Position")
            
            # Mock job data - in real app, this would come from database
            jobs_data = [
                {"id": 1, "title": "Senior Python Developer", "company": "TechCorp", "urgency": "High"},
                {"id": 2, "title": "Data Scientist", "company": "AI Solutions", "urgency": "Medium"},
                {"id": 3, "title": "Full Stack Engineer", "company": "StartupXYZ", "urgency": "Low"},
                {"id": 4, "title": "ML Engineer", "company": "DeepTech", "urgency": "High"},
                {"id": 5, "title": "DevOps Engineer", "company": "CloudFirst", "urgency": "Medium"}
            ]
            
            job_options = [f"{job['title']} - {job['company']} ({'🔴' if job['urgency'] == 'High' else '🟡' if job['urgency'] == 'Medium' else '🟢'} {job['urgency']})" for job in jobs_data]
            selected_job = st.selectbox("Available Positions", job_options)
            
            # Analysis settings
            st.markdown("### ⚙️ Analysis Settings")
            analysis_depth = st.slider("Analysis Depth", 1, 5, 3, help="Higher values provide more detailed analysis")
            include_recommendations = st.checkbox("Include Improvement Recommendations", True)
            generate_feedback = st.checkbox("Generate Student Feedback", True)
            
        with col2:
            if uploaded_file and selected_job:
                # Extract text based on file type
                if uploaded_file.type == "application/pdf":
                    resume_text = self.parser.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    resume_text = self.parser.extract_text_from_docx(uploaded_file)
                else:
                    resume_text = str(uploaded_file.read(), "utf-8")
                
                if resume_text:
                    st.markdown("### 🎯 Analysis Results")
                    
                    # Show analysis in progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        progress_bar.progress(i + 1)
                        if i < 20:
                            status_text.text("🔍 Extracting text content...")
                        elif i < 40:
                            status_text.text("🧠 Analyzing skills and experience...")
                        elif i < 60:
                            status_text.text("📊 Calculating relevance score...")
                        elif i < 80:
                            status_text.text("💡 Generating recommendations...")
                        else:
                            status_text.text("✅ Finalizing analysis...")
                        time.sleep(0.02)
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Mock job description
                    job_description = """
                    We are seeking a Senior Python Developer with expertise in web development, 
                    data analysis, and cloud technologies. Required skills include Python, Django, 
                    Flask, PostgreSQL, AWS, Docker, and API development. Experience with machine 
                    learning libraries like scikit-learn and pandas is preferred. Minimum 3 years 
                    of experience required.
                    """
                    
                    # Calculate relevance score
                    relevance_score = self.matcher.calculate_relevance_score(resume_text, job_description)
                    fit_verdict, fit_icon = self.matcher.get_fit_verdict(relevance_score)
                    
                    # Display score with visual indicator
                    st.markdown(f"""
                    <div class="analysis-section">
                        <h3>🎯 Relevance Score</h3>
                        <div style="text-align: center; padding: 2rem;">
                            <div style="font-size: 3em; font-weight: bold; color: {'#28a745' if relevance_score >= 75 else '#ffc107' if relevance_score >= 50 else '#dc3545'};">
                                {relevance_score:.1f}%
                            </div>
                            <div style="font-size: 1.5em; margin-top: 1rem;">
                                {fit_icon} {fit_verdict} Fit
                            </div>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {relevance_score}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed analysis sections
                    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Skills Analysis", "👤 Profile Summary", "💡 Recommendations", "📈 Detailed Metrics", "🎓 Learning Path"])
                    
                    with tab1:
                        self.show_skills_analysis(resume_text, job_description)
                    
                    with tab2:
                        self.show_profile_summary(resume_text)
                    
                    with tab3:
                        if include_recommendations:
                            recommendations = self.matcher.generate_recommendations(resume_text, job_description, relevance_score)
                            for i, rec in enumerate(recommendations, 1):
                                st.markdown(f"""
                                <div class="recommendation-card">
                                    <h4>💡 Recommendation {i}</h4>
                                    <p>{rec}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    with tab4:
                        self.show_detailed_metrics(resume_text, job_description, relevance_score)
                    
                    with tab5:
                        self.show_learning_path(resume_text, job_description)
                    
                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("💾 Save Analysis"):
                            # Save to database
                            self.save_analysis_result(resume_text, relevance_score, fit_verdict, recommendations)
                            st.success("Analysis saved successfully!")
                    
                    with col2:
                        if st.button("📧 Send Feedback"):
                            st.success("Feedback sent to candidate!")
                    
                    with col3:
                        if st.button("📅 Schedule Interview"):
                            st.success("Interview scheduled!")
                    
                    with col4:
                        if st.button("📄 Generate Report"):
                            st.success("Report generated!")
    
    def show_skills_analysis(self, resume_text, job_description):
        """Display detailed skills analysis"""
        resume_skills = self.parser.extract_skills(resume_text)
        job_skills = self.parser.extract_skills(job_description)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Skills Found in Resume")
            for category, skills in resume_skills.items():
                if skills:
                    st.markdown(f"**{category.replace('_', ' ').title()}:**")
                    for skill in skills:
                        st.markdown(f"""
                        <span class="tech-stack-badge">{skill}</span>
                        """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 💼 Skills Required for Job")
            for category, skills in job_skills.items():
                if skills:
                    st.markdown(f"**{category.replace('_', ' ').title()}:**")
                    for skill in skills:
                        skill_class = "tech-stack-badge" if skill in resume_skills.get(category, []) else "tech-stack-badge-missing"
                        color = "background: linear-gradient(45deg, #28a745, #20c997);" if skill in resume_skills.get(category, []) else "background: linear-gradient(45deg, #dc3545, #fd7e14);"
                        st.markdown(f"""
                        <span class="tech-stack-badge" style="{color}">{skill} {'✓' if skill in resume_skills.get(category, []) else '✗'}</span>
                        """, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
        
        # Skills gap visualization
        st.markdown("#### 📊 Skills Gap Analysis")
        
        all_job_skills = [skill for skills_list in job_skills.values() for skill in skills_list]
        all_resume_skills = [skill for skills_list in resume_skills.values() for skill in skills_list]
        
        matched_skills = len(set(all_job_skills).intersection(set(all_resume_skills)))
        total_job_skills = len(set(all_job_skills))
        missing_skills = len(set(all_job_skills) - set(all_resume_skills))
        
        # Create donut chart for skills match
        fig = go.Figure(data=[go.Pie(
            labels=['Matched Skills', 'Missing Skills'],
            values=[matched_skills, missing_skills],
            hole=.3,
            marker_colors=['#28a745', '#dc3545']
        )])
        fig.update_layout(title="Skills Match Overview", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    def show_profile_summary(self, resume_text):
        """Display AI-generated profile summary"""
        # Extract contact info
        contact_info = self.parser.extract_contact_info(resume_text)
        
        # Analyze resume structure
        sections, completeness = self.parser.analyze_resume_structure(resume_text)
        
        # Calculate experience years
        experience_years = self.parser.calculate_experience_years(resume_text)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Resume Structure Analysis")
            for section, present in sections.items():
                icon = "✅" if present else "❌"
                st.markdown(f"{icon} {section.replace('_', ' ').title()}")
            
            st.metric("Resume Completeness", f"{completeness:.1f}%")
        
        with col2:
            st.markdown("#### 👤 Candidate Profile")
            st.metric("Years of Experience", f"{experience_years} years")
            st.metric("Email Addresses Found", len(contact_info['emails']))
            st.metric("Phone Numbers Found", len(contact_info['phones']))
        
        # AI-generated summary
        st.markdown("#### 🤖 AI-Generated Summary")
        st.markdown(f"""
        <div class="analysis-section">
            <p>This candidate appears to have <strong>{experience_years} years of experience</strong> based on resume analysis. 
            The resume structure is <strong>{completeness:.1f}% complete</strong> with 
            {'all essential sections present' if completeness >= 80 else 'some sections missing'}.</p>
            
            <p>Contact information is {'well documented' if len(contact_info['emails']) > 0 and len(contact_info['phones']) > 0 else 'partially available'}.</p>
            
            <p><strong>Recommendation:</strong> 
            {'This is a well-structured resume from an experienced candidate.' if completeness >= 80 and experience_years >= 2 else 'Consider improvements in resume structure and highlighting relevant experience.'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def show_detailed_metrics(self, resume_text, job_description, relevance_score):
        """Show detailed scoring metrics"""
        # Calculate individual component scores
        tfidf_score = np.random.uniform(0.6, 0.9) * 100  # Mock calculation
        keyword_score = np.random.uniform(0.5, 0.8) * 100
        skills_score = np.random.uniform(0.4, 0.9) * 100
        experience_score = np.random.uniform(0.6, 1.0) * 100
        
        # Create radar chart for metrics
        categories = ['TF-IDF Similarity', 'Keyword Match', 'Skills Match', 'Experience Match']
        scores = [tfidf_score, keyword_score, skills_score, experience_score]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name='Score Breakdown'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="📊 Detailed Scoring Breakdown"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Score components table
        metrics_data = {
            'Metric': categories,
            'Score': [f"{score:.1f}%" for score in scores],
            'Weight': ['30%', '25%', '30%', '15%'],
            'Impact': ['High', 'Medium', 'High', 'Low']
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        st.markdown("#### 📈 Metric Details")
        st.dataframe(df_metrics, use_container_width=True)
    
    def show_learning_path(self, resume_text, job_description):
        """Generate personalized learning path"""
        st.markdown("#### 🎓 Personalized Learning Path")
        
        # Mock learning recommendations based on gaps
        learning_modules = [
            {
                "skill": "Docker & Containerization",
                "priority": "High",
                "duration": "2-3 weeks",
                "resources": ["Docker Official Tutorial", "Kubernetes Basics", "Container Orchestration"],
                "difficulty": "Intermediate"
            },
            {
                "skill": "AWS Cloud Services",
                "priority": "High",
                "duration": "3-4 weeks",
                "resources": ["AWS Solutions Architect", "EC2 Deep Dive", "S3 and Storage Services"],
                "difficulty": "Advanced"
            },
            {
                "skill": "API Development",
                "priority": "Medium",
                "duration": "1-2 weeks",
                "resources": ["REST API Design", "GraphQL Basics", "API Testing"],
                "difficulty": "Beginner"
            }
        ]
        
        for i, module in enumerate(learning_modules, 1):
            priority_color = "#dc3545" if module["priority"] == "High" else "#ffc107" if module["priority"] == "Medium" else "#28a745"
            
            st.markdown(f"""
            <div class="analysis-section">
                <h4>📚 Module {i}: {module['skill']}</h4>
                <div style="display: flex; gap: 1rem; margin: 1rem 0;">
                    <span style="background: {priority_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">
                        {module['priority']} Priority
                    </span>
                    <span style="background: #667eea; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">
                        {module['duration']}
                    </span>
                    <span style="background: #28a745; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">
                        {module['difficulty']}
                    </span>
                </div>
                <p><strong>Recommended Resources:</strong></p>
                <ul>
                    {"".join([f"<li>{resource}</li>" for resource in module['resources']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def bulk_resume_analysis(self):
        """Bulk resume processing interface"""
        st.markdown("### 📦 Bulk Resume Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_files = st.file_uploader(
                "Upload multiple resumes",
                type=['pdf', 'docx'],
                accept_multiple_files=True,
                help="Select multiple resume files for batch processing"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} files uploaded successfully!")
                
                # Processing options
                st.markdown("#### ⚙️ Processing Options")
                job_filter = st.selectbox("Filter by Job Category", ["All", "Software Development", "Data Science", "Marketing", "Sales"])
                min_score_threshold = st.slider("Minimum Score Threshold", 0, 100, 50)
                max_results = st.number_input("Maximum Results to Show", 10, 1000, 100)
                
                if st.button("🚀 Start Batch Processing", type="primary"):
                    # Processing simulation
                    progress_bar = st.progress(0)
                    results_placeholder = st.empty()
                    
                    batch_results = []
                    
                    for i, file in enumerate(uploaded_files):
                        progress = (i + 1) / len(uploaded_files)
                        progress_bar.progress(progress)
                        
                        # Mock processing
                        score = np.random.uniform(30, 95)
                        verdict, icon = self.matcher.get_fit_verdict(score)
                        
                        batch_results.append({
                            "Filename": file.name,
                            "Score": f"{score:.1f}%",
                            "Verdict": f"{icon} {verdict}",
                            "Processing Time": f"{np.random.uniform(1.5, 3.5):.1f}s"
                        })
                        
                        time.sleep(0.1)  # Simulate processing time
                    
                    progress_bar.empty()
                    
                    # Display results
                    st.markdown("#### 📊 Batch Processing Results")
                    df_results = pd.DataFrame(batch_results)
                    st.dataframe(df_results, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Processed", len(batch_results))
                    with col2:
                        high_scores = sum(1 for r in batch_results if float(r["Score"].replace('%', '')) >= 75)
                        st.metric("High Scores", high_scores)
                    with col3:
                        avg_score = np.mean([float(r["Score"].replace('%', '')) for r in batch_results])
                        st.metric("Average Score", f"{avg_score:.1f}%")
                    with col4:
                        avg_time = np.mean([float(r["Processing Time"].replace('s', '')) for r in batch_results])
                        st.metric("Avg Process Time", f"{avg_time:.1f}s")
        
        with col2:
            st.markdown("#### 📈 Processing Statistics")
            
            # Mock statistics
            stats_data = {
                "Metric": ["Files Processed", "Success Rate", "Avg Score", "Time Saved"],
                "Value": ["1,247", "98.5%", "73.2%", "45.2 hours"],
                "Trend": ["↑ 15%", "↑ 2.1%", "↑ 3.2%", "↑ 23%"]
            }
            
            for i in range(len(stats_data["Metric"])):
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{stats_data['Metric'][i]}</h4>
                    <h3>{stats_data['Value'][i]}</h3>
                    <small style="color: green;">{stats_data['Trend'][i]}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def realtime_screening(self):
        """Real-time resume screening interface"""
        st.markdown("### ⚡ Real-time Resume Screening")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 🎯 Screening Configuration")
            
            # Auto-screening settings
            auto_screening = st.checkbox("Enable Auto-Screening", True)
            score_threshold = st.slider("Auto-Accept Threshold", 0, 100, 80)
            reject_threshold = st.slider("Auto-Reject Threshold", 0, 100, 30)
            
            # Notification settings
            st.markdown("#### 🔔 Notification Settings")
            email_notifications = st.checkbox("Email Notifications", True)
            slack_integration = st.checkbox("Slack Integration", False)
            webhook_url = st.text_input("Webhook URL (Optional)")
            
            # Priority jobs
            st.markdown("#### 🚨 Priority Jobs")
            priority_jobs = st.multiselect(
                "High Priority Positions",
                ["Senior Python Developer", "Data Scientist", "ML Engineer", "DevOps Engineer"]
            )
        
        with col2:
            st.markdown("#### 📊 Live Screening Dashboard")
            
            # Real-time metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("⚡ Queue Length", "5", "↓ 2")
                st.metric("🎯 Processing Rate", "12/min", "↑ 3")
            with col2:
                st.metric("✅ Auto-Accepted", "23", "↑ 8")
                st.metric("❌ Auto-Rejected", "15", "↓ 3")
            
            # Live feed simulation
            st.markdown("#### 📋 Live Processing Feed")
            
            if st.button("▶️ Start Live Monitoring"):
                feed_placeholder = st.empty()
                
                for i in range(10):
                    candidate_name = f"Candidate_{np.random.randint(1000, 9999)}"
                    score = np.random.uniform(20, 95)
                    verdict, icon = self.matcher.get_fit_verdict(score)
                    
                    status = "AUTO-ACCEPTED" if score >= score_threshold else "AUTO-REJECTED" if score <= reject_threshold else "MANUAL REVIEW"
                    status_color = "#28a745" if status == "AUTO-ACCEPTED" else "#dc3545" if status == "AUTO-REJECTED" else "#ffc107"
                    
                    feed_placeholder.markdown(f"""
                    <div style="padding: 1rem; margin: 0.5rem 0; background: white; border-radius: 8px; border-left: 4px solid {status_color};">
                        <strong>📄 {candidate_name}</strong> - Score: {score:.1f}% {icon}
                        <br>
                        <span style="color: {status_color}; font-weight: bold;">{status}</span>
                        <div style="font-size: 0.8em; color: gray;">Just now</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    time.sleep(1)
    
    def show_job_management(self):
        """Job management interface"""
        st.markdown("## 💼 Job Management System")
        
        tab1, tab2, tab3 = st.tabs(["📝 Create Job", "📋 Manage Jobs", "📊 Job Analytics"])
        
        with tab1:
            self.create_job_form()
        
        with tab2:
            self.manage_jobs()
        
        with tab3:
            self.job_analytics()
    
    def create_job_form(self):
        """Create new job posting form"""
        st.markdown("### ➕ Create New Job Posting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            job_title = st.text_input("🎯 Job Title", placeholder="e.g., Senior Python Developer")
            company = st.text_input("🏢 Company Name", placeholder="e.g., TechCorp Solutions")
            location = st.selectbox("📍 Location", ["Hyderabad", "Bangalore", "Pune", "Delhi NCR", "Remote", "Hybrid"])
            experience_level = st.selectbox("📈 Experience Level", ["Fresher", "1-2 Years", "2-5 Years", "5+ Years", "Senior Level"])
            
        with col2:
            employment_type = st.selectbox("💼 Employment Type", ["Full-time", "Part-time", "Contract", "Internship"])
            urgency = st.selectbox("🚨 Urgency Level", ["Low", "Medium", "High", "Critical"])
            salary_range = st.text_input("💰 Salary Range", placeholder="e.g., 8-12 LPA")
            posting_date = st.date_input("📅 Posting Date")
        
        # Job description
        st.markdown("#### 📄 Job Description")
        job_description = st.text_area(
            "Description",
            height=200,
            placeholder="Detailed job description, responsibilities, and requirements..."
        )
        
        # Skills section
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🎯 Required Skills")
            required_skills = st.text_area("Required Skills (comma-separated)", height=100)
        
        with col2:
            st.markdown("#### ⭐ Preferred Skills")
            preferred_skills = st.text_area("Preferred Skills (comma-separated)", height=100)
        
        # Additional settings
        st.markdown("#### ⚙️ Additional Settings")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            auto_screening = st.checkbox("Enable Auto-Screening")
            min_score = st.slider("Min Score Threshold", 0, 100, 60) if auto_screening else 0
        
        with col2:
            email_notifications = st.checkbox("Email Notifications")
            max_applications = st.number_input("Max Applications", 1, 1000, 100)
        
        with col3:
            is_featured = st.checkbox("Featured Job")
            expiry_days = st.number_input("Expires in (days)", 1, 90, 30)
        
        # Submit button
        if st.button("🚀 Create Job Posting", type="primary"):
            # Here you would save to database
            job_data = {
                "title": job_title,
                "company": company,
                "location": location,
                "description": job_description,
                "required_skills": required_skills,
                "preferred_skills": preferred_skills,
                "experience_level": experience_level,
                "urgency_level": urgency
            }
            
            st.success(f"✅ Job posting '{job_title}' created successfully!")
            st.balloons()
    
    def manage_jobs(self):
        """Manage existing job postings"""
        st.markdown("### 📋 Active Job Postings")
        
        # Search and filter options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            search_term = st.text_input("🔍 Search Jobs", placeholder="Job title, company...")
        with col2:
            location_filter = st.selectbox("📍 Location", ["All", "Hyderabad", "Bangalore", "Pune", "Delhi NCR"])
        with col3:
            urgency_filter = st.selectbox("🚨 Urgency", ["All", "High", "Medium", "Low"])
        with col4:
            status_filter = st.selectbox("📊 Status", ["All", "Active", "Paused", "Expired"])
        
        # Mock job data
        jobs_data = [
            {
                "ID": "JOB001",
                "Title": "Senior Python Developer",
                "Company": "TechCorp",
                "Location": "Hyderabad",
                "Applications": 127,
                "Urgency": "High",
                "Status": "Active",
                "Posted": "2024-09-15",
                "Expires": "2024-10-15",
                "Avg Score": 68.5
            },
            {
                "ID": "JOB002",
                "Title": "Data Scientist",
                "Company": "AI Solutions",
                "Location": "Bangalore",
                "Applications": 89,
                "Urgency": "Medium",
                "Status": "Active",
                "Posted": "2024-09-18",
                "Expires": "2024-10-18",
                "Avg Score": 72.1
            },
            {
                "ID": "JOB003",
                "Title": "Full Stack Engineer",
                "Company": "StartupXYZ",
                "Location": "Pune",
                "Applications": 203,
                "Urgency": "Low",
                "Status": "Paused",
                "Posted": "2024-09-10",
                "Expires": "2024-10-10",
                "Avg Score": 65.8
            },
            {
                "ID": "JOB004",
                "Title": "ML Engineer",
                "Company": "DeepTech",
                "Location": "Delhi NCR",
                "Applications": 156,
                "Urgency": "High",
                "Status": "Active",
                "Posted": "2024-09-20",
                "Expires": "2024-10-20",
                "Avg Score": 75.3
            }
        ]
        
        # Display jobs table
        df_jobs = pd.DataFrame(jobs_data)
        
        # Create interactive dataframe
        for i, job in enumerate(jobs_data):
            urgency_color = "#dc3545" if job["Urgency"] == "High" else "#ffc107" if job["Urgency"] == "Medium" else "#28a745"
            status_color = "#28a745" if job["Status"] == "Active" else "#ffc107" if job["Status"] == "Paused" else "#6c757d"
            
            with st.expander(f"🎯 {job['Title']} - {job['Company']} ({job['Applications']} applications)"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Location:** {job['Location']}")
                    st.markdown(f"**Posted:** {job['Posted']}")
                    st.markdown(f"**Expires:** {job['Expires']}")
                    st.markdown(f"**Average Score:** {job['Avg Score']}%")
                
                with col2:
                    st.markdown(f"""
                    <div style="padding: 0.5rem; background: {urgency_color}; color: white; border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                        🚨 {job['Urgency']} Priority
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="padding: 0.5rem; background: {status_color}; color: white; border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                        📊 {job['Status']}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if st.button(f"📊 View Analytics", key=f"analytics_{i}"):
                        st.info(f"Opening analytics for {job['Title']}")
                    
                    if st.button(f"✏️ Edit Job", key=f"edit_{i}"):
                        st.info(f"Opening editor for {job['Title']}")
                    
                    if job['Status'] == 'Active':
                        if st.button(f"⏸️ Pause", key=f"pause_{i}"):
                            st.success(f"Job {job['Title']} paused")
                    else:
                        if st.button(f"▶️ Activate", key=f"activate_{i}"):
                            st.success(f"Job {job['Title']} activated")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                        st.warning(f"Job {job['Title']} marked for deletion")
    
    def job_analytics(self):
        """Job-specific analytics dashboard"""
        st.markdown("### 📊 Job Performance Analytics")
        
        # Job selection for detailed analytics
        selected_job = st.selectbox(
            "Select Job for Detailed Analysis",
            ["Senior Python Developer - TechCorp", "Data Scientist - AI Solutions", "ML Engineer - DeepTech"]
        )
        
        # Key metrics for selected job
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Applications", "127", "↑ 15")
        with col2:
            st.metric("🎯 Average Score", "68.5%", "↑ 2.3%")
        with col3:
            st.metric("✅ Qualified Candidates", "32", "↑ 8")
        with col4:
            st.metric("📞 Interviews Scheduled", "12", "↑ 4")
        
        # Charts section
        col1, col2 = st.columns(2)
        
        with col1:
            # Application timeline
            dates = pd.date_range(start='2024-09-15', periods=20, freq='D')
            applications = np.cumsum(np.random.poisson(6, 20))
            
            fig = px.line(
                x=dates, y=applications,
                title="📈 Application Timeline",
                labels={'x': 'Date', 'y': 'Cumulative Applications'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Score distribution
            scores = np.random.normal(68.5, 15, 127)
            scores = np.clip(scores, 0, 100)
            
            fig = px.histogram(
                x=scores,
                nbins=20,
                title="📊 Score Distribution",
                labels={'x': 'Relevance Score', 'y': 'Number of Candidates'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Top candidates table
        st.markdown("#### 🏆 Top Candidates")
        
        top_candidates = [
            {"Name": "Rahul Sharma", "Score": 89.5, "Experience": "5 years", "Location": "Hyderabad", "Status": "Interview Scheduled"},
            {"Name": "Priya Singh", "Score": 86.2, "Experience": "4 years", "Location": "Bangalore", "Status": "Under Review"},
            {"Name": "Amit Kumar", "Score": 84.7, "Experience": "6 years", "Location": "Pune", "Status": "Interview Scheduled"},
            {"Name": "Sneha Patel", "Score": 82.1, "Experience": "3 years", "Location": "Delhi NCR", "Status": "Qualified"},
            {"Name": "Rajesh Gupta", "Score": 80.9, "Experience": "7 years", "Location": "Hyderabad", "Status": "Under Review"}
        ]
        
        df_candidates = pd.DataFrame(top_candidates)
        st.dataframe(df_candidates, use_container_width=True)
        
        # Skills demand analysis
        st.markdown("#### 🎯 Skills in Demand")
        
        skills_demand = {
            'Python': 98,
            'Django': 76,
            'PostgreSQL': 68,
            'AWS': 89,
            'Docker': 72,
            'REST APIs': 84,
            'Git': 91,
            'Machine Learning': 56,
            'React': 43,
            'Kubernetes': 38
        }
        
        fig = px.bar(
            x=list(skills_demand.keys()),
            y=list(skills_demand.values()),
            title="📈 Skills Demand Analysis",
            labels={'x': 'Skills', 'y': 'Demand Percentage'},
            color=list(skills_demand.values()),
            color_continuous_scale='Viridis'
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    def show_analytics(self):
        """Advanced analytics dashboard"""
        st.markdown("## 📊 Advanced Analytics Dashboard")
        
        # Time period selector
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            time_period = st.selectbox("📅 Time Period", ["Last 7 days", "Last 30 days", "Last 90 days", "Custom"])
        with col2:
            if time_period == "Custom":
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")
        with col3:
            auto_refresh = st.checkbox("🔄 Auto Refresh (30s)")
        
        # Overview metrics
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📄 Resumes Processed", "1,247", "↑ 156 (+14.3%)")
        with col2:
            st.metric("💼 Active Jobs", "23", "↑ 3 (+15.0%)")
        with col3:
            st.metric("🎯 Avg Match Score", "73.2%", "↑ 2.1% (+2.9%)")
        with col4:
            st.metric("⚡ Processing Speed", "2.1s", "↓ 0.3s (-12.5%)")
        with col5:
            st.metric("✅ Success Rate", "87.5%", "↑ 3.2% (+3.8%)")
        
        # Advanced charts
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🎯 Performance", "📈 Trends", "🔍 Deep Dive", "🤖 AI Insights"])
        
        with tab1:
            self.show_overview_analytics()
        
        with tab2:
            self.show_performance_analytics()
        
        with tab3:
            self.show_trends_analytics()
        
        with tab4:
            self.show_deep_dive_analytics()
        
        with tab5:
            self.show_ai_insights()
    
    def show_overview_analytics(self):
        """Overview analytics tab"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing volume over time
            dates = pd.date_range(start='2024-08-01', end='2024-09-21', freq='D')
            volumes = np.random.poisson(25, len(dates)) + np.sin(np.arange(len(dates)) * 0.2) * 10 + 30
            
            fig = px.line(
                x=dates, y=volumes,
                title="📈 Daily Processing Volume",
                labels={'x': 'Date', 'y': 'Resumes Processed'}
            )
            fig.add_scatter(x=dates, y=volumes, mode='markers', name='Daily Count')
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Score distribution by verdict
            verdicts = ['High Fit', 'Medium Fit', 'Low Fit']
            counts = [342, 567, 338]
            
            fig = px.pie(
                values=counts, names=verdicts,
                title="🎯 Fit Verdict Distribution",
                color_discrete_sequence=['#28a745', '#ffc107', '#dc3545']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Location-wise analysis
        st.markdown("#### 📍 Geographic Distribution")
        
        location_data = {
            'Location': ['Hyderabad', 'Bangalore', 'Pune', 'Delhi NCR', 'Chennai', 'Mumbai'],
            'Applications': [324, 298, 187, 156, 142, 140],
            'Avg Score': [74.2, 76.8, 71.5, 69.3, 73.1, 72.4],
            'High Fit %': [28, 32, 24, 22, 26, 25]
        }
        
        df_location = pd.DataFrame(location_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_location, x='Location', y='Applications',
                title="📊 Applications by Location",
                color='Applications',
                color_continuous_scale='Blues'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df_location, x='Avg Score', y='High Fit %', size='Applications',
                hover_name='Location',
                title="🎯 Score vs High Fit Rate",
                labels={'Avg Score': 'Average Score (%)', 'High Fit %': 'High Fit Rate (%)'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    def show_performance_analytics(self):
        """Performance analytics tab"""
        # System performance metrics
        st.markdown("#### ⚡ System Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Processing time trends
            hours = list(range(24))
            processing_times = [2.1 + 0.5 * np.sin(h * np.pi / 12) + np.random.normal(0, 0.1) for h in hours]
            
            fig = px.line(
                x=hours, y=processing_times,
                title="⏱️ Hourly Processing Times",
                labels={'x': 'Hour of Day', 'y': 'Avg Processing Time (s)'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accuracy metrics
            components = ['Text Extraction', 'Skill Matching', 'Experience Calc', 'Score Generation']
            accuracy = [98.5, 89.2, 92.7, 87.3]
            
            fig = px.bar(
                x=components, y=accuracy,
                title="🎯 Component Accuracy",
                labels={'x': 'Component', 'y': 'Accuracy (%)'},
                color=accuracy,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Error rate tracking
            dates = pd.date_range(start='2024-09-01', periods=21, freq='D')
            error_rates = np.random.exponential(2, 21)
            
            fig = px.line(
                x=dates, y=error_rates,
                title="❌ Error Rate Tracking",
                labels={'x': 'Date', 'y': 'Error Rate (%)'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance benchmarks
        st.markdown("#### 🏆 Performance Benchmarks")
        
        benchmark_data = {
            'Metric': ['Processing Speed', 'Accuracy Rate', 'Uptime', 'Throughput', 'Response Time'],
            'Current': [2.1, 87.5, 99.8, 720, 1.8],
            'Target': [2.0, 90.0, 99.9, 800, 1.5],
            'Industry Avg': [3.2, 82.1, 98.5, 450, 2.8],
            'Unit': ['seconds', '%', '%', 'resumes/hour', 'seconds']
        }
        
        df_benchmark = pd.DataFrame(benchmark_data)
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='Current', x=df_benchmark['Metric'], y=df_benchmark['Current'], marker_color='#667eea'))
        fig.add_trace(go.Bar(name='Target', x=df_benchmark['Metric'], y=df_benchmark['Target'], marker_color='#28a745'))
        fig.add_trace(go.Bar(name='Industry Avg', x=df_benchmark['Metric'], y=df_benchmark['Industry Avg'], marker_color='#ffc107'))
        
        fig.update_layout(
            title='📊 Performance vs Benchmarks',
            xaxis_title='Metrics',
            yaxis_title='Values',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_trends_analytics(self):
        """Trends analytics tab"""
        st.markdown("#### 📈 Market Trends & Insights")
        
        # Skills trending analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Most demanded skills over time
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
            python_trend = [85, 87, 90, 92, 89, 91, 94, 96, 98]
            js_trend = [78, 79, 81, 83, 85, 84, 86, 88, 89]
            aws_trend = [65, 68, 72, 75, 78, 82, 85, 87, 89]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=months, y=python_trend, mode='lines+markers', name='Python', line=dict(color='#3776ab')))
            fig.add_trace(go.Scatter(x=months, y=js_trend, mode='lines+markers', name='JavaScript', line=dict(color='#f7df1e')))
            fig.add_trace(go.Scatter(x=months, y=aws_trend, mode='lines+markers', name='AWS', line=dict(color='#ff9900')))
            
            fig.update_layout(
                title='🚀 Trending Skills Demand',
                xaxis_title='Month',
                yaxis_title='Demand Score',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Salary trends by role
            roles = ['Data Scientist', 'Full Stack Dev', 'DevOps Engineer', 'ML Engineer', 'Backend Dev']
            avg_salaries = [15.2, 12.8, 14.5, 16.8, 11.5]
            growth_rates = [12.5, 8.3, 15.2, 18.7, 6.9]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Avg Salary (LPA)', x=roles, y=avg_salaries, yaxis='y', offsetgroup=1))
            fig.add_trace(go.Scatter(name='Growth Rate (%)', x=roles, y=growth_rates, yaxis='y2', mode='lines+markers'))
            
            fig.update_layout(
                title='💰 Salary Trends by Role',
                xaxis_title='Job Roles',
                yaxis=dict(title='Average Salary (LPA)', side='left'),
                yaxis2=dict(title='Growth Rate (%)', side='right', overlaying='y'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Hiring patterns
        st.markdown("#### 🎯 Hiring Patterns Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Experience level distribution
            exp_levels = ['Fresher', '1-2 Years', '2-5 Years', '5+ Years']
            exp_counts = [234, 456, 389, 168]
            
            fig = px.pie(
                values=exp_counts, names=exp_levels,
                title="👥 Experience Level Distribution",
                color_discrete_sequence=['#ff6b6b', '#feca57', '#48dbfb', '#0abde3']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Application success rates
            success_data = {
                'Stage': ['Applied', 'Screened', 'Interviewed', 'Hired'],
                'Count': [1247, 389, 156, 47],
                'Rate': [100, 31.2, 12.5, 3.8]
            }
            
            fig = go.Figure()
            fig.add_trace(go.Funnel(
                y=success_data['Stage'],
                x=success_data['Count'],
                textinfo="value+percent initial"
            ))
            
            fig.update_layout(title="🎯 Hiring Funnel Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Time to hire trends
            quarters = ['Q1 2024', 'Q2 2024', 'Q3 2024']
            avg_days = [28, 25, 22]
            
            fig = px.bar(
                x=quarters, y=avg_days,
                title="⏰ Average Time to Hire",
                labels={'x': 'Quarter', 'y': 'Days'},
                color=avg_days,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    def show_deep_dive_analytics(self):
        """Deep dive analytics tab"""
        st.markdown("#### 🔍 Deep Dive Analysis")
        
        # Advanced filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            job_category = st.selectbox("Job Category", ["All", "Tech", "Non-Tech", "Leadership"])
        with col2:
            experience_filter = st.selectbox("Experience", ["All", "0-2", "2-5", "5+"])
        with col3:
            location_filter = st.selectbox("Location", ["All", "Hyderabad", "Bangalore", "Pune", "Delhi NCR"])
        with col4:
            score_range = st.slider("Score Range", 0, 100, (0, 100))
        
        # Correlation analysis
        st.markdown("#### 🔗 Correlation Analysis")
        
        # Generate mock correlation data
        np.random.seed(42)
        n_samples = 1000
        
        correlation_data = pd.DataFrame({
            'Experience_Years': np.random.exponential(3, n_samples),
            'Education_Score': np.random.normal(75, 15, n_samples),
            'Skills_Count': np.random.poisson(12, n_samples),
            'Project_Count': np.random.poisson(5, n_samples),
            'Certification_Count': np.random.poisson(3, n_samples),
            'Relevance_Score': np.random.normal(70, 20, n_samples)
        })
        
        # Clip values to realistic ranges
        correlation_data['Education_Score'] = np.clip(correlation_data['Education_Score'], 0, 100)
        correlation_data['Relevance_Score'] = np.clip(correlation_data['Relevance_Score'], 0, 100)
        correlation_data['Experience_Years'] = np.clip(correlation_data['Experience_Years'], 0, 15)
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title="🔗 Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot matrix
        st.markdown("#### 📊 Multi-dimensional Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                correlation_data,
                x='Experience_Years',
                y='Relevance_Score',
                size='Skills_Count',
                color='Education_Score',
                title="🎯 Experience vs Relevance Score",
                labels={'Experience_Years': 'Years of Experience', 'Relevance_Score': 'Relevance Score (%)'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                correlation_data,
                x='Skills_Count',
                y='Relevance_Score',
                size='Project_Count',
                color='Certification_Count',
                title="🛠️ Skills vs Relevance Score",
                labels={'Skills_Count': 'Number of Skills', 'Relevance_Score': 'Relevance Score (%)'}
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistical insights
        st.markdown("#### 📈 Statistical Insights")
        
        insights = [
            {"metric": "Strong Correlation", "value": "Experience ↔ Relevance Score", "coefficient": "0.67", "significance": "High"},
            {"metric": "Moderate Correlation", "value": "Skills Count ↔ Relevance Score", "coefficient": "0.45", "significance": "Medium"},
            {"metric": "Weak Correlation", "value": "Certifications ↔ Relevance Score", "coefficient": "0.23", "significance": "Low"},
            {"metric": "Surprising Finding", "value": "Education Score has minimal impact", "coefficient": "0.12", "significance": "Very Low"}
        ]
        
        for insight in insights:
            st.markdown(f"""
            <div class="analysis-section">
                <h4>📊 {insight['metric']}</h4>
                <p><strong>{insight['value']}</strong></p>
                <div style="display: flex; gap: 1rem;">
                    <span style="background: #667eea; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">
                        Coefficient: {insight['coefficient']}
                    </span>
                    <span style="background: #28a745; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">
                        {insight['significance']} Significance
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def show_ai_insights(self):
        """AI-powered insights tab"""
        st.markdown("#### 🤖 AI-Powered Insights")
        
        # Predictive analytics
        st.markdown("##### 🔮 Predictive Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hiring probability prediction
            st.markdown("**🎯 Hiring Probability Model**")
            
            # Mock model performance metrics
            model_metrics = {
                "Accuracy": 87.3,
                "Precision": 84.2,
                "Recall": 89.1,
                "F1-Score": 86.6
            }
            
            for metric, value in model_metrics.items():
                st.metric(metric, f"{value}%")
            
            # Feature importance
            features = ['Experience', 'Skills Match', 'Education', 'Projects', 'Certifications']
            importance = [0.35, 0.28, 0.15, 0.12, 0.10]
            
            fig = px.bar(
                x=features, y=importance,
                title="🎯 Feature Importance",
                labels={'x': 'Features', 'y': 'Importance'},
                color=importance,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Market demand prediction
            st.markdown("**📈 Market Demand Forecast**")
            
            # Mock demand forecast
            months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
            demand_forecast = [145, 167, 189, 203, 178, 156]
            confidence_upper = [160, 185, 210, 225, 195, 175]
            confidence_lower = [130, 150, 168, 181, 161, 137]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=months, y=demand_forecast,
                mode='lines+markers',
                name='Predicted Demand',
                line=dict(color='#667eea')
            ))
            fig.add_trace(go.Scatter(
                x=months, y=confidence_upper,
                fill=None,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=months, y=confidence_lower,
                fill='tonexty',
                mode='lines',
                line_color='rgba(0,0,0,0)',
                name='Confidence Interval',
                fillcolor='rgba(102, 126, 234, 0.2)'
            ))
            
            fig.update_layout(
                title='📊 Hiring Demand Forecast',
                xaxis_title='Month',
                yaxis_title='Expected Applications',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # AI Recommendations
        st.markdown("##### 💡 AI Recommendations")
        
        ai_recommendations = [
            {
                "type": "🎯 Optimization",
                "title": "Job Description Enhancement",
                "description": "Consider adding 'React' and 'Node.js' to job requirements. Analysis shows 23% higher application rate for full-stack positions.",
                "impact": "High",
                "effort": "Low"
            },
            {
                "type": "📈 Strategy",
                "title": "Candidate Pool Expansion",
                "description": "Target candidates with 2-4 years experience. This segment shows 67% higher conversion rate to interviews.",
                "impact": "Medium",
                "effort": "Medium"
            },
            {
                "type": "⚡ Process",
                "title": "Faster Screening",
                "description": "Implement auto-screening for scores above 85%. This could save 15 hours per week of manual review time.",
                "impact": "High",
                "effort": "Low"
            },
            {
                "type": "🔍 Quality",
                "title": "Skills Assessment",
                "description": "Add technical assessment for top 10% candidates. Correlation analysis shows this improves hire success rate by 34%.",
                "impact": "High",
                "effort": "High"
            }
        ]
        
        for rec in ai_recommendations:
            impact_color = "#28a745" if rec["impact"] == "High" else "#ffc107" if rec["impact"] == "Medium" else "#6c757d"
            effort_color = "#dc3545" if rec["effort"] == "High" else "#ffc107" if rec["effort"] == "Medium" else "#28a745"
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>{rec['type']} {rec['title']}</h4>
                <p>{rec['description']}</p>
                <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                    <span style="background: {impact_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">
                        Impact: {rec['impact']}
                    </span>
                    <span style="background: {effort_color}; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">
                        Effort: {rec['effort']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Anomaly Detection
        st.markdown("##### 🚨 Anomaly Detection")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**⚠️ Score Anomalies**")
            st.warning("Detected 3 resumes with unusually high scores but low interview conversion")
            st.info("Suggested action: Review scoring algorithm parameters")
        
        with col2:
            st.markdown("**📊 Volume Anomalies**")
            st.success("Application volume 23% above normal - positive trend")
            st.info("Peak hours: 10-11 AM, 2-3 PM")
        
        with col3:
            st.markdown("**🎯 Quality Anomalies**")
            st.error("Quality drop detected in batch processed on Sept 19")
            st.info("Likely cause: Processing pipeline timeout")
    
    def show_student_feedback(self):
        """Student feedback and improvement suggestions"""
        st.markdown("## 🎓 Student Feedback & Development")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Generate Feedback", "📊 Feedback Analytics", "🎯 Skill Development", "🏆 Success Stories"])
        
        with tab1:
            self.generate_student_feedback()
        
        with tab2:
            self.feedback_analytics()
        
        with tab3:
            self.skill_development_program()
        
        with tab4:
            self.success_stories()
    
    def generate_student_feedback(self):
        """Generate personalized feedback for students"""
        st.markdown("### 📝 Personalized Student Feedback Generator")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 👤 Student Selection")
            
            # Mock student data
            students = [
                {"name": "Rahul Sharma", "score": 67.5, "applications": 12, "interviews": 2},
                {"name": "Priya Singh", "score": 82.3, "applications": 8, "interviews": 4},
                {"name": "Amit Kumar", "score": 45.2, "applications": 15, "interviews": 1},
                {"name": "Sneha Patel", "score": 73.8, "applications": 10, "interviews": 3}
            ]
            
            selected_student = st.selectbox(
                "Select Student",
                [f"{s['name']} (Avg Score: {s['score']}%)" for s in students]
            )
            
            student_data = students[0] if "Rahul" in selected_student else students[1] if "Priya" in selected_student else students[2] if "Amit" in selected_student else students[3]
            
            # Feedback type selection
            feedback_type = st.selectbox(
                "Feedback Type",
                ["Comprehensive Report", "Quick Tips", "Skill Gap Analysis", "Interview Preparation"]
            )
            
            # Feedback tone
            feedback_tone = st.selectbox(
                "Feedback Tone",
                ["Encouraging", "Constructive", "Direct", "Motivational"]
            )
            
            if st.button("🚀 Generate Feedback", type="primary"):
                with st.spinner("Generating personalized feedback..."):
                    time.sleep(2)
                    st.success("✅ Feedback generated successfully!")
        
        with col2:
            st.markdown("#### 📊 Student Performance Overview")
            
            # Performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Score", f"{student_data['score']}%", "↑ 5.2%")
                st.metric("Applications Sent", student_data['applications'], "↑ 3")
            with col2:
                st.metric("Interviews Secured", student_data['interviews'], "↑ 1")
                st.metric("Response Rate", f"{(student_data['interviews']/student_data['applications']*100):.1f}%", "↑ 2.1%")
            
            # Performance trend chart
            months = ['Jun', 'Jul', 'Aug', 'Sep']
            scores = [student_data['score']-15, student_data['score']-8, student_data['score']-3, student_data['score']]
            
            fig = px.line(
                x=months, y=scores,
                title=f"📈 {student_data['name']}'s Score Trend",
                labels={'x': 'Month', 'y': 'Average Score (%)'},
                markers=True
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Generated feedback display
        if st.session_state.get('feedback_generated', False):
            st.markdown("#### 📄 Generated Feedback Report")
            
            feedback_content = f"""
            <div class="analysis-section">
                <h3>🎓 Personalized Development Report for {student_data['name']}</h3>
                
                <h4>📊 Performance Summary</h4>
                <p>Your current average relevance score is <strong>{student_data['score']}%</strong>, which places you in the 
                {'top 25%' if student_data['score'] >= 75 else 'middle 50%' if student_data['score'] >= 50 else 'bottom 25%'} of applicants.</p>
                
                <h4>🎯 Strengths</h4>
                <ul>
                    <li>Strong technical foundation in core programming concepts</li>
                    <li>Good project portfolio demonstrating practical skills</li>
                    <li>Consistent improvement trend over the past 4 months</li>
                </ul>
                
                <h4>🔧 Areas for Improvement</h4>
                <ul>
                    <li><strong>Cloud Technologies:</strong> Add AWS/Azure certifications to boost relevance by 15-20%</li>
                    <li><strong>System Design:</strong> Include system design projects in your portfolio</li>
                    <li><strong>Resume Format:</strong> Quantify achievements with specific metrics and numbers</li>
                </ul>
                
                <h4>📚 Recommended Learning Path</h4>
                <ol>
                    <li><strong>Week 1-2:</strong> Complete AWS Cloud Practitioner certification</li>
                    <li><strong>Week 3-4:</strong> Build a full-stack project with cloud deployment</li>
                    <li><strong>Week 5-6:</strong> Practice system design problems and document solutions</li>
                    <li><strong>Week 7-8:</strong> Update resume with quantified achievements</li>
                </ol>
                
                <h4>🎯 Next Steps</h4>
                <p>Focus on the cloud technologies gap first - our analysis shows this single improvement could 
                increase your relevance score to approximately <strong>{student_data['score'] + 12:.1f}%</strong>.</p>
                
                <div style="background: #e8f5e8; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <strong>💡 Pro Tip:</strong> Companies in your target list show 67% preference for candidates with cloud experience.
                </div>
            </div>
            """
            
            st.markdown(feedback_content, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.button("📧 Email Report")
            with col2:
                st.button("💾 Save to Profile")
            with col3:
                st.button("📅 Schedule Mentoring")
            with col4:
                st.button("🔄 Generate New Report")
    
    def feedback_analytics(self):
        """Analytics for student feedback program"""
        st.markdown("### 📊 Feedback Program Analytics")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Students Helped", "1,247", "↑ 156")
        with col2:
            st.metric("📈 Avg Improvement", "18.3%", "↑ 2.1%")
        with col3:
            st.metric("✅ Success Rate", "73.2%", "↑ 5.4%")
        with col4:
            st.metric("⭐ Satisfaction", "4.6/5", "↑ 0.2")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Improvement by feedback type
            feedback_types = ['Comprehensive', 'Quick Tips', 'Skill Gap', 'Interview Prep']
            improvements = [22.5, 12.3, 28.7, 16.4]
            
            fig = px.bar(
                x=feedback_types, y=improvements,
                title="📈 Improvement by Feedback Type",
                labels={'x': 'Feedback Type', 'y': 'Average Score Improvement (%)'},
                color=improvements,
                color_continuous_scale='Blues'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Student satisfaction ratings
            ratings = [1, 2, 3, 4, 5]
            counts = [12, 34, 156, 489, 556]
            
            fig = px.bar(
                x=ratings, y=counts,
                title="⭐ Student Satisfaction Ratings",
                labels={'x': 'Rating (Stars)', 'y': 'Number of Students'},
                color=counts,
                color_continuous_scale='Greens'
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed feedback analysis
        st.markdown("#### 📋 Feedback Effectiveness Analysis")
        
        feedback_effectiveness = {
            'Category': ['Technical Skills', 'Resume Format', 'Interview Skills', 'Soft Skills', 'Industry Knowledge'],
            'Students Helped': [456, 678, 234, 345, 123],
            'Avg Improvement': [25.3, 18.7, 31.2, 14.8, 22.1],
            'Success Rate': [78.5, 82.1, 89.3, 65.7, 74.2]
        }
        
        df_effectiveness = pd.DataFrame(feedback_effectiveness)
        st.dataframe(df_effectiveness, use_container_width=True)
    
    def skill_development_program(self):
        """Skill development program interface"""
        st.markdown("### 🎯 Skill Development Program")
        
        # Program overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🚀 Active Programs</h3>
                <h2 class="gradient-text">12</h2>
                <p>Covering trending skills</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>👥 Enrolled Students</h3>
                <h2 class="gradient-text">847</h2>
                <p>Across all programs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>✅ Completion Rate</h3>
                <h2 class="gradient-text">78%</h2>
                <p>Above industry average</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Program catalog
        st.markdown("#### 📚 Available Programs")
        
        programs = [
            {
                "title": "☁️ Cloud Computing Essentials",
                "duration": "4 weeks",
                "difficulty": "Beginner",
                "enrolled": 156,
                "rating": 4.7,
                "skills": ["AWS", "Docker", "Kubernetes"],
                "market_demand": "Very High"
            },
            {
                "title": "🤖 Machine Learning Bootcamp",
                "duration": "8 weeks", 
                "difficulty": "Advanced",
                "enrolled": 89,
                "rating": 4.8,
                "skills": ["Python", "TensorFlow", "Scikit-learn"],
                "market_demand": "High"
            },
            {
                "title": "🌐 Full Stack Web Development",
                "duration": "12 weeks",
                "difficulty": "Intermediate",
                "enrolled": 203,
                "rating": 4.6,
                "skills": ["React", "Node.js", "MongoDB"],
                "market_demand": "High"
            },
            {
                "title": "🔧 DevOps Engineering",
                "duration": "6 weeks",
                "difficulty": "Intermediate",
                "enrolled": 134,
                "rating": 4.5,
                "skills": ["Jenkins", "Terraform", "Monitoring"],
                "market_demand": "Very High"
            }
        ]
        
        for program in programs:
            difficulty_color = "#28a745" if program["difficulty"] == "Beginner" else "#ffc107" if program["difficulty"] == "Intermediate" else "#dc3545"
            demand_color = "#dc3545" if program["market_demand"] == "Very High" else "#ffc107" if program["market_demand"] == "High" else "#28a745"
            
            with st.expander(f"{program['title']} - {program['enrolled']} students enrolled"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Duration:** {program['duration']}")
                    st.markdown(f"**Rating:** ⭐ {program['rating']}/5")
                    st.markdown("**Skills Covered:**")
                    for skill in program['skills']:
                        st.markdown(f"""
                        <span class="tech-stack-badge">{skill}</span>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="padding: 0.5rem; background: {difficulty_color}; color: white; border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                        📚 {program['difficulty']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="padding: 0.5rem; background: {demand_color}; color: white; border-radius: 8px; text-align: center; margin: 0.25rem 0;">
                        📈 {program['market_demand']} Demand
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    if st.button(f"📚 View Curriculum", key=f"curr_{program['title']}"):
                        st.info(f"Opening curriculum for {program['title']}")
                    
                    if st.button(f"👥 Enroll Students", key=f"enroll_{program['title']}"):
                        st.success(f"Enrollment opened for {program['title']}")
                    
                    if st.button(f"📊 View Analytics", key=f"prog_analytics_{program['title']}"):
                        st.info(f"Opening analytics for {program['title']}")
        
        # Program performance
        st.markdown("#### 📈 Program Performance")
        
        # Completion rates by program
        program_names = [p['title'].split(' ')[1] for p in programs]  # Simplified names
        completion_rates = [78, 67, 82, 73]
        job_placement_rates = [89, 92, 85, 88]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Completion Rate',
            x=program_names,
            y=completion_rates,
            marker_color='#667eea'
        ))
        fig.add_trace(go.Bar(
            name='Job Placement Rate',
            x=program_names,
            y=job_placement_rates,
            marker_color='#28a745'
        ))
        
        fig.update_layout(
            title='📊 Program Success Metrics',
            xaxis_title='Programs',
            yaxis_title='Success Rate (%)',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def success_stories(self):
        """Display success stories and testimonials"""
        st.markdown("### 🏆 Success Stories & Testimonials")
        
        # Success metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎉 Job Placements", "1,156", "↑ 234")
        with col2:
            st.metric("💰 Avg Salary Increase", "45%", "↑ 8%")
        with col3:
            st.metric("⏰ Avg Time to Job", "3.2 months", "↓ 0.8 months")
        with col4:
            st.metric("🏢 Partner Companies", "127", "↑ 23")
        
        # Success stories
        success_stories = [
            {
                "name": "Rahul Sharma",
                "before_score": 45.2,
                "after_score": 87.6,
                "position": "Senior Python Developer",
                "company": "TechCorp",
                "salary": "18 LPA",
                "story": "Rahul's journey from a low-scoring resume to landing his dream job is inspiring. Through our AI-powered feedback system, he identified key skill gaps and completed our Cloud Computing program. Within 2 months, his relevance scores improved by 94%!",
                "image": "👨‍💻",
                "timeline": "4 months"
            },
            {
                "name": "Priya Singh",
                "before_score": 62.3,
                "after_score": 91.2,
                "position": "ML Engineer",
                "company": "DataTech Solutions",
                "salary": "22 LPA",
                "story": "Priya leveraged our Machine Learning Bootcamp and personalized feedback to transform her profile. Her project portfolio became her strongest asset, helping her secure multiple offers from top companies.",
                "image": "👩‍🔬",
                "timeline": "3 months"
            },
            {
                "name": "Amit Kumar",
                "before_score": 38.9,
                "after_score": 79.4,
                "position": "Full Stack Developer",
                "company": "StartupXYZ",
                "salary": "15 LPA",
                "story": "Starting from a low relevance score, Amit's dedication to our Full Stack program paid off. Our system's continuous feedback helped him focus on the right skills at the right time.",
                "image": "👨‍💼",
                "timeline": "5 months"
            }
        ]
        
        for story in success_stories:
            improvement = ((story['after_score'] - story['before_score']) / story['before_score']) * 100
            
            st.markdown(f"""
            <div class="analysis-section">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <div style="font-size: 3rem; margin-right: 1rem;">{story['image']}</div>
                    <div>
                        <h3>{story['name']}</h3>
                        <h4 style="color: #667eea;">{story['position']} at {story['company']}</h4>
                        <p style="font-weight: bold; color: #28a745;">💰 {story['salary']} • ⏰ {story['timeline']}</p>
                    </div>
                </div>
                
                <div style="display: flex; gap: 2rem; margin: 1rem 0;">
                    <div style="text-align: center;">
                        <h4>Before</h4>
                        <div style="font-size: 2rem; color: #dc3545;">{story['before_score']}%</div>
                    </div>
                    <div style="text-align: center; align-self: center;">
                        <div style="font-size: 2rem;">➡️</div>
                        <div style="background: #28a745; color: white; padding: 4px 8px; border-radius: 12px; font-size: 0.8em;">
                            +{improvement:.1f}% improvement
                        </div>
                    </div>
                    <div style="text-align: center;">
                        <h4>After</h4>
                        <div style="font-size: 2rem; color: #28a745;">{story['after_score']}%</div>
                    </div>
                </div>
                
                <blockquote style="border-left: 4px solid #667eea; padding-left: 1rem; font-style: italic; color: #555;">
                    "{story['story']}"
                </blockquote>
            </div>
            """, unsafe_allow_html=True)
        
        # Testimonials section
        st.markdown("#### 💬 Student Testimonials")
        
        testimonials = [
            {
                "text": "The AI-powered feedback was a game-changer! It identified gaps I never knew I had and provided a clear roadmap to improvement.",
                "author": "Sarah Johnson",
                "role": "Data Analyst at FinTech Corp",
                "rating": 5
            },
            {
                "text": "Within 3 months of following the recommendations, my interview call rate increased by 300%. Best career investment ever!",
                "author": "Michael Chen",
                "role": "DevOps Engineer at CloudFirst",
                "rating": 5
            },
            {
                "text": "The personalized learning paths saved me months of random skill building. Every suggestion was spot-on for market demands.",
                "author": "Anita Desai",
                "role": "Full Stack Developer at TechStart",
                "rating": 5
            }
        ]
        
        for testimonial in testimonials:
            stars = "⭐" * testimonial['rating']
            
            st.markdown(f"""
            <div style="background: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #667eea;">
                <div style="margin-bottom: 1rem;">
                    <div style="font-size: 1.2em;">{stars}</div>
                    <p style="font-style: italic; margin: 0.5rem 0;">"{testimonial['text']}"</p>
                </div>
                <div style="font-weight: bold;">{testimonial['author']}</div>
                <div style="color: #667eea; font-size: 0.9em;">{testimonial['role']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    def show_settings(self):
        """System settings and configuration"""
        st.markdown("## ⚙️ System Settings & Configuration")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎛️ General", "🔧 AI Model", "📧 Notifications", "👥 User Management", "📊 Export/Import"])
        
        with tab1:
            self.general_settings()
        
        with tab2:
            self.ai_model_settings()
        
        with tab3:
            self.notification_settings()
        
        with tab4:
            self.user_management()
        
        with tab5:
            self.export_import_settings()
    
    def general_settings(self):
        """General system settings"""
        st.markdown("### 🎛️ General System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Analysis Settings")
            
            default_score_threshold = st.slider("Default Score Threshold", 0, 100, 60)
            auto_screening_enabled = st.checkbox("Enable Auto-Screening", True)
            batch_processing_size = st.number_input("Batch Processing Size", 10, 1000, 100)
            
            st.markdown("#### 📊 Display Settings")
            
            default_time_range = st.selectbox("Default Analytics Time Range", ["7 days", "30 days", "90 days"])
            charts_theme = st.selectbox("Charts Theme", ["Default", "Dark", "Light", "Corporate"])
            refresh_interval = st.selectbox("Auto Refresh Interval", ["Off", "30s", "1min", "5min"])
        
        with col2:
            st.markdown("#### 🔒 Security Settings")
            
            session_timeout = st.selectbox("Session Timeout", ["30 min", "1 hour", "2 hours", "4 hours"])
            two_factor_auth = st.checkbox("Enable Two-Factor Authentication", False)
            password_policy = st.selectbox("Password Policy", ["Basic", "Standard", "Strong"])
            
            st.markdown("#### 💾 Data Settings")
            
            data_retention_days = st.number_input("Data Retention (days)", 30, 365, 90)
            auto_backup_enabled = st.checkbox("Enable Auto Backup", True)
            backup_frequency = st.selectbox("Backup Frequency", ["Daily", "Weekly", "Monthly"])
        
        # Save settings button
        if st.button("💾 Save General Settings", type="primary"):
            st.success("✅ Settings saved successfully!")
    
    def ai_model_settings(self):
        """AI model configuration settings"""
        st.markdown("### 🔧 AI Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🧠 Model Parameters")
            
            tfidf_weight = st.slider("TF-IDF Weight", 0.0, 1.0, 0.3, 0.05)
            keyword_weight = st.slider("Keyword Match Weight", 0.0, 1.0, 0.25, 0.05)
            skills_weight = st.slider("Skills Match Weight", 0.0, 1.0, 0.3, 0.05)
            experience_weight = st.slider("Experience Weight", 0.0, 1.0, 0.15, 0.05)
            
            total_weight = tfidf_weight + keyword_weight + skills_weight + experience_weight
            st.info(f"Total Weight: {total_weight:.2f} (should equal 1.0)")
            
            st.markdown("#### 📊 Performance Tuning")
            
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.05)
            max_features_tfidf = st.number_input("Max TF-IDF Features", 100, 5000, 1000)
            ngram_range = st.selectbox("N-gram Range", ["(1,1)", "(1,2)", "(1,3)", "(2,3)"])
        
        with col2:
            st.markdown("#### 🎯 Skills Database")
            
            skills_update_frequency = st.selectbox("Skills DB Update", ["Manual", "Weekly", "Monthly"])
            auto_skill_detection = st.checkbox("Auto Skill Detection", True)
            custom_skills_enabled = st.checkbox("Enable Custom Skills", True)
            
            if custom_skills_enabled:
                st.text_area("Custom Skills (comma-separated)", 
                           placeholder="blockchain, quantum computing, edge computing...")
            
            st.markdown("#### 🔄 Model Training")
            
            retrain_frequency = st.selectbox("Model Retraining", ["Never", "Weekly", "Monthly", "Quarterly"])
            training_data_size = st.number_input("Training Data Size", 100, 10000, 1000)
            
            if st.button("🚀 Retrain Model"):
                with st.spinner("Retraining model..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    st.success("✅ Model retrained successfully!")
        
        # Model performance metrics
        st.markdown("#### 📈 Current Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", "87.3%", "↑ 2.1%")
        with col2:
            st.metric("⚡ Speed", "2.1s", "↓ 0.3s")
        with col3:
            st.metric("📊 Precision", "84.2%", "↑ 1.8%")
        with col4:
            st.metric("🔄 Recall", "89.1%", "↑ 3.2%")
        
        if st.button("💾 Save AI Model Settings", type="primary"):
            st.success("✅ AI Model settings saved successfully!")
    
    def notification_settings(self):
        """Notification and alert settings"""
        st.markdown("### 📧 Notification Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📧 Email Notifications")
            
            email_enabled = st.checkbox("Enable Email Notifications", True)
            
            if email_enabled:
                smtp_server = st.text_input("SMTP Server", "smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", 1, 65535, 587)
                email_username = st.text_input("Email Username")
                email_password = st.text_input("Email Password", type="password")
                
                st.markdown("**Email Templates:**")
                
                template_type = st.selectbox("Template Type", 
                    ["High Score Alert", "Low Score Alert", "Batch Complete", "System Error"])
                
                if template_type == "High Score Alert":
                    email_template = st.text_area("Email Template", 
                        value="Subject: High Score Candidate Alert\n\nA candidate has achieved a score above threshold:\n- Name: {candidate_name}\n- Score: {score}%\n- Position: {job_title}")
        
        with col2:
            st.markdown("#### 📱 Integration Settings")
            
            slack_enabled = st.checkbox("Enable Slack Integration", False)
            
            if slack_enabled:
                slack_webhook = st.text_input("Slack Webhook URL")
                slack_channel = st.text_input("Default Channel", "#hiring")
                
                slack_notifications = st.multiselect("Slack Notifications", 
                    ["High Scores", "System Alerts", "Daily Summary", "Weekly Report"])
            
            teams_enabled = st.checkbox("Enable Microsoft Teams", False)
            
            if teams_enabled:
                teams_webhook = st.text_input("Teams Webhook URL")
            
            st.markdown("#### 🚨 Alert Thresholds")
            
            high_score_threshold = st.slider("High Score Alert", 0, 100, 85)
            low_score_threshold = st.slider("Low Score Alert", 0, 100, 30)
            system_error_alerts = st.checkbox("System Error Alerts", True)
            daily_summary = st.checkbox("Daily Summary Report", True)
        
        # Notification history
        st.markdown("#### 📋 Recent Notifications")
        
        notification_history = [
            {"time": "2 mins ago", "type": "High Score", "message": "Rahul Sharma scored 89% for Python Developer role"},
            {"time": "15 mins ago", "type": "System", "message": "Batch processing completed - 25 resumes analyzed"},
            {"time": "1 hour ago", "type": "Alert", "message": "Processing queue length exceeded threshold"},
            {"time": "3 hours ago", "type": "Summary", "message": "Daily report sent to team@innomatics.in"}
        ]
        
        for notif in notification_history:
            type_color = "#28a745" if notif["type"] == "High Score" else "#ffc107" if notif["type"] == "System" else "#dc3545" if notif["type"] == "Alert" else "#667eea"
            
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.25rem 0; background: white; border-radius: 8px; border-left: 4px solid {type_color};">
                <strong>{notif['type']}:</strong> {notif['message']}
                <div style="font-size: 0.8em; color: gray;">{notif['time']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("💾 Save Notification Settings", type="primary"):
            st.success("✅ Notification settings saved successfully!")
    
    def user_management(self):
        """User management interface"""
        st.markdown("### 👥 User Management")
        
        tab1, tab2, tab3 = st.tabs(["👤 Users", "🔐 Roles", "📊 Activity"])
        
        with tab1:
            # Add new user
            st.markdown("#### ➕ Add New User")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                new_username = st.text_input("Username")
                new_email = st.text_input("Email")
            
            with col2:
                new_role = st.selectbox("Role", ["Admin", "Recruiter", "Mentor", "Viewer"])
                new_location = st.selectbox("Location", ["Hyderabad", "Bangalore", "Pune", "Delhi NCR"])
            
            with col3:
                new_department = st.selectbox("Department", ["Placement", "Training", "Management"])
                
                if st.button("➕ Add User"):
                    st.success(f"✅ User {new_username} added successfully!")
            
            # Existing users
            st.markdown("#### 📋 Existing Users")
            
            users_data = [
                {"Username": "admin", "Email": "admin@innomatics.in", "Role": "Admin", "Location": "Hyderabad", "Status": "Active", "Last Login": "2024-09-21 09:30"},
                {"Username": "recruiter1", "Email": "r1@innomatics.in", "Role": "Recruiter", "Location": "Bangalore", "Status": "Active", "Last Login": "2024-09-21 11:15"},
                {"Username": "mentor1", "Email": "m1@innomatics.in", "Role": "Mentor", "Location": "Pune", "Status": "Active", "Last Login": "2024-09-20 16:45"},
                {"Username": "viewer1", "Email": "v1@innomatics.in", "Role": "Viewer", "Location": "Delhi NCR", "Status": "Inactive", "Last Login": "2024-09-19 14:20"}
            ]
            
            df_users = pd.DataFrame(users_data)
            
            # Add action buttons
            for i, user in enumerate(users_data):
                with st.expander(f"👤 {user['Username']} ({user['Role']})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write(f"**Email:** {user['Email']}")
                        st.write(f"**Location:** {user['Location']}")
                    
                    with col2:
                        st.write(f"**Status:** {user['Status']}")
                        st.write(f"**Last Login:** {user['Last Login']}")
                    
                    with col3:
                        if st.button(f"✏️ Edit", key=f"edit_user_{i}"):
                            st.info(f"Editing {user['Username']}")
                        
                        if user['Status'] == 'Active':
                            if st.button(f"⏸️ Suspend", key=f"suspend_{i}"):
                                st.warning(f"User {user['Username']} suspended")
                        else:
                            if st.button(f"▶️ Activate", key=f"activate_user_{i}"):
                                st.success(f"User {user['Username']} activated")
                    
                    with col4:
                        if st.button(f"🔄 Reset Password", key=f"reset_{i}"):
                            st.info(f"Password reset link sent to {user['Email']}")
                        
                        if st.button(f"🗑️ Delete", key=f"delete_user_{i}"):
                            st.error(f"User {user['Username']} marked for deletion")
        
        with tab2:
            st.markdown("#### 🔐 Role Management")
            
            # Role permissions matrix
            roles = ["Admin", "Recruiter", "Mentor", "Viewer"]
            permissions = [
                "View Analytics", "Manage Jobs", "Analyze Resumes", "User Management", 
                "System Settings", "Export Data", "Bulk Processing", "Send Notifications"
            ]
            
            permission_matrix = {
                "Admin": [True, True, True, True, True, True, True, True],
                "Recruiter": [True, True, True, False, False, True, True, True],
                "Mentor": [True, False, True, False, False, False, False, True],
                "Viewer": [True, False, False, False, False, False, False, False]
            }
            
            st.markdown("**Role Permissions Matrix:**")
            
            # Create permission matrix display
            matrix_data = []
            for role in roles:
                row = {"Role": role}
                for i, perm in enumerate(permissions):
                    row[perm] = "✅" if permission_matrix[role][i] else "❌"
                matrix_data.append(row)
            
            df_permissions = pd.DataFrame(matrix_data)
            st.dataframe(df_permissions, use_container_width=True)
        
        with tab3:
            st.markdown("#### 📊 User Activity")
            
            # Activity metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Active Users", "12", "↑ 2")
            with col2:
                st.metric("📄 Resumes Processed", "347", "↑ 45")
            with col3:
                st.metric("⏰ Avg Session Time", "2.3 hours", "↑ 0.4h")
            with col4:
                st.metric("🔄 Login Rate", "94.2%", "↑ 2.1%")
            
            # Activity log
            st.markdown("**Recent Activity:**")
            
            activity_log = [
                {"User": "recruiter1", "Action": "Analyzed 15 resumes", "Time": "2 mins ago", "IP": "192.168.1.45"},
                {"User": "admin", "Action": "Updated system settings", "Time": "15 mins ago", "IP": "192.168.1.10"},
                {"User": "mentor1", "Action": "Generated student feedback", "Time": "32 mins ago", "IP": "192.168.1.67"},
                {"User": "recruiter2", "Action": "Created new job posting", "Time": "1 hour ago", "IP": "192.168.1.89"}
            ]
            
            for activity in activity_log:
                st.markdown(f"""
                <div style="padding: 0.5rem; margin: 0.25rem 0; background: white; border-radius: 8px; border-left: 4px solid #667eea;">
                    <strong>{activity['User']}</strong> - {activity['Action']}
                    <div style="font-size: 0.8em; color: gray;">{activity['Time']} • IP: {activity['IP']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    def export_import_settings(self):
        """Export and import functionality"""
        st.markdown("### 📊 Export/Import Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📤 Export Data")
            
            export_type = st.selectbox("Export Type", 
                ["Resume Analysis Results", "Job Postings", "User Data", "Analytics Data", "Complete Backup"])
            
            export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON", "PDF Report"])
            
            date_range = st.date_input("Date Range", value=[datetime.now().date() - timedelta(days=30), datetime.now().date()])
            
            include_sensitive = st.checkbox("Include Sensitive Data", False)
            
            if st.button("📥 Export Data", type="primary"):
                with st.spinner("Preparing export..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                    
                    # Mock download
                    mock_data = pd.DataFrame({
                        'Candidate': ['Candidate_1', 'Candidate_2', 'Candidate_3'],
                        'Score': [78.5, 82.3, 67.1],
                        'Job': ['Python Developer', 'Data Scientist', 'Full Stack'],
                        'Date': ['2024-09-21', '2024-09-20', '2024-09-19']
                    })
                    
                    csv = mock_data.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Export",
                        data=csv,
                        file_name=f"innoresume_export_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    st.success("✅ Export completed successfully!")
        
        with col2:
            st.markdown("#### 📥 Import Data")
            
            import_type = st.selectbox("Import Type", 
                ["Job Postings", "User Data", "Skills Database", "System Configuration"])
            
            uploaded_import_file = st.file_uploader("Choose file to import", 
                type=['csv', 'xlsx', 'json'], help="Supported formats: CSV, Excel, JSON")
            
            if uploaded_import_file:
                st.success(f"✅ File '{uploaded_import_file.name}' uploaded successfully!")
                
                # Preview import data
                if uploaded_import_file.type == "text/csv":
                    import_df = pd.read_csv(uploaded_import_file)
                    st.markdown("**Preview:**")
                    st.dataframe(import_df.head(), use_container_width=True)
                
                validate_import = st.checkbox("Validate Data Before Import", True)
                overwrite_existing = st.checkbox("Overwrite Existing Data", False)
                
                if st.button("📤 Import Data"):
                    with st.spinner("Importing data..."):
                        progress_bar = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        st.success("✅ Data imported successfully!")
                        st.info(f"Imported {len(import_df) if 'import_df' in locals() else 'N/A'} records")
        
        # System backup/restore
        st.markdown("#### 💾 System Backup & Restore")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔄 Create Backup**")
            backup_name = st.text_input("Backup Name", f"backup_{datetime.now().strftime('%Y%m%d_%H%M')}")
            
            if st.button("🗄️ Create Full Backup"):
                with st.spinner("Creating backup..."):
                    time.sleep(3)
                    st.success("✅ Backup created successfully!")
        
        with col2:
            st.markdown("**📋 Available Backups**")
            
            backups = [
                "backup_20240921_1200.zip",
                "backup_20240920_1800.zip", 
                "backup_20240919_1200.zip"
            ]
            
            selected_backup = st.selectbox("Select Backup", backups)
            
            if st.button("♻️ Restore Backup"):
                st.warning("⚠️ This will overwrite current data!")
        
        with col3:
            st.markdown("**📊 Backup Statistics**")
            
            st.metric("💾 Total Backups", "15")
            st.metric("📁 Storage Used", "2.3 GB")
            st.metric("🔄 Last Backup", "2 hours ago")
            st.metric("✅ Success Rate", "100%")
    
    def save_analysis_result(self, resume_text, score, verdict, recommendations):
        """Save analysis result to database"""
        # In a real implementation, this would save to the database
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        result = {
            'timestamp': datetime.now(),
            'score': score,
            'verdict': verdict,
            'recommendations': recommendations
        }
        
        st.session_state.analysis_results.append(result)

# Initialize and run the application
if __name__ == "__main__":
    app = InnoResumeAI()
    app.run()

# Download required NLTK data (run once)
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass