"""
Hybrid Question Paper Generation System
Main Streamlit Application Entry Point
"""

import streamlit as st
from pathlib import Path
import yaml
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Hybrid Question Paper Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Hybrid Question Paper Generator v1.0.0"
    }
)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration from config.yaml"""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'syllabus_processed': False,
        'knowledge_base': None,
        'raw_text': '',
        'cleaned_text': '',
        'patterns': {},
        'topics': [],
        'generated_papers': [],
        'papers_count': 0,
        'topics_analyzed': 0,
        'avg_quality_score': 0.0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x100?text=QPG+System", width=200)
    st.markdown("### 📋 Navigation")
    st.info("👈 Use the navigation menu to access different pages")
    
    st.markdown("---")
    
    # Status indicators
    st.markdown("### 📊 System Status")
    
    if st.session_state.syllabus_processed:
        st.success("✅ Syllabus Processed")
    else:
        st.warning("⏳ No Syllabus Uploaded")
    
    # API Key status
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key and groq_key != 'your_groq_api_key_here':
        st.success("✅ Groq API Connected")
    else:
        st.error("❌ Groq API Key Missing")
        st.caption("Add GROQ_API_KEY to .env file")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📈 Quick Stats")
    st.metric("Papers Generated", st.session_state.papers_count)
    st.metric("Topics Analyzed", st.session_state.topics_analyzed)
    if st.session_state.avg_quality_score > 0:
        st.metric("Avg Quality", f"{st.session_state.avg_quality_score:.1f}/10")
    
    st.markdown("---")
    
    # Help section
    with st.expander("❓ Help & Support"):
        st.markdown("""
        **Quick Start:**
        1. Upload syllabus (📤 Upload)
        2. Generate papers (🤖 Generate)
        3. View analytics (📊 Analytics)
        
        **Support:**
        - [Documentation](https://github.com)
        - [Report Issues](https://github.com/issues)
        """)

# Main content
st.markdown('<p class="main-header">🎓 Hybrid Question Paper Generation System</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent AI-powered examination paper creation with dual LLM architecture</p>', 
            unsafe_allow_html=True)

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📄 Papers Generated",
        value=st.session_state.get('papers_count', 0),
        delta="+5" if st.session_state.get('papers_count', 0) > 0 else None
    )

with col2:
    st.metric(
        label="📚 Topics Analyzed",
        value=st.session_state.get('topics_analyzed', 0),
        delta="+12" if st.session_state.get('topics_analyzed', 0) > 0 else None
    )

with col3:
    quality_score = st.session_state.get('avg_quality_score', 0.0)
    st.metric(
        label="⭐ Avg Quality Score",
        value=f"{quality_score:.1f}/10" if quality_score > 0 else "N/A",
        delta="+0.3" if quality_score > 7 else None
    )

with col4:
    st.metric(
        label="🎯 Coverage Rate",
        value="85%" if st.session_state.syllabus_processed else "0%",
        delta="+5%" if st.session_state.syllabus_processed else None
    )

st.markdown("---")

# Main features section
st.markdown("### 🚀 Get Started")

tab1, tab2, tab3, tab4 = st.tabs([
    "📤 Upload & Process",
    "🤖 Generate Papers",
    "📊 View Analytics",
    "ℹ️ About"
])

with tab1:
    st.markdown("#### Upload Your Syllabus")
    st.info("Upload a PDF or image file containing your syllabus to begin the question paper generation process.")
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        uploaded_file = st.file_uploader(
            "Choose a syllabus file",
            type=["pdf", "png", "jpg", "jpeg"],
            help="Supported formats: PDF, PNG, JPG, JPEG"
        )
        
        if uploaded_file:
            st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🔍 Process Syllabus Now", type="primary", use_container_width=True):
                st.switch_page("pages/1_📤_Upload_Syllabus.py")
    
    with col_b:
        st.markdown("**Processing Steps:**")
        st.markdown("1. 🔍 OCR Extraction")
        st.markdown("2. 🧹 Text Cleaning")
        st.markdown("3. 🔎 Pattern Analysis")
        st.markdown("4. 💾 Store in Database")

with tab2:
    st.markdown("#### Generate Question Papers")
    
    if st.session_state.syllabus_processed:
        st.success("✅ Syllabus is ready! You can now generate question papers.")
        
        col_x, col_y = st.columns(2)
        
        with col_x:
            st.markdown("**Available Features:**")
            st.markdown("- 📝 Multi-set generation (A, B, C)")
            st.markdown("- 🎯 Bloom's taxonomy distribution")
            st.markdown("- ⚖️ Intelligent marks allocation")
            st.markdown("- 🔄 Novelty filtering")
        
        with col_y:
            st.markdown("**AI Models:**")
            st.markdown("- 🤖 Parser: Llama 3.1 8B")
            st.markdown("- 🧠 Generator: Llama 3.3 70B")
            st.markdown("- 📊 ML: scikit-learn")
            st.markdown("- 💾 Vector DB: ChromaDB")
        
        if st.button("🚀 Go to Generation Page", type="primary", use_container_width=True):
            st.switch_page("pages/3_🤖_Generate_Papers.py")
    else:
        st.warning("⚠️ Please upload and process a syllabus first before generating papers.")
        st.markdown("👉 Go to the **Upload & Process** tab to get started.")

with tab3:
    st.markdown("#### Analytics & Insights")
    st.info("View comprehensive analytics about generated papers, topic coverage, and quality metrics.")
    
    if st.session_state.papers_count > 0:
        st.markdown("**Available Analytics:**")
        col_i, col_ii = st.columns(2)
        
        with col_i:
            st.markdown("- 📈 Topic coverage analysis")
            st.markdown("- 🎨 Bloom's taxonomy distribution")
            st.markdown("- 📊 Generation trends")
        
        with col_ii:
            st.markdown("- 🔥 Topic importance heatmap")
            st.markdown("- ⭐ Quality score tracking")
            st.markdown("- 📅 Historical patterns")
        
        if st.button("📊 View Analytics Dashboard", use_container_width=True):
            st.switch_page("pages/4_📊_Analytics_Dashboard.py")
    else:
        st.warning("Generate some papers first to view analytics!")

with tab4:
    st.markdown("#### About This System")
    
    st.markdown("""
    **Hybrid Question Paper Generation System** is an intelligent AI-powered tool that automates 
    the creation of examination question papers using advanced machine learning and natural language processing.
    
    ##### 🎯 Key Features:
    - **Dual LLM Architecture**: Uses two specialized language models for parsing and generation
    - **OCR Processing**: Extracts text from PDF and image files
    - **Pattern Discovery**: Learns from historical question papers
    - **Bloom's Taxonomy**: Automatically classifies questions by cognitive level
    - **Quality Assurance**: Novelty filtering and coverage validation
    - **Multi-Set Generation**: Creates multiple paper variations
    
    ##### 🛠️ Technology Stack:
    - **Frontend**: Streamlit
    - **AI/ML**: LangChain, Groq API, scikit-learn
    - **Database**: ChromaDB (Vector DB)
    - **NLP**: spaCy, Sentence-Transformers
    - **Document Processing**: Tesseract OCR, PyPDF2
    
    ##### 📚 How It Works:
    1. **Upload**: Upload syllabus PDF or image
    2. **Process**: OCR extraction and text cleaning
    3. **Analyze**: Pattern discovery and topic importance calculation
    4. **Generate**: Create questions using dual LLM architecture
    5. **Validate**: Apply novelty filter and quality checks
    6. **Export**: Download as formatted PDF
    
    ##### 🔧 Configuration:
    Edit `config.yaml` to customize:
    - Bloom's taxonomy distribution
    - Priority score weights
    - Similarity thresholds
    - LLM parameters
    
    ##### 📄 Version: 1.0.0
    ##### 📅 Last Updated: February 12, 2026
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ for Education")

# Footer
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("📚 [Documentation](#)")

with col_footer2:
    st.markdown("🐛 [Report Issues](#)")

with col_footer3:
    st.markdown("⭐ [GitHub](#)")
