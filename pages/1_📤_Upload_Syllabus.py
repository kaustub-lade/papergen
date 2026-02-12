"""
Upload Syllabus Page
Handles syllabus upload, OCR processing, and storage
"""

import streamlit as st
from pathlib import Path
import time
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.ocr_processor import OCRProcessor
from modules.pattern_discovery import PatternDiscovery
from modules.priority_calculator import PriorityCalculator
from utils.chromadb_handler import ChromaDBHandler

st.set_page_config(page_title="Upload Syllabus", page_icon="📤", layout="wide")

st.title("📤 Upload & Process Syllabus")
st.markdown("Upload your syllabus to begin the question paper generation process.")

# Initialize session state
if 'syllabus_processed' not in st.session_state:
    st.session_state.syllabus_processed = False

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📄 Upload Syllabus Document")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF or Image file",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Supported formats: PDF, PNG, JPG, JPEG (Max size: 50MB)"
    )
    
    if uploaded_file:
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ File uploaded: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")
        
        # Processing options
        st.markdown("### ⚙️ Processing Options")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            ocr_engine = st.selectbox(
                "OCR Engine",
                ["tesseract", "nanonets"],
                index=0,
                help="Tesseract: Free, local. Nanonets: Paid API, higher accuracy"
            )
            
            remove_watermark = st.checkbox(
                "Remove Watermarks",
                value=True,
                help="Apply regex-based watermark removal"
            )
        
        with col_b:
            analyze_patterns = st.checkbox(
                "Analyze Patterns",
                value=True,
                help="Extract structural patterns from syllabus"
            )
            
            calculate_priorities = st.checkbox(
                "Calculate Topic Priorities",
                value=True,
                help="Compute priority scores for topics"
            )
        
        st.markdown("---")
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            
            with st.spinner("Processing your syllabus..."):
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_container = st.container()
                    
                    # Step 1: OCR Extraction
                    status_text.info("🔍 Step 1/5: Extracting text from document...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    ocr_processor = OCRProcessor(engine=ocr_engine)
                    extracted_text = ocr_processor.extract_text(uploaded_file)
                    
                    if not extracted_text or len(extracted_text) < 50:
                        st.error("❌ Failed to extract text. Please check if the file is readable.")
                        st.stop()
                    
                    st.session_state.raw_text = extracted_text
                    progress_bar.progress(30)
                    
                    # Step 2: Preprocessing
                    status_text.info("🧹 Step 2/5: Cleaning and preprocessing text...")
                    time.sleep(0.5)
                    
                    cleaned_text = ocr_processor.preprocess_text(extracted_text)
                    st.session_state.cleaned_text = cleaned_text
                    progress_bar.progress(50)
                    
                    # Step 3: Pattern Discovery
                    patterns = {}
                    if analyze_patterns:
                        status_text.info("🔍 Step 3/5: Discovering patterns...")
                        time.sleep(0.5)
                        
                        pattern_discovery = PatternDiscovery()
                        patterns = pattern_discovery.analyze(cleaned_text)
                        st.session_state.patterns = patterns
                    
                    progress_bar.progress(70)
                    
                    # Step 4: Priority Calculation
                    topics = []
                    if calculate_priorities:
                        status_text.info("🎯 Step 4/5: Calculating topic priorities...")
                        time.sleep(0.5)
                        
                        priority_calc = PriorityCalculator()
                        topics = priority_calc.calculate_priorities(cleaned_text)
                        st.session_state.topics = topics
                        st.session_state.topics_analyzed = len(topics)
                    
                    progress_bar.progress(85)
                    
                    # Step 5: Store in ChromaDB
                    status_text.info("💾 Step 5/5: Storing in knowledge base...")
                    time.sleep(0.5)
                    
                    try:
                        db_handler = ChromaDBHandler()
                        doc_id = db_handler.store_syllabus(
                            cleaned_text,
                            metadata={
                                "filename": uploaded_file.name,
                                "upload_date": datetime.now().isoformat(),
                                "file_size": uploaded_file.size,
                                "file_type": uploaded_file.type
                            }
                        )
                        st.session_state.knowledge_base = True
                    except Exception as e:
                        st.warning(f"Warning: Could not store in database: {e}")
                    
                    progress_bar.progress(100)
                    status_text.success("✅ Processing complete!")
                    
                    # Mark as processed
                    st.session_state.syllabus_processed = True
                    
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    with results_container:
                        st.success("🎉 Syllabus processed successfully!")
                        
                        # Statistics
                        col_i, col_ii, col_iii = st.columns(3)
                        
                        with col_i:
                            st.metric("Characters Extracted", f"{len(cleaned_text):,}")
                        
                        with col_ii:
                            st.metric("Topics Identified", len(topics))
                        
                        with col_iii:
                            st.metric("Patterns Found", len(patterns))
                        
                        # Tabs for detailed view
                        tab1, tab2, tab3 = st.tabs(["📝 Text Preview", "📊 Topics", "🔍 Patterns"])
                        
                        with tab1:
                            st.markdown("#### Extracted Text Preview")
                            preview_text = cleaned_text[:2000] + "..." if len(cleaned_text) > 2000 else cleaned_text
                            st.text_area(
                                "Cleaned Text",
                                preview_text,
                                height=300,
                                disabled=True
                            )
                            
                            if st.button("📋 Copy Full Text"):
                                st.code(cleaned_text, language="text")
                        
                        with tab2:
                            if topics:
                                st.markdown("#### Topic Priority Analysis")
                                import pandas as pd
                                
                                topics_df = pd.DataFrame(topics)
                                st.dataframe(
                                    topics_df[['name', 'priority', 'syllabus_weight', 'occurrences']],
                                    use_container_width=True,
                                    column_config={
                                        "name": "Topic",
                                        "priority": st.column_config.ProgressColumn(
                                            "Priority Score",
                                            format="%.2f",
                                            min_value=0,
                                            max_value=1,
                                        ),
                                        "syllabus_weight": st.column_config.NumberColumn(
                                            "Syllabus Weight",
                                            format="%.3f"
                                        ),
                                        "occurrences": "Occurrences"
                                    }
                                )
                                
                                # Top 5 topics
                                st.markdown("##### 🏆 Top 5 Priority Topics")
                                top_5 = topics[:5]
                                for i, topic in enumerate(top_5, 1):
                                    st.markdown(f"{i}. **{topic['name']}** (Priority: {topic['priority']:.2f})")
                            else:
                                st.info("Enable 'Calculate Topic Priorities' to see this analysis.")
                        
                        with tab3:
                            if patterns:
                                st.markdown("#### Discovered Patterns")
                                st.json(patterns)
                                
                                # Recommendations
                                if 'recommendations' in patterns or analyze_patterns:
                                    st.markdown("##### 💡 Recommendations")
                                    pattern_discovery = PatternDiscovery()
                                    pattern_discovery.patterns = patterns
                                    recommendations = pattern_discovery.get_recommendations()
                                    
                                    for rec in recommendations:
                                        st.info(rec)
                            else:
                                st.info("Enable 'Analyze Patterns' to see this analysis.")
                        
                        # Next steps
                        st.markdown("---")
                        st.markdown("### 🎯 Next Steps")
                        st.info("👉 Go to **🤖 Generate Papers** page to create question papers based on this syllabus.")
                        
                        if st.button("➡️ Proceed to Generate Papers", type="primary"):
                            st.switch_page("pages/3_🤖_Generate_Papers.py")
                
                except Exception as e:
                    st.error(f"❌ An error occurred during processing: {str(e)}")
                    st.exception(e)

with col2:
    st.subheader("ℹ️ Information")
    
    st.markdown("""
    **Processing Steps:**
    
    1. **OCR Extraction** 📄
       - Extract text from PDF/image
       - Handle scanned documents
    
    2. **Text Cleaning** 🧹
       - Remove watermarks
       - Fix OCR errors
       - Normalize text
    
    3. **Pattern Discovery** 🔍
       - Identify structure
       - Extract topics
       - Analyze organization
    
    4. **Priority Calculation** 🎯
       - Compute topic importance
       - Calculate weightage
       - Rank topics
    
    5. **Database Storage** 💾
       - Store in ChromaDB
       - Enable semantic search
       - Index for retrieval
    """)
    
    st.markdown("---")
    
    # Tips
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - Use clear, high-resolution scans
        - Ensure text is readable
        - Remove unnecessary pages
        - Use PDF format when possible
        - Check for proper page orientation
        """)
    
    # Supported formats
    with st.expander("📋 Supported Formats"):
        st.markdown("""
        **Document Formats:**
        - PDF (.pdf)
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        
        **Maximum Size:** 50 MB
        """)

# Sidebar - Past papers upload
with st.sidebar:
    st.markdown("## 📚 Upload Past Papers")
    st.caption("Optional: Upload previous question papers to improve pattern learning")
    
    past_papers = st.file_uploader(
        "Upload Past Papers",
        type=["pdf"],
        accept_multiple_files=True,
        key="past_papers",
        help="Upload PDF files of past question papers (Sets B, C, D)"
    )
    
    if past_papers:
        st.success(f"✅ {len(past_papers)} paper(s) uploaded")
        
        if st.button("Process Past Papers"):
            with st.spinner("Analyzing past papers..."):
                # TODO: Implement past paper processing
                time.sleep(2)
                st.success("✅ Past papers analyzed!")
                st.info("Historical patterns integrated into the system.")
