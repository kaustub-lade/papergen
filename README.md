# 🎓 Hybrid Question Paper Generation System

An intelligent AI-powered system for automatically generating educational question papers using dual LLM architecture, OCR processing, and ML-based analysis.

## 🚀 Features

- **PDF/Image Syllabus Upload** - Multi-format input support with OCR
- **Intelligent Analysis** - Priority scoring, topic importance estimation
- **Bloom's Taxonomy Classification** - Automatic cognitive level detection
- **Dual LLM Architecture** - Llama 3.1 8B (Parser) + Llama 3.3 70B (Generator)
- **Pattern Discovery** - Cross-subject analysis from past papers
- **Quality Assurance** - Novelty filtering and coverage validation
- **Multi-Set Generation** - Generate Sets A, B, C with variations
- **Analytics Dashboard** - Real-time visualization and insights

## 📁 Project Structure

```
papergen/
├── app.py                          # Main Streamlit entry point
├── pages/                          # Multi-page app
│   ├── 1_📤_Upload_Syllabus.py
│   ├── 2_🔍_Pattern_Analysis.py
│   ├── 3_🤖_Generate_Papers.py
│   ├── 4_📊_Analytics_Dashboard.py
│   └── 5_⚙️_Settings.py
├── modules/
│   ├── __init__.py
│   ├── ocr_processor.py           # OCR & watermark removal
│   ├── pattern_discovery.py       # Cross-subject analysis
│   ├── priority_calculator.py     # Priority scoring
│   ├── llm_engine.py              # Dual LLM orchestration
│   ├── bloom_classifier.py        # Bloom's taxonomy classifier
│   ├── novelty_filter.py          # Similarity checking
│   └── paper_generator.py         # Final generation logic
├── utils/
│   ├── __init__.py
│   ├── chromadb_handler.py        # Vector DB operations
│   ├── pdf_generator.py           # Export to PDF
│   └── visualization.py           # Custom charts
├── models/                         # Trained ML models
│   └── .gitkeep
├── data/
│   ├── syllabi/                   # Uploaded syllabi
│   ├── past_papers/               # Historical papers
│   └── generated_papers/          # Output
├── assets/
│   └── logo.png                   # App logo
├── .env.example                   # Environment variables template
├── .gitignore
├── requirements.txt
├── config.yaml                    # Configuration file
└── README.md
```

## 🛠️ Tech Stack

### Core Framework
- **Streamlit** - Web application framework
- **Python 3.10+** - Programming language

### AI/ML
- **LangChain** - LLM orchestration
- **Groq API** - Fast LLM inference (Llama models)
- **ChromaDB** - Vector database
- **Sentence Transformers** - Embeddings
- **scikit-learn** - ML models (Logistic Regression, K-Means)
- **spaCy** - NLP processing

### Document Processing
- **Tesseract OCR** - Text extraction
- **pdf2image** - PDF to image conversion
- **PyPDF2** - PDF manipulation
- **ReportLab** - PDF generation

### Data & Visualization
- **Pandas** - Data manipulation
- **Plotly** - Interactive charts
- **NumPy** - Numerical operations

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Tesseract OCR installed on system
- Groq API key (get from https://console.groq.com)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd papergen
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR
**Windows:**
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and add to PATH
3. Update `TESSERACT_PATH` in `.env`

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

### Step 5: Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### Step 6: Configure Environment
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required environment variables:
```
GROQ_API_KEY=your_groq_api_key_here
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe  # Windows only
CHROMADB_PATH=./data/chromadb
```

## 🚀 Running the Application

### Development Mode
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Production Mode
```bash
streamlit run app.py --server.port 8080 --server.headless true
```

## 📖 Usage Guide

### 1. Upload Syllabus
- Navigate to "📤 Upload Syllabus" page
- Upload PDF or image file
- Select OCR engine (Tesseract recommended for local use)
- Configure processing options
- Click "🚀 Start Processing"

### 2. Upload Past Papers (Optional)
- Upload previous question papers (B, C, D sets)
- System will analyze patterns for better generation

### 3. Generate Question Papers
- Navigate to "🤖 Generate Papers" page
- Configure generation settings:
  - Number of sets (1-5)
  - Total marks
  - Bloom's Taxonomy distribution
  - LLM models and temperature
- Review syllabus coverage
- Click "🚀 Generate Question Papers"
- Download generated papers as PDF

### 4. View Analytics
- Navigate to "📊 Analytics Dashboard"
- View topic coverage analysis
- Check Bloom's taxonomy distribution
- Monitor generation trends
- Analyze quality metrics

## 🔧 Configuration

Edit `config.yaml` to customize:
- Default Bloom's taxonomy distribution
- Priority score weights
- Similarity thresholds
- PDF formatting options
- Model parameters

## 🧪 Testing

Run tests:
```bash
pytest tests/
```

## 📊 ML Models

### Topic Importance Estimator
- Algorithm: Logistic Regression
- Features: Syllabus weight, historical frequency
- Training: Automatically improves with usage

### Bloom's Taxonomy Classifier
- Algorithm: Multi-class classification
- Categories: Remember, Understand, Apply, Analyze, Evaluate, Create
- Training: Fine-tuned on educational question datasets

### Novelty Filter
- Algorithm: Cosine similarity with Sentence-BERT
- Threshold: 0.85 (configurable)
- Purpose: Avoid duplicate questions

## 🐛 Troubleshooting

### Tesseract OCR Error
```
Error: Tesseract not found
Solution: Install Tesseract and update TESSERACT_PATH in .env
```

### Groq API Rate Limit
```
Error: Rate limit exceeded
Solution: Wait 60 seconds or upgrade Groq plan
```

### ChromaDB Lock Error
```
Error: Database locked
Solution: Close other instances of the app
```

## 🚀 Deployment

### Streamlit Community Cloud (Free)
1. Push code to GitHub
2. Connect repository at https://share.streamlit.io
3. Add secrets in dashboard
4. Deploy

### Docker
```bash
docker build -t papergen .
docker run -p 8501:8501 papergen
```

### Heroku
```bash
heroku create papergen-app
git push heroku main
```

## 📝 API Keys Required

1. **Groq API** (Required)
   - Get from: https://console.groq.com
   - Free tier: 14,400 requests/day
   - Used for: LLM inference

2. **Nanonets OCR** (Optional)
   - Get from: https://nanonets.com
   - Alternative to Tesseract for better accuracy
   - Free tier: 100 requests/month

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License - See LICENSE file for details

## 👥 Authors

- Kaustub Lade

## 🙏 Acknowledgments

- Llama 3.1/3.3 models by Meta
- Groq for fast inference
- Streamlit for the amazing framework
- ChromaDB for vector storage

## 📚 Documentation

For detailed documentation, visit:
- [User Guide](docs/user_guide.md)
- [Developer Guide](docs/developer_guide.md)
- [API Reference](docs/api_reference.md)

## 🔗 Links

- [GitHub Repository](#)
- [Demo Video](#)
- [Documentation](#)
- [Issues & Support](#)

---

**Version:** 1.0.0  
**Last Updated:** February 12, 2026
