# 💼 NOC Job Matcher

An AI-powered web application that matches job descriptions to Canadian National Occupational Classification (NOC) codes using advanced semantic search and duty-by-duty comparison.

## 🌟 Features

- **Advanced Semantic Matching**: Uses MPNet transformer model for deep understanding of job descriptions
- **Duty-by-Duty Analysis**: Compares specific job responsibilities against NOC duties with confidence scores
- **Hybrid Scoring**: Combines overall context (40%) with specific duty matching (60%)
- **Interactive UI**: View matched duties, highlighted keywords, and detailed NOC profiles
- **6,140 Duty Embeddings**: Precise matching across all NOC duties

## 🚀 Live Demo

[Visit the deployed app](your-app-url-here)

## 📊 Data Source

- **NOC 2021 Version 1.0**
- 516 Unit Groups
- Complete with main duties, example titles, and employment requirements
- Data scraped from: https://noc.esdc.gc.ca

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Model**: sentence-transformers (all-mpnet-base-v2)
- **Processing**: NumPy, Pandas, scikit-learn
- **Semantic Search**: Cosine similarity with embeddings

## 💻 Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/noc-job-matcher.git
cd noc-job-matcher

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate embeddings (first time only - takes ~5 minutes)
python prepare_embeddings.py

# Run the app
streamlit run app.py
```

## 📦 Project Structure

```
noc-job-matcher/
├── app.py                      # Main Streamlit application
├── prepare_embeddings.py       # Generate AI embeddings
├── noc_data_full.csv          # NOC data with duties
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── README.md
```

## 🔧 How It Works

1. **Data Processing**: Parses NOC data and creates weighted text representations
2. **Embedding Generation**: 
   - Profile embeddings (516 NOC codes)
   - Duty embeddings (6,140 individual duties)
3. **Matching Process**:
   - Extracts responsibilities from job description
   - Compares against all NOC duties
   - Calculates hybrid similarity scores
4. **Results Display**: Shows top matches with duty-level confidence scores

## 📈 Accuracy Improvements

- **MPNet Model**: 2x embedding dimensions (768 vs 384) for better context understanding
- **Responsibility Extraction**: Automatically identifies key duties from job descriptions
- **Weighted Fields**: 
  - Main duties: 3x weight
  - Title: 2x weight
  - Description: 1.5x weight
  - Requirements: 1x weight

## 🌐 Deployment

### Streamlit Community Cloud (Recommended)

1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy automatically

**Note**: First deployment will take ~5 minutes to generate embeddings.

## 📝 Usage Example

```
Job Description Input:
"We are seeking a Software Developer to design and develop web applications. 
Responsibilities include writing clean code, reviewing pull requests, 
mentoring junior developers, and collaborating with product teams."

Results:
✅ NOC 21232 - Software developers and programmers (87.3% match)
   Matched Duties:
   • Write, modify, integrate and test software code (92% confidence)
   • Maintain existing programs (85% confidence)
```

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Created with ❤️ using GitHub Copilot

## 🙏 Acknowledgments

- Government of Canada for NOC data
- Sentence Transformers for the AI models
- Streamlit for the web framework
