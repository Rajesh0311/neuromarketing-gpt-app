# Neuromarketing GPT App

AI-powered sentiment analysis with professional marketing visual generation. Integrates with Canva Pro for seamless design workflows.

## 🧠 Neuro Deep Research Module

The core module providing deep research capabilities for neurological data analysis with a modern Streamlit interface.

### ✨ Features

- **Dataset Search**: Fetch and browse neurological datasets from OpenNeuro
- **Sentiment Analysis**: AI-powered sentiment analysis using OpenAI GPT models
- **Visual Generation**: Generate marketing visuals based on neurological data
- **Report Export**: Export research findings in multiple formats (PDF, DOCX, HTML, JSON)
- **Security**: Environment-based API key management with validation

### 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/Rajesh0311/neuromarketing-gpt-app.git
cd neuromarketing-gpt-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export OPENAI_API_KEY="your_openai_api_key"
export OPENNEURO_API_KEY="your_openneuro_api_key"
export GITHUB_TOKEN="your_github_token"  # Optional
```

### 🚀 Usage

#### Running the Streamlit Application

```bash
streamlit run neuro_deep_research_module.py
```

#### Using as a Python Module

```python
from neuro_deep_research_module import NeuroResearchModule

# Initialize the module
research = NeuroResearchModule()

# Fetch datasets
datasets = research.fetch_openneuro_datasets()

# Analyze sentiment
result = research.analyze_sentiment_data("Your neurological research text here")

# Generate visuals
visual = research.generate_marketing_visuals({"sentiment": "positive"})

# Export report
report_path = research.export_research_report(findings, format="pdf")
```

### 🔒 Security Features

- **No Hardcoded Credentials**: All API keys are loaded from environment variables
- **Input Validation**: Comprehensive validation for all user inputs
- **Error Handling**: Robust exception handling with custom error types
- **API Key Validation**: Automatic validation of API key formats and presence

### 🛠️ Recent Fixes

This module has been completely refactored to address critical issues:

#### ✅ Syntax Errors Fixed
- Fixed malformed dataclass with incorrect docstring indentation
- Corrected class method declaration order
- Fixed inconsistent indentation throughout the file

#### ✅ Security Vulnerabilities Resolved
- Removed all hardcoded API keys from source code
- Implemented secure environment variable usage for API keys
- Added proper API key validation and format checking

#### ✅ Missing Implementations Completed
- Complete `fetch_openneuro_datasets` method with proper GraphQL queries
- Fixed undefined variables and implemented proper error handling
- Added comprehensive input validation

#### ✅ Code Structure Improvements
- Fixed import statements and removed unused imports
- Added proper type hints and typing imports
- Implemented consistent exception handling with custom error classes
- Fixed async/await usage for Streamlit compatibility

#### ✅ Logic Errors Corrected
- Fixed variables used without definition (node, response)
- Completed all missing method bodies
- Implemented proper data flow and error propagation

#### ✅ Best Practices Implemented
- Added comprehensive input validation
- Implemented structured logging throughout
- Added detailed docstrings for all functions and classes
- Applied consistent coding standards and PEP 8 compliance

### 🧪 Testing

Run the test suite to verify all functionality:

```bash
python test_neuro_module.py
```

Run the Streamlit demo:

```bash
python demo_streamlit.py
```

### 📋 API Reference

#### NeuroDataset Class
```python
@dataclass
class NeuroDataset:
    id: str
    name: str
    description: str
    participants: int
    data_type: str
```

#### NeuroResearchModule Class

**Methods:**
- `fetch_openneuro_datasets(query_params=None)`: Fetch datasets from OpenNeuro
- `analyze_sentiment_data(text_data)`: Analyze sentiment of neurological text
- `generate_marketing_visuals(data)`: Generate marketing visuals
- `export_research_report(findings, format='pdf')`: Export research reports

**Custom Exceptions:**
- `APIKeyError`: API key validation failures
- `DataFetchError`: Data fetching operation failures
- `ValidationError`: Input validation failures

### 🔧 Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENNEURO_API_KEY=your_openneuro_api_key_here
GITHUB_TOKEN=your_github_token_here
```

### 📦 Dependencies

- `streamlit>=1.28.0`: Web application framework
- `requests>=2.31.0`: HTTP library for API calls
- `pandas>=2.0.0`: Data manipulation and analysis
- `python-dotenv>=1.0.0`: Environment variable loading

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure everything works
5. Submit a pull request

### 📄 License

MIT License - see LICENSE file for details.

### 🆘 Support

For support and questions, please refer to the support.html page or contact the development team.
