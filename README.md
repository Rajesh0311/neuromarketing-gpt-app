# NeuroMarketing GPT App

🧠 AI-powered sentiment analysis with professional marketing visual generation and comprehensive neuroscience dataset integration.

## Features

### 🧠 Deep Research Module
- **Multi-source Dataset Integration**: Access OpenNeuro, Zenodo, PhysioNet, and more
- **Literature Search**: Search PubMed for relevant neuroscience research
- **No API Keys Required**: Most features work with open datasets (no authentication needed)
- **OpenNeuro Fixed**: Resolved API key issues - now works with public GraphQL access

### 📊 Sentiment Analysis
- **Multi-dimensional Analysis**: 20+ emotional, business, and psychological dimensions
- **Graceful Degradation**: Works with or without OpenAI API key
- **Real-time Processing**: Instant analysis and visualization
- **Professional Insights**: Marketing-focused metrics and recommendations

### 🎨 Visual Generation
- **Interactive Charts**: Plotly-powered visualizations
- **Marketing Dashboards**: Professional analytics displays
- **Export Ready**: Download reports and visualizations
- **Multiple Formats**: Support for various output types

### 📋 Research Management
- **Project Tracking**: Organize research activities
- **Comprehensive Reports**: Automated report generation
- **Data Quality Assessment**: Real-time validation
- **Export Capabilities**: Multiple output formats

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Rajesh0311/neuromarketing-gpt-app.git
cd neuromarketing-gpt-app

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Configuration (Optional)

Set your OpenAI API key for enhanced sentiment analysis:
- Go to Settings in the app
- Enter your OpenAI API key
- Or set the environment variable: `OPENAI_API_KEY=your_key_here`

**Note**: Most features work without any API keys using open datasets and basic analysis.

## Dataset Sources

### ✅ Currently Integrated
- **OpenNeuro**: Open neuroscience datasets (GraphQL API - no auth required)
- **Zenodo**: EEG and consumer neuroscience datasets
- **PhysioNet**: Comprehensive EEG databases
- **PubMed**: Neuroscience research papers

### 🔄 In Development
- IEEE DataPort Neuromarketing datasets
- OSF (Open Science Framework) studies
- Google Dataset Search integration

## Architecture

```
app.py                    # Main Streamlit application
deep_research_module.py   # Core research functionality
requirements.txt          # Python dependencies
.gitignore               # Git ignore rules
```

### Key Components

- **NeuroResearchModule**: Main research class with dataset integrations
- **render_deep_research_module()**: UI rendering function
- **integrate_deep_research_module()**: Integration function for main app

## API Integration Status

| Service | Status | Authentication | Notes |
|---------|--------|----------------|-------|
| OpenNeuro | ✅ Working | None required | Fixed GraphQL queries |
| Zenodo | ✅ Working | None required | Public API access |
| PhysioNet | ✅ Working | None required | Curated dataset list |
| PubMed | ✅ Working | None required | NCBI E-utilities |
| OpenAI | 🔄 Optional | API key required | Enhanced analysis only |

## Usage Examples

### 1. Dataset Discovery
```python
from deep_research_module import NeuroResearchModule

# Initialize module
research = NeuroResearchModule()

# Search OpenNeuro datasets
results = research.fetch_openneuro_datasets("EEG")
print(f"Found {results['count']} datasets")
```

### 2. Sentiment Analysis
```python
# Basic analysis (no API key required)
analysis = research.analyze_sentiment_data("Your marketing text here")
print(f"Sentiment: {analysis['basic']['overall_sentiment']}")
```

### 3. Literature Search
```python
# Search research papers
papers = research.search_pubmed_papers("neuromarketing")
print(f"Found {papers['count']} papers")
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **OpenNeuro API Error**: This has been fixed - no API key is needed
3. **Streamlit Issues**: Update Streamlit with `pip install --upgrade streamlit`

### Getting Help

- Check the in-app Support section
- Review error messages in the Streamlit interface
- Ensure internet connectivity for dataset access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the terms in the LICENSE file.

## Acknowledgments

- OpenNeuro for providing open neuroscience datasets
- Zenodo for hosting research datasets
- PhysioNet for EEG databases
- NCBI for PubMed access
