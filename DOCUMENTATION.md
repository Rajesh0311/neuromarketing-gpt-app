# 🧠 NeuroMarketing GPT Platform - Complete Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Installation Guide](#installation-guide)
3. [API Documentation](#api-documentation)
4. [Module Reference](#module-reference)
5. [Usage Examples](#usage-examples)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Deployment Instructions](#deployment-instructions)
8. [Performance Optimization](#performance-optimization)
9. [Environment Configuration](#environment-configuration)
10. [Contributing Guidelines](#contributing-guidelines)

---

## 📖 Overview

The **NeuroMarketing GPT Platform** is a revolutionary neuromarketing intelligence platform that integrates advanced sentiment analysis, neural simulation, environmental analysis, and deep research capabilities. This unified application combines all features from previous PRs into a comprehensive 10-tab interface.

### 🎯 Key Features

- **📊 Advanced Sentiment Analysis** - Multi-dimensional emotional and psychological profiling
- **🎭 Sarcasm & Irony Detection** - Advanced contextual analysis with cultural adaptation
- **🔬 Basic Neural Monitoring** - Real-time EEG processing and neural pattern analysis
- **🎨 Professional Visuals** - Publication-ready charts and marketing visualizations
- **📁 Media Input Hub** - Text, image, video, audio, and URL analysis capabilities
- **🏪 Environmental Simulation** - Advanced walkthrough recording and spatial analysis
- **📋 Reports & Export** - Professional multi-format report generation
- **🧠 NeuroInsight-Africa Platform** - Cultural adaptation for African markets
- **🔬 Deep Research Engine** - OpenNeuro, Zenodo, PhysioNet dataset integration
- **⚙️ System Integration** - Unified configuration and cross-module data management

---

## 🚀 Installation Guide

### Prerequisites

- **Python 3.8 or higher**
- **pip package manager**
- **Git** (for cloning repository)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/Rajesh0311/neuromarketing-gpt-app.git
cd neuromarketing-gpt-app

# Create virtual environment (recommended)
python -m venv neuromarketing_env
source neuromarketing_env/bin/activate  # On Windows: neuromarketing_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run main_neuromarketing_app.py
```

### Docker Installation

```bash
# Build Docker image
docker build -t neuromarketing-gpt .

# Run container
docker run -p 8501:8501 neuromarketing-gpt
```

### Verification

The application should open in your browser at `http://localhost:8501`. You should see the main dashboard with 10 tabs.

---

## 🔧 API Documentation

### Core Classes

#### AdvancedSentimentAnalyzer

Performs comprehensive sentiment analysis with multiple dimensions.

```python
from advanced_sentiment_module import AdvancedSentimentAnalyzer

analyzer = AdvancedSentimentAnalyzer()
result = analyzer.analyze_comprehensive_sentiment(
    text="Your marketing text here",
    analysis_type="comprehensive"
)
```

**Methods:**

- `analyze_comprehensive_sentiment(text: str, analysis_type: str) -> Dict[str, Any]`
  - **Parameters:**
    - `text`: Input text to analyze
    - `analysis_type`: Type of analysis ('basic', 'advanced', 'marketing', 'cultural')
  - **Returns:** Dictionary containing comprehensive analysis results

#### DigitalBrainTwin

Digital brain twin for marketing response simulation.

```python
from neural_simulation import DigitalBrainTwin

brain_twin = DigitalBrainTwin()
simulation = brain_twin.simulate_marketing_response(
    stimulus_text="Marketing content",
    consumer_type="analytical_buyer",
    duration=10.0
)
```

**Methods:**

- `simulate_marketing_response(stimulus_text: str, consumer_type: str, duration: float) -> Dict[str, Any]`
  - **Parameters:**
    - `stimulus_text`: Marketing content to analyze
    - `consumer_type`: Type of consumer profile
    - `duration`: Simulation duration in seconds
  - **Returns:** Dictionary containing simulation results

#### NeuroResearchModule

Multi-source dataset integration and research synthesis.

```python
from neuro_deep_research_module import NeuroResearchModule

research = NeuroResearchModule()
datasets = research.fetch_openneuro_datasets(query="EEG", limit=20)
```

**Methods:**

- `fetch_openneuro_datasets(query: str, limit: int) -> Dict[str, Any]`
- `fetch_zenodo_datasets(query: str, limit: int) -> Dict[str, Any]`
- `search_pubmed_papers(query: str, limit: int) -> Dict[str, Any]`

#### ProfessionalExporter

Professional report exporter for analysis results.

```python
from export_module import ProfessionalExporter

exporter = ProfessionalExporter()
report = exporter.export_comprehensive_report(
    analysis_data=data,
    format_type='html',
    template_style='professional'
)
```

### API Endpoints

The platform uses internal APIs for module communication. All APIs follow REST principles:

- **GET** `/api/sentiment/analyze` - Sentiment analysis
- **POST** `/api/neural/simulate` - Neural simulation
- **GET** `/api/research/datasets` - Dataset search
- **POST** `/api/export/report` - Report generation

---

## 📚 Module Reference

### Advanced Sentiment Module (`advanced_sentiment_module.py`)

**Purpose:** Multi-dimensional sentiment analysis with emotional profiling

**Key Features:**
- Emotion lexicon with 8 primary emotions
- Marketing-specific keyword analysis
- Cultural adaptation capabilities
- Psychological profiling (PAD model)
- Linguistic feature extraction

**Usage Example:**
```python
from advanced_sentiment_module import AdvancedSentimentAnalyzer

analyzer = AdvancedSentimentAnalyzer()
results = analyzer.analyze_comprehensive_sentiment(
    "This amazing product will revolutionize your life!",
    analysis_type="marketing"
)
print(f"Brand Appeal: {results['marketing_metrics']['brand_appeal']:.1%}")
```

### Neural Simulation Module (`neural_simulation.py`)

**Purpose:** Digital brain twin simulation for marketing response prediction

**Key Features:**
- 8-region brain model simulation
- Multiple consumer personality profiles
- Neural oscillation modeling
- Behavioral outcome prediction
- Marketing insights generation

**Usage Example:**
```python
from neural_simulation import DigitalBrainTwin

brain_twin = DigitalBrainTwin()
simulation = brain_twin.simulate_marketing_response(
    "Buy now and save 50%!",
    consumer_type="impulse_buyer",
    duration=15.0
)
print(f"Purchase Probability: {simulation['behavioral_outcomes']['purchase_probability']:.1%}")
```

### Deep Research Module (`neuro_deep_research_module.py`)

**Purpose:** Multi-source neuroscience dataset integration

**Key Features:**
- OpenNeuro dataset access
- Zenodo research data
- PhysioNet EEG datasets
- PubMed literature search
- Research synthesis capabilities

**Usage Example:**
```python
from neuro_deep_research_module import NeuroResearchModule

research = NeuroResearchModule()
datasets = research.fetch_openneuro_datasets("neuromarketing", 10)
if datasets['success']:
    print(f"Found {datasets['count']} datasets")
```

### Export Module (`export_module.py`)

**Purpose:** Professional report generation and export

**Key Features:**
- Multiple export formats (HTML, PDF, JSON, CSV)
- Professional templates
- Comprehensive visualizations
- Data aggregation across modules

**Usage Example:**
```python
from export_module import ProfessionalExporter

exporter = ProfessionalExporter()
report = exporter.export_comprehensive_report(
    analysis_data,
    format_type='html',
    template_style='professional',
    include_visuals=True
)
```

---

## 💡 Usage Examples

### Basic Sentiment Analysis

```python
import streamlit as st
from advanced_sentiment_module import AdvancedSentimentAnalyzer

# Initialize analyzer
analyzer = AdvancedSentimentAnalyzer()

# Analyze text
text = "This incredible product exceeded all my expectations!"
results = analyzer.analyze_comprehensive_sentiment(text, "marketing")

# Display results
st.metric("Overall Sentiment", results['basic_sentiment']['polarity'])
st.metric("Brand Appeal", f"{results['marketing_metrics']['brand_appeal']:.1%}")
```

### Neural Simulation Workflow

```python
from neural_simulation import DigitalBrainTwin

# Initialize brain twin
brain_twin = DigitalBrainTwin()

# Run simulation
stimulus = "Limited time offer - buy 2 get 1 free!"
simulation = brain_twin.simulate_marketing_response(
    stimulus,
    consumer_type="price_conscious",
    duration=10.0
)

# Extract insights
purchase_prob = simulation['behavioral_outcomes']['purchase_probability']
neuro_score = simulation['marketing_insights']['neuro_score']

print(f"Purchase Probability: {purchase_prob:.1%}")
print(f"Neuro Score: {neuro_score:.1f}/100")
```

### Research Data Integration

```python
from neuro_deep_research_module import NeuroResearchModule

# Initialize research module
research = NeuroResearchModule()

# Search multiple sources
openneuro_data = research.fetch_openneuro_datasets("EEG consumer", 5)
zenodo_data = research.fetch_zenodo_datasets("neuromarketing", 5)
papers = research.search_pubmed_papers("consumer neuroscience", 10)

# Process results
total_datasets = openneuro_data['count'] + zenodo_data['count']
print(f"Found {total_datasets} datasets and {papers['count']} papers")
```

### Report Generation

```python
from export_module import ProfessionalExporter

# Collect analysis data
analysis_data = {
    'sentiment_results': sentiment_results,
    'neural_simulation_results': neural_results,
    'dataset_results': research_results
}

# Generate report
exporter = ProfessionalExporter()
report = exporter.export_comprehensive_report(
    analysis_data,
    format_type='html',
    template_style='professional',
    include_visuals=True,
    include_raw_data=False
)

# Save or display report
with open('neuromarketing_report.html', 'w') as f:
    f.write(report['content'])
```

---

## 🛠 Troubleshooting Guide

### Common Issues and Solutions

#### 1. **Import Errors**

**Problem:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
# Ensure you're in the correct environment
source neuromarketing_env/bin/activate  # Linux/Mac
# OR
neuromarketing_env\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. **Streamlit Port Issues**

**Problem:** `Port 8501 is already in use`

**Solution:**
```bash
# Use different port
streamlit run main_neuromarketing_app.py --server.port 8502

# OR kill existing processes
pkill -f streamlit  # Linux/Mac
taskkill /f /im "streamlit.exe"  # Windows
```

#### 3. **Memory Issues with Large Datasets**

**Problem:** `MemoryError` when processing large files

**Solution:**
```python
# Add to your environment variables
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=1000
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=1000

# Or process in chunks
def process_large_file(file_path, chunk_size=1000):
    with open(file_path, 'r') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
```

#### 4. **API Connection Timeouts**

**Problem:** Research module APIs timing out

**Solution:**
```python
# Increase timeout in requests
import requests

# Configure longer timeouts
session = requests.Session()
session.timeout = 30  # 30 seconds

# Add retry logic
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

retry_strategy = Retry(
    total=3,
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

#### 5. **Visualization Rendering Issues**

**Problem:** Plotly charts not displaying correctly

**Solution:**
```python
# Update plotly and dependencies
pip install --upgrade plotly streamlit

# Clear Streamlit cache
streamlit cache clear

# Use specific plotly renderer
import plotly.io as pio
pio.renderers.default = "browser"
```

### Performance Issues

#### 1. **Slow Loading Times**

**Causes and Solutions:**
- **Large datasets:** Implement pagination and lazy loading
- **Heavy computations:** Use caching with `@st.cache_data`
- **Network requests:** Implement connection pooling

```python
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def expensive_computation(data):
    # Your expensive operation here
    return processed_data
```

#### 2. **High Memory Usage**

**Solutions:**
```python
# Use memory-efficient data structures
import pandas as pd

# Read in chunks
chunk_iter = pd.read_csv('large_file.csv', chunksize=1000)
for chunk in chunk_iter:
    process_chunk(chunk)

# Clear variables when done
import gc
del large_variable
gc.collect()
```

### Data Issues

#### 1. **Missing API Keys**

**Problem:** Enhanced features not working without API keys

**Solution:**
```python
# Check if API keys are available
import os

openai_key = os.getenv('OPENAI_API_KEY')
if not openai_key:
    st.warning("Enhanced features require OpenAI API key")
    # Fallback to basic functionality
```

#### 2. **Invalid Data Formats**

**Problem:** Uploaded files in unsupported formats

**Solution:**
```python
# Validate file types
def validate_file_type(file, allowed_types):
    if file.type not in allowed_types:
        st.error(f"Unsupported file type: {file.type}")
        return False
    return True

# Graceful error handling
try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {str(e)}")
    st.info("Please ensure file is a valid CSV format")
```

---

## 🚀 Deployment Instructions

### Streamlit Cloud Deployment

#### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account

#### Steps:

1. **Prepare Repository**
```bash
# Ensure all files are committed
git add .
git commit -m "Prepare for deployment"
git push origin main
```

2. **Create requirements.txt**
```text
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
requests>=2.31.0
openai>=1.0.0
python-dotenv>=1.0.0
Pillow>=10.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
```

3. **Add .streamlit/config.toml**
```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
maxUploadSize = 1000
maxMessageSize = 1000
```

4. **Deploy on Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Connect your GitHub repository
- Select `main_neuromarketing_app.py` as the main file
- Add environment variables if needed

### Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "main_neuromarketing_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Create docker-compose.yml
```yaml
version: '3.8'

services:
  neuromarketing-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

#### Deploy with Docker
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Scale if needed
docker-compose up -d --scale neuromarketing-app=3
```

### AWS/GCP/Azure Deployment

#### AWS EC2 Deployment
```bash
# Launch EC2 instance (Ubuntu 20.04)
# Install Docker
sudo apt update
sudo apt install docker.io docker-compose
sudo usermod -aG docker ubuntu

# Clone repository
git clone https://github.com/Rajesh0311/neuromarketing-gpt-app.git
cd neuromarketing-gpt-app

# Set environment variables
echo "OPENAI_API_KEY=your_key_here" > .env

# Deploy
docker-compose up -d

# Configure security group to allow port 8501
```

#### GCP Cloud Run Deployment
```bash
# Build for Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/neuromarketing-app

# Deploy
gcloud run deploy neuromarketing-app \
    --image gcr.io/PROJECT_ID/neuromarketing-app \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --port 8501
```

### Environment Variables

Create a `.env` file for local development:
```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Database URLs (if using external databases)
DATABASE_URL=your_database_url_here

# Feature Flags
ENABLE_ADVANCED_FEATURES=true
ENABLE_CACHING=true
DEBUG_MODE=false

# Performance Settings
MAX_UPLOAD_SIZE=1000
CACHE_TTL=3600
```

### Production Optimizations

#### 1. **Enable Caching**
```python
# Use Redis for production caching
import redis
import streamlit as st

@st.cache_data(ttl=3600)
def cached_api_call(query):
    # Your API call here
    return result
```

#### 2. **Add Monitoring**
```python
# Add application monitoring
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add metrics collection
def track_usage(action, user_id=None):
    logging.info(f"Action: {action}, User: {user_id}")
```

#### 3. **Security Considerations**
```python
# Add rate limiting
from streamlit_extras.app_logo import add_logo
import time

def rate_limit(max_requests=100, window=3600):
    # Implement rate limiting logic
    pass

# Sanitize user input
import html

def sanitize_input(user_input):
    return html.escape(user_input)
```

---

## ⚡ Performance Optimization

### Caching Strategies

#### 1. **Function-level Caching**
```python
import streamlit as st

@st.cache_data(ttl=3600, max_entries=100)
def expensive_analysis(text):
    # Your expensive computation
    return results

@st.cache_resource
def load_model():
    # Load ML model once
    return model
```

#### 2. **Session State Management**
```python
# Efficient session state usage
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

def get_cached_result(key):
    return st.session_state.analysis_cache.get(key)

def set_cached_result(key, value):
    st.session_state.analysis_cache[key] = value
```

#### 3. **Database Optimization**
```python
# Use connection pooling for database access
import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db_connection():
    conn = sqlite3.connect('app.db')
    try:
        yield conn
    finally:
        conn.close()

# Batch operations
def batch_insert(data_list):
    with get_db_connection() as conn:
        conn.executemany(
            "INSERT INTO results VALUES (?, ?, ?)",
            data_list
        )
```

### Memory Management

#### 1. **Large Dataset Handling**
```python
# Process in chunks
def process_large_dataset(file_path, chunk_size=1000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield process_chunk(chunk)

# Use generators for memory efficiency
def generate_analysis_results(data):
    for item in data:
        yield analyze_item(item)
```

#### 2. **Memory Monitoring**
```python
import psutil
import streamlit as st

def display_memory_usage():
    memory_percent = psutil.virtual_memory().percent
    st.sidebar.metric("Memory Usage", f"{memory_percent:.1f}%")
```

### Network Optimization

#### 1. **Async Operations**
```python
import asyncio
import aiohttp

async def fetch_multiple_apis(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.json()
```

#### 2. **Connection Pooling**
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.3)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
```

---

## 🔧 Environment Configuration

### Development Environment

#### .env File Setup
```bash
# Development settings
NODE_ENV=development
DEBUG=true
LOG_LEVEL=debug

# API Keys
OPENAI_API_KEY=your_development_key

# Database
DATABASE_URL=sqlite:///dev.db

# Feature Flags
ENABLE_EXPERIMENTAL_FEATURES=true
```

#### IDE Configuration

**VS Code Settings (.vscode/settings.json):**
```json
{
    "python.defaultInterpreterPath": "./neuromarketing_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "files.associations": {
        "*.py": "python"
    }
}
```

### Production Environment

#### Environment Variables
```bash
# Production settings
NODE_ENV=production
DEBUG=false
LOG_LEVEL=info

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Caching
REDIS_URL=redis://localhost:6379/0

# Monitoring
SENTRY_DSN=your_sentry_dsn_here
```

#### systemd Service (Linux)
```ini
[Unit]
Description=NeuroMarketing GPT Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/neuromarketing-gpt-app
Environment=NODE_ENV=production
ExecStart=/opt/neuromarketing-gpt-app/venv/bin/streamlit run main_neuromarketing_app.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

### Configuration Management

#### config.py
```python
import os
from typing import Optional

class Config:
    """Application configuration"""
    
    # Basic settings
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'info').upper()
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    
    # Performance
    CACHE_TTL = int(os.getenv('CACHE_TTL', '3600'))
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', '1000'))
    
    # Feature flags
    ENABLE_ADVANCED_FEATURES = os.getenv('ENABLE_ADVANCED_FEATURES', 'true').lower() == 'true'

# Load configuration
config = Config()
```

---

## 🤝 Contributing Guidelines

### Development Setup

1. **Fork and Clone**
```bash
git clone https://github.com/your-username/neuromarketing-gpt-app.git
cd neuromarketing-gpt-app
```

2. **Create Development Environment**
```bash
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements-dev.txt
```

3. **Install Pre-commit Hooks**
```bash
pre-commit install
```

### Code Standards

#### Python Style Guide
- Follow PEP 8
- Use type hints
- Add docstrings for all functions
- Maximum line length: 100 characters

#### Example:
```python
def analyze_sentiment(
    text: str, 
    analysis_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Analyze sentiment of input text.
    
    Args:
        text: Input text to analyze
        analysis_type: Type of analysis to perform
        
    Returns:
        Dictionary containing analysis results
    """
    # Implementation here
    pass
```

### Testing

#### Unit Tests
```python
import unittest
from advanced_sentiment_module import AdvancedSentimentAnalyzer

class TestSentimentAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.analyzer = AdvancedSentimentAnalyzer()
    
    def test_positive_sentiment(self):
        result = self.analyzer.analyze_comprehensive_sentiment(
            "This is amazing!"
        )
        self.assertEqual(result['basic_sentiment']['polarity'], 'positive')
```

#### Integration Tests
```python
def test_full_workflow():
    """Test complete analysis workflow"""
    # Setup
    text = "Great product, highly recommend!"
    
    # Execute
    sentiment_result = analyze_sentiment(text)
    neural_result = simulate_neural_response(text)
    
    # Verify
    assert sentiment_result['success'] is True
    assert neural_result['purchase_probability'] > 0.5
```

### Pull Request Process

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Changes and Test**
```bash
# Make your changes
python -m pytest tests/
black .
flake8 .
```

3. **Commit and Push**
```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

4. **Create Pull Request**
- Use descriptive title
- Include summary of changes
- Reference related issues
- Add screenshots for UI changes

### Issue Reporting

Use the issue template:
```markdown
## Bug Report

**Description:** Brief description of the issue

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior:** What should happen

**Actual Behavior:** What actually happens

**Environment:**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python Version: [e.g., 3.9.0]
- Streamlit Version: [e.g., 1.28.0]

**Additional Context:** Any other relevant information
```

---

## 📞 Support and Resources

### Getting Help

1. **Documentation:** This comprehensive guide
2. **GitHub Issues:** For bug reports and feature requests
3. **Discussions:** For general questions and community support

### External Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)
- [OpenNeuro API](https://openneuro.org/docs)
- [Zenodo API](https://developers.zenodo.org/)

### Community

- **Discord:** Join our community discussions
- **Twitter:** Follow @NeuroMarketingGPT for updates
- **Blog:** Read our technical blog posts

---

**Last Updated:** January 2025  
**Version:** 1.0.0  
**Maintainers:** NeuroMarketing GPT Team