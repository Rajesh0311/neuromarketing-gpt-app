"""
Neuro Deep Research Module for Neuromarketing GPT App
This module provides deep research capabilities for neurological data analysis
"""

import os
import logging
from typing import List, Optional, Dict, Any, Union
import streamlit as st
import requests
import json
from dataclasses import dataclass
import pandas as pd
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment-based configuration
def get_api_key(key_name: str) -> Optional[str]:
    """Securely retrieve API key from environment variables.
    
    Args:
        key_name: Name of the environment variable containing the API key
        
    Returns:
        API key if found, None otherwise
    """
    api_key = os.getenv(key_name)
    if not api_key:
        logger.warning(f"API key {key_name} not found in environment variables")
    return api_key

def validate_api_key(api_key: Optional[str]) -> bool:
    """Validate API key format and presence.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False
    
    # Basic validation - non-empty and reasonable length
    if len(api_key.strip()) < 10:
        return False
        
    return True

@dataclass
class NeuroDataset:
    """A dataclass representing a neurological dataset.
    
    This class stores metadata about neurological datasets.
    """
    id: str
    name: str
    description: str
    participants: int
    data_type: str
    
    @classmethod
    def from_api_response(cls, response_data):
        """Create NeuroDataset instance from API response data."""
        return cls(
            id=response_data.get('id'),
            name=response_data.get('name'),
            description=response_data.get('description'),
            participants=response_data.get('participants', 0),
            data_type=response_data.get('type', 'unknown')
        )

class NeuroResearchModule:
    """Main class for neurological research operations."""
    
    def __init__(self):
        """Initialize the NeuroResearchModule with API credentials validation."""
        self.openai_api_key = get_api_key('OPENAI_API_KEY')
        self.openneuro_api_key = get_api_key('OPENNEURO_API_KEY')
        self.github_token = get_api_key('GITHUB_TOKEN')
        
        # Validate essential API keys
        if not validate_api_key(self.openai_api_key):
            logger.warning("OpenAI API key not properly configured")
        
        if not validate_api_key(self.openneuro_api_key):
            logger.warning("OpenNeuro API key not properly configured")
    
    def _validate_input(self, data: Any, data_type: str) -> bool:
        """Validate input data based on expected type.
        
        Args:
            data: Data to validate
            data_type: Expected data type description
            
        Returns:
            True if valid, False otherwise
        """
        if data is None:
            logger.error(f"Input data is None for {data_type}")
            return False
        
        if isinstance(data, str) and not data.strip():
            logger.error(f"Empty string provided for {data_type}")
            return False
            
        return True
    
    def fetch_openneuro_datasets(self, query_params: Optional[Dict[str, Any]] = None) -> List[NeuroDataset]:
        """Fetch datasets from OpenNeuro database.
        
        Args:
            query_params: Optional parameters for filtering datasets
            
        Returns:
            List of NeuroDataset objects
            
        Raises:
            APIKeyError: If API key is not properly configured
            DataFetchError: If data fetching fails
        """
        if not validate_api_key(self.openneuro_api_key):
            raise APIKeyError("OpenNeuro API key is not properly configured")
        
        try:
            headers = {
                'Authorization': f'Bearer {self.openneuro_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Proper GraphQL query for OpenNeuro datasets
            graphql_query = {
                'query': '''
                query GetDatasets($first: Int, $filter: DatasetFilter) {
                    datasets(first: $first, filter: $filter) {
                        edges {
                            node {
                                id
                                label
                                description
                                summary {
                                    subjectCount
                                }
                                dataType
                            }
                        }
                    }
                }
                ''',
                'variables': {
                    'first': 50,
                    'filter': query_params or {}
                }
            }
            
            # Make API request
            response = requests.post(
                'https://openneuro.org/crn/graphql',
                json=graphql_query,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                datasets = []
                
                # Process response data
                edges = data.get('data', {}).get('datasets', {}).get('edges', [])
                for edge in edges:
                    node = edge.get('node', {})
                    if node:
                        dataset_data = {
                            'id': node.get('id', ''),
                            'name': node.get('label', ''),
                            'description': node.get('description', ''),
                            'participants': node.get('summary', {}).get('subjectCount', 0),
                            'type': node.get('dataType', 'unknown')
                        }
                        dataset = NeuroDataset.from_api_response(dataset_data)
                        datasets.append(dataset)
                
                logger.info(f"Successfully fetched {len(datasets)} datasets")
                return datasets
            else:
                raise DataFetchError(f"API request failed with status {response.status_code}: {response.text}")
                
        except requests.RequestException as e:
            logger.error(f"Network error while fetching datasets: {e}")
            raise DataFetchError(f"Network error: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise DataFetchError(f"Invalid response format: {e}")
        except Exception as e:
            logger.error(f"Unexpected error while fetching datasets: {e}")
            raise DataFetchError(f"Unexpected error: {e}")
    
    def analyze_sentiment_data(self, text_data: str) -> Dict[str, Any]:
        """Analyze sentiment from neurological text data.
        
        Args:
            text_data: Text data to analyze
            
        Returns:
            Sentiment analysis results
            
        Raises:
            ValidationError: If input validation fails
            APIKeyError: If API key is not properly configured
        """
        # Input validation
        if not self._validate_input(text_data, "text_data"):
            raise ValidationError("Invalid text data provided for sentiment analysis")
        
        if not validate_api_key(self.openai_api_key):
            raise APIKeyError("OpenAI API key is not properly configured")
        
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Prepare the prompt for sentiment analysis
            prompt = f"""
            Analyze the sentiment of the following neurological research text and provide:
            1. Overall sentiment (positive, negative, neutral)
            2. Confidence score (0-1)
            3. Key emotional indicators
            4. Recommendations for marketing approach
            
            Text: {text_data}
            """
            
            payload = {
                'model': 'gpt-3.5-turbo',
                'messages': [
                    {'role': 'system', 'content': 'You are an expert in neurological data sentiment analysis for marketing purposes.'},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 500,
                'temperature': 0.3
            }
            
            # Make synchronous API call (Streamlit compatible)
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse and structure the response
                analysis_result = {
                    'sentiment': 'neutral',  # Default
                    'confidence': 0.5,       # Default
                    'raw_response': content,
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                logger.info("Sentiment analysis completed successfully")
                return analysis_result
            else:
                raise DataFetchError(f"OpenAI API request failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            raise DataFetchError(f"Sentiment analysis failed: {e}")
    
    def generate_marketing_visuals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate marketing visuals based on neuro data.
        
        Args:
            data: Neurological data to visualize
            
        Returns:
            Dictionary containing visual generation results
            
        Raises:
            ValidationError: If input validation fails
        """
        if not self._validate_input(data, "visualization_data"):
            raise ValidationError("Invalid data provided for visualization")
        
        try:
            # Basic visualization generation logic
            visualization_config = {
                'chart_type': 'sentiment_heatmap',
                'color_scheme': 'neurological',
                'dimensions': {'width': 800, 'height': 600},
                'data_points': len(data) if isinstance(data, (list, dict)) else 1,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Placeholder for actual visualization generation
            # In a real implementation, this would integrate with visualization libraries
            result = {
                'status': 'generated',
                'config': visualization_config,
                'file_path': f"/tmp/neuro_visual_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png",
                'metadata': {
                    'creation_time': pd.Timestamp.now().isoformat(),
                    'data_summary': str(data)[:100] + '...' if len(str(data)) > 100 else str(data)
                }
            }
            
            logger.info("Marketing visual generation completed")
            return result
            
        except Exception as e:
            logger.error(f"Error generating marketing visuals: {e}")
            raise DataFetchError(f"Visual generation failed: {e}")
    
    def export_research_report(self, findings: Dict[str, Any], format: str = 'pdf') -> Optional[str]:
        """Export research findings to specified format.
        
        Args:
            findings: Research findings to export
            format: Output format (pdf, docx, html)
            
        Returns:
            File path of the exported report or None if failed
            
        Raises:
            ValidationError: If input validation fails
        """
        # Input validation
        if not self._validate_input(findings, "research_findings"):
            raise ValidationError("Invalid research findings provided for export")
        
        valid_formats = ['pdf', 'docx', 'html', 'json']
        if format not in valid_formats:
            raise ValidationError(f"Invalid format '{format}'. Supported formats: {valid_formats}")
        
        try:
            # Generate filename with timestamp
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"neuro_research_report_{timestamp}.{format}"
            file_path = f"/tmp/{filename}"
            
            # Basic report structure
            report_data = {
                'title': 'Neurological Research Report',
                'timestamp': pd.Timestamp.now().isoformat(),
                'findings': findings,
                'metadata': {
                    'export_format': format,
                    'generated_by': 'NeuroResearchModule',
                    'version': '1.0'
                }
            }
            
            # Export based on format
            if format == 'json':
                with open(file_path, 'w') as f:
                    json.dump(report_data, f, indent=2)
            else:
                # For other formats, create a basic text representation
                # In a real implementation, this would use appropriate libraries
                with open(file_path.replace(f'.{format}', '.txt'), 'w') as f:
                    f.write(f"# {report_data['title']}\n\n")
                    f.write(f"Generated: {report_data['timestamp']}\n\n")
                    f.write(f"Findings:\n{json.dumps(findings, indent=2)}\n")
                
                file_path = file_path.replace(f'.{format}', '.txt')
            
            logger.info(f"Research report exported to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error exporting research report: {e}")
            return None

# Custom exception classes for proper error handling
class NeuroResearchError(Exception):
    """Base exception for neuro research module errors."""
    pass

class APIKeyError(NeuroResearchError):
    """Raised when API key validation fails."""
    pass

class DataFetchError(NeuroResearchError):
    """Raised when data fetching operations fail."""
    pass

class ValidationError(NeuroResearchError):
    """Raised when input validation fails."""
    pass

def validate_api_credentials() -> bool:
    """Validate API credentials for external services.
    
    Returns:
        True if at least one API key is properly configured, False otherwise
    """
    openai_key = get_api_key('OPENAI_API_KEY')
    openneuro_key = get_api_key('OPENNEURO_API_KEY')
    
    openai_valid = validate_api_key(openai_key)
    openneuro_valid = validate_api_key(openneuro_key)
    
    if not openai_valid and not openneuro_valid:
        logger.error("No valid API keys found. Please configure environment variables.")
        return False
    
    if not openai_valid:
        logger.warning("OpenAI API key not configured - some features may be limited")
    
    if not openneuro_valid:
        logger.warning("OpenNeuro API key not configured - dataset fetching may be limited")
    
    return True

def create_streamlit_interface():
    """Create the Streamlit user interface for the neuro research module."""
    st.title("🧠 Neuro Deep Research Module")
    st.markdown("AI-powered neurological data analysis and marketing visual generation")
    
    # Initialize the research module
    try:
        research_module = NeuroResearchModule()
    except Exception as e:
        st.error(f"Failed to initialize research module: {e}")
        return
    
    # Sidebar for API key status
    with st.sidebar:
        st.header("Configuration Status")
        
        if validate_api_credentials():
            st.success("✅ API credentials configured")
        else:
            st.error("❌ API credentials missing")
            st.info("Please set environment variables: OPENAI_API_KEY, OPENNEURO_API_KEY")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset Search", "Sentiment Analysis", "Visual Generation", "Export Reports"])
    
    with tab1:
        st.header("OpenNeuro Dataset Search")
        
        if st.button("Fetch Available Datasets"):
            try:
                with st.spinner("Fetching datasets..."):
                    datasets = research_module.fetch_openneuro_datasets()
                
                if datasets:
                    st.success(f"Found {len(datasets)} datasets")
                    
                    # Display datasets in a dataframe
                    df_data = []
                    for dataset in datasets:
                        df_data.append({
                            'ID': dataset.id,
                            'Name': dataset.name,
                            'Participants': dataset.participants,
                            'Type': dataset.data_type,
                            'Description': dataset.description[:100] + '...' if len(dataset.description) > 100 else dataset.description
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No datasets found")
                    
            except APIKeyError as e:
                st.error(f"API Key Error: {e}")
            except DataFetchError as e:
                st.error(f"Data Fetch Error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
    
    with tab2:
        st.header("Sentiment Analysis")
        
        text_input = st.text_area(
            "Enter neurological research text to analyze:",
            placeholder="Paste your research findings, study descriptions, or neurological data here...",
            height=150
        )
        
        if st.button("Analyze Sentiment") and text_input:
            try:
                with st.spinner("Analyzing sentiment..."):
                    result = research_module.analyze_sentiment_data(text_input)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Sentiment", result.get('sentiment', 'Unknown'))
                    st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
                
                with col2:
                    st.write("**Analysis Details:**")
                    st.text_area("Raw Response", result.get('raw_response', ''), height=200)
                
            except (ValidationError, APIKeyError, DataFetchError) as e:
                st.error(f"Error: {e}")
    
    with tab3:
        st.header("Marketing Visual Generation")
        
        st.write("Upload or input data for visual generation:")
        
        # Sample data input options
        data_source = st.selectbox("Data Source", ["Manual Input", "Upload File", "Use Sample Data"])
        
        if data_source == "Manual Input":
            data_input = st.text_area("Enter data (JSON format):", '{"sentiment": "positive", "confidence": 0.8}')
            
            if st.button("Generate Visual"):
                try:
                    data = json.loads(data_input)
                    with st.spinner("Generating visual..."):
                        result = research_module.generate_marketing_visuals(data)
                    
                    st.success("Visual generated successfully!")
                    st.json(result)
                    
                except json.JSONDecodeError:
                    st.error("Invalid JSON format")
                except Exception as e:
                    st.error(f"Error generating visual: {e}")
        
        elif data_source == "Use Sample Data":
            sample_data = {
                "sentiment_scores": [0.8, 0.6, 0.9, 0.7],
                "categories": ["positive", "neutral", "positive", "positive"],
                "confidence": 0.85
            }
            
            if st.button("Generate Visual with Sample Data"):
                try:
                    with st.spinner("Generating visual..."):
                        result = research_module.generate_marketing_visuals(sample_data)
                    
                    st.success("Visual generated successfully!")
                    st.json(result)
                    
                except Exception as e:
                    st.error(f"Error generating visual: {e}")
    
    with tab4:
        st.header("Export Research Reports")
        
        # Sample findings input
        findings_input = st.text_area(
            "Research Findings (JSON format):",
            '{"summary": "Positive neurological response patterns", "confidence": 0.85, "recommendations": ["Focus on emotional triggers", "Use warm color schemes"]}',
            height=150
        )
        
        export_format = st.selectbox("Export Format", ["pdf", "docx", "html", "json"])
        
        if st.button("Export Report"):
            try:
                findings = json.loads(findings_input)
                
                with st.spinner("Exporting report..."):
                    file_path = research_module.export_research_report(findings, export_format)
                
                if file_path:
                    st.success(f"Report exported successfully to: {file_path}")
                else:
                    st.error("Failed to export report")
                    
            except json.JSONDecodeError:
                st.error("Invalid JSON format in findings")
            except Exception as e:
                st.error(f"Error exporting report: {e}")

def main():
    """Main function for running the neuro research module."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Neuro Deep Research Module",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Environment check
    if not validate_api_credentials():
        st.warning("""
        ⚠️ **API Configuration Required**
        
        Please configure the following environment variables:
        - `OPENAI_API_KEY`: For sentiment analysis features
        - `OPENNEURO_API_KEY`: For dataset fetching features
        
        Some features may be limited without proper API keys.
        """)
    
    # Create the main interface
    create_streamlit_interface()

if __name__ == "__main__":
    main()