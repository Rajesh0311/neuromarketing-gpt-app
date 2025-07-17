"""
OpenNeuro GraphQL API Integration Module
Handles communication with OpenNeuro database for neural data access
"""

import requests
import json
from typing import Dict, List, Optional, Any
import streamlit as st

class OpenNeuroClient:
    """Client for OpenNeuro GraphQL API integration"""
    
    def __init__(self, api_token: Optional[str] = None):
        self.base_url = "https://openneuro.org/crn/graphql"
        self.api_token = api_token or st.session_state.api_keys.get('openneuro', '')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_token}' if self.api_token else ''
        }
    
    def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Execute GraphQL query against OpenNeuro API"""
        payload = {
            'query': query,
            'variables': variables or {}
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"OpenNeuro API Error: {str(e)}")
            return {'errors': [{'message': str(e)}]}
    
    def search_datasets(self, keywords: List[str], limit: int = 10) -> Dict:
        """Search for relevant datasets in OpenNeuro"""
        query = """
        query SearchDatasets($keywords: [String!], $limit: Int) {
            datasets(
                filter: {
                    description: {contains: $keywords}
                }
                limit: $limit
            ) {
                id
                label
                description
                modality
                tags
                summary {
                    subjectCount
                    sessionCount
                    size
                }
                analytics {
                    downloads
                    views
                }
            }
        }
        """
        
        variables = {
            'keywords': keywords,
            'limit': limit
        }
        
        return self.execute_query(query, variables)
    
    def get_dataset_details(self, dataset_id: str) -> Dict:
        """Get detailed information about a specific dataset"""
        query = """
        query GetDataset($id: ID!) {
            dataset(id: $id) {
                id
                label
                description
                modality
                authors {
                    name
                    email
                    orcid
                }
                summary {
                    subjectCount
                    sessionCount
                    taskCount
                    size
                    dataTypes
                }
                files(prefix: "") {
                    name
                    size
                    objectType
                }
                analytics {
                    downloads
                    views
                }
                permissions {
                    userPermissions
                    public
                }
            }
        }
        """
        
        variables = {'id': dataset_id}
        return self.execute_query(query, variables)
    
    def get_eeg_datasets(self, limit: int = 20) -> Dict:
        """Get EEG-specific datasets for analysis"""
        query = """
        query GetEEGDatasets($limit: Int) {
            datasets(
                filter: {
                    modality: {contains: ["eeg"]}
                }
                orderBy: DOWNLOADS_DESC
                limit: $limit
            ) {
                id
                label
                description
                modality
                tags
                summary {
                    subjectCount
                    sessionCount
                    size
                    dataTypes
                }
                analytics {
                    downloads
                    views
                }
            }
        }
        """
        
        variables = {'limit': limit}
        return self.execute_query(query, variables)
    
    def get_task_based_datasets(self, task_keywords: List[str]) -> Dict:
        """Get datasets based on specific experimental tasks"""
        query = """
        query GetTaskDatasets($keywords: [String!]) {
            datasets(
                filter: {
                    description: {contains: $keywords}
                    modality: {contains: ["eeg", "meg", "fmri"]}
                }
            ) {
                id
                label
                description
                modality
                tags
                summary {
                    subjectCount
                    sessionCount
                    taskCount
                    dataTypes
                }
                files(prefix: "task-") {
                    name
                    size
                }
            }
        }
        """
        
        variables = {'keywords': task_keywords}
        return self.execute_query(query, variables)

def format_dataset_info(dataset: Dict) -> str:
    """Format dataset information for display"""
    info = f"**{dataset.get('label', 'Unknown Dataset')}**\n"
    info += f"ID: {dataset.get('id', 'N/A')}\n"
    info += f"Description: {dataset.get('description', 'No description available')[:200]}...\n"
    info += f"Modality: {', '.join(dataset.get('modality', []))}\n"
    
    summary = dataset.get('summary', {})
    info += f"Subjects: {summary.get('subjectCount', 'N/A')}\n"
    info += f"Sessions: {summary.get('sessionCount', 'N/A')}\n"
    info += f"Size: {summary.get('size', 'N/A')} bytes\n"
    
    analytics = dataset.get('analytics', {})
    info += f"Downloads: {analytics.get('downloads', 'N/A')}\n"
    info += f"Views: {analytics.get('views', 'N/A')}\n"
    
    return info

def suggest_relevant_datasets(research_query: str) -> List[str]:
    """Suggest relevant datasets based on research query"""
    # Simple keyword extraction for dataset search
    marketing_keywords = [
        'emotion', 'decision', 'attention', 'visual', 'auditory',
        'memory', 'learning', 'reward', 'social', 'language',
        'face', 'recognition', 'preference', 'choice', 'valence'
    ]
    
    query_lower = research_query.lower()
    relevant_keywords = [kw for kw in marketing_keywords if kw in query_lower]
    
    # Add general neural keywords if none found
    if not relevant_keywords:
        relevant_keywords = ['emotion', 'decision', 'attention']
    
    return relevant_keywords

# Streamlit integration functions
def render_openneuro_search():
    """Render OpenNeuro dataset search interface"""
    st.markdown("### 🧠 OpenNeuro Dataset Integration")
    
    if not st.session_state.api_keys.get('openneuro'):
        st.warning("⚠️ OpenNeuro API token not configured. Some features may be limited.")
        return
    
    client = OpenNeuroClient()
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_terms = st.text_input(
            "Search datasets:",
            placeholder="Enter keywords like 'emotion', 'decision-making', 'visual attention'"
        )
    
    with col2:
        search_limit = st.selectbox("Results", [5, 10, 20], index=1)
    
    if st.button("🔍 Search OpenNeuro"):
        if search_terms:
            keywords = [term.strip() for term in search_terms.split(',')]
            
            with st.spinner("Searching OpenNeuro database..."):
                results = client.search_datasets(keywords, search_limit)
                
                if 'errors' in results:
                    st.error("Failed to search datasets. Please check your API token.")
                    return
                
                datasets = results.get('data', {}).get('datasets', [])
                
                if not datasets:
                    st.info("No datasets found for the given keywords.")
                    return
                
                st.success(f"Found {len(datasets)} relevant datasets:")
                
                for i, dataset in enumerate(datasets):
                    with st.expander(f"Dataset {i+1}: {dataset.get('label', 'Unknown')}"):
                        st.markdown(format_dataset_info(dataset))
                        
                        if st.button(f"Use Dataset {i+1}", key=f"use_dataset_{i}"):
                            st.session_state.selected_dataset = dataset
                            st.success("✅ Dataset selected for analysis!")

def get_neural_insights_from_openneuro(research_query: str) -> Dict:
    """Get neural insights by querying relevant OpenNeuro datasets"""
    client = OpenNeuroClient()
    
    # Suggest keywords based on research query
    keywords = suggest_relevant_datasets(research_query)
    
    # Search for relevant datasets
    search_results = client.search_datasets(keywords, limit=5)
    
    if 'errors' in search_results:
        return {
            'success': False,
            'error': 'Failed to connect to OpenNeuro',
            'insights': []
        }
    
    datasets = search_results.get('data', {}).get('datasets', [])
    
    # Generate insights based on found datasets
    insights = []
    for dataset in datasets[:3]:  # Use top 3 datasets
        summary = dataset.get('summary', {})
        modality = dataset.get('modality', [])
        
        insight = {
            'dataset_id': dataset.get('id'),
            'dataset_name': dataset.get('label'),
            'modality': modality,
            'subject_count': summary.get('subjectCount', 0),
            'relevance_score': calculate_relevance_score(dataset, keywords),
            'neural_findings': generate_neural_findings(dataset, research_query)
        }
        insights.append(insight)
    
    return {
        'success': True,
        'total_datasets': len(datasets),
        'insights': insights,
        'keywords_used': keywords
    }

def calculate_relevance_score(dataset: Dict, keywords: List[str]) -> float:
    """Calculate relevance score for a dataset based on keywords"""
    description = dataset.get('description', '').lower()
    tags = [tag.lower() for tag in dataset.get('tags', [])]
    
    score = 0.0
    for keyword in keywords:
        if keyword in description:
            score += 0.3
        if keyword in ' '.join(tags):
            score += 0.2
    
    # Boost score based on dataset popularity
    analytics = dataset.get('analytics', {})
    downloads = analytics.get('downloads', 0)
    if downloads > 100:
        score += 0.1
    if downloads > 1000:
        score += 0.1
    
    return min(score, 1.0)

def generate_neural_findings(dataset: Dict, research_query: str) -> List[str]:
    """Generate relevant neural findings based on dataset and query"""
    modality = dataset.get('modality', [])
    findings = []
    
    if 'eeg' in modality:
        findings.extend([
            "Alpha wave patterns indicate relaxed attention states",
            "Beta activity suggests active cognitive processing",
            "Gamma oscillations correlate with conscious awareness"
        ])
    
    if 'fmri' in modality:
        findings.extend([
            "Prefrontal cortex activation during decision-making",
            "Limbic system engagement in emotional processing",
            "Default mode network deactivation during focused tasks"
        ])
    
    if 'meg' in modality:
        findings.extend([
            "Millisecond-precision neural timing effects",
            "Cortical source localization of cognitive processes",
            "Neural connectivity patterns in task networks"
        ])
    
    # Limit to 2-3 most relevant findings
    return findings[:3]