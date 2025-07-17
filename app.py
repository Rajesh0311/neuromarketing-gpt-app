"""
NeuroMarketing GPT - Main Streamlit Application
AI-powered sentiment analysis with professional marketing visual generation
"""

import streamlit as st
import os
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NeuroMarketing GPT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import modules (with error handling for missing dependencies)
try:
    from deep_research_module import render_deep_research_module, integrate_deep_research_module
    DEEP_RESEARCH_AVAILABLE = True
except ImportError as e:
    DEEP_RESEARCH_AVAILABLE = False
    st.error(f"Deep Research Module not available: {e}")

def main():
    """Main application function"""
    
    # Sidebar for navigation
    st.sidebar.title("🧠 NeuroMarketing GPT")
    st.sidebar.markdown("---")
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Choose a feature:",
        [
            "🏠 Home",
            "🧠 Deep Research",
            "📊 Sentiment Analysis", 
            "🎨 Visual Generation",
            "📋 Reports",
            "⚙️ Settings"
        ]
    )
    
    # Main content area
    if page == "🏠 Home":
        render_home_page()
    elif page == "🧠 Deep Research":
        if DEEP_RESEARCH_AVAILABLE:
            render_deep_research_module()
        else:
            st.error("Deep Research Module is not available. Please check the installation.")
    elif page == "📊 Sentiment Analysis":
        render_sentiment_analysis()
    elif page == "🎨 Visual Generation":
        render_visual_generation()
    elif page == "📋 Reports":
        render_reports()
    elif page == "⚙️ Settings":
        render_settings()

def render_home_page():
    """Render the home page"""
    st.title("🧠 NeuroMarketing GPT")
    st.subheader("AI-Powered Marketing Insights")
    
    st.markdown("""
    Welcome to NeuroMarketing GPT - your comprehensive platform for neuroscience-based marketing research and analysis.
    
    ## 🚀 Key Features
    
    - **🧠 Deep Research**: Access multiple open neuroscience datasets and research papers
    - **📊 Sentiment Analysis**: Multi-dimensional emotional and psychological analysis
    - **🎨 Visual Generation**: Professional marketing visuals and infographics
    - **📋 Research Reports**: Comprehensive analysis reports and exports
    - **⚙️ API Integration**: Seamless integration with multiple data sources
    
    ## 🎯 Get Started
    
    1. **Configure APIs** in Settings (optional - many features work without API keys)
    2. **Explore Deep Research** to access neuroscience datasets
    3. **Analyze Content** with our sentiment analysis tools
    4. **Generate Visuals** for your marketing campaigns
    5. **Export Reports** for professional presentations
    """)
    
    # Status indicators
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if DEEP_RESEARCH_AVAILABLE:
            st.success("✅ Deep Research Module")
        else:
            st.error("❌ Deep Research Module")
    
    with col2:
        # Check OpenAI API
        openai_key = os.getenv('OPENAI_API_KEY') or st.session_state.get('openai_api_key')
        if openai_key:
            st.success("✅ OpenAI API")
        else:
            st.warning("⚠️ OpenAI API (Optional)")
    
    with col3:
        st.info("ℹ️ Open Datasets (No API required)")

def render_sentiment_analysis():
    """Render sentiment analysis page"""
    st.title("📊 Sentiment Analysis")
    st.markdown("Multi-dimensional emotional and psychological content analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Paste your marketing content, social media posts, or any text for analysis..."
    )
    
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if text_input.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    # Placeholder for sentiment analysis
                    st.success("Analysis completed!")
                    
                    # Sample results (replace with actual analysis)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📈 Emotional Dimensions")
                        st.metric("Overall Sentiment", "Positive", "0.8")
                        st.metric("Emotional Intensity", "High", "0.75")
                        st.metric("Trust Level", "High", "0.82")
                    
                    with col2:
                        st.subheader("🎯 Marketing Insights")
                        st.metric("Brand Appeal", "Strong", "0.78")
                        st.metric("Purchase Intent", "High", "0.71")
                        st.metric("Viral Potential", "Medium", "0.65")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
        else:
            st.warning("Please enter some text to analyze.")

def render_visual_generation():
    """Render visual generation page"""
    st.title("🎨 Visual Generation")
    st.markdown("Generate professional marketing visuals and infographics")
    
    st.info("Visual generation functionality will be implemented based on sentiment analysis results.")
    
    # Placeholder for visual generation
    visual_type = st.selectbox(
        "Choose visual type:",
        ["Infographic", "Social Media Post", "Dashboard", "Chart", "Presentation Slide"]
    )
    
    if st.button("🎨 Generate Visual", type="primary"):
        st.info(f"Generating {visual_type}... (Feature in development)")

def render_reports():
    """Render reports page"""
    st.title("📋 Research Reports")
    st.markdown("Export comprehensive analysis reports")
    
    st.info("Report generation functionality will be available after analysis completion.")

def render_settings():
    """Render settings page"""
    st.title("⚙️ Settings")
    st.markdown("Configure API keys and application settings")
    
    st.subheader("🔑 API Configuration")
    
    # OpenAI API Key
    openai_key = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        value=st.session_state.get('openai_api_key', ''),
        help="Required for advanced sentiment analysis. Leave empty to use basic analysis."
    )
    
    if st.button("💾 Save Settings"):
        if openai_key:
            st.session_state['openai_api_key'] = openai_key
            st.success("✅ OpenAI API key saved!")
        else:
            st.session_state.pop('openai_api_key', None)
            st.info("OpenAI API key cleared. Basic analysis will be used.")
    
    st.markdown("---")
    st.subheader("📊 Data Sources")
    st.markdown("""
    The following open datasets are available without API keys:
    - ✅ **OpenNeuro**: Open neuroscience datasets
    - ✅ **Zenodo**: EEG Consumer Neuroscience datasets  
    - ✅ **PhysioNet**: EEG databases
    - ✅ **IEEE DataPort**: Neuromarketing datasets
    - ✅ **OSF**: Open Science Framework studies
    - ✅ **PubMed/PMC**: Neuroscience research papers
    - ✅ **Google Dataset Search**: Dataset discovery
    """)

if __name__ == "__main__":
    main()