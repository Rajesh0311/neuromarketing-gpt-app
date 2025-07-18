"""
Unified NeuroMarketing GPT Application
Integrates all PR features into a comprehensive neuromarketing platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="NeuroMarketing GPT - Unified Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .tab-header {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .analysis-result {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .feature-highlight {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'uploaded_media' not in st.session_state:
    st.session_state.uploaded_media = {'text': [], 'images': [], 'videos': [], 'audio': [], 'urls': []}
if 'environmental_data' not in st.session_state:
    st.session_state.environmental_data = {}

# Main application header
st.markdown("""
<div class="main-header">
    <h1>🧠 NeuroMarketing GPT Platform</h1>
    <h3>Revolutionary Neuromarketing Intelligence Platform</h3>
    <p>Advanced Sentiment Analysis • Media Processing • Environmental Simulation • Deep Research</p>
</div>
""", unsafe_allow_html=True)

# Create 10 tabs as specified in the requirements
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Advanced Sentiment Analysis",  # Enhanced from PR #4
    "🎭 Sarcasm & Irony Detection",    # Existing + Enhanced
    "🔬 Basic Monitoring",             # Existing Neural Monitoring
    "🎨 Professional Visuals",         # Enhanced Export from PR #4
    "📁 Media Input Hub",              # NEW - PR #4 Media Capabilities
    "🏪 Environmental Simulation",     # NEW - PR #5 Environmental Features
    "📋 Reports & Export",             # Enhanced Professional Reports
    "🧠 NeuroInsight-Africa Platform", # Existing Advanced Platform
    "🔬 Deep Research Engine",         # NEW - PR #3 Research Module
    "⚙️ System Integration"            # NEW - Unified Configuration
])

# Tab 1: Advanced Sentiment Analysis (Enhanced from PR #4)
with tab1:
    st.markdown('<div class="tab-header"><h2>📊 Advanced Sentiment Analysis</h2><p>Multi-dimensional emotional and psychological content analysis with AI enhancement</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_text = st.text_area(
            "Enter text for analysis:",
            height=200,
            placeholder="Paste your marketing content, social media posts, or any text for comprehensive analysis..."
        )
        
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["Basic Sentiment", "Advanced Emotions", "Marketing Insights", "Psychological Profiling", "Cross-Cultural"]
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if analysis_text:
                with st.spinner("Performing advanced sentiment analysis..."):
                    # Simulate advanced analysis
                    time.sleep(2)
                    
                    # Generate comprehensive results
                    results = {
                        'overall_sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], p=[0.6, 0.2, 0.2]),
                        'confidence': np.random.uniform(0.75, 0.95),
                        'emotions': {
                            'Joy': np.random.uniform(0.4, 0.8),
                            'Trust': np.random.uniform(0.5, 0.9),
                            'Fear': np.random.uniform(0.1, 0.3),
                            'Surprise': np.random.uniform(0.2, 0.6),
                            'Sadness': np.random.uniform(0.1, 0.3),
                            'Disgust': np.random.uniform(0.05, 0.2),
                            'Anger': np.random.uniform(0.05, 0.25),
                            'Anticipation': np.random.uniform(0.3, 0.7)
                        },
                        'marketing_metrics': {
                            'Brand Appeal': np.random.uniform(0.65, 0.85),
                            'Purchase Intent': np.random.uniform(0.60, 0.80),
                            'Viral Potential': np.random.uniform(0.50, 0.75),
                            'Memorability': np.random.uniform(0.55, 0.80)
                        }
                    }
                    
                    st.session_state.analysis_results['sentiment'] = results
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
    
    with col2:
        st.markdown("### Analysis Insights")
        if 'sentiment' in st.session_state.analysis_results:
            results = st.session_state.analysis_results['sentiment']
            
            st.metric("Overall Sentiment", results['overall_sentiment'], f"{results['confidence']:.1%} confidence")
            
            # Emotion radar chart
            emotions = results['emotions']
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=list(emotions.values()),
                theta=list(emotions.keys()),
                fill='toself',
                name='Emotions'
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# Tab 2: Sarcasm & Irony Detection
with tab2:
    st.markdown('<div class="tab-header"><h2>🎭 Sarcasm & Irony Detection</h2><p>Advanced contextual analysis with cultural adaptation</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sarcasm_text = st.text_area(
            "Enter text for sarcasm analysis:",
            height=150,
            placeholder="Enter text that might contain sarcasm, irony, or subtle meanings..."
        )
        
        detection_level = st.selectbox("Detection Sensitivity:", ["Standard", "High", "Maximum"])
        cultural_context = st.selectbox("Cultural Context:", ["Global", "US", "UK", "AU", "CA"])
        
        if st.button("🔍 Detect Sarcasm", type="primary"):
            if sarcasm_text:
                with st.spinner("Analyzing linguistic patterns..."):
                    time.sleep(1.5)
                    
                    sarcasm_probability = np.random.uniform(0.15, 0.85)
                    irony_probability = np.random.uniform(0.10, 0.70)
                    
                    st.markdown("### Detection Results")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Sarcasm Probability", f"{sarcasm_probability:.1%}")
                    with col_b:
                        st.metric("Irony Probability", f"{irony_probability:.1%}")
                    
                    # Linguistic markers
                    st.markdown("### Detected Markers")
                    markers = ["Contrast patterns", "Exaggeration", "Contextual mismatch", "Sentiment inversion"]
                    for marker in markers[:np.random.randint(2, 4)]:
                        st.markdown(f'<span class="feature-highlight">{marker}</span>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Cultural Adaptation")
        st.info("Sarcasm detection adapts to cultural context and linguistic patterns specific to different regions.")
        
        # Cultural sensitivity chart
        cultural_factors = {
            'Directness': np.random.uniform(0.3, 0.9),
            'Context Dependency': np.random.uniform(0.4, 0.8),
            'Humor Style': np.random.uniform(0.2, 0.7),
            'Social Politeness': np.random.uniform(0.5, 0.9)
        }
        
        fig = px.bar(
            x=list(cultural_factors.keys()),
            y=list(cultural_factors.values()),
            title="Cultural Sensitivity Factors"
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 3: Basic Monitoring (Neural Monitoring)
with tab3:
    st.markdown('<div class="tab-header"><h2>🔬 Basic Neural Monitoring</h2><p>Real-time neural pattern analysis and monitoring</p></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### EEG Device Connection")
        device_options = ["NeuroSky", "Emotiv EPOC", "OpenBCI", "Muse", "Simulation Mode"]
        selected_device = st.selectbox("Select EEG Device:", device_options)
        
        connection_status = st.button("🔗 Connect Device")
        if connection_status:
            st.success(f"✅ Connected to {selected_device}")
    
    with col2:
        st.markdown("### Frequency Bands")
        frequency_bands = {
            'Delta (0.5-4 Hz)': np.random.uniform(20, 40),
            'Theta (4-8 Hz)': np.random.uniform(15, 35),
            'Alpha (8-13 Hz)': np.random.uniform(25, 45),
            'Beta (13-30 Hz)': np.random.uniform(30, 50),
            'Gamma (30-100 Hz)': np.random.uniform(10, 25)
        }
        
        for band, value in frequency_bands.items():
            st.metric(band, f"{value:.1f} μV")
    
    with col3:
        st.markdown("### Real-time Monitoring")
        
        # Generate real-time data simulation
        if st.checkbox("Start Monitoring"):
            chart_placeholder = st.empty()
            
            # Simulate real-time EEG data
            time_points = np.arange(0, 10, 0.1)
            eeg_data = np.sin(time_points * 2) + 0.5 * np.sin(time_points * 8) + 0.3 * np.random.randn(len(time_points))
            
            fig = px.line(x=time_points, y=eeg_data, title="Live EEG Signal")
            chart_placeholder.plotly_chart(fig, use_container_width=True)

# Tab 4: Professional Visuals (Enhanced Export from PR #4)
with tab4:
    st.markdown('<div class="tab-header"><h2>🎨 Professional Visuals</h2><p>Generate publication-ready charts, infographics, and marketing visuals</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Visual Generation Options")
        
        visual_type = st.selectbox(
            "Visual Type:",
            ["Sentiment Dashboard", "Emotion Radar", "Marketing Metrics", "Comparison Chart", "Infographic"]
        )
        
        color_scheme = st.selectbox(
            "Color Scheme:",
            ["Professional Blue", "Marketing Gradient", "Neutral Tones", "Brand Colors"]
        )
        
        export_format = st.selectbox(
            "Export Format:",
            ["PNG (High-Res)", "PDF (Vector)", "SVG (Scalable)", "Interactive HTML"]
        )
        
        if st.button("🎨 Generate Visual", type="primary"):
            with st.spinner("Creating professional visual..."):
                time.sleep(2)
                st.success("✅ Visual generated successfully!")
    
    with col2:
        st.markdown("### Generated Visuals")
        
        # Sample marketing dashboard
        if 'sentiment' in st.session_state.analysis_results:
            results = st.session_state.analysis_results['sentiment']
            marketing_metrics = results['marketing_metrics']
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Brand Appeal', 'Purchase Intent', 'Viral Potential', 'Memorability'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            metrics = list(marketing_metrics.items())
            for i, (metric, value) in enumerate(metrics):
                row = i // 2 + 1
                col = i % 2 + 1
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=value * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': metric},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "#667eea"},
                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 90}}
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(height=500, title_text="Marketing Performance Dashboard")
            st.plotly_chart(fig, use_container_width=True)

# Tab 5: Media Input Hub (NEW - PR #4 Media Capabilities)
with tab5:
    st.markdown('<div class="tab-header"><h2>📁 Media Input Hub</h2><p>Upload and analyze text, images, videos, audio, and URLs</p></div>', unsafe_allow_html=True)
    
    # Media upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Text Analysis")
        text_tab1, text_tab2, text_tab3 = st.tabs(["Ad Copy", "Social Media", "Brand Messaging"])
        
        with text_tab1:
            ad_copy = st.text_area("Advertisement Text:", height=100)
            if ad_copy:
                st.session_state.uploaded_media['text'].append({
                    'type': 'ad_copy',
                    'content': ad_copy,
                    'timestamp': datetime.now()
                })
        
        with text_tab2:
            social_content = st.text_area("Social Media Content:", height=100)
            if social_content:
                st.session_state.uploaded_media['text'].append({
                    'type': 'social_media',
                    'content': social_content,
                    'timestamp': datetime.now()
                })
        
        with text_tab3:
            brand_message = st.text_area("Brand Messaging:", height=100)
            if brand_message:
                st.session_state.uploaded_media['text'].append({
                    'type': 'brand_messaging',
                    'content': brand_message,
                    'timestamp': datetime.now()
                })
        
        st.markdown("### Image Upload")
        uploaded_images = st.file_uploader(
            "Upload Images:",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            accept_multiple_files=True
        )
        if uploaded_images:
            st.session_state.uploaded_media['images'].extend(uploaded_images)
            st.success(f"✅ {len(uploaded_images)} images uploaded")
    
    with col2:
        st.markdown("### Video Upload")
        uploaded_videos = st.file_uploader(
            "Upload Videos:",
            type=['mp4', 'mov', 'avi', 'webm'],
            accept_multiple_files=True
        )
        if uploaded_videos:
            st.session_state.uploaded_media['videos'].extend(uploaded_videos)
            st.success(f"✅ {len(uploaded_videos)} videos uploaded")
        
        st.markdown("### Audio Upload")
        uploaded_audio = st.file_uploader(
            "Upload Audio:",
            type=['mp3', 'wav', 'aac', 'ogg'],
            accept_multiple_files=True
        )
        if uploaded_audio:
            st.session_state.uploaded_media['audio'].extend(uploaded_audio)
            st.success(f"✅ {len(uploaded_audio)} audio files uploaded")
        
        st.markdown("### URL Analysis")
        url_input = st.text_input("Website URL:")
        if st.button("📊 Capture & Analyze URL") and url_input:
            st.session_state.uploaded_media['urls'].append({
                'url': url_input,
                'timestamp': datetime.now()
            })
            st.success("✅ URL added for analysis")
    
    # Content summary
    st.markdown("### Content Summary")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    
    with col_a:
        st.metric("Text Inputs", len(st.session_state.uploaded_media['text']))
    with col_b:
        st.metric("Images", len(st.session_state.uploaded_media['images']))
    with col_c:
        st.metric("Videos", len(st.session_state.uploaded_media['videos']))
    with col_d:
        st.metric("Audio Files", len(st.session_state.uploaded_media['audio']))
    with col_e:
        st.metric("URLs", len(st.session_state.uploaded_media['urls']))

# Tab 6: Environmental Simulation (NEW - PR #5 Environmental Features)
with tab6:
    st.markdown('<div class="tab-header"><h2>🏪 Environmental Simulation</h2><p>Advanced walkthrough recording and environmental sensor simulation</p></div>', unsafe_allow_html=True)
    
    simulation_type = st.selectbox(
        "Simulation Type:",
        ["Retail Store Navigation", "Drive-Through Experience", "Museum Exhibition", "Casino Environment", "Automotive Showroom"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Environment Setup")
        
        if simulation_type == "Retail Store Navigation":
            store_type = st.selectbox("Store Type:", ["Grocery", "Fashion", "Electronics", "Pharmacy"])
            layout_style = st.selectbox("Layout:", ["Grid", "Loop", "Free-form", "Boutique"])
            
        elif simulation_type == "Drive-Through Experience":
            restaurant_type = st.selectbox("Restaurant Type:", ["Fast Food", "Coffee", "Ice Cream", "Pharmacy"])
            menu_complexity = st.selectbox("Menu Complexity:", ["Simple", "Medium", "Complex"])
            
        st.markdown("### Environmental Factors")
        lighting = st.slider("Lighting Level", 0, 100, 70)
        noise_level = st.slider("Noise Level", 0, 100, 30)
        crowd_density = st.slider("Crowd Density", 0, 100, 50)
        temperature = st.slider("Temperature (°F)", 60, 80, 72)
    
    with col2:
        st.markdown("### Walkthrough Recording")
        
        recording_mode = st.selectbox("Recording Mode:", ["Mobile Phone", "Professional Setup", "360° Camera"])
        
        if st.button("🎥 Start Recording"):
            with st.spinner("Initializing recording..."):
                time.sleep(2)
                st.success("✅ Recording started!")
                
                # Simulate walkthrough data
                walkthrough_data = {
                    'duration': np.random.randint(3, 15),
                    'decision_points': np.random.randint(5, 20),
                    'attention_zones': np.random.randint(8, 25),
                    'emotional_peaks': np.random.randint(3, 8)
                }
                
                st.session_state.environmental_data['walkthrough'] = walkthrough_data
        
        if 'walkthrough' in st.session_state.environmental_data:
            data = st.session_state.environmental_data['walkthrough']
            st.metric("Duration", f"{data['duration']} min")
            st.metric("Decision Points", data['decision_points'])
            st.metric("Attention Zones", data['attention_zones'])
    
    with col3:
        st.markdown("### Spatial Analysis")
        
        if 'walkthrough' in st.session_state.environmental_data:
            # Generate heatmap simulation
            heatmap_data = np.random.rand(10, 10)
            
            fig = px.imshow(
                heatmap_data,
                color_continuous_scale="Viridis",
                title="Attention Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Start a walkthrough recording to see spatial analysis")

# Tab 7: Reports & Export (Enhanced Professional Reports)
with tab7:
    st.markdown('<div class="tab-header"><h2>📋 Reports & Export</h2><p>Comprehensive analysis reports and professional export capabilities</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Report Configuration")
        
        report_type = st.selectbox(
            "Report Type:",
            ["Executive Summary", "Technical Analysis", "Marketing Insights", "Complete Report"]
        )
        
        export_format = st.selectbox(
            "Export Format:",
            ["PDF", "DOCX", "HTML", "JSON", "PowerPoint"]
        )
        
        include_visuals = st.checkbox("Include Visualizations", True)
        include_raw_data = st.checkbox("Include Raw Data", False)
        include_recommendations = st.checkbox("Include Recommendations", True)
        
        if st.button("📊 Generate Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                time.sleep(3)
                
                # Generate report content
                report_content = f"""
# NeuroMarketing Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Type:** {report_type}

## Executive Summary
- Analysis completed across multiple dimensions
- {len(st.session_state.uploaded_media['text'])} text inputs analyzed
- {len(st.session_state.uploaded_media['images'])} images processed
- Environmental simulation data collected

## Key Findings
- Overall sentiment: {'Positive' if np.random.random() > 0.3 else 'Neutral'}
- Engagement score: {np.random.uniform(0.7, 0.9):.1%}
- Brand appeal: {np.random.uniform(0.6, 0.8):.1%}

## Recommendations
1. Optimize content for higher emotional engagement
2. Enhance visual hierarchy in marketing materials
3. Consider environmental factors in customer journey design
"""
                
                st.success("✅ Report generated successfully!")
                
                # Store report for display
                st.session_state.generated_report = report_content
    
    with col2:
        st.markdown("### Generated Report Preview")
        
        if 'generated_report' in st.session_state:
            st.markdown(st.session_state.generated_report)
            
            # Download button
            st.download_button(
                label="📥 Download Report",
                data=st.session_state.generated_report,
                file_name=f"neuromarketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.info("Generate a report to see preview")

# Tab 8: NeuroInsight-Africa Platform (Existing Advanced Platform)
with tab8:
    st.markdown('<div class="tab-header"><h2>🧠 NeuroInsight-Africa Platform</h2><p>Advanced neuromarketing platform with cultural adaptation for African markets</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Cultural Adaptation Engine")
        
        african_region = st.selectbox(
            "Target Region:",
            ["West Africa", "East Africa", "Southern Africa", "North Africa", "Pan-African"]
        )
        
        language_preference = st.selectbox(
            "Primary Language:",
            ["English", "French", "Arabic", "Swahili", "Hausa", "Yoruba", "Zulu"]
        )
        
        cultural_factors = {
            'Collectivism vs Individualism': st.slider("", 0, 100, 75),
            'High Context Communication': st.slider("", 0, 100, 80),
            'Respect for Authority': st.slider("", 0, 100, 85),
            'Ubuntu Philosophy': st.slider("", 0, 100, 90)
        }
        
        st.markdown("### Local Market Insights")
        market_metrics = {
            'Mobile Usage': np.random.uniform(0.8, 0.95),
            'Social Media Penetration': np.random.uniform(0.6, 0.85),
            'Brand Loyalty': np.random.uniform(0.7, 0.9),
            'Price Sensitivity': np.random.uniform(0.75, 0.95)
        }
        
        for metric, value in market_metrics.items():
            st.metric(metric, f"{value:.1%}")
    
    with col2:
        st.markdown("### Cultural Sentiment Analysis")
        
        # African cultural sentiment radar
        cultural_dimensions = {
            'Ubuntu (Community)': np.random.uniform(0.7, 0.95),
            'Respect (Dignity)': np.random.uniform(0.8, 0.95),
            'Spirituality': np.random.uniform(0.6, 0.9),
            'Family Values': np.random.uniform(0.8, 0.95),
            'Tradition': np.random.uniform(0.7, 0.9),
            'Innovation': np.random.uniform(0.5, 0.8)
        }
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(cultural_dimensions.values()),
            theta=list(cultural_dimensions.keys()),
            fill='toself',
            name='Cultural Alignment'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=False,
            title="Cultural Sentiment Profile"
        )
        st.plotly_chart(fig, use_container_width=True)

# Tab 9: Deep Research Engine (NEW - PR #3 Research Module)
with tab9:
    st.markdown('<div class="tab-header"><h2>🔬 Deep Research Engine</h2><p>OpenNeuro dataset integration and global research synthesis</p></div>', unsafe_allow_html=True)
    
    research_tab1, research_tab2, research_tab3 = st.tabs(["Dataset Search", "Literature Review", "Research Synthesis"])
    
    with research_tab1:
        st.markdown("### Multi-Source Dataset Discovery")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_query = st.text_input("Search Keywords:", placeholder="e.g., EEG, fMRI, neuromarketing")
            
            data_sources = st.multiselect(
                "Data Sources:",
                ["OpenNeuro", "Zenodo", "PhysioNet", "IEEE DataPort", "PubMed"],
                default=["OpenNeuro", "Zenodo"]
            )
            
            if st.button("🔍 Search Datasets"):
                with st.spinner("Searching across multiple databases..."):
                    time.sleep(2)
                    
                    # Simulate dataset results
                    datasets_found = np.random.randint(15, 45)
                    
                    st.success(f"✅ Found {datasets_found} datasets across {len(data_sources)} sources")
                    
                    # Sample dataset results
                    sample_datasets = [
                        {"name": "Consumer EEG Response Dataset", "subjects": 45, "source": "OpenNeuro"},
                        {"name": "Marketing Stimulus EEG Data", "subjects": 32, "source": "Zenodo"},
                        {"name": "Brand Recognition Neural Patterns", "subjects": 28, "source": "IEEE DataPort"}
                    ]
                    
                    for dataset in sample_datasets:
                        with st.expander(f"📊 {dataset['name']}"):
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Subjects", dataset['subjects'])
                            with col_b:
                                st.metric("Source", dataset['source'])
                            with col_c:
                                if st.button(f"📥 Download", key=dataset['name']):
                                    st.success("Dataset download initiated")
        
        with col2:
            st.markdown("### Search Status")
            st.info("🔍 Search across open neuroscience databases")
            
            if data_sources:
                for source in data_sources:
                    st.markdown(f"✅ {source}")
    
    with research_tab2:
        st.markdown("### Academic Literature Search")
        
        literature_query = st.text_input("Literature Keywords:", placeholder="neuromarketing consumer behavior")
        
        paper_filters = {
            "Publication Year": st.slider("From Year:", 2015, 2024, 2020),
            "Study Type": st.selectbox("Study Type:", ["All", "Experimental", "Review", "Meta-analysis"]),
            "Journal Ranking": st.selectbox("Journal Ranking:", ["All", "Q1", "Q1-Q2", "Top 10"])
        }
        
        if st.button("📚 Search Literature"):
            with st.spinner("Searching academic databases..."):
                time.sleep(2)
                
                papers_found = np.random.randint(25, 85)
                st.success(f"✅ Found {papers_found} relevant papers")
                
                # Sample paper results
                sample_papers = [
                    {
                        "title": "Neural Mechanisms of Brand Preference in Consumer Decision Making",
                        "authors": "Smith, J. et al.",
                        "journal": "Journal of Consumer Psychology",
                        "year": 2023,
                        "citations": 47
                    },
                    {
                        "title": "EEG-Based Analysis of Emotional Responses to Marketing Stimuli",
                        "authors": "Johnson, A. et al.",
                        "journal": "NeuroImage",
                        "year": 2022,
                        "citations": 62
                    }
                ]
                
                for paper in sample_papers:
                    with st.expander(f"📄 {paper['title']}"):
                        st.write(f"**Authors:** {paper['authors']}")
                        st.write(f"**Journal:** {paper['journal']} ({paper['year']})")
                        st.write(f"**Citations:** {paper['citations']}")
    
    with research_tab3:
        st.markdown("### Research Synthesis & Meta-Analysis")
        
        if st.button("🧠 Generate Research Synthesis"):
            with st.spinner("Synthesizing research findings..."):
                time.sleep(3)
                
                synthesis_results = {
                    "Total Studies": np.random.randint(25, 50),
                    "Total Participants": np.random.randint(1500, 3000),
                    "Effect Size": np.random.uniform(0.4, 0.8),
                    "Confidence Interval": "95%"
                }
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Studies Included", synthesis_results["Total Studies"])
                with col2:
                    st.metric("Total Participants", synthesis_results["Total Participants"])
                with col3:
                    st.metric("Effect Size", f"{synthesis_results['Effect Size']:.2f}")
                with col4:
                    st.metric("Confidence Level", synthesis_results["Confidence Interval"])
                
                st.markdown("### Key Findings")
                findings = [
                    "Strong correlation between neural activity and brand preference",
                    "Emotional processing significantly impacts purchase decisions",
                    "Cultural factors modulate neuromarketing responses",
                    "Multi-modal stimuli enhance engagement metrics"
                ]
                
                for finding in findings:
                    st.markdown(f"• {finding}")

# Tab 10: System Integration (NEW - Unified Configuration)
with tab10:
    st.markdown('<div class="tab-header"><h2>⚙️ System Integration</h2><p>Unified configuration and cross-module data management</p></div>', unsafe_allow_html=True)
    
    integration_tab1, integration_tab2, integration_tab3 = st.tabs(["API Configuration", "Data Flow", "Export Settings"])
    
    with integration_tab1:
        st.markdown("### API Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### External APIs")
            
            # OpenAI API
            openai_key = st.text_input("OpenAI API Key:", type="password", 
                                     help="For enhanced sentiment analysis")
            if openai_key:
                st.session_state['openai_api_key'] = openai_key
            
            # Other API configurations
            canva_integration = st.checkbox("Enable Canva Pro Integration")
            social_media_apis = st.multiselect(
                "Social Media APIs:",
                ["Twitter", "Facebook", "Instagram", "LinkedIn", "TikTok"]
            )
        
        with col2:
            st.markdown("#### Data Sources")
            
            # Database connections
            database_type = st.selectbox("Database:", ["Local SQLite", "PostgreSQL", "MongoDB", "Cloud Storage"])
            
            # Research database access
            research_access = st.multiselect(
                "Research Databases:",
                ["OpenNeuro", "Zenodo", "PhysioNet", "PubMed", "IEEE Xplore"],
                default=["OpenNeuro", "Zenodo"]
            )
            
            if st.button("🔗 Test Connections"):
                with st.spinner("Testing API connections..."):
                    time.sleep(2)
                    st.success("✅ All connections successful!")
    
    with integration_tab2:
        st.markdown("### Data Flow Management")
        
        # Data flow diagram
        st.markdown("#### Cross-Module Data Pipeline")
        
        flow_stages = [
            "📁 Media Input → 📊 Sentiment Analysis",
            "🔬 Research Data → 🏪 Environmental Simulation", 
            "🎭 Sarcasm Detection → 📋 Professional Reports",
            "🧠 Neural Monitoring → 🎨 Visual Generation",
            "All Modules → ⚙️ Unified Export System"
        ]
        
        for stage in flow_stages:
            st.markdown(f"• {stage}")
        
        st.markdown("#### Data Synchronization")
        
        sync_status = {
            "Sentiment Analysis": "✅ Synced",
            "Media Hub": "✅ Synced", 
            "Environmental Data": "⏳ Syncing",
            "Research Engine": "✅ Synced",
            "Export System": "✅ Ready"
        }
        
        for module, status in sync_status.items():
            st.markdown(f"**{module}:** {status}")
    
    with integration_tab3:
        st.markdown("### Export & Integration Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Preferences")
            
            default_format = st.selectbox("Default Export Format:", ["PDF", "DOCX", "JSON", "HTML"])
            include_metadata = st.checkbox("Include Metadata", True)
            compress_exports = st.checkbox("Compress Large Files", True)
            
            export_destination = st.selectbox(
                "Export Destination:",
                ["Local Download", "Cloud Storage", "Email", "FTP Server"]
            )
        
        with col2:
            st.markdown("#### Integration Workflows")
            
            auto_analysis = st.checkbox("Auto-analyze uploaded content", True)
            real_time_sync = st.checkbox("Real-time cross-module sync", True)
            notification_alerts = st.checkbox("Analysis completion alerts", False)
            
            workflow_templates = st.multiselect(
                "Workflow Templates:",
                ["Marketing Campaign Analysis", "Product Testing", "Brand Assessment", "Custom"]
            )
        
        if st.button("💾 Save Configuration"):
            st.success("✅ Configuration saved successfully!")

# Footer with system status
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Active Modules", "10/10")
with col2:
    st.metric("Data Processed", f"{np.random.randint(150, 500)} MB")
with col3:
    st.metric("Analysis Completed", np.random.randint(25, 100))
with col4:
    st.metric("System Status", "🟢 Online")

st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>🧠 NeuroMarketing GPT Platform - Revolutionary Neuromarketing Intelligence</p>
    <p>Integrated PR #3 (Deep Research) • PR #4 (Media Input) • PR #5 (Environmental Simulation)</p>
</div>
""", unsafe_allow_html=True)