"""
Professional Export Module - Enhanced export capabilities from PR #4
Multi-format report generation with professional styling
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime
import base64
from io import BytesIO, StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

class ProfessionalExporter:
    """Professional report exporter for NeuroMarketing analysis"""
    
    def __init__(self):
        self.export_formats = ['PDF', 'DOCX', 'HTML', 'JSON', 'CSV', 'PowerPoint']
        self.template_styles = ['Professional', 'Modern', 'Minimal', 'Corporate', 'Academic']
        
    def export_comprehensive_report(
        self, 
        analysis_data: Dict[str, Any], 
        format_type: str = 'html',
        template_style: str = 'professional',
        include_visuals: bool = True,
        include_raw_data: bool = False
    ) -> Dict[str, Any]:
        """
        Export comprehensive analysis report in specified format
        
        Args:
            analysis_data: Complete analysis results from all modules
            format_type: Export format (pdf, docx, html, json, csv, pptx)
            template_style: Report styling template
            include_visuals: Whether to include charts and visualizations
            include_raw_data: Whether to include raw analysis data
            
        Returns:
            Dictionary containing export results and download information
        """
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format_type.lower() == 'html':
                content = self._generate_html_report(
                    analysis_data, template_style, include_visuals, include_raw_data
                )
                filename = f"neuromarketing_report_{timestamp}.html"
                mime_type = "text/html"
                
            elif format_type.lower() == 'json':
                content = self._generate_json_export(analysis_data, include_raw_data)
                filename = f"neuromarketing_data_{timestamp}.json"
                mime_type = "application/json"
                
            elif format_type.lower() == 'csv':
                content = self._generate_csv_export(analysis_data)
                filename = f"neuromarketing_data_{timestamp}.csv"
                mime_type = "text/csv"
                
            elif format_type.lower() == 'markdown':
                content = self._generate_markdown_report(
                    analysis_data, include_visuals, include_raw_data
                )
                filename = f"neuromarketing_report_{timestamp}.md"
                mime_type = "text/markdown"
                
            else:
                # Default to HTML
                content = self._generate_html_report(
                    analysis_data, template_style, include_visuals, include_raw_data
                )
                filename = f"neuromarketing_report_{timestamp}.html"
                mime_type = "text/html"
            
            return {
                'success': True,
                'content': content,
                'filename': filename,
                'mime_type': mime_type,
                'size': len(content) if isinstance(content, str) else len(str(content)),
                'format': format_type,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_html_report(
        self, 
        data: Dict[str, Any], 
        template_style: str,
        include_visuals: bool,
        include_raw_data: bool
    ) -> str:
        """Generate comprehensive HTML report"""
        
        # Get template CSS based on style
        css_styles = self._get_template_css(template_style)
        
        # Generate report sections
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NeuroMarketing Analysis Report</title>
    <style>
        {css_styles}
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="container">
        {self._generate_report_header()}
        {self._generate_executive_summary(data)}
        {self._generate_sentiment_analysis_section(data)}
        {self._generate_neural_simulation_section(data)}
        {self._generate_research_findings_section(data)}
        {self._generate_marketing_insights_section(data)}
        {self._generate_recommendations_section(data)}
        {self._generate_visualizations_section(data) if include_visuals else ''}
        {self._generate_raw_data_section(data) if include_raw_data else ''}
        {self._generate_report_footer()}
    </div>
</body>
</html>
"""
        return html_content
    
    def _get_template_css(self, template_style: str) -> str:
        """Get CSS styles for different templates"""
        
        base_css = """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .header {
            text-align: center;
            padding: 2rem 0;
            border-bottom: 3px solid #667eea;
            margin-bottom: 2rem;
        }
        
        .header h1 {
            font-size: 2.5rem;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .header .subtitle {
            font-size: 1.2rem;
            color: #666;
        }
        
        .section {
            margin: 2rem 0;
            padding: 1.5rem;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .section h2 {
            color: #333;
            margin-bottom: 1rem;
            font-size: 1.8rem;
        }
        
        .section h3 {
            color: #555;
            margin: 1rem 0 0.5rem 0;
            font-size: 1.3rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
        
        .insight-list {
            list-style: none;
            padding: 0;
        }
        
        .insight-list li {
            background: white;
            margin: 0.5rem 0;
            padding: 1rem;
            border-radius: 4px;
            border-left: 3px solid #28a745;
        }
        
        .recommendation-list li {
            border-left-color: #ffc107;
        }
        
        .risk-list li {
            border-left-color: #dc3545;
        }
        
        .chart-container {
            margin: 1rem 0;
            padding: 1rem;
            background: white;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        .data-table th,
        .data-table td {
            padding: 0.75rem;
            border: 1px solid #e9ecef;
            text-align: left;
        }
        
        .data-table th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }
        
        .data-table tr:nth-child(even) {
            background: #f8f9fa;
        }
        
        .footer {
            text-align: center;
            padding: 2rem 0;
            border-top: 2px solid #e9ecef;
            margin-top: 3rem;
            color: #666;
        }
        
        @media print {
            body { background: white; }
            .container { box-shadow: none; }
        }
        """
        
        if template_style.lower() == 'modern':
            return base_css + """
            .header h1 { color: #6c5ce7; }
            .section { border-left-color: #6c5ce7; }
            .metric-value { color: #6c5ce7; }
            """
            
        elif template_style.lower() == 'minimal':
            return base_css + """
            .section { background: white; border: 1px solid #e9ecef; }
            .header { border-bottom-color: #333; }
            .metric-value { color: #333; }
            """
            
        elif template_style.lower() == 'corporate':
            return base_css + """
            .header h1 { color: #2c3e50; }
            .section { border-left-color: #2c3e50; }
            .metric-value { color: #2c3e50; }
            body { font-family: Georgia, serif; }
            """
            
        else:  # Professional (default)
            return base_css
    
    def _generate_report_header(self) -> str:
        """Generate report header"""
        timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
        return f"""
        <div class="header">
            <h1>🧠 NeuroMarketing Analysis Report</h1>
            <div class="subtitle">Comprehensive Analysis Report</div>
            <div class="subtitle">Generated on {timestamp}</div>
        </div>
        """
    
    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary section"""
        
        # Extract key metrics from various analysis modules
        summary_metrics = {}
        
        if 'sentiment_results' in data:
            sentiment = data['sentiment_results']
            if 'basic_sentiment' in sentiment:
                summary_metrics['Overall Sentiment'] = sentiment['basic_sentiment'].get('polarity', 'N/A')
                summary_metrics['Confidence Score'] = f"{sentiment['basic_sentiment'].get('confidence', 0):.1%}"
        
        if 'neural_simulation_results' in data:
            neural = data['neural_simulation_results']
            if 'marketing_insights' in neural:
                summary_metrics['Neuro Score'] = f"{neural['marketing_insights'].get('neuro_score', 0):.1f}/100"
                summary_metrics['Purchase Probability'] = f"{neural['behavioral_outcomes'].get('purchase_probability', 0):.1%}"
        
        if 'dataset_results' in data:
            total_datasets = sum(
                result.get('count', 0) for result in data['dataset_results'].values() 
                if result.get('success')
            )
            summary_metrics['Datasets Analyzed'] = str(total_datasets)
        
        metrics_html = ""
        for metric, value in summary_metrics.items():
            metrics_html += f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{metric}</div>
            </div>
            """
        
        return f"""
        <div class="section">
            <h2>📋 Executive Summary</h2>
            <p>This comprehensive NeuroMarketing analysis report presents findings from advanced sentiment analysis, 
            neural simulation modeling, and multi-source research data integration. The analysis combines cutting-edge 
            neuroscience research with practical marketing insights to provide actionable recommendations.</p>
            
            <h3>Key Performance Metrics</h3>
            <div class="metrics-grid">
                {metrics_html}
            </div>
        </div>
        """
    
    def _generate_sentiment_analysis_section(self, data: Dict[str, Any]) -> str:
        """Generate sentiment analysis section"""
        
        if 'sentiment_results' not in data:
            return ""
        
        sentiment = data['sentiment_results']
        
        content = """
        <div class="section">
            <h2>📊 Sentiment Analysis Results</h2>
        """
        
        if 'basic_sentiment' in sentiment:
            basic = sentiment['basic_sentiment']
            content += f"""
            <h3>Core Sentiment Metrics</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{basic.get('polarity', 'N/A')}</div>
                    <div class="metric-label">Overall Sentiment</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{basic.get('confidence', 0):.1%}</div>
                    <div class="metric-label">Confidence Level</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{basic.get('positive_indicators', 0)}</div>
                    <div class="metric-label">Positive Indicators</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{basic.get('negative_indicators', 0)}</div>
                    <div class="metric-label">Negative Indicators</div>
                </div>
            </div>
            """
        
        if 'emotional_profile' in sentiment:
            emotions = sentiment['emotional_profile']
            content += """
            <h3>Emotional Dimensions</h3>
            <ul class="insight-list">
            """
            for emotion, score in emotions.items():
                content += f"<li><strong>{emotion.title()}:</strong> {score:.1%} activation</li>"
            content += "</ul>"
        
        if 'marketing_metrics' in sentiment:
            marketing = sentiment['marketing_metrics']
            content += f"""
            <h3>Marketing Performance Indicators</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{marketing.get('brand_appeal', 0):.1%}</div>
                    <div class="metric-label">Brand Appeal</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{marketing.get('purchase_intent', 0):.1%}</div>
                    <div class="metric-label">Purchase Intent</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{marketing.get('trust_score', 0):.1%}</div>
                    <div class="metric-label">Trust Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{marketing.get('viral_potential', 0):.1%}</div>
                    <div class="metric-label">Viral Potential</div>
                </div>
            </div>
            """
        
        content += "</div>"
        return content
    
    def _generate_neural_simulation_section(self, data: Dict[str, Any]) -> str:
        """Generate neural simulation section"""
        
        if 'neural_simulation_results' not in data:
            return ""
        
        neural = data['neural_simulation_results']
        
        content = """
        <div class="section">
            <h2>🧠 Neural Simulation Analysis</h2>
        """
        
        if 'behavioral_outcomes' in neural:
            outcomes = neural['behavioral_outcomes']
            content += f"""
            <h3>Behavioral Predictions</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{outcomes.get('purchase_probability', 0):.1%}</div>
                    <div class="metric-label">Purchase Probability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{outcomes.get('attention_score', 0):.1%}</div>
                    <div class="metric-label">Attention Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{outcomes.get('emotional_response', 0):.1%}</div>
                    <div class="metric-label">Emotional Response</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{outcomes.get('memory_retention', 0):.1%}</div>
                    <div class="metric-label">Memory Retention</div>
                </div>
            </div>
            """
        
        if 'marketing_insights' in neural:
            insights = neural['marketing_insights']
            
            content += f"""
            <h3>Neural Insights</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{insights.get('neuro_score', 0):.1f}</div>
                    <div class="metric-label">Neuro Score (0-100)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{insights.get('effectiveness_rating', 'N/A')}</div>
                    <div class="metric-label">Effectiveness Rating</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{insights.get('predicted_conversion_rate', 0):.1f}%</div>
                    <div class="metric-label">Predicted Conversion</div>
                </div>
            </div>
            """
            
            if 'top_strengths' in insights:
                content += """
                <h3>Top Strengths</h3>
                <ul class="insight-list">
                """
                for strength in insights['top_strengths']:
                    content += f"<li>{strength}</li>"
                content += "</ul>"
            
            if 'improvement_areas' in insights:
                content += """
                <h3>Areas for Improvement</h3>
                <ul class="insight-list recommendation-list">
                """
                for area in insights['improvement_areas']:
                    content += f"<li>{area}</li>"
                content += "</ul>"
        
        content += "</div>"
        return content
    
    def _generate_research_findings_section(self, data: Dict[str, Any]) -> str:
        """Generate research findings section"""
        
        if 'dataset_results' not in data:
            return ""
        
        datasets = data['dataset_results']
        
        content = """
        <div class="section">
            <h2>🔬 Research Data Analysis</h2>
        """
        
        total_datasets = 0
        active_sources = 0
        
        content += "<h3>Data Source Summary</h3>"
        content += '<table class="data-table"><thead><tr><th>Source</th><th>Status</th><th>Datasets Found</th><th>Notes</th></tr></thead><tbody>'
        
        for source, result in datasets.items():
            if result.get('success'):
                status = "✅ Active"
                count = result.get('count', 0)
                total_datasets += count
                active_sources += 1
                notes = "Successfully connected"
            else:
                status = "❌ Error"
                count = 0
                notes = result.get('error', 'Unknown error')[:50] + '...' if len(result.get('error', '')) > 50 else result.get('error', 'Unknown error')
            
            content += f"<tr><td>{source}</td><td>{status}</td><td>{count}</td><td>{notes}</td></tr>"
        
        content += "</tbody></table>"
        
        content += f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total_datasets}</div>
                <div class="metric-label">Total Datasets Found</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{active_sources}</div>
                <div class="metric-label">Active Data Sources</div>
            </div>
        </div>
        """
        
        content += "</div>"
        return content
    
    def _generate_marketing_insights_section(self, data: Dict[str, Any]) -> str:
        """Generate marketing insights section"""
        
        insights = []
        
        # Collect insights from various modules
        if 'sentiment_results' in data and 'marketing_metrics' in data['sentiment_results']:
            marketing = data['sentiment_results']['marketing_metrics']
            if marketing.get('brand_appeal', 0) > 0.7:
                insights.append("Strong brand appeal detected - leverage for premium positioning")
            if marketing.get('purchase_intent', 0) > 0.7:
                insights.append("High purchase intent - optimize call-to-action placement")
            if marketing.get('viral_potential', 0) > 0.6:
                insights.append("Good viral potential - consider social media amplification")
        
        if 'neural_simulation_results' in data:
            neural = data['neural_simulation_results']
            if 'marketing_insights' in neural:
                neural_insights = neural['marketing_insights']
                if neural_insights.get('neuro_score', 0) > 75:
                    insights.append("Excellent neural engagement - content resonates well with target audience")
                if neural_insights.get('predicted_conversion_rate', 0) > 60:
                    insights.append("High conversion probability - consider increasing budget allocation")
        
        if not insights:
            insights = [
                "Continue monitoring engagement metrics for optimization opportunities",
                "Consider A/B testing variations to improve performance",
                "Implement tracking to measure real-world conversion rates"
            ]
        
        content = """
        <div class="section">
            <h2>💡 Strategic Marketing Insights</h2>
            <ul class="insight-list">
        """
        
        for insight in insights:
            content += f"<li>{insight}</li>"
        
        content += """
            </ul>
        </div>
        """
        
        return content
    
    def _generate_recommendations_section(self, data: Dict[str, Any]) -> str:
        """Generate recommendations section"""
        
        recommendations = []
        
        # Collect recommendations from various modules
        if 'neural_simulation_results' in data:
            neural = data['neural_simulation_results']
            if 'marketing_insights' in neural and 'recommendations' in neural['marketing_insights']:
                recommendations.extend(neural['marketing_insights']['recommendations'])
        
        # Add general recommendations
        if not recommendations:
            recommendations = [
                "Optimize content for higher emotional engagement",
                "Enhance visual hierarchy in marketing materials",
                "Consider environmental factors in customer journey design",
                "Implement multi-touchpoint analysis for comprehensive insights",
                "Conduct follow-up studies to validate initial findings"
            ]
        
        content = """
        <div class="section">
            <h2>🎯 Action Recommendations</h2>
            <ul class="insight-list recommendation-list">
        """
        
        for rec in recommendations:
            content += f"<li>{rec}</li>"
        
        content += """
            </ul>
        </div>
        """
        
        return content
    
    def _generate_visualizations_section(self, data: Dict[str, Any]) -> str:
        """Generate visualizations section"""
        
        # Note: In a full implementation, this would include embedded charts
        # For now, we'll include placeholders and descriptions
        
        content = """
        <div class="section">
            <h2>📈 Data Visualizations</h2>
            <p>This section contains interactive charts and visualizations generated from the analysis data.</p>
            
            <div class="chart-container">
                <h3>Sentiment Distribution Chart</h3>
                <p>Pie chart showing the distribution of positive, negative, and neutral sentiment indicators.</p>
            </div>
            
            <div class="chart-container">
                <h3>Neural Activity Simulation</h3>
                <p>Time-series visualization of simulated brain region activity during stimulus exposure.</p>
            </div>
            
            <div class="chart-container">
                <h3>Marketing Metrics Dashboard</h3>
                <p>Gauge charts showing brand appeal, purchase intent, trust score, and viral potential.</p>
            </div>
        </div>
        """
        
        return content
    
    def _generate_raw_data_section(self, data: Dict[str, Any]) -> str:
        """Generate raw data section"""
        
        content = """
        <div class="section">
            <h2>📋 Raw Data Export</h2>
            <p>This section contains the raw analysis data in structured format for further processing.</p>
            
            <h3>Analysis Parameters</h3>
            <table class="data-table">
                <thead>
                    <tr><th>Parameter</th><th>Value</th></tr>
                </thead>
                <tbody>
        """
        
        # Add analysis parameters
        if 'sentiment_results' in data:
            sentiment = data['sentiment_results']
            content += f"<tr><td>Analysis Type</td><td>{sentiment.get('analysis_type', 'N/A')}</td></tr>"
            content += f"<tr><td>Text Length</td><td>{sentiment.get('text_length', 'N/A')} characters</td></tr>"
            content += f"<tr><td>Timestamp</td><td>{sentiment.get('timestamp', 'N/A')}</td></tr>"
        
        if 'neural_simulation_results' in data:
            neural = data['neural_simulation_results']
            content += f"<tr><td>Consumer Type</td><td>{neural.get('consumer_type', 'N/A')}</td></tr>"
            content += f"<tr><td>Simulation Duration</td><td>{neural.get('duration', 'N/A')} seconds</td></tr>"
            content += f"<tr><td>Stimulus Type</td><td>{neural.get('stimulus_type', 'N/A')}</td></tr>"
        
        content += """
                </tbody>
            </table>
        </div>
        """
        
        return content
    
    def _generate_report_footer(self) -> str:
        """Generate report footer"""
        
        return """
        <div class="footer">
            <p><strong>NeuroMarketing GPT Platform</strong></p>
            <p>Advanced Sentiment Analysis • Neural Simulation • Research Integration</p>
            <p>This report was generated automatically using AI-powered analysis tools.</p>
            <p>For questions or support, please contact your analysis team.</p>
        </div>
        """
    
    def _generate_json_export(self, data: Dict[str, Any], include_raw_data: bool) -> str:
        """Generate JSON export of analysis data"""
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'export_format': 'json',
                'platform': 'NeuroMarketing GPT',
                'version': '1.0'
            }
        }
        
        # Include analysis data
        if include_raw_data:
            export_data.update(data)
        else:
            # Include only summary data
            summary_data = {}
            
            if 'sentiment_results' in data:
                sentiment = data['sentiment_results']
                summary_data['sentiment_summary'] = {
                    'overall_sentiment': sentiment.get('basic_sentiment', {}).get('polarity', 'N/A'),
                    'confidence': sentiment.get('basic_sentiment', {}).get('confidence', 0),
                    'analysis_type': sentiment.get('analysis_type', 'N/A')
                }
            
            if 'neural_simulation_results' in data:
                neural = data['neural_simulation_results']
                summary_data['neural_summary'] = {
                    'neuro_score': neural.get('marketing_insights', {}).get('neuro_score', 0),
                    'purchase_probability': neural.get('behavioral_outcomes', {}).get('purchase_probability', 0),
                    'effectiveness_rating': neural.get('marketing_insights', {}).get('effectiveness_rating', 'N/A')
                }
            
            export_data['analysis_summary'] = summary_data
        
        return json.dumps(export_data, indent=2, default=str)
    
    def _generate_csv_export(self, data: Dict[str, Any]) -> str:
        """Generate CSV export of analysis metrics"""
        
        csv_data = []
        
        # Flatten data for CSV format
        if 'sentiment_results' in data:
            sentiment = data['sentiment_results']
            if 'basic_sentiment' in sentiment:
                basic = sentiment['basic_sentiment']
                csv_data.append({
                    'Metric': 'Overall Sentiment',
                    'Value': basic.get('polarity', 'N/A'),
                    'Category': 'Sentiment Analysis'
                })
                csv_data.append({
                    'Metric': 'Confidence Score',
                    'Value': basic.get('confidence', 0),
                    'Category': 'Sentiment Analysis'
                })
            
            if 'marketing_metrics' in sentiment:
                marketing = sentiment['marketing_metrics']
                for metric, value in marketing.items():
                    csv_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': value,
                        'Category': 'Marketing Metrics'
                    })
        
        if 'neural_simulation_results' in data:
            neural = data['neural_simulation_results']
            if 'behavioral_outcomes' in neural:
                outcomes = neural['behavioral_outcomes']
                for metric, value in outcomes.items():
                    csv_data.append({
                        'Metric': metric.replace('_', ' ').title(),
                        'Value': value,
                        'Category': 'Neural Simulation'
                    })
        
        # Convert to CSV string
        if csv_data:
            df = pd.DataFrame(csv_data)
            return df.to_csv(index=False)
        else:
            return "Metric,Value,Category\nNo Data,N/A,N/A"
    
    def _generate_markdown_report(
        self, 
        data: Dict[str, Any], 
        include_visuals: bool,
        include_raw_data: bool
    ) -> str:
        """Generate Markdown report"""
        
        timestamp = datetime.now().strftime('%B %d, %Y at %I:%M %p')
        
        content = f"""# 🧠 NeuroMarketing Analysis Report

**Generated:** {timestamp}

## 📋 Executive Summary

This comprehensive NeuroMarketing analysis report presents findings from advanced sentiment analysis, neural simulation modeling, and multi-source research data integration.

"""
        
        # Add sentiment analysis section
        if 'sentiment_results' in data:
            sentiment = data['sentiment_results']
            content += "## 📊 Sentiment Analysis Results\n\n"
            
            if 'basic_sentiment' in sentiment:
                basic = sentiment['basic_sentiment']
                content += f"""### Core Metrics
- **Overall Sentiment:** {basic.get('polarity', 'N/A')}
- **Confidence Level:** {basic.get('confidence', 0):.1%}
- **Positive Indicators:** {basic.get('positive_indicators', 0)}
- **Negative Indicators:** {basic.get('negative_indicators', 0)}

"""
            
            if 'marketing_metrics' in sentiment:
                marketing = sentiment['marketing_metrics']
                content += f"""### Marketing Performance
- **Brand Appeal:** {marketing.get('brand_appeal', 0):.1%}
- **Purchase Intent:** {marketing.get('purchase_intent', 0):.1%}
- **Trust Score:** {marketing.get('trust_score', 0):.1%}
- **Viral Potential:** {marketing.get('viral_potential', 0):.1%}

"""
        
        # Add neural simulation section
        if 'neural_simulation_results' in data:
            neural = data['neural_simulation_results']
            content += "## 🧠 Neural Simulation Analysis\n\n"
            
            if 'behavioral_outcomes' in neural:
                outcomes = neural['behavioral_outcomes']
                content += f"""### Behavioral Predictions
- **Purchase Probability:** {outcomes.get('purchase_probability', 0):.1%}
- **Attention Score:** {outcomes.get('attention_score', 0):.1%}
- **Emotional Response:** {outcomes.get('emotional_response', 0):.1%}
- **Memory Retention:** {outcomes.get('memory_retention', 0):.1%}

"""
            
            if 'marketing_insights' in neural:
                insights = neural['marketing_insights']
                content += f"""### Neural Insights
- **Neuro Score:** {insights.get('neuro_score', 0):.1f}/100
- **Effectiveness Rating:** {insights.get('effectiveness_rating', 'N/A')}
- **Predicted Conversion:** {insights.get('predicted_conversion_rate', 0):.1f}%

"""
                
                if 'recommendations' in insights:
                    content += "### Recommendations\n"
                    for rec in insights['recommendations']:
                        content += f"- {rec}\n"
                    content += "\n"
        
        # Add research findings
        if 'dataset_results' in data:
            datasets = data['dataset_results']
            content += "## 🔬 Research Data Analysis\n\n"
            
            total_datasets = sum(
                result.get('count', 0) for result in datasets.values() 
                if result.get('success')
            )
            active_sources = sum(1 for result in datasets.values() if result.get('success'))
            
            content += f"- **Total Datasets Found:** {total_datasets}\n"
            content += f"- **Active Data Sources:** {active_sources}\n\n"
            
            content += "### Data Source Status\n"
            for source, result in datasets.items():
                status = "✅ Active" if result.get('success') else "❌ Error"
                count = result.get('count', 0) if result.get('success') else 0
                content += f"- **{source}:** {status} ({count} datasets)\n"
            content += "\n"
        
        content += """## 💡 Strategic Insights

- Continue monitoring engagement metrics for optimization opportunities
- Consider A/B testing variations to improve performance
- Implement tracking to measure real-world conversion rates

## 🎯 Next Steps

1. Validate findings with additional testing
2. Implement recommended optimizations
3. Monitor performance metrics
4. Conduct follow-up analysis

---

*Report generated by NeuroMarketing GPT Platform*
*Advanced Sentiment Analysis • Neural Simulation • Research Integration*
"""
        
        return content

# Utility functions for Streamlit integration
def render_export_interface():
    """Render the export interface in Streamlit"""
    
    st.subheader("📋 Professional Export")
    
    exporter = ProfessionalExporter()
    
    # Check if there's data to export
    exportable_data = {}
    
    if 'sentiment_results' in st.session_state:
        exportable_data['sentiment_results'] = st.session_state['sentiment_results']
    
    if 'neural_simulation_results' in st.session_state:
        exportable_data['neural_simulation_results'] = st.session_state['neural_simulation_results']
    
    if 'dataset_results' in st.session_state:
        exportable_data['dataset_results'] = st.session_state['dataset_results']
    
    if not exportable_data:
        st.info("No analysis data available for export. Please run some analysis first.")
        return
    
    # Export configuration
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format:",
            ["HTML", "JSON", "CSV", "Markdown"],
            help="Choose the format for your export"
        )
        
        template_style = st.selectbox(
            "Template Style:",
            ["Professional", "Modern", "Minimal", "Corporate"],
            help="Choose the visual style for the report"
        )
    
    with col2:
        include_visuals = st.checkbox("Include Visualizations", True)
        include_raw_data = st.checkbox("Include Raw Data", False)
        
        if st.button("📊 Generate Export", type="primary"):
            with st.spinner("Generating export..."):
                result = exporter.export_comprehensive_report(
                    exportable_data,
                    export_format.lower(),
                    template_style.lower(),
                    include_visuals,
                    include_raw_data
                )
                
                if result['success']:
                    st.success(f"✅ Export generated successfully! ({result['size']} bytes)")
                    
                    # Display download button
                    st.download_button(
                        label=f"📥 Download {export_format} Report",
                        data=result['content'],
                        file_name=result['filename'],
                        mime=result['mime_type']
                    )
                    
                    # Show preview for HTML/Markdown
                    if export_format.lower() in ['html', 'markdown']:
                        with st.expander("📄 Preview"):
                            if export_format.lower() == 'html':
                                st.components.v1.html(result['content'], height=600, scrolling=True)
                            else:
                                st.markdown(result['content'])
                    
                    # Show preview for JSON
                    elif export_format.lower() == 'json':
                        with st.expander("📄 JSON Preview"):
                            st.json(json.loads(result['content']))
                    
                    # Show preview for CSV
                    elif export_format.lower() == 'csv':
                        with st.expander("📄 CSV Preview"):
                            df = pd.read_csv(StringIO(result['content']))
                            st.dataframe(df)
                
                else:
                    st.error(f"Export failed: {result.get('error', 'Unknown error')}")
    
    # Show current data summary
    st.markdown("---")
    st.markdown("### 📊 Available Data Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'sentiment_results' in exportable_data:
            st.success("✅ Sentiment Analysis Data")
        else:
            st.info("ℹ️ No Sentiment Data")
    
    with col2:
        if 'neural_simulation_results' in exportable_data:
            st.success("✅ Neural Simulation Data")
        else:
            st.info("ℹ️ No Neural Data")
    
    with col3:
        if 'dataset_results' in exportable_data:
            st.success("✅ Research Data")
        else:
            st.info("ℹ️ No Research Data")