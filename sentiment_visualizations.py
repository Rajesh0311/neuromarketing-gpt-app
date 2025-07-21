import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class SentimentVisualizer:
    def __init__(self):
        self.colors = {
            'positive': '#00C851',
            'negative': '#FF4444', 
            'neutral': '#FFA726',
            'background': '#f8f9fa'
        }
    
    def create_sentiment_gauge(self, sentiment_score, sentiment_label, confidence=0.85):
        """Create circular sentiment gauge with confidence indicator"""
        
        # Convert sentiment to 0-100 scale for gauge
        if sentiment_label.lower() == 'positive':
            gauge_value = 50 + (sentiment_score * 50)
            color = self.colors['positive']
        elif sentiment_label.lower() == 'negative':
            gauge_value = 50 - (abs(sentiment_score) * 50)
            color = self.colors['negative']
        else:
            gauge_value = 50
            color = self.colors['neutral']
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = gauge_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Sentiment: {sentiment_label.title()}"},
            delta = {'reference': 50, 'increasing': {'color': self.colors['positive']}, 
                    'decreasing': {'color': self.colors['negative']}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': '#FFE5E5'},
                    {'range': [25, 50], 'color': '#FFF3E0'},
                    {'range': [50, 75], 'color': '#E8F5E8'},
                    {'range': [75, 100], 'color': '#E0F2F1'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': confidence * 100}}))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor=self.colors['background'],
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig
    
    def create_emotion_radar_chart(self, emotions_dict):
        """Create radar chart for emotion analysis"""
        
        emotions = list(emotions_dict.keys())
        values = list(emotions_dict.values())
        
        # Convert to percentage scale
        values_pct = [v * 100 for v in values]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_pct,
            theta=emotions,
            fill='toself',
            name='Emotion Intensity',
            line_color='rgb(32, 201, 151)',
            fillcolor='rgba(32, 201, 151, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )),
            showlegend=True,
            title="Emotion Analysis Radar",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def create_confidence_heatmap(self, confidence_data):
        """Create confidence heatmap for sentiment analysis"""
        
        # Sample confidence matrix
        if not confidence_data:
            confidence_data = {
                'Positive': [0.85, 0.12, 0.03],
                'Neutral': [0.10, 0.75, 0.15], 
                'Negative': [0.05, 0.13, 0.82]
            }
        
        df = pd.DataFrame(confidence_data, 
                         index=['Predicted Positive', 'Predicted Neutral', 'Predicted Negative'])
        
        fig = go.Figure(data=go.Heatmap(
            z=df.values,
            x=df.columns,
            y=df.index,
            colorscale='RdYlGn',
            text=df.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Sentiment Confidence Matrix",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig

# Initialize the visualizer
sentiment_viz = SentimentVisualizer()