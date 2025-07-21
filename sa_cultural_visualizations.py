import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class SACulturalVisualizer:
    def __init__(self):
        self.sa_colors = {
            'ubuntu': '#FF6B35',      # Ubuntu orange
            'cultural': '#4ECDC4',    # Teal for cultural
            'language': '#45B7D1',    # Blue for language
            'regional': '#96CEB4',    # Green for regional
            'sensitivity': '#FFEAA7', # Yellow for sensitivity
            'background': '#f8f9fa'
        }
        
        self.ubuntu_principles = [
            'Community Focus',
            'Shared Humanity', 
            'Collective Responsibility',
            'Compassion',
            'Respect for Others',
            'Interconnectedness'
        ]
    
    def create_ubuntu_compatibility_gauge(self, ubuntu_score, detailed_scores=None):
        """Create Ubuntu compatibility circular gauge"""
        
        # Convert to percentage for display
        ubuntu_percentage = ubuntu_score * 100
        
        # Determine color based on score
        if ubuntu_percentage >= 70:
            gauge_color = self.sa_colors['ubuntu']
            status = "High Ubuntu Alignment"
        elif ubuntu_percentage >= 50:
            gauge_color = self.sa_colors['cultural']
            status = "Moderate Ubuntu Alignment"
        else:
            gauge_color = '#FF7675'
            status = "Low Ubuntu Alignment"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = ubuntu_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Ubuntu Compatibility<br><span style='font-size:0.8em;color:gray'>{status}</span>"},
            delta = {'reference': 50, 'increasing': {'color': self.sa_colors['ubuntu']}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_color, 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#FFE5E5'},
                    {'range': [30, 50], 'color': '#FFF3E0'},
                    {'range': [50, 70], 'color': '#E8F5E8'},
                    {'range': [70, 100], 'color': '#E0F7FA'}],
                'threshold': {
                    'line': {'color': self.sa_colors['ubuntu'], 'width': 4},
                    'thickness': 0.75,
                    'value': 75}}))
        
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor=self.sa_colors['background'],
            font={'color': "darkblue", 'family': "Arial", 'size': 12}
        )
        
        return fig
    
    def create_cultural_sensitivity_radar(self, cultural_scores):
        """Create SA Cultural Sensitivity Radar Chart"""
        
        # Default cultural dimensions if not provided
        if not cultural_scores:
            cultural_scores = {
                'Language Respect': 0.75,
                'Cultural Awareness': 0.68,
                'Regional Sensitivity': 0.82,
                'Ubuntu Philosophy': 0.71,
                'Cross-Cultural Understanding': 0.79,
                'Historical Context': 0.65,
                'Social Harmony': 0.73,
                'Community Values': 0.77
            }
        
        categories = list(cultural_scores.keys())
        values = [v * 100 for v in cultural_scores.values()]  # Convert to percentage
        
        # Close the radar chart by adding first value at the end
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            fill='toself',
            name='Cultural Sensitivity',
            line=dict(color=self.sa_colors['cultural'], width=3),
            fillcolor=f"rgba(78, 205, 196, 0.3)",  # Transparent fill
            marker=dict(size=8, color=self.sa_colors['cultural'])
        ))
        
        # Add Ubuntu benchmark line
        ubuntu_benchmark = [70] * len(categories_closed)
        fig.add_trace(go.Scatterpolar(
            r=ubuntu_benchmark,
            theta=categories_closed,
            mode='lines',
            name='Ubuntu Benchmark',
            line=dict(color=self.sa_colors['ubuntu'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%',
                    tickfont=dict(size=10),
                    gridcolor='lightgray'
                ),
                angularaxis=dict(
                    tickfont=dict(size=11)
                )
            ),
            showlegend=True,
            title={
                'text': "South African Cultural Sensitivity Analysis",
                'x': 0.5,
                'font': {'size': 16, 'color': 'darkblue'}
            },
            height=450,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor=self.sa_colors['background']
        )
        
        return fig
    
    def create_language_respect_meter(self, language_scores):
        """Create language respect visualization"""
        
        if not language_scores:
            # Sample data for 11 SA official languages
            language_scores = {
                'English': 0.85,
                'Afrikaans': 0.72,
                'isiZulu': 0.78,
                'isiXhosa': 0.76,
                'Sepedi': 0.68,
                'Setswana': 0.71,
                'Sesotho': 0.69,
                'Xitsonga': 0.65,
                'siSwati': 0.67,
                'Tshivenda': 0.63,
                'isiNdebele': 0.66
            }
        
        languages = list(language_scores.keys())
        scores = [v * 100 for v in language_scores.values()]
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=scores,
            y=languages,
            orientation='h',
            marker=dict(
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Respect Level %")
            ),
            text=[f"{score:.0f}%" for score in scores],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Language Respect Index - SA Official Languages",
            xaxis_title="Respect Level (%)",
            yaxis_title="Languages",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor=self.sa_colors['background']
        )
        
        return fig
    
    def create_regional_appropriateness_chart(self, provincial_data):
        """Create provincial appropriateness visualization"""
        
        if not provincial_data:
            # Sample data for 9 SA provinces
            provincial_data = {
                'Gauteng': 0.82,
                'Western Cape': 0.79,
                'KwaZulu-Natal': 0.85,
                'Eastern Cape': 0.76,
                'Limpopo': 0.73,
                'Mpumalanga': 0.71,
                'North West': 0.68,
                'Free State': 0.74,
                'Northern Cape': 0.69
            }
        
        provinces = list(provincial_data.keys())
        appropriateness = [v * 100 for v in provincial_data.values()]
        
        # Create donut chart
        fig = go.Figure(data=[go.Pie(
            labels=provinces, 
            values=appropriateness,
            hole=.4,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_layout(
            title="Regional Appropriateness by Province",
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor=self.sa_colors['background'],
            annotations=[dict(text='Regional<br>Fit', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig

# Initialize the SA cultural visualizer
sa_cultural_viz = SACulturalVisualizer()