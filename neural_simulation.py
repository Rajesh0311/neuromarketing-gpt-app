"""
Neural Simulation and Digital Brain Twin Module
Advanced neural modeling for marketing response simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import json
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import norm, gamma
import warnings
warnings.filterwarnings('ignore')

class DigitalBrainTwin:
    """Digital Brain Twin for marketing response simulation"""
    
    def __init__(self):
        # Brain region definitions
        self.brain_regions = {
            'frontal_cortex': {
                'function': 'executive_control',
                'marketing_role': 'decision_making',
                'base_activation': 0.3,
                'response_latency': 150  # milliseconds
            },
            'limbic_system': {
                'function': 'emotion_processing',
                'marketing_role': 'emotional_response',
                'base_activation': 0.4,
                'response_latency': 80
            },
            'visual_cortex': {
                'function': 'visual_processing',
                'marketing_role': 'visual_attention',
                'base_activation': 0.5,
                'response_latency': 50
            },
            'auditory_cortex': {
                'function': 'auditory_processing',
                'marketing_role': 'audio_attention',
                'base_activation': 0.3,
                'response_latency': 60
            },
            'memory_centers': {
                'function': 'memory_encoding',
                'marketing_role': 'brand_recall',
                'base_activation': 0.35,
                'response_latency': 200
            },
            'reward_system': {
                'function': 'reward_processing',
                'marketing_role': 'purchase_motivation',
                'base_activation': 0.25,
                'response_latency': 120
            },
            'attention_networks': {
                'function': 'attention_control',
                'marketing_role': 'focus_maintenance',
                'base_activation': 0.4,
                'response_latency': 100
            },
            'social_brain': {
                'function': 'social_cognition',
                'marketing_role': 'social_influence',
                'base_activation': 0.3,
                'response_latency': 180
            }
        }
        
        # Neural oscillation frequencies (Hz)
        self.frequency_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # Consumer profiles
        self.consumer_profiles = {
            'analytical': {
                'frontal_cortex': 1.2,
                'limbic_system': 0.8,
                'decision_threshold': 0.7
            },
            'emotional': {
                'frontal_cortex': 0.8,
                'limbic_system': 1.3,
                'decision_threshold': 0.5
            },
            'impulsive': {
                'reward_system': 1.4,
                'frontal_cortex': 0.7,
                'decision_threshold': 0.4
            },
            'conservative': {
                'frontal_cortex': 1.1,
                'reward_system': 0.7,
                'decision_threshold': 0.8
            }
        }
    
    def simulate_marketing_response(self, 
                                  stimulus_text: str,
                                  consumer_type: str = 'balanced',
                                  duration: float = 10.0,
                                  sampling_rate: int = 250) -> Dict:
        """Simulate neural response to marketing stimulus"""
        
        # Analyze stimulus properties
        stimulus_features = self._analyze_stimulus_features(stimulus_text)
        
        # Generate time series
        time_points = np.arange(0, duration, 1/sampling_rate)
        n_samples = len(time_points)
        
        # Initialize results
        simulation_results = {
            'stimulus_text': stimulus_text,
            'consumer_type': consumer_type,
            'duration': duration,
            'sampling_rate': sampling_rate,
            'time_points': time_points,
            'stimulus_features': stimulus_features,
            'neural_activity': {},
            'oscillations': {},
            'connectivity': {},
            'behavioral_outcomes': {},
            'marketing_metrics': {}
        }
        
        # Simulate neural activity for each brain region
        for region, properties in self.brain_regions.items():
            activity = self._simulate_region_activity(
                region, properties, stimulus_features, 
                consumer_type, time_points
            )
            simulation_results['neural_activity'][region] = activity
        
        # Simulate neural oscillations
        simulation_results['oscillations'] = self._simulate_oscillations(
            stimulus_features, time_points, sampling_rate
        )
        
        # Calculate connectivity patterns
        simulation_results['connectivity'] = self._calculate_connectivity(
            simulation_results['neural_activity']
        )
        
        # Predict behavioral outcomes
        simulation_results['behavioral_outcomes'] = self._predict_behavior(
            simulation_results['neural_activity'], 
            stimulus_features,
            consumer_type
        )
        
        # Calculate marketing metrics
        simulation_results['marketing_metrics'] = self._calculate_marketing_metrics(
            simulation_results
        )
        
        return simulation_results
    
    def _analyze_stimulus_features(self, text: str) -> Dict:
        """Analyze marketing stimulus features"""
        features = {}
        
        text_lower = text.lower()
        
        # Emotional intensity
        emotional_words = ['amazing', 'incredible', 'fantastic', 'love', 'hate', 'fear', 'excited']
        emotional_intensity = sum(1 for word in emotional_words if word in text_lower) / len(text.split())
        features['emotional_intensity'] = min(emotional_intensity * 2, 1.0)
        
        # Urgency signals
        urgency_words = ['now', 'today', 'limited', 'hurry', 'urgent', 'sale', 'offer']
        urgency_level = sum(1 for word in urgency_words if word in text_lower) / len(text.split())
        features['urgency'] = min(urgency_level * 3, 1.0)
        
        # Social proof
        social_words = ['everyone', 'customers', 'reviews', 'popular', 'bestseller']
        social_proof = sum(1 for word in social_words if word in text_lower) / len(text.split())
        features['social_proof'] = min(social_proof * 2, 1.0)
        
        # Reward signals
        reward_words = ['free', 'discount', 'save', 'bonus', 'gift', 'reward']
        reward_intensity = sum(1 for word in reward_words if word in text_lower) / len(text.split())
        features['reward_intensity'] = min(reward_intensity * 2, 1.0)
        
        # Cognitive load
        complex_words = [word for word in text.split() if len(word) > 8]
        cognitive_load = len(complex_words) / len(text.split())
        features['cognitive_load'] = min(cognitive_load * 2, 1.0)
        
        # Visual appeal (estimated from text characteristics)
        appeal_words = ['beautiful', 'stunning', 'gorgeous', 'sleek', 'elegant']
        visual_appeal = sum(1 for word in appeal_words if word in text_lower) / len(text.split())
        features['visual_appeal'] = min(visual_appeal * 2, 1.0)
        
        return features
    
    def _simulate_region_activity(self, 
                                region: str, 
                                properties: Dict, 
                                stimulus_features: Dict,
                                consumer_type: str,
                                time_points: np.ndarray) -> np.ndarray:
        """Simulate neural activity for a specific brain region"""
        
        base_activation = properties['base_activation']
        response_latency = properties['response_latency'] / 1000  # Convert to seconds
        
        # Consumer type modulation
        consumer_modulation = 1.0
        if consumer_type in self.consumer_profiles:
            consumer_modulation = self.consumer_profiles[consumer_type].get(region, 1.0)
        
        # Stimulus-specific activation
        stimulus_activation = self._calculate_stimulus_activation(region, stimulus_features)
        
        # Create response curve
        n_samples = len(time_points)
        activity = np.zeros(n_samples)
        
        # Add stimulus response with latency
        latency_samples = int(response_latency * len(time_points) / (time_points[-1] - time_points[0]))
        
        if latency_samples < n_samples:
            # Create response curve (gamma distribution)
            response_duration = min(3.0, time_points[-1] - response_latency)
            response_samples = int(response_duration * len(time_points) / (time_points[-1] - time_points[0]))
            
            if response_samples > 0:
                # Gamma function for realistic neural response
                x = np.linspace(0, 5, response_samples)
                gamma_response = gamma.pdf(x, a=2, scale=1)
                gamma_response = gamma_response / np.max(gamma_response)
                
                # Scale by activation level
                total_activation = (base_activation + stimulus_activation) * consumer_modulation
                gamma_response *= total_activation
                
                # Insert response into activity array
                end_idx = min(latency_samples + response_samples, n_samples)
                actual_samples = end_idx - latency_samples
                activity[latency_samples:end_idx] = gamma_response[:actual_samples]
        
        # Add baseline activity
        baseline = base_activation * 0.3 * consumer_modulation
        activity += baseline
        
        # Add noise
        noise_level = 0.1
        activity += np.random.normal(0, noise_level, n_samples)
        
        # Ensure non-negative
        activity = np.maximum(activity, 0)
        
        return activity
    
    def _calculate_stimulus_activation(self, region: str, features: Dict) -> float:
        """Calculate stimulus-specific activation for a brain region"""
        
        activation_maps = {
            'frontal_cortex': {
                'cognitive_load': 0.8,
                'urgency': 0.6
            },
            'limbic_system': {
                'emotional_intensity': 1.0,
                'reward_intensity': 0.8
            },
            'visual_cortex': {
                'visual_appeal': 1.0,
                'emotional_intensity': 0.4
            },
            'auditory_cortex': {
                'emotional_intensity': 0.6,
                'urgency': 0.5
            },
            'memory_centers': {
                'emotional_intensity': 0.7,
                'social_proof': 0.6
            },
            'reward_system': {
                'reward_intensity': 1.0,
                'emotional_intensity': 0.6
            },
            'attention_networks': {
                'urgency': 0.8,
                'visual_appeal': 0.6
            },
            'social_brain': {
                'social_proof': 1.0,
                'emotional_intensity': 0.5
            }
        }
        
        region_map = activation_maps.get(region, {})
        activation = 0.0
        
        for feature, weight in region_map.items():
            activation += features.get(feature, 0) * weight
        
        return min(activation, 1.0)
    
    def _simulate_oscillations(self, 
                             stimulus_features: Dict, 
                             time_points: np.ndarray,
                             sampling_rate: int) -> Dict:
        """Simulate neural oscillations in different frequency bands"""
        
        oscillations = {}
        n_samples = len(time_points)
        
        for band, (low_freq, high_freq) in self.frequency_bands.items():
            # Generate oscillation based on stimulus features
            base_power = self._get_band_baseline(band)
            stimulus_modulation = self._get_stimulus_band_modulation(band, stimulus_features)
            
            # Generate frequency within band
            center_freq = (low_freq + high_freq) / 2
            
            # Create oscillation with time-varying amplitude
            amplitude_envelope = self._create_amplitude_envelope(
                time_points, stimulus_modulation, base_power
            )
            
            # Generate oscillation
            oscillation = amplitude_envelope * np.sin(2 * np.pi * center_freq * time_points)
            
            # Add phase noise
            phase_noise = np.random.uniform(0, 2*np.pi, n_samples)
            oscillation += 0.1 * amplitude_envelope * np.sin(2 * np.pi * center_freq * time_points + phase_noise)
            
            oscillations[band] = {
                'signal': oscillation,
                'power': amplitude_envelope**2,
                'frequency': center_freq,
                'base_power': base_power,
                'stimulus_modulation': stimulus_modulation
            }
        
        return oscillations
    
    def _get_band_baseline(self, band: str) -> float:
        """Get baseline power for frequency band"""
        baselines = {
            'delta': 0.3,
            'theta': 0.4,
            'alpha': 0.6,
            'beta': 0.5,
            'gamma': 0.2
        }
        return baselines.get(band, 0.4)
    
    def _get_stimulus_band_modulation(self, band: str, features: Dict) -> float:
        """Get stimulus modulation for frequency band"""
        
        modulations = {
            'delta': features.get('cognitive_load', 0) * 0.5,
            'theta': features.get('emotional_intensity', 0) * 0.8,
            'alpha': (1 - features.get('urgency', 0)) * 0.6,  # Alpha decreases with urgency
            'beta': features.get('urgency', 0) * 0.7 + features.get('cognitive_load', 0) * 0.5,
            'gamma': features.get('emotional_intensity', 0) * 0.6 + features.get('reward_intensity', 0) * 0.4
        }
        
        return modulations.get(band, 0.3)
    
    def _create_amplitude_envelope(self, 
                                 time_points: np.ndarray, 
                                 modulation: float, 
                                 baseline: float) -> np.ndarray:
        """Create time-varying amplitude envelope"""
        
        # Start with baseline
        envelope = np.full_like(time_points, baseline)
        
        # Add stimulus response (peaks around 1-2 seconds)
        peak_time = 1.5
        peak_width = 2.0
        
        for i, t in enumerate(time_points):
            if t >= 0.5:  # Response delay
                response = modulation * np.exp(-((t - peak_time) / peak_width) ** 2)
                envelope[i] += response
        
        # Add slight decay over time
        decay_factor = np.exp(-time_points / 10)
        envelope *= decay_factor
        
        return envelope
    
    def _calculate_connectivity(self, neural_activity: Dict) -> Dict:
        """Calculate connectivity between brain regions"""
        
        connectivity = {}
        regions = list(neural_activity.keys())
        
        # Calculate correlation matrix
        activity_matrix = np.array([neural_activity[region] for region in regions])
        correlation_matrix = np.corrcoef(activity_matrix)
        
        # Create connectivity dictionary
        for i, region1 in enumerate(regions):
            connectivity[region1] = {}
            for j, region2 in enumerate(regions):
                if i != j:
                    connectivity[region1][region2] = correlation_matrix[i, j]
        
        # Calculate key connectivity metrics
        connectivity['summary'] = {
            'frontal_limbic': correlation_matrix[
                regions.index('frontal_cortex'), 
                regions.index('limbic_system')
            ] if 'frontal_cortex' in regions and 'limbic_system' in regions else 0,
            
            'attention_visual': correlation_matrix[
                regions.index('attention_networks'), 
                regions.index('visual_cortex')
            ] if 'attention_networks' in regions and 'visual_cortex' in regions else 0,
            
            'reward_memory': correlation_matrix[
                regions.index('reward_system'), 
                regions.index('memory_centers')
            ] if 'reward_system' in regions and 'memory_centers' in regions else 0
        }
        
        return connectivity
    
    def _predict_behavior(self, 
                        neural_activity: Dict, 
                        stimulus_features: Dict,
                        consumer_type: str) -> Dict:
        """Predict behavioral outcomes from neural activity"""
        
        # Calculate key behavioral metrics
        behavioral_outcomes = {}
        
        # Attention level (visual cortex + attention networks)
        attention_score = 0
        if 'visual_cortex' in neural_activity:
            attention_score += np.mean(neural_activity['visual_cortex']) * 0.6
        if 'attention_networks' in neural_activity:
            attention_score += np.mean(neural_activity['attention_networks']) * 0.4
        
        behavioral_outcomes['attention_level'] = min(attention_score, 1.0)
        
        # Emotional engagement (limbic system)
        emotional_engagement = 0
        if 'limbic_system' in neural_activity:
            emotional_engagement = np.mean(neural_activity['limbic_system'])
        
        behavioral_outcomes['emotional_engagement'] = min(emotional_engagement, 1.0)
        
        # Purchase intention (reward system + frontal cortex)
        purchase_intention = 0
        if 'reward_system' in neural_activity:
            purchase_intention += np.mean(neural_activity['reward_system']) * 0.7
        if 'frontal_cortex' in neural_activity:
            purchase_intention += np.mean(neural_activity['frontal_cortex']) * 0.3
        
        behavioral_outcomes['purchase_intention'] = min(purchase_intention, 1.0)
        
        # Memory encoding (memory centers + emotional intensity)
        memory_encoding = 0
        if 'memory_centers' in neural_activity:
            memory_encoding = np.mean(neural_activity['memory_centers'])
            # Boost with emotional intensity
            memory_encoding *= (1 + stimulus_features.get('emotional_intensity', 0) * 0.5)
        
        behavioral_outcomes['memory_encoding'] = min(memory_encoding, 1.0)
        
        # Decision confidence (frontal cortex strength)
        decision_confidence = 0
        if 'frontal_cortex' in neural_activity:
            decision_confidence = np.std(neural_activity['frontal_cortex'])  # Consistent activation = confidence
            decision_confidence = 1 - min(decision_confidence, 1.0)  # Invert std
        
        behavioral_outcomes['decision_confidence'] = decision_confidence
        
        # Social influence susceptibility
        social_susceptibility = 0
        if 'social_brain' in neural_activity:
            social_susceptibility = np.mean(neural_activity['social_brain'])
            social_susceptibility *= stimulus_features.get('social_proof', 0.5)
        
        behavioral_outcomes['social_susceptibility'] = min(social_susceptibility, 1.0)
        
        return behavioral_outcomes
    
    def _calculate_marketing_metrics(self, simulation_results: Dict) -> Dict:
        """Calculate marketing-specific metrics from simulation"""
        
        behavioral = simulation_results['behavioral_outcomes']
        neural = simulation_results['neural_activity']
        stimulus = simulation_results['stimulus_features']
        
        marketing_metrics = {}
        
        # Overall engagement score
        engagement_components = [
            behavioral.get('attention_level', 0) * 0.3,
            behavioral.get('emotional_engagement', 0) * 0.4,
            behavioral.get('memory_encoding', 0) * 0.3
        ]
        marketing_metrics['overall_engagement'] = sum(engagement_components)
        
        # Purchase likelihood
        purchase_components = [
            behavioral.get('purchase_intention', 0) * 0.6,
            behavioral.get('decision_confidence', 0) * 0.2,
            stimulus.get('reward_intensity', 0) * 0.2
        ]
        marketing_metrics['purchase_likelihood'] = sum(purchase_components)
        
        # Brand recall potential
        recall_components = [
            behavioral.get('memory_encoding', 0) * 0.5,
            behavioral.get('emotional_engagement', 0) * 0.3,
            stimulus.get('visual_appeal', 0) * 0.2
        ]
        marketing_metrics['brand_recall_potential'] = sum(recall_components)
        
        # Viral potential (sharing likelihood)
        viral_components = [
            behavioral.get('emotional_engagement', 0) * 0.4,
            behavioral.get('social_susceptibility', 0) * 0.3,
            stimulus.get('social_proof', 0) * 0.3
        ]
        marketing_metrics['viral_potential'] = sum(viral_components)
        
        # Message clarity (inverse of cognitive load with good comprehension)
        comprehension = 1 - stimulus.get('cognitive_load', 0)
        attention = behavioral.get('attention_level', 0)
        marketing_metrics['message_clarity'] = (comprehension + attention) / 2
        
        # Urgency response
        urgency_response = stimulus.get('urgency', 0) * behavioral.get('decision_confidence', 0)
        marketing_metrics['urgency_response'] = urgency_response
        
        # Emotional impact
        marketing_metrics['emotional_impact'] = behavioral.get('emotional_engagement', 0)
        
        # Cognitive ease
        marketing_metrics['cognitive_ease'] = 1 - stimulus.get('cognitive_load', 0)
        
        return marketing_metrics

class NeuralVisualizationEngine:
    """Create advanced visualizations for neural simulation results"""
    
    @staticmethod
    def create_brain_activity_heatmap(simulation_results: Dict) -> go.Figure:
        """Create brain activity heatmap"""
        
        neural_activity = simulation_results['neural_activity']
        time_points = simulation_results['time_points']
        
        # Create matrix for heatmap
        regions = list(neural_activity.keys())
        activity_matrix = np.array([neural_activity[region] for region in regions])
        
        # Downsample for visualization
        downsample_factor = max(1, len(time_points) // 100)
        time_downsampled = time_points[::downsample_factor]
        activity_downsampled = activity_matrix[:, ::downsample_factor]
        
        fig = go.Figure(data=go.Heatmap(
            z=activity_downsampled,
            x=time_downsampled,
            y=regions,
            colorscale='Viridis',
            colorbar=dict(title="Neural Activity")
        ))
        
        fig.update_layout(
            title="Brain Activity Heatmap",
            xaxis_title="Time (seconds)",
            yaxis_title="Brain Regions",
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_oscillation_plot(simulation_results: Dict) -> go.Figure:
        """Create neural oscillation visualization"""
        
        oscillations = simulation_results['oscillations']
        time_points = simulation_results['time_points']
        
        fig = make_subplots(
            rows=len(oscillations), cols=1,
            subplot_titles=[f"{band.upper()} ({osc['frequency']:.1f} Hz)" 
                          for band, osc in oscillations.items()],
            shared_xaxes=True
        )
        
        for i, (band, osc_data) in enumerate(oscillations.items()):
            # Downsample for visualization
            downsample_factor = max(1, len(time_points) // 1000)
            time_ds = time_points[::downsample_factor]
            signal_ds = osc_data['signal'][::downsample_factor]
            
            fig.add_trace(
                go.Scatter(
                    x=time_ds,
                    y=signal_ds,
                    mode='lines',
                    name=band.upper(),
                    line=dict(width=1)
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=800,
            title="Neural Oscillations by Frequency Band",
            showlegend=False
        )
        fig.update_xaxes(title_text="Time (seconds)", row=len(oscillations), col=1)
        
        return fig
    
    @staticmethod
    def create_connectivity_network(simulation_results: Dict) -> go.Figure:
        """Create brain connectivity network visualization"""
        
        connectivity = simulation_results['connectivity']
        regions = list(simulation_results['neural_activity'].keys())
        
        # Create network layout (circular)
        n_regions = len(regions)
        angles = np.linspace(0, 2*np.pi, n_regions, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Create edges for strong connections
        edge_x = []
        edge_y = []
        edge_weights = []
        
        for i, region1 in enumerate(regions):
            for j, region2 in enumerate(regions):
                if i < j and region1 in connectivity and region2 in connectivity[region1]:
                    correlation = connectivity[region1][region2]
                    if abs(correlation) > 0.3:  # Only show strong connections
                        edge_x.extend([x_pos[i], x_pos[j], None])
                        edge_y.extend([y_pos[i], y_pos[j], None])
                        edge_weights.append(abs(correlation))
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='lightgray'),
            hoverinfo='none',
            showlegend=False
        ))
        
        # Add nodes
        node_colors = [np.mean(simulation_results['neural_activity'][region]) for region in regions]
        
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(
                size=30,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Activity Level")
            ),
            text=[region.replace('_', '<br>') for region in regions],
            textposition="middle center",
            textfont=dict(size=10),
            hovertemplate='<b>%{text}</b><br>Activity: %{marker.color:.3f}<extra></extra>',
            showlegend=False
        ))
        
        fig.update_layout(
            title="Brain Connectivity Network",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            width=600
        )
        
        return fig
    
    @staticmethod
    def create_marketing_metrics_dashboard(simulation_results: Dict) -> go.Figure:
        """Create marketing metrics dashboard"""
        
        marketing_metrics = simulation_results['marketing_metrics']
        behavioral_outcomes = simulation_results['behavioral_outcomes']
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Marketing Metrics', 'Behavioral Outcomes', 
                          'Neural Response Profile', 'Time Series Analysis'],
            specs=[[{"type": "bar"}, {"type": "radar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Marketing metrics bar chart
        metrics_names = list(marketing_metrics.keys())
        metrics_values = list(marketing_metrics.values())
        
        fig.add_trace(
            go.Bar(
                x=metrics_names,
                y=metrics_values,
                name='Marketing Metrics',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Behavioral outcomes radar chart
        behavioral_names = list(behavioral_outcomes.keys())
        behavioral_values = list(behavioral_outcomes.values())
        
        fig.add_trace(
            go.Scatterpolar(
                r=behavioral_values,
                theta=behavioral_names,
                fill='toself',
                name='Behavioral Outcomes'
            ),
            row=1, col=2
        )
        
        # Neural response profile
        neural_activity = simulation_results['neural_activity']
        region_names = list(neural_activity.keys())
        mean_activity = [np.mean(activity) for activity in neural_activity.values()]
        
        fig.add_trace(
            go.Bar(
                x=region_names,
                y=mean_activity,
                name='Neural Activity',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Time series of key metrics
        time_points = simulation_results['time_points']
        
        # Calculate time-varying engagement
        if 'limbic_system' in neural_activity and 'attention_networks' in neural_activity:
            engagement_timeseries = (
                neural_activity['limbic_system'] * 0.6 + 
                neural_activity['attention_networks'] * 0.4
            )
            
            # Downsample for visualization
            downsample_factor = max(1, len(time_points) // 200)
            time_ds = time_points[::downsample_factor]
            engagement_ds = engagement_timeseries[::downsample_factor]
            
            fig.add_trace(
                go.Scatter(
                    x=time_ds,
                    y=engagement_ds,
                    mode='lines',
                    name='Engagement Over Time',
                    line=dict(color='red')
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Comprehensive Marketing Analysis Dashboard",
            showlegend=False
        )
        
        return fig

# Streamlit integration functions
def render_neural_simulation_interface():
    """Render neural simulation interface in Streamlit"""
    
    st.markdown("### 🧠 Digital Brain Twin Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        consumer_type = st.selectbox(
            "Consumer Profile:",
            ["balanced", "analytical", "emotional", "impulsive", "conservative"]
        )
        
        simulation_duration = st.slider(
            "Simulation Duration (seconds):",
            min_value=5.0,
            max_value=30.0,
            value=10.0,
            step=1.0
        )
    
    with col2:
        include_oscillations = st.checkbox("Include Neural Oscillations", value=True)
        include_connectivity = st.checkbox("Show Connectivity Analysis", value=True)
        
        visualization_type = st.selectbox(
            "Visualization Focus:",
            ["Comprehensive Dashboard", "Brain Activity Heatmap", 
             "Neural Oscillations", "Connectivity Network"]
        )
    
    return consumer_type, simulation_duration, include_oscillations, include_connectivity, visualization_type

def run_neural_simulation_analysis(stimulus_text: str, 
                                 consumer_type: str, 
                                 duration: float,
                                 include_oscillations: bool,
                                 include_connectivity: bool,
                                 visualization_type: str):
    """Run complete neural simulation analysis"""
    
    # Initialize digital brain twin
    brain_twin = DigitalBrainTwin()
    
    # Run simulation
    with st.spinner("🧠 Running digital brain twin simulation..."):
        simulation_results = brain_twin.simulate_marketing_response(
            stimulus_text=stimulus_text,
            consumer_type=consumer_type,
            duration=duration,
            sampling_rate=250
        )
    
    # Display results
    st.success("✅ Neural simulation completed!")
    
    # Key metrics summary
    st.markdown("#### 📊 Simulation Summary")
    
    marketing_metrics = simulation_results['marketing_metrics']
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Engagement", f"{marketing_metrics.get('overall_engagement', 0):.3f}")
    
    with col2:
        st.metric("Purchase Likelihood", f"{marketing_metrics.get('purchase_likelihood', 0):.3f}")
    
    with col3:
        st.metric("Brand Recall", f"{marketing_metrics.get('brand_recall_potential', 0):.3f}")
    
    with col4:
        st.metric("Viral Potential", f"{marketing_metrics.get('viral_potential', 0):.3f}")
    
    # Visualizations
    viz_engine = NeuralVisualizationEngine()
    
    if visualization_type == "Comprehensive Dashboard":
        fig = viz_engine.create_marketing_metrics_dashboard(simulation_results)
        st.plotly_chart(fig, use_container_width=True)
    
    elif visualization_type == "Brain Activity Heatmap":
        fig = viz_engine.create_brain_activity_heatmap(simulation_results)
        st.plotly_chart(fig, use_container_width=True)
    
    elif visualization_type == "Neural Oscillations" and include_oscillations:
        fig = viz_engine.create_oscillation_plot(simulation_results)
        st.plotly_chart(fig, use_container_width=True)
    
    elif visualization_type == "Connectivity Network" and include_connectivity:
        fig = viz_engine.create_connectivity_network(simulation_results)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results in expandable sections
    with st.expander("🔍 Detailed Neural Analysis"):
        behavioral = simulation_results['behavioral_outcomes']
        
        st.markdown("**Behavioral Predictions:**")
        for metric, value in behavioral.items():
            st.write(f"• {metric.replace('_', ' ').title()}: {value:.3f}")
    
    with st.expander("📈 Marketing Insights"):
        insights = generate_marketing_insights(simulation_results)
        for insight in insights:
            st.write(f"• {insight}")
    
    return simulation_results

def generate_marketing_insights(simulation_results: Dict) -> List[str]:
    """Generate actionable marketing insights from simulation"""
    
    insights = []
    marketing_metrics = simulation_results['marketing_metrics']
    behavioral = simulation_results['behavioral_outcomes']
    stimulus_features = simulation_results['stimulus_features']
    
    # Engagement insights
    engagement = marketing_metrics.get('overall_engagement', 0)
    if engagement > 0.7:
        insights.append("High engagement detected - this content effectively captures attention")
    elif engagement < 0.4:
        insights.append("Low engagement - consider increasing emotional appeal or visual elements")
    
    # Purchase intention insights
    purchase_likelihood = marketing_metrics.get('purchase_likelihood', 0)
    if purchase_likelihood > 0.6:
        insights.append("Strong purchase intent signals - optimal for conversion campaigns")
    elif purchase_likelihood < 0.3:
        insights.append("Weak purchase signals - add stronger call-to-action or incentives")
    
    # Memory insights
    recall = marketing_metrics.get('brand_recall_potential', 0)
    if recall > 0.6:
        insights.append("High brand recall potential - good for brand awareness campaigns")
    elif recall < 0.4:
        insights.append("Low memorability - increase emotional intensity or distinctive elements")
    
    # Emotional insights
    emotion = behavioral.get('emotional_engagement', 0)
    if emotion > 0.7:
        insights.append("Strong emotional response - leverage for viral marketing")
    elif emotion < 0.3:
        insights.append("Weak emotional connection - add storytelling or emotional triggers")
    
    # Cognitive load insights
    cognitive_ease = marketing_metrics.get('cognitive_ease', 0)
    if cognitive_ease < 0.5:
        insights.append("High cognitive load detected - simplify message for better comprehension")
    
    return insights