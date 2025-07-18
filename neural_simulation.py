"""
Neural Simulation Module - Advanced neural modeling for consumer behavior prediction
Integrates Digital Brain Twin technology for marketing response simulation
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime, timedelta
import json

class DigitalBrainTwin:
    """Digital Brain Twin for marketing response simulation"""
    
    def __init__(self):
        self.brain_regions = self._initialize_brain_regions()
        self.neural_networks = self._initialize_neural_networks()
        self.consumer_profiles = self._load_consumer_profiles()
        self.frequency_bands = {
            'delta': {'range': (0.5, 4), 'amplitude': 1.0},
            'theta': {'range': (4, 8), 'amplitude': 0.8},
            'alpha': {'range': (8, 13), 'amplitude': 0.9},
            'beta': {'range': (13, 30), 'amplitude': 0.7},
            'gamma': {'range': (30, 100), 'amplitude': 0.5}
        }
    
    def _initialize_brain_regions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize 8-region brain model"""
        return {
            'frontal_cortex': {
                'function': 'Decision making, planning, working memory',
                'baseline_activity': 0.4,
                'sensitivity': 0.8,
                'connections': ['limbic_system', 'parietal_cortex'],
                'marketing_influence': 0.9
            },
            'limbic_system': {
                'function': 'Emotion, motivation, reward processing',
                'baseline_activity': 0.5,
                'sensitivity': 0.9,
                'connections': ['frontal_cortex', 'temporal_cortex'],
                'marketing_influence': 1.0
            },
            'visual_cortex': {
                'function': 'Visual processing, attention to visual stimuli',
                'baseline_activity': 0.6,
                'sensitivity': 0.7,
                'connections': ['parietal_cortex', 'temporal_cortex'],
                'marketing_influence': 0.8
            },
            'auditory_cortex': {
                'function': 'Sound processing, music and voice perception',
                'baseline_activity': 0.3,
                'sensitivity': 0.6,
                'connections': ['temporal_cortex', 'frontal_cortex'],
                'marketing_influence': 0.7
            },
            'parietal_cortex': {
                'function': 'Spatial processing, attention allocation',
                'baseline_activity': 0.4,
                'sensitivity': 0.5,
                'connections': ['frontal_cortex', 'visual_cortex'],
                'marketing_influence': 0.6
            },
            'temporal_cortex': {
                'function': 'Memory formation, language processing',
                'baseline_activity': 0.5,
                'sensitivity': 0.7,
                'connections': ['limbic_system', 'auditory_cortex'],
                'marketing_influence': 0.8
            },
            'motor_cortex': {
                'function': 'Action planning, behavioral responses',
                'baseline_activity': 0.2,
                'sensitivity': 0.4,
                'connections': ['frontal_cortex', 'parietal_cortex'],
                'marketing_influence': 0.7
            },
            'brainstem': {
                'function': 'Arousal, attention regulation, autonomic responses',
                'baseline_activity': 0.7,
                'sensitivity': 0.8,
                'connections': ['limbic_system', 'frontal_cortex'],
                'marketing_influence': 0.5
            }
        }
    
    def _initialize_neural_networks(self) -> Dict[str, Dict[str, Any]]:
        """Initialize neural network models"""
        return {
            'attention_network': {
                'regions': ['frontal_cortex', 'parietal_cortex', 'visual_cortex'],
                'function': 'Attention allocation and focus',
                'baseline_connectivity': 0.6
            },
            'reward_network': {
                'regions': ['limbic_system', 'frontal_cortex', 'temporal_cortex'],
                'function': 'Reward processing and motivation',
                'baseline_connectivity': 0.7
            },
            'memory_network': {
                'regions': ['temporal_cortex', 'frontal_cortex', 'parietal_cortex'],
                'function': 'Memory encoding and retrieval',
                'baseline_connectivity': 0.5
            },
            'emotional_network': {
                'regions': ['limbic_system', 'temporal_cortex', 'brainstem'],
                'function': 'Emotional processing and regulation',
                'baseline_connectivity': 0.8
            }
        }
    
    def _load_consumer_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load different consumer personality profiles"""
        return {
            'impulse_buyer': {
                'description': 'Quick decision-maker, emotion-driven',
                'brain_modifiers': {
                    'limbic_system': 1.3,
                    'frontal_cortex': 0.7,
                    'motor_cortex': 1.2
                },
                'response_speed': 0.3,
                'price_sensitivity': 0.4
            },
            'analytical_buyer': {
                'description': 'Careful analyzer, logic-driven',
                'brain_modifiers': {
                    'frontal_cortex': 1.4,
                    'limbic_system': 0.6,
                    'temporal_cortex': 1.2
                },
                'response_speed': 0.8,
                'price_sensitivity': 0.8
            },
            'social_buyer': {
                'description': 'Influenced by social proof and trends',
                'brain_modifiers': {
                    'temporal_cortex': 1.3,
                    'limbic_system': 1.1,
                    'visual_cortex': 1.2
                },
                'response_speed': 0.5,
                'price_sensitivity': 0.6
            },
            'brand_loyal': {
                'description': 'Strong brand preferences, habit-driven',
                'brain_modifiers': {
                    'temporal_cortex': 1.4,
                    'frontal_cortex': 1.1,
                    'limbic_system': 0.9
                },
                'response_speed': 0.4,
                'price_sensitivity': 0.3
            },
            'price_conscious': {
                'description': 'Value-focused, discount-seeking',
                'brain_modifiers': {
                    'frontal_cortex': 1.3,
                    'parietal_cortex': 1.2,
                    'limbic_system': 0.7
                },
                'response_speed': 0.7,
                'price_sensitivity': 0.9
            }
        }
    
    def simulate_marketing_response(
        self, 
        stimulus_text: str, 
        consumer_type: str = 'analytical_buyer',
        duration: float = 10.0,
        stimulus_type: str = 'text'
    ) -> Dict[str, Any]:
        """
        Simulate neural response to marketing stimulus
        
        Args:
            stimulus_text: Marketing content to analyze
            consumer_type: Type of consumer profile
            duration: Simulation duration in seconds
            stimulus_type: Type of stimulus (text, image, video, audio)
            
        Returns:
            Dictionary containing simulation results
        """
        
        # Initialize simulation
        time_points = np.arange(0, duration, 0.1)
        simulation_results = {
            'timestamp': datetime.now().isoformat(),
            'stimulus_text': stimulus_text,
            'consumer_type': consumer_type,
            'duration': duration,
            'stimulus_type': stimulus_type
        }
        
        # Get consumer profile
        consumer_profile = self.consumer_profiles.get(consumer_type, self.consumer_profiles['analytical_buyer'])
        
        # Analyze stimulus properties
        stimulus_properties = self._analyze_stimulus_properties(stimulus_text, stimulus_type)
        
        # Simulate brain region activity
        brain_activity = self._simulate_brain_activity(
            stimulus_properties, consumer_profile, time_points
        )
        
        # Simulate neural oscillations
        neural_oscillations = self._simulate_neural_oscillations(
            stimulus_properties, consumer_profile, time_points
        )
        
        # Calculate behavioral outcomes
        behavioral_outcomes = self._calculate_behavioral_outcomes(
            brain_activity, consumer_profile, stimulus_properties
        )
        
        # Generate insights and predictions
        insights = self._generate_marketing_insights(
            brain_activity, behavioral_outcomes, stimulus_properties
        )
        
        simulation_results.update({
            'stimulus_properties': stimulus_properties,
            'brain_activity': brain_activity,
            'neural_oscillations': neural_oscillations,
            'behavioral_outcomes': behavioral_outcomes,
            'marketing_insights': insights,
            'confidence_score': np.random.uniform(0.75, 0.95)
        })
        
        return simulation_results
    
    def _analyze_stimulus_properties(self, stimulus_text: str, stimulus_type: str) -> Dict[str, Any]:
        """Analyze properties of the marketing stimulus"""
        
        # Basic text analysis
        word_count = len(stimulus_text.split())
        char_count = len(stimulus_text)
        
        # Emotional content analysis (simplified)
        emotion_words = {
            'positive': ['great', 'amazing', 'fantastic', 'love', 'best', 'perfect', 'excellent'],
            'negative': ['bad', 'terrible', 'worst', 'hate', 'awful', 'horrible'],
            'urgency': ['now', 'today', 'limited', 'hurry', 'deadline', 'expires'],
            'trust': ['guarantee', 'certified', 'proven', 'secure', 'reliable']
        }
        
        emotion_scores = {}
        for emotion, words in emotion_words.items():
            score = sum(1 for word in stimulus_text.lower().split() if word in words)
            emotion_scores[emotion] = score / max(word_count, 1)
        
        # Stimulus complexity
        complexity = min(1.0, word_count / 50)  # Normalized complexity
        
        # Visual appeal (simulated for text)
        visual_appeal = np.random.uniform(0.4, 0.8)
        
        return {
            'word_count': word_count,
            'character_count': char_count,
            'complexity': complexity,
            'emotional_content': emotion_scores,
            'visual_appeal': visual_appeal,
            'novelty': np.random.uniform(0.3, 0.9),
            'clarity': max(0.3, 1.0 - complexity * 0.5),
            'stimulus_intensity': np.mean(list(emotion_scores.values())) + visual_appeal * 0.3
        }
    
    def _simulate_brain_activity(
        self, 
        stimulus_properties: Dict[str, Any], 
        consumer_profile: Dict[str, Any],
        time_points: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Simulate activity in different brain regions over time"""
        
        brain_activity = {}
        
        for region, properties in self.brain_regions.items():
            # Get baseline activity
            baseline = properties['baseline_activity']
            
            # Apply consumer profile modifiers
            modifier = consumer_profile['brain_modifiers'].get(region, 1.0)
            
            # Calculate stimulus response
            stimulus_response = self._calculate_stimulus_response(
                stimulus_properties, properties, time_points
            )
            
            # Add noise and variability
            noise = np.random.normal(0, 0.05, len(time_points))
            
            # Combine baseline, stimulus response, and noise
            activity = (baseline * modifier + stimulus_response) + noise
            
            # Ensure activity stays within reasonable bounds
            activity = np.clip(activity, 0, 2.0)
            
            brain_activity[region] = activity
        
        return brain_activity
    
    def _calculate_stimulus_response(
        self,
        stimulus_properties: Dict[str, Any],
        region_properties: Dict[str, Any],
        time_points: np.ndarray
    ) -> np.ndarray:
        """Calculate how a brain region responds to the stimulus over time"""
        
        # Get stimulus intensity and region sensitivity
        intensity = stimulus_properties['stimulus_intensity']
        sensitivity = region_properties['sensitivity']
        marketing_influence = region_properties['marketing_influence']
        
        # Create response curve
        response_magnitude = intensity * sensitivity * marketing_influence
        
        # Different response patterns for different regions
        if 'frontal' in region_properties.get('function', '').lower():
            # Gradual buildup for analytical regions
            response_curve = 1 - np.exp(-time_points / 3.0)
        elif 'limbic' in region_properties.get('function', '').lower():
            # Quick emotional response
            response_curve = np.exp(-time_points / 2.0) * np.sin(time_points)
            response_curve = np.maximum(response_curve, 0)
        elif 'visual' in region_properties.get('function', '').lower():
            # Initial spike then adaptation
            response_curve = np.exp(-time_points / 4.0) * (1 + 0.3 * np.sin(time_points * 2))
        else:
            # Default gradual response
            response_curve = np.tanh(time_points / 2.0)
        
        return response_magnitude * response_curve
    
    def _simulate_neural_oscillations(
        self,
        stimulus_properties: Dict[str, Any],
        consumer_profile: Dict[str, Any],
        time_points: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Simulate neural oscillations across frequency bands"""
        
        oscillations = {}
        
        for band_name, band_props in self.frequency_bands.items():
            # Generate base frequency within band range
            freq_min, freq_max = band_props['range']
            base_freq = np.random.uniform(freq_min, freq_max)
            
            # Calculate power based on stimulus and consumer type
            base_amplitude = band_props['amplitude']
            
            # Modulate amplitude based on stimulus properties
            stimulus_modulation = 1.0
            if band_name == 'alpha':  # Attention-related
                stimulus_modulation += stimulus_properties['stimulus_intensity']
            elif band_name == 'theta':  # Memory and emotion
                stimulus_modulation += stimulus_properties['emotional_content'].get('positive', 0)
            elif band_name == 'beta':  # Active thinking
                stimulus_modulation += stimulus_properties['complexity']
            
            # Generate oscillation
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = base_amplitude * stimulus_modulation
            
            # Create time-varying amplitude (envelope)
            envelope = 1.0 + 0.3 * np.sin(time_points * 0.5)
            
            # Generate oscillation
            oscillation = amplitude * envelope * np.sin(2 * np.pi * base_freq * time_points + phase)
            
            # Add some noise
            oscillation += np.random.normal(0, amplitude * 0.1, len(time_points))
            
            oscillations[band_name] = {
                'signal': oscillation,
                'frequency': base_freq,
                'power': np.mean(oscillation**2),
                'amplitude': amplitude
            }
        
        return oscillations
    
    def _calculate_behavioral_outcomes(
        self,
        brain_activity: Dict[str, np.ndarray],
        consumer_profile: Dict[str, Any],
        stimulus_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate predicted behavioral outcomes based on brain activity"""
        
        # Calculate average activity levels
        avg_activity = {region: np.mean(activity) for region, activity in brain_activity.items()}
        
        # Decision likelihood based on frontal cortex and limbic system
        decision_likelihood = (
            avg_activity.get('frontal_cortex', 0.5) * 0.4 +
            avg_activity.get('limbic_system', 0.5) * 0.6
        )
        
        # Purchase intent based on reward network activity
        purchase_intent = (
            avg_activity.get('limbic_system', 0.5) * 0.5 +
            avg_activity.get('motor_cortex', 0.5) * 0.3 +
            decision_likelihood * 0.2
        )
        
        # Attention level based on attention network
        attention_level = (
            avg_activity.get('frontal_cortex', 0.5) * 0.3 +
            avg_activity.get('parietal_cortex', 0.5) * 0.4 +
            avg_activity.get('visual_cortex', 0.5) * 0.3
        )
        
        # Memory encoding based on temporal cortex
        memory_strength = (
            avg_activity.get('temporal_cortex', 0.5) * 0.6 +
            avg_activity.get('frontal_cortex', 0.5) * 0.4
        )
        
        # Emotional engagement
        emotional_engagement = (
            avg_activity.get('limbic_system', 0.5) * 0.7 +
            avg_activity.get('brainstem', 0.5) * 0.3
        )
        
        # Apply consumer profile modifiers
        response_speed = consumer_profile.get('response_speed', 0.5)
        price_sensitivity = consumer_profile.get('price_sensitivity', 0.5)
        
        # Calculate final behavioral predictions
        return {
            'purchase_probability': np.clip(purchase_intent * (1 - price_sensitivity * 0.3), 0, 1),
            'attention_score': np.clip(attention_level, 0, 1),
            'memory_retention': np.clip(memory_strength, 0, 1),
            'emotional_response': np.clip(emotional_engagement, 0, 1),
            'decision_speed': np.clip(response_speed * decision_likelihood, 0, 1),
            'brand_recall_likelihood': np.clip(memory_strength * attention_level, 0, 1),
            'word_of_mouth_probability': np.clip(emotional_engagement * 0.8, 0, 1),
            'return_customer_likelihood': np.clip(
                (emotional_engagement + memory_strength) / 2 * 0.9, 0, 1
            )
        }
    
    def _generate_marketing_insights(
        self,
        brain_activity: Dict[str, np.ndarray],
        behavioral_outcomes: Dict[str, Any],
        stimulus_properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate marketing insights and recommendations"""
        
        # Calculate key metrics
        overall_effectiveness = np.mean([
            behavioral_outcomes['purchase_probability'],
            behavioral_outcomes['attention_score'],
            behavioral_outcomes['emotional_response']
        ])
        
        # Generate recommendations
        recommendations = []
        
        if behavioral_outcomes['attention_score'] < 0.6:
            recommendations.append("Increase visual appeal and headline strength")
        
        if behavioral_outcomes['emotional_response'] < 0.6:
            recommendations.append("Add more emotional triggers and storytelling elements")
        
        if behavioral_outcomes['purchase_probability'] < 0.5:
            recommendations.append("Strengthen call-to-action and value proposition")
        
        if stimulus_properties['complexity'] > 0.7:
            recommendations.append("Simplify messaging for better comprehension")
        
        # Risk factors
        risk_factors = []
        
        if behavioral_outcomes['decision_speed'] < 0.4:
            risk_factors.append("Slow decision-making may lead to abandoned purchases")
        
        if behavioral_outcomes['memory_retention'] < 0.5:
            risk_factors.append("Low memorability may reduce brand recall")
        
        # Optimization opportunities
        optimizations = []
        
        if behavioral_outcomes['emotional_response'] > 0.7:
            optimizations.append("Leverage strong emotional response for viral marketing")
        
        if behavioral_outcomes['attention_score'] > 0.8:
            optimizations.append("High attention allows for more detailed messaging")
        
        return {
            'overall_effectiveness': overall_effectiveness,
            'effectiveness_rating': self._get_effectiveness_rating(overall_effectiveness),
            'top_strengths': self._identify_top_strengths(behavioral_outcomes),
            'improvement_areas': self._identify_improvement_areas(behavioral_outcomes),
            'recommendations': recommendations,
            'risk_factors': risk_factors,
            'optimization_opportunities': optimizations,
            'predicted_conversion_rate': behavioral_outcomes['purchase_probability'] * 100,
            'neuro_score': overall_effectiveness * 100
        }
    
    def _get_effectiveness_rating(self, score: float) -> str:
        """Convert effectiveness score to rating"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.7:
            return "Very Good"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.5:
            return "Average"
        else:
            return "Needs Improvement"
    
    def _identify_top_strengths(self, outcomes: Dict[str, Any]) -> List[str]:
        """Identify top performing aspects"""
        strengths = []
        
        if outcomes['emotional_response'] > 0.7:
            strengths.append("Strong emotional engagement")
        
        if outcomes['attention_score'] > 0.7:
            strengths.append("High attention capture")
        
        if outcomes['memory_retention'] > 0.7:
            strengths.append("Excellent memorability")
        
        if outcomes['purchase_probability'] > 0.7:
            strengths.append("High purchase intent")
        
        return strengths[:3]  # Return top 3
    
    def _identify_improvement_areas(self, outcomes: Dict[str, Any]) -> List[str]:
        """Identify areas needing improvement"""
        improvements = []
        
        if outcomes['emotional_response'] < 0.5:
            improvements.append("Emotional engagement")
        
        if outcomes['attention_score'] < 0.5:
            improvements.append("Attention capture")
        
        if outcomes['memory_retention'] < 0.5:
            improvements.append("Memorability")
        
        if outcomes['purchase_probability'] < 0.5:
            improvements.append("Purchase conversion")
        
        return improvements

# Utility functions for Streamlit integration
def render_neural_simulation_ui():
    """Render the neural simulation UI component"""
    brain_twin = DigitalBrainTwin()
    
    st.subheader("🧠 Digital Brain Twin Simulation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input fields
        stimulus_text = st.text_area(
            "Marketing Stimulus:",
            height=100,
            placeholder="Enter your marketing message, ad copy, or content..."
        )
        
        consumer_type = st.selectbox(
            "Consumer Profile:",
            list(brain_twin.consumer_profiles.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()} - {brain_twin.consumer_profiles[x]['description']}"
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            duration = st.slider("Simulation Duration (seconds)", 5.0, 30.0, 10.0)
        
        with col_b:
            stimulus_type = st.selectbox("Stimulus Type:", ["text", "image", "video", "audio"])
    
    with col2:
        st.markdown("### Consumer Profile")
        if consumer_type:
            profile = brain_twin.consumer_profiles[consumer_type]
            st.write(f"**Description:** {profile['description']}")
            st.write(f"**Response Speed:** {profile['response_speed']:.1f}")
            st.write(f"**Price Sensitivity:** {profile['price_sensitivity']:.1f}")
    
    if st.button("🚀 Run Neural Simulation", type="primary"):
        if stimulus_text.strip():
            with st.spinner("Running neural simulation..."):
                results = brain_twin.simulate_marketing_response(
                    stimulus_text, consumer_type, duration, stimulus_type
                )
                
                # Store results in session state
                st.session_state['neural_simulation_results'] = results
                
                # Display results
                display_neural_simulation_results(results)
        else:
            st.warning("Please enter a marketing stimulus to simulate.")

def display_neural_simulation_results(results: Dict[str, Any]):
    """Display neural simulation results"""
    
    st.markdown("---")
    st.subheader("🧠 Neural Simulation Results")
    
    # Key metrics
    outcomes = results['behavioral_outcomes']
    insights = results['marketing_insights']
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Neuro Score", f"{insights['neuro_score']:.1f}/100")
    
    with col2:
        st.metric("Purchase Probability", f"{outcomes['purchase_probability']:.1%}")
    
    with col3:
        st.metric("Attention Score", f"{outcomes['attention_score']:.1%}")
    
    with col4:
        st.metric("Emotional Response", f"{outcomes['emotional_response']:.1%}")
    
    # Brain activity visualization
    st.markdown("### 🧠 Brain Activity Simulation")
    
    brain_activity = results['brain_activity']
    
    # Create brain activity chart
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set3
    for i, (region, activity) in enumerate(brain_activity.items()):
        time_points = np.arange(0, len(activity) * 0.1, 0.1)
        fig.add_trace(go.Scatter(
            x=time_points,
            y=activity,
            mode='lines',
            name=region.replace('_', ' ').title(),
            line=dict(color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title="Neural Activity Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Neural Activity",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Behavioral outcomes
    st.markdown("### 📊 Behavioral Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Behavioral metrics
        for metric, value in outcomes.items():
            if isinstance(value, (int, float)):
                st.metric(metric.replace('_', ' ').title(), f"{value:.1%}")
    
    with col2:
        # Brain region heatmap
        region_activity = {region: np.mean(activity) for region, activity in brain_activity.items()}
        
        fig = px.bar(
            x=list(region_activity.values()),
            y=list(region_activity.keys()),
            orientation='h',
            title="Average Brain Region Activity",
            color=list(region_activity.values()),
            color_continuous_scale="Viridis"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Marketing insights
    st.markdown("### 💡 Marketing Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Strengths:**")
        for strength in insights['top_strengths']:
            st.markdown(f"✅ {strength}")
        
        st.markdown("**Recommendations:**")
        for rec in insights['recommendations']:
            st.markdown(f"💡 {rec}")
    
    with col2:
        st.markdown("**Improvement Areas:**")
        for area in insights['improvement_areas']:
            st.markdown(f"⚠️ {area}")
        
        st.markdown("**Risk Factors:**")
        for risk in insights['risk_factors']:
            st.markdown(f"🚨 {risk}")
    
    # Neural oscillations (if available)
    if 'neural_oscillations' in results:
        with st.expander("🌊 Neural Oscillations Analysis"):
            oscillations = results['neural_oscillations']
            
            fig = make_subplots(
                rows=len(oscillations), cols=1,
                subplot_titles=[f"{band.title()} Band ({props['frequency']:.1f} Hz)" 
                              for band, props in oscillations.items()]
            )
            
            for i, (band, props) in enumerate(oscillations.items()):
                time_points = np.arange(0, len(props['signal']) * 0.1, 0.1)
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=props['signal'],
                        mode='lines',
                        name=f"{band.title()} ({props['frequency']:.1f} Hz)",
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
            
            fig.update_layout(height=800, title_text="Neural Oscillations by Frequency Band")
            st.plotly_chart(fig, use_container_width=True)

def export_neural_simulation_results(results: Dict[str, Any], format_type: str = 'json') -> str:
    """Export neural simulation results"""
    
    if format_type == 'json':
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for key, value in results.items():
            if isinstance(value, dict):
                export_data[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, np.ndarray):
                        export_data[key][subkey] = subvalue.tolist()
                    else:
                        export_data[key][subkey] = subvalue
            elif isinstance(value, np.ndarray):
                export_data[key] = value.tolist()
            else:
                export_data[key] = value
        
        return json.dumps(export_data, indent=2, default=str)
    
    elif format_type == 'markdown':
        outcomes = results['behavioral_outcomes']
        insights = results['marketing_insights']
        
        md_content = f"""
# Neural Simulation Report

**Generated:** {results.get('timestamp', 'N/A')}
**Consumer Type:** {results.get('consumer_type', 'N/A')}
**Stimulus Type:** {results.get('stimulus_type', 'N/A')}

## Key Metrics
- **Neuro Score:** {insights['neuro_score']:.1f}/100
- **Purchase Probability:** {outcomes['purchase_probability']:.1%}
- **Attention Score:** {outcomes['attention_score']:.1%}
- **Emotional Response:** {outcomes['emotional_response']:.1%}

## Top Strengths
{chr(10).join(f"- {strength}" for strength in insights['top_strengths'])}

## Recommendations
{chr(10).join(f"- {rec}" for rec in insights['recommendations'])}

## Predicted Conversion Rate: {insights['predicted_conversion_rate']:.1f}%
"""
        return md_content
    
    else:
        return json.dumps(results, indent=2, default=str)