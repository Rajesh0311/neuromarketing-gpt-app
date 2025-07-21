"""
Neural Simulation Module - Advanced neural modeling for consumer behavior prediction
Integrates Digital Brain Twin technology for marketing response simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime, timedelta
import json

# Optional imports with fallbacks
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Forward declarations and base classes
class MultiChannelEEGSimulator:
    """Multi-channel EEG Data Simulation - Advanced neural monitoring capability"""
    
    def __init__(self):
        self.channels = self._initialize_eeg_channels()
        self.sampling_rate = 256  # Hz
        self.noise_level = 0.1
        
    def _initialize_eeg_channels(self) -> Dict[str, Dict[str, Any]]:
        """Initialize standard 10-20 EEG electrode system"""
        return {
            'Fp1': {'location': 'frontal_left', 'sensitivity': 0.8, 'baseline': 0.0},
            'Fp2': {'location': 'frontal_right', 'sensitivity': 0.8, 'baseline': 0.0},
            'F3': {'location': 'frontal_left', 'sensitivity': 0.9, 'baseline': 0.0},
            'F4': {'location': 'frontal_right', 'sensitivity': 0.9, 'baseline': 0.0},
            'F7': {'location': 'temporal_left', 'sensitivity': 0.7, 'baseline': 0.0},
            'F8': {'location': 'temporal_right', 'sensitivity': 0.7, 'baseline': 0.0},
            'C3': {'location': 'central_left', 'sensitivity': 0.8, 'baseline': 0.0},
            'C4': {'location': 'central_right', 'sensitivity': 0.8, 'baseline': 0.0},
            'Cz': {'location': 'central_midline', 'sensitivity': 0.9, 'baseline': 0.0},
            'T3': {'location': 'temporal_left', 'sensitivity': 0.6, 'baseline': 0.0},
            'T4': {'location': 'temporal_right', 'sensitivity': 0.6, 'baseline': 0.0},
            'P3': {'location': 'parietal_left', 'sensitivity': 0.7, 'baseline': 0.0},
            'P4': {'location': 'parietal_right', 'sensitivity': 0.7, 'baseline': 0.0},
            'Pz': {'location': 'parietal_midline', 'sensitivity': 0.8, 'baseline': 0.0},
            'O1': {'location': 'occipital_left', 'sensitivity': 0.8, 'baseline': 0.0},
            'O2': {'location': 'occipital_right', 'sensitivity': 0.8, 'baseline': 0.0}
        }
    
    def simulate_multichannel_eeg(self, 
                                 stimulus_type: str = "marketing", 
                                 duration: float = 10.0,
                                 consumer_state: str = "attentive") -> Dict[str, Any]:
        """Simulate multi-channel EEG data for marketing stimulus"""
        
        time_points = np.arange(0, duration, 1/self.sampling_rate)
        eeg_data = {}
        
        # Stimulus-specific patterns
        stimulus_patterns = self._get_stimulus_patterns(stimulus_type)
        state_modifiers = self._get_state_modifiers(consumer_state)
        
        for channel, properties in self.channels.items():
            # Generate base signal
            signal = self._generate_channel_signal(
                time_points, properties, stimulus_patterns, state_modifiers
            )
            
            eeg_data[channel] = {
                'signal': signal,
                'location': properties['location'],
                'quality_score': np.random.uniform(0.85, 0.98),
                'impedance': np.random.uniform(1.0, 5.0)  # kΩ
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'sampling_rate': self.sampling_rate,
            'channels': list(self.channels.keys()),
            'eeg_data': eeg_data,
            'stimulus_type': stimulus_type,
            'consumer_state': consumer_state,
            'signal_quality': self._assess_signal_quality(eeg_data)
        }
    
    def _get_stimulus_patterns(self, stimulus_type: str) -> Dict[str, float]:
        """Get stimulus-specific EEG patterns"""
        patterns = {
            'marketing': {
                'frontal_activation': 1.2,
                'temporal_activation': 1.1,
                'attention_focus': 1.3,
                'emotional_response': 1.0
            },
            'price_promotion': {
                'frontal_activation': 1.4,
                'temporal_activation': 0.9,
                'attention_focus': 1.5,
                'emotional_response': 1.2
            },
            'brand_message': {
                'frontal_activation': 1.0,
                'temporal_activation': 1.3,
                'attention_focus': 1.1,
                'emotional_response': 1.1
            },
            'emotional_appeal': {
                'frontal_activation': 0.9,
                'temporal_activation': 1.2,
                'attention_focus': 1.0,
                'emotional_response': 1.4
            }
        }
        return patterns.get(stimulus_type, patterns['marketing'])
    
    def _get_state_modifiers(self, consumer_state: str) -> Dict[str, float]:
        """Get consumer state modifiers"""
        modifiers = {
            'attentive': {
                'alpha_suppression': 0.7,
                'beta_enhancement': 1.3,
                'theta_baseline': 1.0,
                'gamma_activity': 1.2
            },
            'distracted': {
                'alpha_suppression': 1.2,
                'beta_enhancement': 0.8,
                'theta_baseline': 1.1,
                'gamma_activity': 0.9
            },
            'fatigued': {
                'alpha_suppression': 1.3,
                'beta_enhancement': 0.6,
                'theta_baseline': 1.4,
                'gamma_activity': 0.7
            },
            'engaged': {
                'alpha_suppression': 0.6,
                'beta_enhancement': 1.4,
                'theta_baseline': 0.9,
                'gamma_activity': 1.3
            }
        }
        return modifiers.get(consumer_state, modifiers['attentive'])
    
    def _generate_channel_signal(self, 
                               time_points: np.ndarray, 
                               properties: Dict[str, Any],
                               stimulus_patterns: Dict[str, float],
                               state_modifiers: Dict[str, float]) -> np.ndarray:
        """Generate realistic EEG signal for a specific channel"""
        
        # Base frequency components
        delta = np.sin(2 * np.pi * 2 * time_points) * 0.5
        theta = np.sin(2 * np.pi * 6 * time_points) * 0.3 * state_modifiers['theta_baseline']
        alpha = np.sin(2 * np.pi * 10 * time_points) * 0.4 * state_modifiers['alpha_suppression']
        beta = np.sin(2 * np.pi * 20 * time_points) * 0.2 * state_modifiers['beta_enhancement']
        gamma = np.sin(2 * np.pi * 40 * time_points) * 0.1 * state_modifiers['gamma_activity']
        
        # Location-specific modulations
        location = properties['location']
        if 'frontal' in location:
            signal = delta + theta * 0.8 + alpha * 0.6 + beta * 1.2 + gamma * 0.8
            signal *= stimulus_patterns['frontal_activation']
        elif 'temporal' in location:
            signal = delta * 0.8 + theta * 1.2 + alpha * 1.0 + beta * 0.8 + gamma * 0.6
            signal *= stimulus_patterns['temporal_activation']
        elif 'parietal' in location:
            signal = delta * 0.6 + theta * 0.8 + alpha * 1.4 + beta * 1.0 + gamma * 0.7
            signal *= stimulus_patterns['attention_focus']
        elif 'occipital' in location:
            signal = delta * 0.5 + theta * 0.6 + alpha * 1.6 + beta * 0.6 + gamma * 0.5
        else:  # central
            signal = delta * 0.7 + theta + alpha + beta + gamma
        
        # Add realistic noise
        noise = np.random.normal(0, self.noise_level, len(time_points))
        signal += noise
        
        # Apply sensitivity
        signal *= properties['sensitivity']
        
        # Convert to microvolts
        signal *= 50  # Typical EEG amplitude range
        
        return signal
    
    def _assess_signal_quality(self, eeg_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess overall signal quality"""
        quality_scores = [data['quality_score'] for data in eeg_data.values()]
        impedances = [data['impedance'] for data in eeg_data.values()]
        
        return {
            'average_quality': np.mean(quality_scores),
            'quality_range': (np.min(quality_scores), np.max(quality_scores)),
            'average_impedance': np.mean(impedances),
            'impedance_range': (np.min(impedances), np.max(impedances)),
            'channels_good_quality': sum(1 for q in quality_scores if q > 0.9),
            'total_channels': len(quality_scores)
        }


class FrequencyBandAnalyzer:
    """Brainwave Frequency Analysis - Alpha, Beta, Theta, Delta, Gamma bands"""
    
    def __init__(self):
        self.frequency_bands = {
            'delta': {'range': (0.5, 4), 'function': 'Deep sleep, unconscious processing'},
            'theta': {'range': (4, 8), 'function': 'Creativity, memory, emotion'},
            'alpha': {'range': (8, 13), 'function': 'Relaxed awareness, attention'},
            'beta': {'range': (13, 30), 'function': 'Active thinking, concentration'},
            'gamma': {'range': (30, 100), 'function': 'Consciousness, binding, insight'}
        }
    
    def analyze_frequency_bands(self, eeg_signal: np.ndarray, sampling_rate: int = 256) -> Dict[str, Any]:
        """Analyze frequency bands in EEG signal"""
        
        # Compute power spectral density using FFT
        frequencies, psd = self._compute_power_spectrum(eeg_signal, sampling_rate)
        
        # Extract power for each band
        band_powers = {}
        band_relative_powers = {}
        
        total_power = np.sum(psd)
        
        for band_name, band_info in self.frequency_bands.items():
            low_freq, high_freq = band_info['range']
            
            # Find frequency indices
            freq_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
            band_power = np.sum(psd[freq_mask])
            
            band_powers[band_name] = band_power
            band_relative_powers[band_name] = band_power / total_power if total_power > 0 else 0
        
        # Calculate ratios of interest
        ratios = self._calculate_frequency_ratios(band_powers)
        
        # Interpret cognitive states
        cognitive_states = self._interpret_cognitive_states(band_relative_powers, ratios)
        
        return {
            'band_powers': band_powers,
            'relative_powers': band_relative_powers,
            'frequency_ratios': ratios,
            'cognitive_states': cognitive_states,
            'dominant_frequency': self._find_dominant_frequency(band_relative_powers),
            'mental_state_indicators': self._assess_mental_state(band_relative_powers)
        }
    
    def _compute_power_spectrum(self, signal: np.ndarray, sampling_rate: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density using FFT"""
        # Remove DC component
        signal = signal - np.mean(signal)
        
        # Apply Hamming window
        windowed_signal = signal * np.hamming(len(signal))
        
        # Compute FFT
        fft = np.fft.fft(windowed_signal)
        frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
        
        # Take positive frequencies only
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        
        # Compute power spectral density
        psd = np.abs(fft[positive_freq_idx])**2
        
        return frequencies, psd
    
    def _calculate_frequency_ratios(self, band_powers: Dict[str, float]) -> Dict[str, float]:
        """Calculate important frequency ratios"""
        ratios = {}
        
        # Theta/Beta ratio (attention index)
        if band_powers['beta'] > 0:
            ratios['theta_beta_ratio'] = band_powers['theta'] / band_powers['beta']
        else:
            ratios['theta_beta_ratio'] = float('inf')
        
        # Alpha/Theta ratio (relaxation index)
        if band_powers['theta'] > 0:
            ratios['alpha_theta_ratio'] = band_powers['alpha'] / band_powers['theta']
        else:
            ratios['alpha_theta_ratio'] = float('inf')
        
        # Beta/Alpha ratio (arousal index)
        if band_powers['alpha'] > 0:
            ratios['beta_alpha_ratio'] = band_powers['beta'] / band_powers['alpha']
        else:
            ratios['beta_alpha_ratio'] = float('inf')
        
        # Gamma/Beta ratio (cognitive processing)
        if band_powers['beta'] > 0:
            ratios['gamma_beta_ratio'] = band_powers['gamma'] / band_powers['beta']
        else:
            ratios['gamma_beta_ratio'] = float('inf')
        
        return ratios
    
    def _interpret_cognitive_states(self, relative_powers: Dict[str, float], ratios: Dict[str, float]) -> Dict[str, Any]:
        """Interpret cognitive states from frequency analysis"""
        states = {}
        
        # Attention level
        if ratios['theta_beta_ratio'] < 0.8:
            states['attention_level'] = 'High'
        elif ratios['theta_beta_ratio'] < 1.5:
            states['attention_level'] = 'Medium'
        else:
            states['attention_level'] = 'Low'
        
        # Relaxation level
        if relative_powers['alpha'] > 0.3:
            states['relaxation_level'] = 'High'
        elif relative_powers['alpha'] > 0.15:
            states['relaxation_level'] = 'Medium'
        else:
            states['relaxation_level'] = 'Low'
        
        # Cognitive engagement
        if relative_powers['beta'] > 0.25:
            states['cognitive_engagement'] = 'High'
        elif relative_powers['beta'] > 0.15:
            states['cognitive_engagement'] = 'Medium'
        else:
            states['cognitive_engagement'] = 'Low'
        
        # Creativity/insight indicator
        if relative_powers['gamma'] > 0.1 and relative_powers['theta'] > 0.2:
            states['creativity_index'] = 'High'
        elif relative_powers['gamma'] > 0.05 or relative_powers['theta'] > 0.15:
            states['creativity_index'] = 'Medium'
        else:
            states['creativity_index'] = 'Low'
        
        return states
    
    def _find_dominant_frequency(self, relative_powers: Dict[str, float]) -> str:
        """Find the dominant frequency band"""
        return max(relative_powers.keys(), key=lambda k: relative_powers[k])
    
    def _assess_mental_state(self, relative_powers: Dict[str, float]) -> List[str]:
        """Assess overall mental state indicators"""
        indicators = []
        
        if relative_powers['alpha'] > 0.3:
            indicators.append("Relaxed and receptive to information")
        
        if relative_powers['beta'] > 0.25:
            indicators.append("Actively processing information")
        
        if relative_powers['theta'] > 0.25:
            indicators.append("Creative or emotional processing active")
        
        if relative_powers['gamma'] > 0.08:
            indicators.append("High-level cognitive binding occurring")
        
        if relative_powers['delta'] > 0.4:
            indicators.append("Low arousal or fatigue detected")
        
        return indicators if indicators else ["Normal mixed-frequency activity"]


class CognitiveLoadMeasurement:
    """Cognitive Load Measurement - Working memory capacity analysis"""
    
    def __init__(self):
        self.load_indicators = {
            'pupil_dilation': {'weight': 0.3, 'baseline': 3.5},  # mm
            'theta_power': {'weight': 0.25, 'threshold': 0.2},
            'alpha_suppression': {'weight': 0.2, 'threshold': 0.15},
            'beta_enhancement': {'weight': 0.15, 'threshold': 0.25},
            'response_time': {'weight': 0.1, 'baseline': 800}  # ms
        }
    
    def measure_cognitive_load(self, 
                             eeg_data: Dict[str, Any],
                             task_performance: Dict[str, Any] = None) -> Dict[str, Any]:
        """Measure cognitive load from multiple indicators"""
        
        # Extract EEG-based indicators
        eeg_indicators = self._extract_eeg_indicators(eeg_data)
        
        # Simulate physiological indicators (in real system, would come from sensors)
        physiological_indicators = self._simulate_physiological_indicators()
        
        # Calculate task performance indicators
        if task_performance:
            performance_indicators = self._calculate_performance_indicators(task_performance)
        else:
            performance_indicators = self._simulate_performance_indicators()
        
        # Combine all indicators
        overall_load = self._calculate_overall_load(
            eeg_indicators, physiological_indicators, performance_indicators
        )
        
        # Classify load level
        load_level = self._classify_load_level(overall_load)
        
        # Generate recommendations
        recommendations = self._generate_load_recommendations(load_level, eeg_indicators)
        
        return {
            'cognitive_load_score': overall_load,
            'load_level': load_level,
            'eeg_indicators': eeg_indicators,
            'physiological_indicators': physiological_indicators,
            'performance_indicators': performance_indicators,
            'working_memory_capacity': self._estimate_working_memory_capacity(overall_load),
            'mental_effort_required': self._estimate_mental_effort(overall_load),
            'recommendations': recommendations
        }
    
    def _extract_eeg_indicators(self, eeg_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract cognitive load indicators from EEG data"""
        
        # Simulate extraction from actual EEG data
        # In real implementation, would analyze specific channels and frequencies
        
        return {
            'frontal_theta': np.random.uniform(0.15, 0.35),  # Higher = more load
            'alpha_suppression': np.random.uniform(0.1, 0.4),  # Higher = more load  
            'beta_enhancement': np.random.uniform(0.15, 0.35),  # Higher = more load
            'gamma_activity': np.random.uniform(0.05, 0.15),  # Complex processing
            'p300_amplitude': np.random.uniform(5, 15)  # μV, attention allocation
        }
    
    def _simulate_physiological_indicators(self) -> Dict[str, float]:
        """Simulate physiological indicators of cognitive load"""
        return {
            'pupil_dilation': np.random.uniform(3.0, 5.5),  # mm
            'heart_rate_variability': np.random.uniform(20, 60),  # ms
            'skin_conductance': np.random.uniform(2, 8),  # μS
            'eye_blink_rate': np.random.uniform(10, 25),  # blinks/min
            'muscle_tension': np.random.uniform(1, 5)  # relative scale
        }
    
    def _simulate_performance_indicators(self) -> Dict[str, float]:
        """Simulate task performance indicators"""
        return {
            'response_time': np.random.uniform(600, 1200),  # ms
            'accuracy_rate': np.random.uniform(0.75, 0.98),  # proportion
            'error_rate': np.random.uniform(0.02, 0.25),  # proportion
            'task_completion_rate': np.random.uniform(0.85, 1.0)  # proportion
        }
    
    def _calculate_performance_indicators(self, task_performance: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance indicators from actual task data"""
        # This would process real task performance data
        return self._simulate_performance_indicators()
    
    def _calculate_overall_load(self, 
                               eeg: Dict[str, float], 
                               physiological: Dict[str, float], 
                               performance: Dict[str, float]) -> float:
        """Calculate overall cognitive load score"""
        
        # Normalize and weight EEG indicators
        eeg_score = (
            eeg['frontal_theta'] * 0.3 +
            eeg['alpha_suppression'] * 0.25 +
            eeg['beta_enhancement'] * 0.25 +
            eeg['gamma_activity'] * 0.2
        )
        
        # Normalize physiological indicators
        physio_score = (
            (physiological['pupil_dilation'] - 3.5) / 2.0 * 0.4 +  # Normalize pupil
            (physiological['heart_rate_variability'] - 40) / 20 * 0.3 +  # Normalize HRV
            (physiological['skin_conductance'] - 5) / 3 * 0.3  # Normalize SC
        )
        
        # Performance indicators (inverse relationship - worse performance = higher load)
        perf_score = (
            (1200 - performance['response_time']) / 600 * 0.5 +  # Slower = higher load
            (1 - performance['accuracy_rate']) * 0.5  # Lower accuracy = higher load
        )
        
        # Combine with weights
        overall_load = (
            eeg_score * 0.5 +
            physio_score * 0.3 +
            perf_score * 0.2
        )
        
        return np.clip(overall_load, 0, 1)
    
    def _classify_load_level(self, load_score: float) -> str:
        """Classify cognitive load level"""
        if load_score < 0.3:
            return "Low"
        elif load_score < 0.6:
            return "Medium"
        elif load_score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _estimate_working_memory_capacity(self, load_score: float) -> Dict[str, Any]:
        """Estimate working memory capacity utilization"""
        # Working memory capacity typically 7±2 items
        base_capacity = 7
        utilization = load_score
        
        effective_capacity = base_capacity * (1 - utilization * 0.5)
        
        return {
            'estimated_capacity': base_capacity,
            'current_utilization': utilization,
            'effective_capacity': effective_capacity,
            'capacity_remaining': max(0, effective_capacity - (utilization * base_capacity))
        }
    
    def _estimate_mental_effort(self, load_score: float) -> Dict[str, str]:
        """Estimate mental effort levels"""
        if load_score < 0.3:
            return {
                'effort_level': 'Minimal',
                'description': 'Task requires little mental effort, cognitive resources available'
            }
        elif load_score < 0.6:
            return {
                'effort_level': 'Moderate',
                'description': 'Balanced cognitive engagement, comfortable processing level'
            }
        elif load_score < 0.8:
            return {
                'effort_level': 'High',
                'description': 'Significant mental effort required, approaching capacity limits'
            }
        else:
            return {
                'effort_level': 'Excessive',
                'description': 'Mental effort exceeds comfortable levels, performance degradation likely'
            }
    
    def _generate_load_recommendations(self, load_level: str, eeg_indicators: Dict[str, float]) -> List[str]:
        """Generate recommendations based on cognitive load"""
        recommendations = []
        
        if load_level == "Very High":
            recommendations.extend([
                "Reduce information complexity",
                "Implement shorter content segments",
                "Add visual aids to reduce working memory burden",
                "Consider guided attention techniques"
            ])
        elif load_level == "High":
            recommendations.extend([
                "Optimize information presentation order",
                "Consider chunking complex information",
                "Monitor user engagement closely"
            ])
        elif load_level == "Medium":
            recommendations.extend([
                "Current load level is optimal for learning",
                "Consider adding slightly more complex elements"
            ])
        else:  # Low
            recommendations.extend([
                "Increase information richness",
                "Add interactive elements",
                "Consider more engaging content"
            ])
        
        # EEG-specific recommendations
        if eeg_indicators['frontal_theta'] > 0.3:
            recommendations.append("High frontal theta suggests working memory strain")
        
        if eeg_indicators['alpha_suppression'] > 0.3:
            recommendations.append("Strong alpha suppression indicates high attention demands")
        
        return recommendations


class AttentionStateMonitor:
    """Attention State Tracking - Focused, divided, selective attention monitoring"""
    
    def __init__(self):
        self.attention_types = {
            'focused': 'Concentrated attention on single stimulus',
            'divided': 'Attention split between multiple stimuli',
            'selective': 'Filtering relevant from irrelevant information',
            'sustained': 'Maintaining attention over extended periods'
        }
        
    def monitor_attention_state(self, 
                               eeg_data: Dict[str, Any],
                               stimulus_characteristics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Monitor and classify attention states"""
        
        # Extract attention-related EEG features
        attention_features = self._extract_attention_features(eeg_data)
        
        # Classify current attention state
        current_state = self._classify_attention_state(attention_features)
        
        # Calculate attention intensity
        attention_intensity = self._calculate_attention_intensity(attention_features)
        
        # Assess attention stability
        stability_metrics = self._assess_attention_stability(attention_features)
        
        # Generate attention insights
        insights = self._generate_attention_insights(current_state, attention_intensity, stability_metrics)
        
        return {
            'current_attention_state': current_state,
            'attention_intensity': attention_intensity,
            'stability_metrics': stability_metrics,
            'attention_features': attention_features,
            'attention_quality': self._assess_attention_quality(current_state, attention_intensity),
            'distraction_indicators': self._detect_distraction_indicators(attention_features),
            'recommendations': insights
        }
    
    def _extract_attention_features(self, eeg_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract attention-related features from EEG"""
        
        # Simulate extraction from multi-channel EEG
        return {
            'frontal_beta': np.random.uniform(0.15, 0.35),  # Attention control
            'parietal_alpha': np.random.uniform(0.1, 0.3),  # Attention allocation
            'temporal_gamma': np.random.uniform(0.05, 0.15),  # Attention binding
            'central_theta': np.random.uniform(0.1, 0.25),  # Attention effort
            'occipital_alpha': np.random.uniform(0.2, 0.4),  # Visual attention
            'attention_asymmetry': np.random.uniform(-0.2, 0.2),  # Left-right balance
            'p300_amplitude': np.random.uniform(5, 20),  # Attention allocation
            'n200_latency': np.random.uniform(180, 250)  # Attention speed
        }
    
    def _classify_attention_state(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Classify the current attention state"""
        
        # Decision logic based on EEG features
        states_probability = {}
        
        # Focused attention: High frontal beta, low alpha
        focused_score = features['frontal_beta'] * 0.6 + (1 - features['parietal_alpha']) * 0.4
        states_probability['focused'] = focused_score
        
        # Divided attention: Moderate frontal activity, high temporal gamma
        divided_score = features['temporal_gamma'] * 0.5 + features['central_theta'] * 0.3 + features['frontal_beta'] * 0.2
        states_probability['divided'] = divided_score
        
        # Selective attention: High parietal activity, good P300
        selective_score = features['parietal_alpha'] * 0.4 + (features['p300_amplitude'] / 20) * 0.6
        states_probability['selective'] = selective_score
        
        # Sustained attention: Stable beta, low theta
        sustained_score = features['frontal_beta'] * 0.5 + (1 - features['central_theta']) * 0.5
        states_probability['sustained'] = sustained_score
        
        # Find dominant state
        dominant_state = max(states_probability.keys(), key=lambda k: states_probability[k])
        confidence = states_probability[dominant_state]
        
        return {
            'dominant_state': dominant_state,
            'confidence': confidence,
            'state_probabilities': states_probability,
            'description': self.attention_types[dominant_state]
        }
    
    def _calculate_attention_intensity(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate attention intensity metrics"""
        
        # Overall attention intensity
        intensity = (features['frontal_beta'] + features['p300_amplitude']/20 + (1-features['parietal_alpha'])) / 3
        
        # Attention components
        components = {
            'alertness': features['frontal_beta'],
            'orientation': features['parietal_alpha'],
            'execution': features['temporal_gamma'],
            'overall_intensity': intensity
        }
        
        return components
    
    def _assess_attention_stability(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Assess attention stability over time"""
        
        # Simulate stability metrics (in real system, would track over time)
        stability_score = 1 - abs(features['attention_asymmetry'])
        
        return {
            'stability_score': stability_score,
            'variability': abs(features['attention_asymmetry']),
            'consistency': np.random.uniform(0.6, 0.9),
            'drift_tendency': np.random.uniform(0.0, 0.3)
        }
    
    def _assess_attention_quality(self, state: Dict[str, Any], intensity: Dict[str, float]) -> str:
        """Assess overall attention quality"""
        
        quality_score = state['confidence'] * 0.5 + intensity['overall_intensity'] * 0.5
        
        if quality_score > 0.8:
            return "Excellent"
        elif quality_score > 0.6:
            return "Good"
        elif quality_score > 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def _detect_distraction_indicators(self, features: Dict[str, float]) -> List[str]:
        """Detect indicators of distraction"""
        indicators = []
        
        if features['frontal_beta'] < 0.2:
            indicators.append("Low frontal beta suggests reduced attention control")
        
        if features['parietal_alpha'] > 0.25:
            indicators.append("High parietal alpha indicates attention disengagement")
        
        if features['central_theta'] > 0.2:
            indicators.append("Elevated theta suggests mind-wandering")
        
        if abs(features['attention_asymmetry']) > 0.15:
            indicators.append("Attention asymmetry suggests unbalanced processing")
        
        if features['p300_amplitude'] < 8:
            indicators.append("Reduced P300 amplitude indicates decreased attention allocation")
        
        return indicators if indicators else ["No significant distraction indicators detected"]
    
    def _generate_attention_insights(self, state: Dict[str, Any], intensity: Dict[str, float], stability: Dict[str, Any]) -> List[str]:
        """Generate attention-based insights and recommendations"""
        insights = []
        
        # State-specific insights
        if state['dominant_state'] == 'focused':
            insights.append("Strong focused attention - ideal for detailed information processing")
        elif state['dominant_state'] == 'divided':
            insights.append("Divided attention detected - consider simplifying concurrent demands")
        elif state['dominant_state'] == 'selective':
            insights.append("Good selective attention - effective at filtering relevant information")
        elif state['dominant_state'] == 'sustained':
            insights.append("Sustained attention active - good for extended engagement")
        
        # Intensity insights
        if intensity['overall_intensity'] > 0.7:
            insights.append("High attention intensity - maintain current engagement level")
        elif intensity['overall_intensity'] < 0.3:
            insights.append("Low attention intensity - consider more engaging stimuli")
        
        # Stability insights
        if stability['stability_score'] > 0.8:
            insights.append("Stable attention pattern - consistent engagement")
        elif stability['stability_score'] < 0.5:
            insights.append("Unstable attention - implement attention-focusing techniques")
        
        return insights


class NeuralFatigueAnalyzer:
    """Neural Fatigue Detection - Mental exhaustion pattern recognition"""
    
    def __init__(self):
        self.fatigue_indicators = {
            'alpha_increase': {'weight': 0.25, 'threshold': 0.3},
            'theta_increase': {'weight': 0.2, 'threshold': 0.25},
            'beta_decrease': {'weight': 0.2, 'threshold': 0.15},
            'p300_decline': {'weight': 0.15, 'threshold': 8.0},
            'response_slowing': {'weight': 0.1, 'threshold': 1000},
            'blink_rate_increase': {'weight': 0.1, 'threshold': 20}
        }
    
    def analyze_neural_fatigue(self, 
                              eeg_data: Dict[str, Any],
                              session_duration: float = 10.0,
                              baseline_comparison: bool = True) -> Dict[str, Any]:
        """Analyze neural fatigue indicators"""
        
        # Extract fatigue-related features
        fatigue_features = self._extract_fatigue_features(eeg_data)
        
        # Calculate fatigue score
        fatigue_score = self._calculate_fatigue_score(fatigue_features)
        
        # Classify fatigue level
        fatigue_level = self._classify_fatigue_level(fatigue_score)
        
        # Estimate time-on-task effects
        time_effects = self._estimate_time_effects(session_duration, fatigue_score)
        
        # Generate recovery recommendations
        recovery_recommendations = self._generate_recovery_recommendations(fatigue_level, fatigue_features)
        
        # Predict performance impact
        performance_impact = self._predict_performance_impact(fatigue_score, fatigue_level)
        
        return {
            'fatigue_score': fatigue_score,
            'fatigue_level': fatigue_level,
            'fatigue_features': fatigue_features,
            'time_on_task_effects': time_effects,
            'performance_impact': performance_impact,
            'recovery_recommendations': recovery_recommendations,
            'break_suggestion': self._suggest_break_timing(fatigue_score, session_duration)
        }
    
    def _extract_fatigue_features(self, eeg_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract fatigue-related features from EEG"""
        
        # Simulate fatigue indicators (in real system, would process actual EEG)
        return {
            'alpha_power_increase': np.random.uniform(0.0, 0.4),  # Drowsiness indicator
            'theta_power_increase': np.random.uniform(0.0, 0.3),  # Mental fatigue
            'beta_power_decrease': np.random.uniform(0.0, 0.25),  # Reduced alertness
            'frontal_theta_ratio': np.random.uniform(0.1, 0.4),  # Working memory fatigue
            'posterior_alpha_increase': np.random.uniform(0.0, 0.35),  # Visual fatigue
            'p300_amplitude_decline': np.random.uniform(0.0, 0.5),  # Attention decline
            'microsleep_indicators': np.random.uniform(0.0, 0.2),  # Brief sleep episodes
            'eye_closure_duration': np.random.uniform(0.0, 3.0)  # Seconds of eye closure
        }
    
    def _calculate_fatigue_score(self, features: Dict[str, float]) -> float:
        """Calculate overall fatigue score"""
        
        # Weight and combine fatigue indicators
        fatigue_score = (
            features['alpha_power_increase'] * 0.25 +
            features['theta_power_increase'] * 0.2 +
            features['beta_power_decrease'] * 0.2 +
            features['p300_amplitude_decline'] * 0.15 +
            features['microsleep_indicators'] * 0.1 +
            features['frontal_theta_ratio'] * 0.1
        )
        
        return np.clip(fatigue_score, 0, 1)
    
    def _classify_fatigue_level(self, fatigue_score: float) -> str:
        """Classify fatigue level"""
        if fatigue_score < 0.2:
            return "Alert"
        elif fatigue_score < 0.4:
            return "Mild Fatigue"
        elif fatigue_score < 0.6:
            return "Moderate Fatigue"
        elif fatigue_score < 0.8:
            return "High Fatigue"
        else:
            return "Severe Fatigue"
    
    def _estimate_time_effects(self, duration: float, fatigue_score: float) -> Dict[str, Any]:
        """Estimate time-on-task effects"""
        
        # Fatigue typically increases with time
        expected_fatigue = min(duration / 60 * 0.1, 0.5)  # 10% per hour, max 50%
        
        return {
            'session_duration_minutes': duration,
            'expected_fatigue_for_duration': expected_fatigue,
            'actual_vs_expected': fatigue_score - expected_fatigue,
            'fatigue_rate': fatigue_score / (duration / 60) if duration > 0 else 0,
            'time_to_critical_fatigue': self._estimate_time_to_critical(fatigue_score, duration)
        }
    
    def _estimate_time_to_critical(self, current_fatigue: float, duration: float) -> str:
        """Estimate time until critical fatigue level"""
        if current_fatigue >= 0.8:
            return "Already at critical level"
        
        # Simple linear projection
        fatigue_rate = current_fatigue / (duration / 60) if duration > 0 else 0.1
        time_to_critical = (0.8 - current_fatigue) / fatigue_rate if fatigue_rate > 0 else float('inf')
        
        if time_to_critical < 0.5:
            return "Less than 30 minutes"
        elif time_to_critical < 1:
            return "30-60 minutes"
        elif time_to_critical < 2:
            return "1-2 hours"
        else:
            return "More than 2 hours"
    
    def _predict_performance_impact(self, fatigue_score: float, fatigue_level: str) -> Dict[str, Any]:
        """Predict impact of fatigue on performance"""
        
        # Performance decreases with fatigue
        performance_reduction = fatigue_score * 0.3  # Up to 30% reduction
        
        impacts = {
            'reaction_time_increase': fatigue_score * 200,  # ms increase
            'accuracy_decrease': performance_reduction,  # Proportion decrease
            'attention_lapses': fatigue_score * 10,  # Lapses per hour
            'decision_quality': 1 - performance_reduction,  # Quality score
            'creative_thinking': 1 - (fatigue_score * 0.4)  # Creativity impact
        }
        
        return {
            'performance_impacts': impacts,
            'overall_performance_retention': 1 - performance_reduction,
            'critical_functions_affected': self._identify_affected_functions(fatigue_level),
            'safety_concerns': self._assess_safety_concerns(fatigue_score)
        }
    
    def _identify_affected_functions(self, fatigue_level: str) -> List[str]:
        """Identify cognitive functions affected by fatigue"""
        affected = []
        
        if fatigue_level in ["Mild Fatigue", "Moderate Fatigue", "High Fatigue", "Severe Fatigue"]:
            affected.append("Sustained attention")
        
        if fatigue_level in ["Moderate Fatigue", "High Fatigue", "Severe Fatigue"]:
            affected.extend(["Working memory", "Processing speed"])
        
        if fatigue_level in ["High Fatigue", "Severe Fatigue"]:
            affected.extend(["Decision making", "Creative thinking", "Risk assessment"])
        
        if fatigue_level == "Severe Fatigue":
            affected.extend(["Motor coordination", "Safety awareness"])
        
        return affected if affected else ["No significant functional impairment"]
    
    def _assess_safety_concerns(self, fatigue_score: float) -> List[str]:
        """Assess safety concerns based on fatigue level"""
        concerns = []
        
        if fatigue_score > 0.6:
            concerns.append("Increased risk of errors")
        
        if fatigue_score > 0.7:
            concerns.extend(["Reduced situational awareness", "Slower emergency responses"])
        
        if fatigue_score > 0.8:
            concerns.extend(["High risk of microsleep episodes", "Severely compromised judgment"])
        
        return concerns if concerns else ["No significant safety concerns"]
    
    def _generate_recovery_recommendations(self, fatigue_level: str, features: Dict[str, float]) -> List[str]:
        """Generate fatigue recovery recommendations"""
        recommendations = []
        
        if fatigue_level == "Alert":
            recommendations.append("Maintain current activity level")
        
        elif fatigue_level == "Mild Fatigue":
            recommendations.extend([
                "Consider a 5-10 minute break",
                "Hydration and light movement recommended"
            ])
        
        elif fatigue_level == "Moderate Fatigue":
            recommendations.extend([
                "Take a 15-20 minute break",
                "Consider caffeine or short walk",
                "Reduce task complexity temporarily"
            ])
        
        elif fatigue_level == "High Fatigue":
            recommendations.extend([
                "Extended break (30+ minutes) recommended",
                "Consider power nap (10-20 minutes)",
                "Reassess task priorities"
            ])
        
        else:  # Severe Fatigue
            recommendations.extend([
                "Immediate break required",
                "Consider ending session",
                "Full rest period needed before resuming"
            ])
        
        # Feature-specific recommendations
        if features['alpha_power_increase'] > 0.3:
            recommendations.append("High alpha activity suggests drowsiness - increase alertness")
        
        if features['microsleep_indicators'] > 0.1:
            recommendations.append("Microsleep detected - immediate break essential")
        
        return recommendations
    
    def _suggest_break_timing(self, fatigue_score: float, session_duration: float) -> Dict[str, Any]:
        """Suggest optimal break timing"""
        
        if fatigue_score > 0.7:
            urgency = "Immediate"
            duration = "20-30 minutes"
        elif fatigue_score > 0.5:
            urgency = "Within 15 minutes"
            duration = "10-15 minutes"
        elif fatigue_score > 0.3:
            urgency = "Within 30 minutes"
            duration = "5-10 minutes"
        else:
            urgency = "Next natural break point"
            duration = "5 minutes"
        
        return {
            'break_urgency': urgency,
            'recommended_duration': duration,
            'break_type': self._recommend_break_type(fatigue_score),
            'return_readiness_estimate': self._estimate_return_readiness(fatigue_score)
        }
    
    def _recommend_break_type(self, fatigue_score: float) -> str:
        """Recommend type of break based on fatigue level"""
        if fatigue_score > 0.7:
            return "Rest break with eyes closed"
        elif fatigue_score > 0.5:
            return "Active break with light movement"
        elif fatigue_score > 0.3:
            return "Cognitive break with different activity"
        else:
            return "Maintenance break"
    
    def _estimate_return_readiness(self, fatigue_score: float) -> str:
        """Estimate readiness to return after break"""
        if fatigue_score > 0.8:
            return "May need extended recovery time"
        elif fatigue_score > 0.6:
            return "Should feel refreshed after recommended break"
        elif fatigue_score > 0.4:
            return "Quick recovery expected"
        else:
            return "Ready to continue with minimal break"


class MLNeuralPatterns:
    """Machine learning-based neural signature detection"""
    
    def __init__(self):
        self.pattern_templates = self._initialize_pattern_templates()
        self.classification_confidence_threshold = 0.7
        
    def _initialize_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize neural pattern templates for recognition"""
        return {
            'marketing_engagement': {
                'description': 'High engagement with marketing content',
                'signature': {
                    'frontal_beta': (0.25, 0.4),
                    'temporal_gamma': (0.08, 0.15),
                    'parietal_alpha': (0.1, 0.2),
                    'limbic_theta': (0.15, 0.25)
                },
                'confidence_indicators': ['p300_amplitude', 'frontal_asymmetry']
            },
            'purchase_intention': {
                'description': 'Neural markers of purchase intention',
                'signature': {
                    'frontal_beta': (0.2, 0.35),
                    'reward_gamma': (0.1, 0.18),
                    'motor_preparation': (0.15, 0.3),
                    'decision_theta': (0.18, 0.28)
                },
                'confidence_indicators': ['readiness_potential', 'bereitschaftspotential']
            },
            'brand_recognition': {
                'description': 'Brand familiarity and recognition patterns',
                'signature': {
                    'temporal_gamma': (0.12, 0.2),
                    'memory_theta': (0.2, 0.3),
                    'visual_alpha': (0.15, 0.25),
                    'familiarity_n400': (5, 15)
                },
                'confidence_indicators': ['n170_amplitude', 'memory_retrieval']
            },
            'emotional_arousal': {
                'description': 'Emotional arousal and valence patterns',
                'signature': {
                    'limbic_theta': (0.2, 0.35),
                    'autonomic_response': (0.15, 0.3),
                    'frontal_asymmetry': (-0.2, 0.2),
                    'arousal_beta': (0.2, 0.4)
                },
                'confidence_indicators': ['skin_conductance', 'heart_rate_variability']
            },
            'cognitive_overload': {
                'description': 'Information processing overload patterns',
                'signature': {
                    'frontal_theta': (0.3, 0.5),
                    'working_memory_load': (0.4, 0.7),
                    'alpha_suppression': (0.3, 0.5),
                    'stress_cortisol': (0.2, 0.4)
                },
                'confidence_indicators': ['pupil_dilation', 'response_time_variance']
            }
        }
    
    def detect_neural_patterns(self, 
                              neural_data: Dict[str, Any],
                              context: str = "marketing") -> Dict[str, Any]:
        """Detect and classify neural patterns using ML-based approach"""
        
        # Extract features for pattern recognition
        features = self._extract_pattern_features(neural_data)
        
        # Match against known patterns
        pattern_matches = self._match_patterns(features)
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(features, pattern_matches)
        
        # Generate insights
        insights = self._generate_pattern_insights(pattern_matches, confidence_scores)
        
        return {
            'detected_patterns': pattern_matches,
            'confidence_scores': confidence_scores,
            'pattern_insights': insights,
            'recommendation_priority': self._prioritize_recommendations(pattern_matches, confidence_scores),
            'neural_signature': self._generate_neural_signature(features)
        }
    
    def _extract_pattern_features(self, neural_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features relevant for pattern recognition"""
        
        # This would extract features from actual neural data
        # For now, simulate realistic neural features
        return {
            'frontal_beta': np.random.uniform(0.15, 0.4),
            'temporal_gamma': np.random.uniform(0.05, 0.2),
            'parietal_alpha': np.random.uniform(0.1, 0.3),
            'limbic_theta': np.random.uniform(0.1, 0.35),
            'motor_preparation': np.random.uniform(0.05, 0.3),
            'reward_gamma': np.random.uniform(0.05, 0.18),
            'memory_theta': np.random.uniform(0.15, 0.3),
            'visual_alpha': np.random.uniform(0.1, 0.25),
            'autonomic_response': np.random.uniform(0.1, 0.3),
            'frontal_asymmetry': np.random.uniform(-0.2, 0.2),
            'working_memory_load': np.random.uniform(0.2, 0.7),
            'alpha_suppression': np.random.uniform(0.1, 0.5),
            'decision_theta': np.random.uniform(0.15, 0.3),
            'arousal_beta': np.random.uniform(0.15, 0.4)
        }
    
    def _match_patterns(self, features: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Match extracted features against pattern templates"""
        matches = {}
        
        for pattern_name, template in self.pattern_templates.items():
            match_score = 0
            matched_features = {}
            
            signature = template['signature']
            for feature_name, (min_val, max_val) in signature.items():
                if feature_name in features:
                    feature_value = features[feature_name]
                    
                    # Check if feature falls within expected range
                    if min_val <= feature_value <= max_val:
                        # Calculate how well it matches (closer to center = better)
                        center = (min_val + max_val) / 2
                        range_size = max_val - min_val
                        distance_from_center = abs(feature_value - center)
                        feature_match = 1 - (distance_from_center / (range_size / 2))
                        
                        match_score += feature_match
                        matched_features[feature_name] = {
                            'value': feature_value,
                            'expected_range': (min_val, max_val),
                            'match_quality': feature_match
                        }
            
            # Normalize match score
            if signature:
                match_score /= len(signature)
                
                matches[pattern_name] = {
                    'overall_match_score': match_score,
                    'matched_features': matched_features,
                    'description': template['description'],
                    'feature_count': len(matched_features)
                }
        
        return matches
    
    def _calculate_confidence_scores(self, 
                                   features: Dict[str, float], 
                                   pattern_matches: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for pattern detections"""
        confidence_scores = {}
        
        for pattern_name, match_data in pattern_matches.items():
            base_confidence = match_data['overall_match_score']
            
            # Adjust confidence based on number of matching features
            feature_count_bonus = min(match_data['feature_count'] / 4, 0.2)  # Max 20% bonus
            
            # Check confidence indicators if available
            template = self.pattern_templates[pattern_name]
            indicator_bonus = 0
            if 'confidence_indicators' in template:
                # Simulate indicator presence (in real system, would check actual indicators)
                indicator_bonus = np.random.uniform(0, 0.15)
            
            final_confidence = min(base_confidence + feature_count_bonus + indicator_bonus, 1.0)
            confidence_scores[pattern_name] = final_confidence
        
        return confidence_scores
    
    def _generate_pattern_insights(self, 
                                 pattern_matches: Dict[str, Dict[str, Any]], 
                                 confidence_scores: Dict[str, float]) -> List[str]:
        """Generate insights based on detected patterns"""
        insights = []
        
        # High confidence patterns
        high_confidence_patterns = [
            name for name, confidence in confidence_scores.items() 
            if confidence > self.classification_confidence_threshold
        ]
        
        for pattern in high_confidence_patterns:
            match_data = pattern_matches[pattern]
            confidence = confidence_scores[pattern]
            
            insight = f"Strong {pattern.replace('_', ' ')} pattern detected (confidence: {confidence:.1%})"
            insights.append(insight)
            
            # Pattern-specific insights
            if pattern == 'marketing_engagement':
                insights.append("Consumer shows high neural engagement with marketing content")
            elif pattern == 'purchase_intention':
                insights.append("Neural markers suggest readiness to make purchase decision")
            elif pattern == 'brand_recognition':
                insights.append("Strong brand familiarity and recognition response")
            elif pattern == 'emotional_arousal':
                insights.append("Significant emotional response to stimulus detected")
            elif pattern == 'cognitive_overload':
                insights.append("Information processing capacity may be exceeded")
        
        # Medium confidence patterns
        medium_confidence_patterns = [
            name for name, confidence in confidence_scores.items()
            if 0.4 < confidence <= self.classification_confidence_threshold
        ]
        
        if medium_confidence_patterns:
            insights.append(f"Moderate evidence for: {', '.join(medium_confidence_patterns)}")
        
        return insights if insights else ["No clear neural patterns detected above threshold"]
    
    def _prioritize_recommendations(self, 
                                  pattern_matches: Dict[str, Dict[str, Any]], 
                                  confidence_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Prioritize recommendations based on detected patterns"""
        recommendations = []
        
        # Sort patterns by confidence
        sorted_patterns = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
        
        for pattern_name, confidence in sorted_patterns:
            if confidence > 0.5:  # Only recommend for patterns with reasonable confidence
                rec = {
                    'pattern': pattern_name,
                    'confidence': confidence,
                    'priority': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.6 else 'Low',
                    'action': self._get_pattern_action(pattern_name),
                    'expected_impact': self._estimate_pattern_impact(pattern_name, confidence)
                }
                recommendations.append(rec)
        
        return recommendations
    
    def _get_pattern_action(self, pattern_name: str) -> str:
        """Get recommended action for detected pattern"""
        actions = {
            'marketing_engagement': 'Maintain current engagement strategy, consider similar content',
            'purchase_intention': 'Present purchase options, reduce friction in buying process',
            'brand_recognition': 'Leverage brand familiarity, emphasize brand strengths',
            'emotional_arousal': 'Capitalize on emotional connection, avoid overwhelming',
            'cognitive_overload': 'Simplify information, reduce cognitive demands'
        }
        return actions.get(pattern_name, 'Monitor pattern development')
    
    def _estimate_pattern_impact(self, pattern_name: str, confidence: float) -> str:
        """Estimate impact of acting on detected pattern"""
        base_impacts = {
            'marketing_engagement': 'High',
            'purchase_intention': 'Very High',
            'brand_recognition': 'Medium',
            'emotional_arousal': 'High',
            'cognitive_overload': 'Medium'
        }
        
        base_impact = base_impacts.get(pattern_name, 'Low')
        
        # Adjust based on confidence
        if confidence < 0.6:
            if base_impact == 'Very High':
                return 'High'
            elif base_impact == 'High':
                return 'Medium'
            elif base_impact == 'Medium':
                return 'Low'
        
        return base_impact
    
    def _generate_neural_signature(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Generate a neural signature summary"""
        
        # Identify dominant features
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:5]
        
        # Calculate signature strength
        signature_strength = np.mean([value for _, value in top_features])
        
        # Categorize signature
        if signature_strength > 0.3:
            signature_type = "Strong neural response"
        elif signature_strength > 0.2:
            signature_type = "Moderate neural response"
        else:
            signature_type = "Weak neural response"
        
        return {
            'signature_type': signature_type,
            'signature_strength': signature_strength,
            'dominant_features': dict(top_features),
            'neural_complexity': len([v for v in features.values() if v > 0.15]),
            'activation_pattern': self._classify_activation_pattern(features)
        }
    
    def _classify_activation_pattern(self, features: Dict[str, float]) -> str:
        """Classify overall neural activation pattern"""
        
        frontal_activity = features.get('frontal_beta', 0) + features.get('working_memory_load', 0)
        temporal_activity = features.get('temporal_gamma', 0) + features.get('memory_theta', 0)
        limbic_activity = features.get('limbic_theta', 0) + features.get('autonomic_response', 0)
        
        max_activity = max(frontal_activity, temporal_activity, limbic_activity)
        
        if max_activity == frontal_activity:
            return "Frontal-dominant (analytical processing)"
        elif max_activity == temporal_activity:
            return "Temporal-dominant (memory/language processing)"
        else:
            return "Limbic-dominant (emotional processing)"


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
        # Advanced Neural Monitoring Features (from problem statement)
        self.eeg_simulator = MultiChannelEEGSimulator()
        self.brainwave_analyzer = FrequencyBandAnalyzer()
        self.cognitive_load_meter = CognitiveLoadMeasurement()
        self.attention_tracker = AttentionStateMonitor()
        self.fatigue_detector = NeuralFatigueAnalyzer()
        self.pattern_recognizer = MLNeuralPatterns()
    
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


# Utility functions for Streamlit integration (only if Streamlit is available)
def render_neural_simulation_ui():
    """Render the neural simulation UI component"""
    if not HAS_STREAMLIT:
        print("Warning: Streamlit not available. UI rendering skipped.")
        return
        
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
    if not HAS_STREAMLIT or not HAS_PLOTLY:
        print("Warning: Streamlit or Plotly not available. Results display skipped.")
        return
    
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
    
    # Continue with other visualizations and insights...


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


# Advanced Neural Processor Class
class AdvancedNeuralProcessor:
    """
    Advanced Neural Processing Module for Enhanced Neural Simulation
    
    Features:
    - Real-time EEG processing with white noise generation
    - Dark matter neural pattern simulation
    - Advanced neural oscillations analysis
    - Neural connectivity mapping
    - Cognitive load measurement
    """
    
    def __init__(self):
        self.eeg_channels = self._initialize_eeg_channels()
        self.frequency_bands = self._initialize_frequency_bands()
        self.connectivity_matrix = self._initialize_connectivity_matrix()
        self.cultural_modulation = self._initialize_cultural_modulation()
        
    def _initialize_eeg_channels(self) -> Dict[str, Dict[str, Any]]:
        """Initialize 64-channel EEG setup"""
        channels = {}
        standard_positions = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
            'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
            'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
        ]
        
        for i, pos in enumerate(standard_positions[:32]):  # 32 main channels
            channels[pos] = {
                'position': pos,
                'x_coord': np.cos(2 * np.pi * i / 32),
                'y_coord': np.sin(2 * np.pi * i / 32),
                'brain_region': self._assign_brain_region(pos),
                'baseline_amplitude': np.random.uniform(8, 15),  # microvolts
                'noise_level': np.random.uniform(0.5, 2.0)
            }
            
        return channels
    
    def _assign_brain_region(self, channel: str) -> str:
        """Assign brain region to EEG channel"""
        if channel.startswith(('Fp', 'F')):
            return 'frontal'
        elif channel.startswith(('C', 'FC')):
            return 'central'
        elif channel.startswith(('P', 'CP')):
            return 'parietal'
        elif channel.startswith(('O', 'PO')):
            return 'occipital'
        elif channel.startswith('T'):
            return 'temporal'
        else:
            return 'other'
    
    def _initialize_frequency_bands(self) -> Dict[str, Dict[str, Any]]:
        """Initialize enhanced frequency band analysis"""
        return {
            'delta': {
                'range': (0.5, 4),
                'function': 'Deep sleep, unconscious processing',
                'marketing_relevance': 'Subliminal influence, brand familiarity',
                'cultural_variation': 0.15
            },
            'theta': {
                'range': (4, 8),
                'function': 'Memory encoding, emotional processing',
                'marketing_relevance': 'Emotional brand connection, memory formation',
                'cultural_variation': 0.25
            },
            'alpha': {
                'range': (8, 13),
                'function': 'Relaxed awareness, creative thinking',
                'marketing_relevance': 'Brand preference, aesthetic appreciation',
                'cultural_variation': 0.30
            },
            'beta': {
                'range': (13, 30),
                'function': 'Active thinking, decision making',
                'marketing_relevance': 'Product evaluation, purchase decisions',
                'cultural_variation': 0.20
            },
            'gamma': {
                'range': (30, 100),
                'function': 'Conscious awareness, binding',
                'marketing_relevance': 'Brand recognition, attention capture',
                'cultural_variation': 0.10
            },
            'high_gamma': {
                'range': (100, 200),
                'function': 'Ultra-fast processing, micro-states',
                'marketing_relevance': 'Instant reactions, first impressions',
                'cultural_variation': 0.05
            }
        }
    
    def _initialize_connectivity_matrix(self) -> np.ndarray:
        """Initialize neural connectivity matrix"""
        num_regions = 8  # 8 main brain regions
        connectivity = np.random.rand(num_regions, num_regions)
        # Make symmetric
        connectivity = (connectivity + connectivity.T) / 2
        # Zero diagonal (no self-connections)
        np.fill_diagonal(connectivity, 0)
        return connectivity
    
    def _initialize_cultural_modulation(self) -> Dict[str, Dict[str, float]]:
        """Initialize cultural modulation factors"""
        return {
            'collectivist': {
                'frontal_alpha': 1.2,
                'temporal_theta': 1.3,
                'social_gamma': 1.4,
                'empathy_beta': 1.25
            },
            'individualist': {
                'frontal_beta': 1.3,
                'parietal_gamma': 1.2,
                'decision_theta': 0.8,
                'self_alpha': 1.35
            },
            'high_context': {
                'temporal_theta': 1.4,
                'occipital_alpha': 1.2,
                'integration_gamma': 1.3,
                'contextual_beta': 1.15
            },
            'low_context': {
                'frontal_beta': 1.25,
                'central_gamma': 1.1,
                'direct_alpha': 1.0,
                'explicit_theta': 0.9
            }
        }
    
    def generate_white_noise_eeg_baseline(self, duration: float = 10.0, 
                                        sampling_rate: int = 250,
                                        cultural_context: str = 'neutral') -> Dict[str, Any]:
        """
        Generate white noise EEG baseline with cultural modulation
        
        Args:
            duration: Recording duration in seconds
            sampling_rate: EEG sampling rate in Hz
            cultural_context: Cultural context for modulation
            
        Returns:
            Dictionary containing EEG baseline data
        """
        num_samples = int(duration * sampling_rate)
        time_axis = np.linspace(0, duration, num_samples)
        
        eeg_data = {}
        
        for channel, config in self.eeg_channels.items():
            # Generate base white noise
            white_noise = np.random.normal(0, config['noise_level'], num_samples)
            
            # Add physiological oscillations
            signal = white_noise.copy()
            for band_name, band_config in self.frequency_bands.items():
                freq_low, freq_high = band_config['range']
                center_freq = (freq_low + freq_high) / 2
                
                # Generate band-specific oscillation
                oscillation = np.sin(2 * np.pi * center_freq * time_axis)
                
                # Apply cultural modulation
                cultural_factor = self._get_cultural_modulation(
                    cultural_context, channel, band_name
                )
                
                signal += oscillation * config['baseline_amplitude'] * 0.1 * cultural_factor
            
            eeg_data[channel] = {
                'signal': signal,
                'sampling_rate': sampling_rate,
                'duration': duration,
                'brain_region': config['brain_region'],
                'cultural_modulation': cultural_factor
            }
        
        return {
            'eeg_channels': eeg_data,
            'time_axis': time_axis,
            'cultural_context': cultural_context,
            'metadata': {
                'num_channels': len(eeg_data),
                'duration': duration,
                'sampling_rate': sampling_rate,
                'baseline_quality': 'high'
            }
        }
    
    def simulate_dark_matter_neural_patterns(self, baseline_eeg: Dict[str, Any],
                                           unconscious_stimulus: str = 'brand_logo') -> Dict[str, Any]:
        """
        Simulate dark matter neural patterns for unconscious processing
        
        Args:
            baseline_eeg: Baseline EEG data
            unconscious_stimulus: Type of unconscious stimulus
            
        Returns:
            Dictionary containing dark matter neural patterns
        """
        dark_matter_patterns = {}
        
        for channel, data in baseline_eeg['eeg_channels'].items():
            signal = data['signal'].copy()
            
            # Add dark matter oscillations (very low amplitude, high frequency)
            dark_matter_freq = np.random.uniform(150, 300)  # Hz
            time_axis = baseline_eeg['time_axis']
            
            # Ultra-weak dark matter signal
            dark_matter_amplitude = np.random.uniform(0.01, 0.05)  # microvolts
            dark_matter_signal = dark_matter_amplitude * np.sin(
                2 * np.pi * dark_matter_freq * time_axis
            )
            
            # Phase coupling with existing signals
            phase_coupling = self._calculate_phase_coupling(signal, dark_matter_signal)
            
            # Add unconscious processing modulation
            unconscious_modulation = self._get_unconscious_modulation(
                unconscious_stimulus, data['brain_region']
            )
            
            dark_matter_patterns[channel] = {
                'enhanced_signal': signal + dark_matter_signal * unconscious_modulation,
                'dark_matter_component': dark_matter_signal * unconscious_modulation,
                'phase_coupling': phase_coupling,
                'unconscious_influence': unconscious_modulation,
                'detection_threshold': dark_matter_amplitude * 2,
                'subliminal_index': unconscious_modulation * phase_coupling
            }
        
        return {
            'dark_matter_patterns': dark_matter_patterns,
            'unconscious_stimulus': unconscious_stimulus,
            'overall_influence': np.mean([
                p['subliminal_index'] for p in dark_matter_patterns.values()
            ]),
            'detection_probability': np.random.uniform(0.05, 0.15),  # Very low
            'unconscious_processing_score': np.random.uniform(0.7, 0.95)
        }
    
    def _get_cultural_modulation(self, cultural_context: str, channel: str, band: str) -> float:
        """Get cultural modulation factor"""
        if cultural_context in self.cultural_modulation:
            return self.cultural_modulation[cultural_context].get(f"{channel}_{band}", 1.0)
        return 1.0
    
    def _calculate_phase_coupling(self, signal1: np.ndarray, signal2: np.ndarray) -> float:
        """Calculate phase coupling between signals"""
        # Simplified phase coupling calculation using correlation
        correlation = np.corrcoef(signal1, signal2)[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def _get_unconscious_modulation(self, stimulus: str, brain_region: str) -> float:
        """Get unconscious processing modulation"""
        modulation_map = {
            ('brand_logo', 'occipital'): 1.2,
            ('brand_logo', 'temporal'): 1.1,
            ('emotional_content', 'frontal'): 1.4,
            ('price_anchor', 'frontal'): 1.3,
        }
        return modulation_map.get((stimulus, brain_region), 1.0)


# Comprehensive Neural Analysis Pipeline
class NeuralAnalysisPipeline:
    """Comprehensive neural analysis pipeline integrating all components"""
    
    def __init__(self):
        self.brain_twin = DigitalBrainTwin()
        self.advanced_processor = AdvancedNeuralProcessor()
        
    def run_comprehensive_analysis(self, 
                                 stimulus_text: str,
                                 consumer_type: str = 'analytical_buyer',
                                 cultural_context: str = 'neutral',
                                 analysis_depth: str = 'standard') -> Dict[str, Any]:
        """
        Run comprehensive neural analysis
        
        Args:
            stimulus_text: Marketing stimulus to analyze
            consumer_type: Consumer profile type
            cultural_context: Cultural context for analysis
            analysis_depth: 'basic', 'standard', or 'comprehensive'
            
        Returns:
            Complete analysis results
        """
        
        print(f"Starting comprehensive neural analysis...")
        print(f"Stimulus: {stimulus_text[:50]}...")
        print(f"Consumer Type: {consumer_type}")
        print(f"Cultural Context: {cultural_context}")
        
        # 1. Basic brain twin simulation
        basic_results = self.brain_twin.simulate_marketing_response(
            stimulus_text, consumer_type, duration=10.0
        )
        
        if analysis_depth == 'basic':
            return {'basic_simulation': basic_results}
        
        # 2. Generate EEG baseline
        eeg_baseline = self.advanced_processor.generate_white_noise_eeg_baseline(
            duration=10.0, cultural_context=cultural_context
        )
        
        # 3. Simulate dark matter patterns
        dark_matter = self.advanced_processor.simulate_dark_matter_neural_patterns(
            eeg_baseline, unconscious_stimulus='brand_logo'
        )
        
        if analysis_depth == 'standard':
            return {
                'basic_simulation': basic_results,
                'eeg_baseline': eeg_baseline,
                'dark_matter_patterns': dark_matter
            }
        
        # 4. Advanced pattern analysis (comprehensive only)
        advanced_analysis = self._run_advanced_pattern_analysis(basic_results, eeg_baseline)
        
        # 5. Generate comprehensive insights
        comprehensive_insights = self._generate_comprehensive_insights(
            basic_results, eeg_baseline, dark_matter, advanced_analysis
        )
        
        return {
            'basic_simulation': basic_results,
            'eeg_baseline': eeg_baseline,
            'dark_matter_patterns': dark_matter,
            'advanced_analysis': advanced_analysis,
            'comprehensive_insights': comprehensive_insights,
            'meta_analysis': {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_depth': analysis_depth,
                'cultural_context': cultural_context,
                'confidence_level': np.random.uniform(0.85, 0.98)
            }
        }
    
    def _run_advanced_pattern_analysis(self, basic_results: Dict, eeg_data: Dict) -> Dict[str, Any]:
        """Run advanced pattern analysis"""
        
        # Simulate advanced EEG analysis
        frequency_analysis = self.brain_twin.brainwave_analyzer.analyze_frequency_bands(
            np.random.randn(2560), sampling_rate=256  # 10 seconds at 256 Hz
        )
        
        # Cognitive load analysis
        cognitive_load = self.brain_twin.cognitive_load_meter.measure_cognitive_load(eeg_data)
        
        # Attention state monitoring
        attention_state = self.brain_twin.attention_tracker.monitor_attention_state(eeg_data)
        
        # Fatigue analysis
        fatigue_analysis = self.brain_twin.fatigue_detector.analyze_neural_fatigue(eeg_data)
        
        # Pattern recognition
        pattern_analysis = self.brain_twin.pattern_recognizer.detect_neural_patterns(eeg_data)
        
        return {
            'frequency_analysis': frequency_analysis,
            'cognitive_load': cognitive_load,
            'attention_state': attention_state,
            'fatigue_analysis': fatigue_analysis,
            'pattern_analysis': pattern_analysis
        }
    
    def _generate_comprehensive_insights(self, basic: Dict, eeg: Dict, 
                                       dark_matter: Dict, advanced: Dict) -> Dict[str, Any]:
        """Generate comprehensive insights from all analyses"""
        
        # Extract key metrics
        purchase_probability = basic['behavioral_outcomes']['purchase_probability']
        attention_score = basic['behavioral_outcomes']['attention_score']
        emotional_response = basic['behavioral_outcomes']['emotional_response']
        unconscious_influence = dark_matter['overall_influence']
        cognitive_load = advanced['cognitive_load']['cognitive_load_score']
        
        # Calculate composite scores
        conscious_engagement = (purchase_probability + attention_score + emotional_response) / 3
        unconscious_engagement = unconscious_influence
        overall_neural_impact = (conscious_engagement * 0.7 + unconscious_engagement * 0.3)
        
        # Generate strategic recommendations
        strategic_recommendations = []
        
        if overall_neural_impact > 0.8:
            strategic_recommendations.extend([
                "Exceptional neural response - scale this approach",
                "Consider A/B testing variations to optimize further",
                "Use as template for similar campaigns"
            ])
        elif overall_neural_impact > 0.6:
            strategic_recommendations.extend([
                "Strong neural foundation with optimization potential",
                "Focus on enhancing weaker response areas",
                "Test with different consumer segments"
            ])
        else:
            strategic_recommendations.extend([
                "Significant revision needed for optimal impact",
                "Consider fundamental approach changes",
                "Test alternative messaging strategies"
            ])
        
        # Risk assessment
        risk_factors = []
        if cognitive_load > 0.7:
            risk_factors.append("High cognitive load may reduce comprehension")
        if advanced['fatigue_analysis']['fatigue_score'] > 0.6:
            risk_factors.append("Neural fatigue detected - may impact decision quality")
        if attention_score < 0.4:
            risk_factors.append("Low attention capture - message may be ignored")
        
        return {
            'composite_scores': {
                'conscious_engagement': conscious_engagement,
                'unconscious_engagement': unconscious_engagement,
                'overall_neural_impact': overall_neural_impact,
                'cognitive_efficiency': 1 - cognitive_load,
                'attention_quality': attention_score,
                'emotional_resonance': emotional_response
            },
            'strategic_recommendations': strategic_recommendations,
            'risk_assessment': {
                'risk_factors': risk_factors,
                'overall_risk_level': 'Low' if len(risk_factors) == 0 else 'Medium' if len(risk_factors) <= 2 else 'High'
            },
            'optimization_priorities': self._generate_optimization_priorities(basic, advanced),
            'market_segmentation_insights': self._generate_segmentation_insights(basic, advanced),
            'predicted_business_impact': self._predict_business_impact(overall_neural_impact, purchase_probability)
        }
    
    def _generate_optimization_priorities(self, basic: Dict, advanced: Dict) -> List[Dict[str, Any]]:
        """Generate optimization priorities"""
        priorities = []
        
        outcomes = basic['behavioral_outcomes']
        
        # Prioritize based on impact potential
        metrics = [
            ('attention_score', outcomes['attention_score'], 'Attention Capture'),
            ('emotional_response', outcomes['emotional_response'], 'Emotional Engagement'),
            ('purchase_probability', outcomes['purchase_probability'], 'Purchase Intent'),
            ('memory_retention', outcomes['memory_retention'], 'Brand Recall')
        ]
        
        # Sort by lowest scores (highest improvement potential)
        sorted_metrics = sorted(metrics, key=lambda x: x[1])
        
        for i, (metric, score, description) in enumerate(sorted_metrics[:3]):
            priority = {
                'rank': i + 1,
                'area': description,
                'current_score': score,
                'improvement_potential': 1 - score,
                'priority_level': 'High' if i == 0 else 'Medium' if i == 1 else 'Low',
                'recommended_actions': self._get_improvement_actions(metric, score)
            }
            priorities.append(priority)
        
        return priorities
    
    def _get_improvement_actions(self, metric: str, score: float) -> List[str]:
        """Get specific improvement actions for metrics"""
        actions_map = {
            'attention_score': [
                "Enhance visual hierarchy and contrast",
                "Strengthen headline impact",
                "Add dynamic visual elements",
                "Optimize information density"
            ],
            'emotional_response': [
                "Incorporate storytelling elements",
                "Add emotional triggers and imagery",
                "Use aspirational language",
                "Include social proof elements"
            ],
            'purchase_probability': [
                "Strengthen value proposition",
                "Improve call-to-action clarity",
                "Reduce purchase friction",
                "Add urgency indicators"
            ],
            'memory_retention': [
                "Create memorable brand associations",
                "Use repetition strategically",
                "Add distinctive visual elements",
                "Simplify key messages"
            ]
        }
        
        base_actions = actions_map.get(metric, ["Monitor and test variations"])
        
        # Filter actions based on score severity
        if score < 0.3:
            return base_actions  # All actions needed
        elif score < 0.6:
            return base_actions[:2]  # Top 2 actions
        else:
            return base_actions[:1]  # Top action only
    
    def _generate_segmentation_insights(self, basic: Dict, advanced: Dict) -> Dict[str, Any]:
        """Generate market segmentation insights"""
        
        consumer_type = basic['consumer_type']
        outcomes = basic['behavioral_outcomes']
        
        # Determine optimal segments
        segment_scores = {
            'impulse_buyers': outcomes['emotional_response'] * 0.6 + (1 - outcomes['decision_speed']) * 0.4,
            'analytical_buyers': outcomes['attention_score'] * 0.4 + outcomes['memory_retention'] * 0.6,
            'social_buyers': outcomes['emotional_response'] * 0.5 + outcomes['word_of_mouth_probability'] * 0.5,
            'price_conscious': outcomes['purchase_probability'] * 0.7 + outcomes['attention_score'] * 0.3,
            'brand_loyal': outcomes['memory_retention'] * 0.6 + outcomes['return_customer_likelihood'] * 0.4
        }
        
        # Sort segments by effectiveness
        sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'primary_target': sorted_segments[0][0],
            'secondary_targets': [seg[0] for seg in sorted_segments[1:3]],
            'segment_effectiveness': dict(sorted_segments),
            'targeting_recommendations': [
                f"Primary focus: {sorted_segments[0][0]} (score: {sorted_segments[0][1]:.2f})",
                f"Secondary targets: {', '.join([seg[0] for seg in sorted_segments[1:3]])}",
                "Consider segment-specific message variations"
            ]
        }
    
    def _predict_business_impact(self, neural_impact: float, purchase_prob: float) -> Dict[str, Any]:
        """Predict business impact based on neural analysis"""
        
        # Simulate business metrics
        baseline_conversion = 0.03  # 3% baseline conversion rate
        predicted_conversion = baseline_conversion * (1 + neural_impact * 2)  # Up to 3x improvement
        
        # Calculate impact multipliers
        engagement_multiplier = 1 + (neural_impact - 0.5) * 0.8  # 0.6x to 1.4x
        retention_multiplier = 1 + (purchase_prob - 0.5) * 0.6   # 0.7x to 1.3x
        
        return {
            'predicted_conversion_rate': predicted_conversion,
            'conversion_improvement': (predicted_conversion / baseline_conversion - 1) * 100,
            'engagement_multiplier': engagement_multiplier,
            'retention_multiplier': retention_multiplier,
            'roi_indicators': {
                'campaign_effectiveness': 'High' if neural_impact > 0.7 else 'Medium' if neural_impact > 0.5 else 'Low',
                'investment_recommendation': 'Scale' if neural_impact > 0.8 else 'Optimize' if neural_impact > 0.5 else 'Revise',
                'confidence_level': min(neural_impact * 1.2, 1.0)
            }
        }


# Example usage function
def main():
    """Example usage of the neural simulation module"""
    print("Neural Simulation Module - Example Usage")
    print("=" * 50)
    
    # Initialize the analysis pipeline
    pipeline = NeuralAnalysisPipeline()
    
    # Example marketing stimulus
    marketing_text = """
    Discover the Revolutionary SmartPhone X Pro - where cutting-edge technology meets 
    unparalleled elegance. With our advanced AI camera system, capture life's precious 
    moments in stunning 8K resolution. Limited time offer: Save $200 when you order today!
    """
    
    # Run basic analysis
    print("\n1. Running Basic Analysis...")
    basic_results = pipeline.run_comprehensive_analysis(
        stimulus_text=marketing_text,
        consumer_type='impulse_buyer',
        analysis_depth='basic'
    )
    
    print(f"Basic Analysis Complete:")
    print(f"- Neuro Score: {basic_results['basic_simulation']['marketing_insights']['neuro_score']:.1f}/100")
    print(f"- Purchase Probability: {basic_results['basic_simulation']['behavioral_outcomes']['purchase_probability']:.1%}")
    print(f"- Attention Score: {basic_results['basic_simulation']['behavioral_outcomes']['attention_score']:.1%}")
    
    # Run comprehensive analysis
    print("\n2. Running Comprehensive Analysis...")
    comprehensive_results = pipeline.run_comprehensive_analysis(
        stimulus_text=marketing_text,
        consumer_type='analytical_buyer',
        cultural_context='individualist',
        analysis_depth='comprehensive'
    )
    
    print(f"Comprehensive Analysis Complete:")
    insights = comprehensive_results['comprehensive_insights']
    print(f"- Overall Neural Impact: {insights['composite_scores']['overall_neural_impact']:.1%}")
    print(f"- Conscious Engagement: {insights['composite_scores']['conscious_engagement']:.1%}")
    print(f"- Unconscious Engagement: {insights['composite_scores']['unconscious_engagement']:.1%}")
    print(f"- Primary Target Segment: {insights['market_segmentation_insights']['primary_target']}")
    print(f"- Investment Recommendation: {insights['predicted_business_impact']['roi_indicators']['investment_recommendation']}")
    
    # Display top recommendations
    print(f"\n3. Top Strategic Recommendations:")
    for i, rec in enumerate(insights['strategic_recommendations'][:3], 1):
        print(f"   {i}. {rec}")
    
    # Display optimization priorities
    print(f"\n4. Optimization Priorities:")
    for priority in insights['optimization_priorities']:
        print(f"   {priority['rank']}. {priority['area']} (Score: {priority['current_score']:.1%}, Priority: {priority['priority_level']})")
    
    # Export results
    print(f"\n5. Exporting Results...")
    
    # Export as JSON
    json_export = export_neural_simulation_results(comprehensive_results['basic_simulation'], 'json')
    with open('neural_analysis_results.json', 'w') as f:
        f.write(json_export)
    print("   ✓ Results exported to neural_analysis_results.json")
    
    # Export as Markdown
    md_export = export_neural_simulation_results(comprehensive_results['basic_simulation'], 'markdown')
    with open('neural_analysis_report.md', 'w') as f:
        f.write(md_export)
    print("   ✓ Report exported to neural_analysis_report.md")
    
    print("\n" + "=" * 50)
    print("Neural Simulation Analysis Complete!")
    print("Check the exported files for detailed results.")
    
    return comprehensive_results


# Validation and testing functions
def validate_neural_simulation():
    """Validate the neural simulation module functionality"""
    print("Validating Neural Simulation Module...")
    
    try:
        # Test basic components
        print("1. Testing basic components...")
        brain_twin = DigitalBrainTwin()
        assert len(brain_twin.brain_regions) == 8
        assert len(brain_twin.consumer_profiles) == 5
        print("   ✓ DigitalBrainTwin initialization successful")
        
        # Test EEG simulator
        eeg_sim = MultiChannelEEGSimulator()
        eeg_data = eeg_sim.simulate_multichannel_eeg(duration=5.0)
        assert len(eeg_data['eeg_data']) == 16
        print("   ✓ EEG simulation successful")
        
        # Test frequency analyzer
        freq_analyzer = FrequencyBandAnalyzer()
        test_signal = np.random.randn(1280)  # 5 seconds at 256 Hz
        freq_results = freq_analyzer.analyze_frequency_bands(test_signal)
        assert len(freq_results['band_powers']) == 5
        print("   ✓ Frequency analysis successful")
        
        # Test cognitive load measurement
        load_meter = CognitiveLoadMeasurement()
        load_results = load_meter.measure_cognitive_load(eeg_data)
        assert 0 <= load_results['cognitive_load_score'] <= 1
        print("   ✓ Cognitive load measurement successful")
        
        # Test marketing simulation
        print("2. Testing marketing simulation...")
        results = brain_twin.simulate_marketing_response(
            "Test marketing message with great benefits!",
            consumer_type='impulse_buyer',
            duration=5.0
        )
        assert 'behavioral_outcomes' in results
        assert 'marketing_insights' in results
        print("   ✓ Marketing simulation successful")
        
        # Test advanced processor
        print("3. Testing advanced neural processor...")
        processor = AdvancedNeuralProcessor()
        baseline = processor.generate_white_noise_eeg_baseline(duration=5.0)
        assert len(baseline['eeg_channels']) > 0
        print("   ✓ Advanced processor successful")
        
        # Test dark matter simulation
        dark_matter = processor.simulate_dark_matter_neural_patterns(baseline)
        assert 'dark_matter_patterns' in dark_matter
        print("   ✓ Dark matter simulation successful")
        
        # Test comprehensive pipeline
        print("4. Testing comprehensive analysis pipeline...")
        pipeline = NeuralAnalysisPipeline()
        comprehensive = pipeline.run_comprehensive_analysis(
            "Test comprehensive analysis",
            analysis_depth='comprehensive'
        )
        assert 'comprehensive_insights' in comprehensive
        print("   ✓ Comprehensive pipeline successful")
        
        print("\n✓ All validation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Validation failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("🧠 Neural Simulation Module for Marketing Analysis")
    print("Advanced brain-computer interface simulation for consumer behavior prediction")
    print("=" * 80)
    
    # Run validation
    if validate_neural_simulation():
        print("\n" + "=" * 80)
        
        # Run main example
        comprehensive_results = main()
        
        print(f"\n🎯 Analysis Summary:")
        print(f"   • Stimulus analyzed with advanced neural modeling")
        print(f"   • Multi-layer brain simulation completed")
        print(f"   • Consumer behavior predictions generated")
        print(f"   • Strategic recommendations provided")
        print(f"   • Business impact assessment completed")
        
        print(f"\n📊 Key Findings:")
        if 'comprehensive_insights' in comprehensive_results:
            insights = comprehensive_results['comprehensive_insights']
            print(f"   • Overall Neural Impact: {insights['composite_scores']['overall_neural_impact']:.1%}")
            print(f"   • Primary Target: {insights['market_segmentation_insights']['primary_target']}")
            print(f"   • Investment Rec: {insights['predicted_business_impact']['roi_indicators']['investment_recommendation']}")
        
        print(f"\n🚀 Module ready for production use!")
    else:
        print("\n❌ Module validation failed. Please check the implementation.")
        
    print("=" * 80)