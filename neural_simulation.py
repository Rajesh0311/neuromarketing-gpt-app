#!/usr/bin/env python3
"""
Neural Simulation Engine for NeuroMarketing GPT Platform
========================================================

This module provides advanced neural simulation capabilities including:
- Digital Brain Twin technology
- EEG signal simulation and analysis
- Consumer behavior prediction
- Neural response modeling
- Real-time brain activity monitoring

Authors: NeuroMarketing GPT Team
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import asyncio
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BrainRegion:
    """Represents a brain region in the digital twin"""
    name: str
    location: Tuple[float, float, float]  # 3D coordinates
    base_frequency: float  # Hz
    activation_level: float  # 0.0 to 1.0
    connections: List[str]  # Connected regions
    function: str  # Primary function

@dataclass
class NeuralSignal:
    """Represents a neural signal measurement"""
    timestamp: datetime
    frequency_bands: Dict[str, float]  # Delta, Theta, Alpha, Beta, Gamma
    amplitude: float
    quality: float  # Signal quality 0.0 to 1.0
    region: str
    noise_level: float

@dataclass
class SimulationResult:
    """Results from neural simulation"""
    participant_id: str
    stimulus_type: str
    simulation_duration: float
    brain_regions: Dict[str, BrainRegion]
    neural_signals: List[NeuralSignal]
    behavioral_predictions: Dict[str, float]
    summary_metrics: Dict[str, float]
    timestamp: datetime

class DigitalBrainTwin:
    """Digital Brain Twin for marketing response simulation"""
    
    def __init__(self):
        self.brain_regions = self._initialize_brain_regions()
        self.frequency_bands = {
            'delta': (0.5, 4),      # Deep sleep, unconscious processing
            'theta': (4, 8),        # Drowsiness, creativity, meditation
            'alpha': (8, 13),       # Relaxed awareness, visual processing
            'beta': (13, 30),       # Active thinking, concentration
            'gamma': (30, 100)      # Binding consciousness, insight
        }
        self.simulation_history = []
        
    def _initialize_brain_regions(self) -> Dict[str, BrainRegion]:
        """Initialize brain regions for the digital twin"""
        regions = {
            'frontal_cortex': BrainRegion(
                name='Frontal Cortex',
                location=(0.0, 0.5, 0.0),
                base_frequency=20.0,
                activation_level=0.5,
                connections=['parietal_cortex', 'temporal_cortex', 'limbic_system'],
                function='Executive control, decision making, planning'
            ),
            'parietal_cortex': BrainRegion(
                name='Parietal Cortex',
                location=(-0.3, 0.0, 0.0),
                base_frequency=10.0,
                activation_level=0.4,
                connections=['frontal_cortex', 'temporal_cortex', 'occipital_cortex'],
                function='Spatial processing, attention, sensory integration'
            ),
            'temporal_cortex': BrainRegion(
                name='Temporal Cortex',
                location=(0.3, 0.0, 0.0),
                base_frequency=15.0,
                activation_level=0.3,
                connections=['frontal_cortex', 'parietal_cortex', 'limbic_system'],
                function='Auditory processing, language, memory'
            ),
            'occipital_cortex': BrainRegion(
                name='Occipital Cortex',
                location=(0.0, -0.5, 0.0),
                base_frequency=12.0,
                activation_level=0.6,
                connections=['parietal_cortex', 'temporal_cortex'],
                function='Visual processing, image recognition'
            ),
            'limbic_system': BrainRegion(
                name='Limbic System',
                location=(0.0, 0.0, -0.3),
                base_frequency=8.0,
                activation_level=0.7,
                connections=['frontal_cortex', 'temporal_cortex'],
                function='Emotion, motivation, memory formation'
            ),
            'cerebellum': BrainRegion(
                name='Cerebellum',
                location=(0.0, -0.3, -0.4),
                base_frequency=25.0,
                activation_level=0.4,
                connections=['frontal_cortex', 'parietal_cortex'],
                function='Motor control, balance, coordination'
            ),
            'brainstem': BrainRegion(
                name='Brainstem',
                location=(0.0, 0.0, -0.5),
                base_frequency=6.0,
                activation_level=0.8,
                connections=['limbic_system', 'cerebellum'],
                function='Vital functions, arousal, attention'
            ),
            'insula': BrainRegion(
                name='Insula',
                location=(0.2, 0.1, -0.1),
                base_frequency=18.0,
                activation_level=0.5,
                connections=['frontal_cortex', 'limbic_system'],
                function='Interoception, empathy, decision making'
            )
        }
        return regions
    
    def simulate_marketing_response(self, stimulus_text: str, 
                                  consumer_type: str = "average",
                                  duration: float = 30.0,
                                  sampling_rate: int = 250) -> SimulationResult:
        """
        Simulate neural response to marketing stimulus
        
        Args:
            stimulus_text: Marketing content to analyze
            consumer_type: Type of consumer (average, impulsive, analytical, etc.)
            duration: Simulation duration in seconds
            sampling_rate: Sampling rate in Hz
            
        Returns:
            SimulationResult with complete neural simulation data
        """
        start_time = datetime.now()
        participant_id = f"sim_{int(start_time.timestamp())}"
        
        logger.info(f"Starting neural simulation for {duration}s at {sampling_rate}Hz")
        
        # Analyze stimulus characteristics
        stimulus_features = self._analyze_stimulus(stimulus_text)
        
        # Adjust brain regions based on stimulus and consumer type
        self._adjust_brain_regions(stimulus_features, consumer_type)
        
        # Generate neural signals
        neural_signals = self._generate_neural_signals(duration, sampling_rate)
        
        # Predict behavioral outcomes
        behavioral_predictions = self._predict_behavior(neural_signals, stimulus_features)
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(neural_signals, behavioral_predictions)
        
        result = SimulationResult(
            participant_id=participant_id,
            stimulus_type="marketing_text",
            simulation_duration=duration,
            brain_regions=self.brain_regions.copy(),
            neural_signals=neural_signals,
            behavioral_predictions=behavioral_predictions,
            summary_metrics=summary_metrics,
            timestamp=start_time
        )
        
        self.simulation_history.append(result)
        
        logger.info(f"Neural simulation completed in {(datetime.now() - start_time).total_seconds():.2f}s")
        return result
    
    def _analyze_stimulus(self, stimulus_text: str) -> Dict[str, float]:
        """Analyze marketing stimulus characteristics"""
        features = {
            'emotional_intensity': 0.5,
            'cognitive_load': 0.5,
            'visual_complexity': 0.3,
            'urgency_level': 0.4,
            'trust_signals': 0.6,
            'social_proof': 0.3
        }
        
        text_lower = stimulus_text.lower()
        
        # Emotional intensity
        emotional_words = ['amazing', 'incredible', 'fantastic', 'love', 'hate', 'excited']
        emotional_count = sum(1 for word in emotional_words if word in text_lower)
        features['emotional_intensity'] = min(emotional_count / 10.0, 1.0)
        
        # Cognitive load (complexity)
        word_count = len(stimulus_text.split())
        avg_word_length = np.mean([len(word) for word in stimulus_text.split()])
        features['cognitive_load'] = min((word_count * avg_word_length) / 1000.0, 1.0)
        
        # Urgency level
        urgency_words = ['now', 'limited', 'urgent', 'hurry', 'today', 'immediately']
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        features['urgency_level'] = min(urgency_count / 5.0, 1.0)
        
        # Trust signals
        trust_words = ['guarantee', 'certified', 'proven', 'trusted', 'secure', 'reliable']
        trust_count = sum(1 for word in trust_words if word in text_lower)
        features['trust_signals'] = min(trust_count / 5.0, 1.0)
        
        return features
    
    def _adjust_brain_regions(self, stimulus_features: Dict[str, float], 
                             consumer_type: str):
        """Adjust brain region activation based on stimulus and consumer type"""
        # Consumer type modifiers
        type_modifiers = {
            'impulsive': {
                'limbic_system': 1.3,
                'frontal_cortex': 0.7,
                'temporal_cortex': 1.1
            },
            'analytical': {
                'frontal_cortex': 1.4,
                'parietal_cortex': 1.2,
                'limbic_system': 0.8
            },
            'visual': {
                'occipital_cortex': 1.5,
                'parietal_cortex': 1.2,
                'frontal_cortex': 0.9
            },
            'average': {region: 1.0 for region in self.brain_regions.keys()}
        }
        
        modifiers = type_modifiers.get(consumer_type, type_modifiers['average'])
        
        for region_name, region in self.brain_regions.items():
            # Apply consumer type modifier
            base_activation = region.activation_level * modifiers.get(region_name, 1.0)
            
            # Apply stimulus-specific adjustments
            if region_name == 'limbic_system':
                # Emotional processing
                base_activation += stimulus_features['emotional_intensity'] * 0.3
            elif region_name == 'frontal_cortex':
                # Decision making and cognitive processing
                base_activation += stimulus_features['cognitive_load'] * 0.2
                base_activation += stimulus_features['trust_signals'] * 0.2
            elif region_name == 'temporal_cortex':
                # Language processing
                base_activation += stimulus_features['cognitive_load'] * 0.15
            elif region_name == 'insula':
                # Empathy and emotional awareness
                base_activation += stimulus_features['emotional_intensity'] * 0.25
            
            # Ensure activation stays within bounds
            region.activation_level = np.clip(base_activation, 0.0, 1.0)
    
    def _generate_neural_signals(self, duration: float, 
                                sampling_rate: int) -> List[NeuralSignal]:
        """Generate realistic neural signals for each brain region"""
        signals = []
        time_points = int(duration * sampling_rate)
        time_vector = np.linspace(0, duration, time_points)
        
        for region_name, region in self.brain_regions.items():
            for i, t in enumerate(time_vector):
                # Generate base signal with region-specific characteristics
                base_freq = region.base_frequency
                activation = region.activation_level
                
                # Create realistic EEG-like signal
                signal_components = []
                
                # Add frequency band components
                for band_name, (low_freq, high_freq) in self.frequency_bands.items():
                    # Determine band power based on region and activation
                    band_power = self._calculate_band_power(region_name, band_name, activation)
                    
                    # Generate band-specific signal
                    freq = np.random.uniform(low_freq, high_freq)
                    amplitude = band_power * np.random.uniform(0.8, 1.2)
                    phase = np.random.uniform(0, 2 * np.pi)
                    
                    component = amplitude * np.sin(2 * np.pi * freq * t + phase)
                    signal_components.append(component)
                
                # Combine all components
                total_signal = np.sum(signal_components)
                
                # Add realistic noise
                noise_level = 0.1 * np.random.normal(0, 1)
                total_signal += noise_level
                
                # Calculate frequency band powers
                frequency_bands = self._extract_frequency_bands(signal_components)
                
                # Signal quality (affected by noise and artifact)
                quality = max(0.0, 1.0 - abs(noise_level) / 0.5)
                
                neural_signal = NeuralSignal(
                    timestamp=datetime.now() + timedelta(seconds=t),
                    frequency_bands=frequency_bands,
                    amplitude=float(total_signal),
                    quality=quality,
                    region=region_name,
                    noise_level=abs(noise_level)
                )
                
                signals.append(neural_signal)
        
        return signals
    
    def _calculate_band_power(self, region_name: str, band_name: str, 
                             activation: float) -> float:
        """Calculate power for specific frequency band in a brain region"""
        # Base power levels for different regions and bands
        base_powers = {
            'frontal_cortex': {
                'delta': 0.2, 'theta': 0.3, 'alpha': 0.4, 'beta': 0.8, 'gamma': 0.6
            },
            'parietal_cortex': {
                'delta': 0.2, 'theta': 0.4, 'alpha': 0.9, 'beta': 0.6, 'gamma': 0.4
            },
            'temporal_cortex': {
                'delta': 0.3, 'theta': 0.5, 'alpha': 0.6, 'beta': 0.7, 'gamma': 0.5
            },
            'occipital_cortex': {
                'delta': 0.2, 'theta': 0.3, 'alpha': 1.0, 'beta': 0.5, 'gamma': 0.3
            },
            'limbic_system': {
                'delta': 0.4, 'theta': 0.8, 'alpha': 0.5, 'beta': 0.6, 'gamma': 0.7
            },
            'cerebellum': {
                'delta': 0.1, 'theta': 0.2, 'alpha': 0.3, 'beta': 0.5, 'gamma': 0.8
            },
            'brainstem': {
                'delta': 0.8, 'theta': 0.6, 'alpha': 0.3, 'beta': 0.4, 'gamma': 0.2
            },
            'insula': {
                'delta': 0.3, 'theta': 0.4, 'alpha': 0.5, 'beta': 0.7, 'gamma': 0.6
            }
        }
        
        base_power = base_powers.get(region_name, {}).get(band_name, 0.5)
        return base_power * activation
    
    def _extract_frequency_bands(self, signal_components: List[float]) -> Dict[str, float]:
        """Extract frequency band powers from signal components"""
        # Simplified extraction - in reality would use FFT
        return {
            'delta': abs(signal_components[0]) if len(signal_components) > 0 else 0.0,
            'theta': abs(signal_components[1]) if len(signal_components) > 1 else 0.0,
            'alpha': abs(signal_components[2]) if len(signal_components) > 2 else 0.0,
            'beta': abs(signal_components[3]) if len(signal_components) > 3 else 0.0,
            'gamma': abs(signal_components[4]) if len(signal_components) > 4 else 0.0
        }
    
    def _predict_behavior(self, neural_signals: List[NeuralSignal], 
                         stimulus_features: Dict[str, float]) -> Dict[str, float]:
        """Predict behavioral outcomes from neural activity"""
        # Group signals by region
        region_signals = {}
        for signal in neural_signals:
            if signal.region not in region_signals:
                region_signals[signal.region] = []
            region_signals[signal.region].append(signal)
        
        # Calculate average activations
        region_activations = {}
        for region, signals in region_signals.items():
            avg_activation = np.mean([s.amplitude for s in signals])
            region_activations[region] = avg_activation
        
        # Behavioral predictions based on neuroscience research
        predictions = {}
        
        # Purchase intention (frontal cortex + limbic system)
        frontal_activity = abs(region_activations.get('frontal_cortex', 0))
        limbic_activity = abs(region_activations.get('limbic_system', 0))
        purchase_intent = (frontal_activity * 0.6 + limbic_activity * 0.4) * 0.5
        purchase_intent += stimulus_features['urgency_level'] * 0.2
        predictions['purchase_intention'] = np.clip(purchase_intent, 0.0, 1.0)
        
        # Brand recall (temporal cortex + hippocampus approximation)
        temporal_activity = abs(region_activations.get('temporal_cortex', 0))
        brand_recall = temporal_activity * 0.4 + stimulus_features['cognitive_load'] * 0.3
        predictions['brand_recall'] = np.clip(brand_recall, 0.0, 1.0)
        
        # Emotional engagement (limbic system + insula)
        insula_activity = abs(region_activations.get('insula', 0))
        emotional_engagement = (limbic_activity * 0.7 + insula_activity * 0.3) * 0.6
        emotional_engagement += stimulus_features['emotional_intensity'] * 0.3
        predictions['emotional_engagement'] = np.clip(emotional_engagement, 0.0, 1.0)
        
        # Attention level (parietal cortex + frontal cortex)
        parietal_activity = abs(region_activations.get('parietal_cortex', 0))
        attention_level = (parietal_activity * 0.6 + frontal_activity * 0.4) * 0.5
        predictions['attention_level'] = np.clip(attention_level, 0.0, 1.0)
        
        # Cognitive load (frontal cortex activity)
        cognitive_load = frontal_activity * 0.7 + stimulus_features['cognitive_load'] * 0.3
        predictions['cognitive_load'] = np.clip(cognitive_load, 0.0, 1.0)
        
        # Trust response (frontal cortex + stimulus trust signals)
        trust_response = frontal_activity * 0.5 + stimulus_features['trust_signals'] * 0.5
        predictions['trust_response'] = np.clip(trust_response, 0.0, 1.0)
        
        return predictions
    
    def _calculate_summary_metrics(self, neural_signals: List[NeuralSignal],
                                  behavioral_predictions: Dict[str, float]) -> Dict[str, float]:
        """Calculate summary metrics for the simulation"""
        # Signal quality metrics
        avg_quality = np.mean([s.quality for s in neural_signals])
        avg_noise = np.mean([s.noise_level for s in neural_signals])
        
        # Amplitude statistics
        amplitudes = [s.amplitude for s in neural_signals]
        avg_amplitude = np.mean(amplitudes)
        amplitude_variance = np.var(amplitudes)
        
        # Frequency band analysis
        all_alpha = [s.frequency_bands.get('alpha', 0) for s in neural_signals]
        all_beta = [s.frequency_bands.get('beta', 0) for s in neural_signals]
        alpha_beta_ratio = np.mean(all_alpha) / (np.mean(all_beta) + 1e-6)
        
        # Overall activation index
        overall_activation = np.mean([abs(a) for a in amplitudes])
        
        # Behavioral summary
        avg_behavioral_response = np.mean(list(behavioral_predictions.values()))
        
        return {
            'signal_quality': avg_quality,
            'noise_level': avg_noise,
            'average_amplitude': avg_amplitude,
            'amplitude_variance': amplitude_variance,
            'alpha_beta_ratio': alpha_beta_ratio,
            'overall_activation': overall_activation,
            'behavioral_response_strength': avg_behavioral_response,
            'data_points_collected': len(neural_signals)
        }
    
    def get_region_connectivity(self) -> Dict[str, List[str]]:
        """Get brain region connectivity map"""
        connectivity = {}
        for region_name, region in self.brain_regions.items():
            connectivity[region_name] = region.connections
        return connectivity
    
    def simulate_real_time_monitoring(self, duration: float = 60.0) -> Dict[str, Any]:
        """Simulate real-time brain monitoring session"""
        logger.info(f"Starting real-time monitoring for {duration} seconds")
        
        monitoring_data = {
            'session_start': datetime.now(),
            'duration': duration,
            'real_time_metrics': [],
            'alerts': [],
            'summary': {}
        }
        
        # Simulate real-time data collection
        time_points = int(duration * 4)  # 4 Hz sampling for real-time display
        
        for i in range(time_points):
            timestamp = datetime.now()
            
            # Generate current metrics
            current_metrics = {
                'timestamp': timestamp,
                'attention_level': np.random.uniform(0.4, 0.9),
                'engagement_level': np.random.uniform(0.3, 0.8),
                'stress_level': np.random.uniform(0.1, 0.6),
                'cognitive_load': np.random.uniform(0.2, 0.8),
                'emotional_state': np.random.choice(['positive', 'neutral', 'negative']),
                'signal_quality': np.random.uniform(0.7, 1.0)
            }
            
            monitoring_data['real_time_metrics'].append(current_metrics)
            
            # Generate alerts if needed
            if current_metrics['stress_level'] > 0.8:
                monitoring_data['alerts'].append({
                    'timestamp': timestamp,
                    'type': 'high_stress',
                    'message': 'High stress level detected'
                })
            
            if current_metrics['signal_quality'] < 0.5:
                monitoring_data['alerts'].append({
                    'timestamp': timestamp,
                    'type': 'poor_signal',
                    'message': 'Poor signal quality detected'
                })
        
        # Calculate session summary
        metrics_df = pd.DataFrame(monitoring_data['real_time_metrics'])
        monitoring_data['summary'] = {
            'avg_attention': float(metrics_df['attention_level'].mean()),
            'avg_engagement': float(metrics_df['engagement_level'].mean()),
            'avg_stress': float(metrics_df['stress_level'].mean()),
            'avg_cognitive_load': float(metrics_df['cognitive_load'].mean()),
            'signal_quality': float(metrics_df['signal_quality'].mean()),
            'total_alerts': len(monitoring_data['alerts']),
            'session_duration': duration
        }
        
        return monitoring_data
    
    def export_simulation_data(self, result: SimulationResult, 
                              format_type: str = 'json') -> str:
        """Export simulation results in various formats"""
        try:
            if format_type.lower() == 'json':
                # Convert to JSON-serializable format
                export_data = {
                    'participant_id': result.participant_id,
                    'stimulus_type': result.stimulus_type,
                    'simulation_duration': result.simulation_duration,
                    'timestamp': result.timestamp.isoformat(),
                    'brain_regions': {
                        name: {
                            'name': region.name,
                            'location': region.location,
                            'base_frequency': region.base_frequency,
                            'activation_level': region.activation_level,
                            'connections': region.connections,
                            'function': region.function
                        } for name, region in result.brain_regions.items()
                    },
                    'behavioral_predictions': result.behavioral_predictions,
                    'summary_metrics': result.summary_metrics,
                    'signal_count': len(result.neural_signals)
                }
                
                return json.dumps(export_data, indent=2)
            
            elif format_type.lower() == 'csv':
                # Convert neural signals to CSV
                signals_data = []
                for signal in result.neural_signals:
                    row = {
                        'timestamp': signal.timestamp.isoformat(),
                        'region': signal.region,
                        'amplitude': signal.amplitude,
                        'quality': signal.quality,
                        'noise_level': signal.noise_level
                    }
                    # Add frequency bands
                    for band, power in signal.frequency_bands.items():
                        row[f'band_{band}'] = power
                    
                    signals_data.append(row)
                
                df = pd.DataFrame(signals_data)
                return df.to_csv(index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting simulation data: {str(e)}")
            return f"Error: {str(e)}"

# Convenience functions for easy integration
def simulate_consumer_response(stimulus: str, **kwargs) -> SimulationResult:
    """Quick consumer response simulation"""
    brain_twin = DigitalBrainTwin()
    return brain_twin.simulate_marketing_response(stimulus, **kwargs)

def get_neural_insights(stimulus: str, consumer_type: str = "average") -> Dict[str, float]:
    """Get neural insights from marketing stimulus"""
    result = simulate_consumer_response(stimulus, consumer_type=consumer_type)
    return result.behavioral_predictions

def start_real_time_monitoring(duration: float = 60.0) -> Dict[str, Any]:
    """Start real-time neural monitoring session"""
    brain_twin = DigitalBrainTwin()
    return brain_twin.simulate_real_time_monitoring(duration)

# Example usage and testing
if __name__ == "__main__":
    # Test the neural simulation engine
    brain_twin = DigitalBrainTwin()
    
    test_stimuli = [
        "Amazing new product! Limited time offer - buy now and save 50%!",
        "Our premium quality service ensures customer satisfaction.",
        "Join thousands of happy customers who trust our brand.",
        "Luxury redefined. Experience the difference.",
        "Don't miss out! Only 24 hours left to get this exclusive deal!"
    ]
    
    print("Neural Simulation Engine Results:")
    print("=" * 50)
    
    for i, stimulus in enumerate(test_stimuli, 1):
        print(f"\nStimulus {i}: {stimulus}")
        
        # Test different consumer types
        for consumer_type in ['average', 'impulsive', 'analytical']:
            result = brain_twin.simulate_marketing_response(
                stimulus, consumer_type=consumer_type, duration=10.0
            )
            
            print(f"\nConsumer Type: {consumer_type}")
            print(f"Behavioral Predictions:")
            for prediction, value in result.behavioral_predictions.items():
                print(f"  {prediction}: {value:.3f}")
            
            print(f"Summary Metrics:")
            print(f"  Signal Quality: {result.summary_metrics['signal_quality']:.3f}")
            print(f"  Overall Activation: {result.summary_metrics['overall_activation']:.3f}")
            print(f"  Behavioral Response: {result.summary_metrics['behavioral_response_strength']:.3f}")
    
    # Test real-time monitoring
    print(f"\n\nReal-time Monitoring Test:")
    print("=" * 30)
    
    monitoring_session = brain_twin.simulate_real_time_monitoring(duration=10.0)
    summary = monitoring_session['summary']
    
    print(f"Session Duration: {summary['session_duration']} seconds")
    print(f"Average Attention: {summary['avg_attention']:.3f}")
    print(f"Average Engagement: {summary['avg_engagement']:.3f}")
    print(f"Average Stress: {summary['avg_stress']:.3f}")
    print(f"Signal Quality: {summary['signal_quality']:.3f}")
    print(f"Total Alerts: {summary['total_alerts']}")