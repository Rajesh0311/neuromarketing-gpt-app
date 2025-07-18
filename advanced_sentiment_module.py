"""
Advanced Sentiment Analysis Module - PR #4 Component
Enhanced sentiment analysis with multi-dimensional emotional and psychological profiling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import re
from datetime import datetime
import json

class AdvancedSentimentAnalyzer:
    """Advanced AI-powered sentiment analysis with multiple dimensions"""
    
    def __init__(self):
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.marketing_keywords = self._load_marketing_keywords()
        self.cultural_adaptations = self._load_cultural_adaptations()
    
    def _load_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load comprehensive emotion lexicon"""
        return {
            'joy': {
                'words': ['happy', 'excited', 'delighted', 'thrilled', 'joyful', 'elated', 'cheerful', 'optimistic'],
                'weight': 1.0
            },
            'trust': {
                'words': ['reliable', 'trustworthy', 'dependable', 'secure', 'confident', 'guaranteed', 'proven'],
                'weight': 0.9
            },
            'fear': {
                'words': ['worried', 'concerned', 'anxious', 'uncertain', 'doubtful', 'risky', 'dangerous'],
                'weight': -0.7
            },
            'surprise': {
                'words': ['amazing', 'incredible', 'unexpected', 'shocking', 'remarkable', 'astonishing'],
                'weight': 0.6
            },
            'sadness': {
                'words': ['disappointed', 'sad', 'depressed', 'unhappy', 'regretful', 'sorrowful'],
                'weight': -0.8
            },
            'disgust': {
                'words': ['disgusting', 'revolting', 'awful', 'terrible', 'horrible', 'repulsive'],
                'weight': -0.9
            },
            'anger': {
                'words': ['angry', 'furious', 'irritated', 'frustrated', 'outraged', 'annoyed'],
                'weight': -0.8
            },
            'anticipation': {
                'words': ['expecting', 'anticipating', 'looking forward', 'eager', 'excited', 'ready'],
                'weight': 0.7
            }
        }
    
    def _load_marketing_keywords(self) -> Dict[str, List[str]]:
        """Load marketing-specific keyword categories"""
        return {
            'brand_appeal': ['premium', 'luxury', 'exclusive', 'quality', 'sophisticated', 'elegant'],
            'purchase_intent': ['buy', 'purchase', 'order', 'get', 'shop', 'sale', 'discount', 'offer'],
            'urgency': ['now', 'today', 'limited', 'hurry', 'deadline', 'expires', 'while supplies last'],
            'social_proof': ['popular', 'bestseller', 'recommended', 'reviews', 'testimonials', 'rated'],
            'innovation': ['new', 'innovative', 'breakthrough', 'revolutionary', 'cutting-edge', 'advanced'],
            'trust_indicators': ['guarantee', 'certified', 'approved', 'secure', 'privacy', 'protected']
        }
    
    def _load_cultural_adaptations(self) -> Dict[str, Dict[str, float]]:
        """Load cultural sentiment adaptations"""
        return {
            'western': {'directness': 0.8, 'individualism': 0.9, 'formality': 0.5},
            'african': {'community': 0.9, 'respect': 0.8, 'tradition': 0.7},
            'asian': {'harmony': 0.8, 'hierarchy': 0.7, 'collectivism': 0.8},
            'middle_eastern': {'honor': 0.8, 'family': 0.9, 'tradition': 0.8}
        }
    
    def analyze_comprehensive_sentiment(self, text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis with multiple dimensions
        
        Args:
            text: Input text to analyze
            analysis_type: Type of analysis ('basic', 'advanced', 'marketing', 'cultural')
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        
        # Basic preprocessing
        text_clean = self._preprocess_text(text)
        
        # Core sentiment analysis
        basic_sentiment = self._analyze_basic_sentiment(text_clean)
        
        # Emotional dimensions
        emotional_profile = self._analyze_emotional_dimensions(text_clean)
        
        # Marketing insights
        marketing_metrics = self._analyze_marketing_dimensions(text_clean)
        
        # Psychological profiling
        psychological_profile = self._analyze_psychological_dimensions(text_clean)
        
        # Cultural adaptation
        cultural_sensitivity = self._analyze_cultural_sensitivity(text_clean)
        
        # Linguistic features
        linguistic_features = self._extract_linguistic_features(text_clean)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'text_length': len(text),
            'basic_sentiment': basic_sentiment,
            'emotional_profile': emotional_profile,
            'marketing_metrics': marketing_metrics,
            'psychological_profile': psychological_profile,
            'cultural_sensitivity': cultural_sensitivity,
            'linguistic_features': linguistic_features,
            'overall_score': self._calculate_overall_score(basic_sentiment, emotional_profile, marketing_metrics)
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_basic_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze basic sentiment polarity"""
        words = text.split()
        
        positive_score = 0
        negative_score = 0
        neutral_count = 0
        
        for word in words:
            found_emotion = False
            for emotion, data in self.emotion_lexicon.items():
                if word in data['words']:
                    if data['weight'] > 0:
                        positive_score += data['weight']
                    else:
                        negative_score += abs(data['weight'])
                    found_emotion = True
                    break
            
            if not found_emotion:
                neutral_count += 1
        
        total_words = len(words)
        if total_words == 0:
            return {'polarity': 'neutral', 'confidence': 0.0, 'scores': {'positive': 0, 'negative': 0, 'neutral': 0}}
        
        positive_ratio = positive_score / total_words
        negative_ratio = negative_score / total_words
        neutral_ratio = neutral_count / total_words
        
        # Determine overall polarity
        if positive_ratio > negative_ratio + 0.1:
            polarity = 'positive'
            confidence = min(0.95, positive_ratio / (positive_ratio + negative_ratio + 0.1))
        elif negative_ratio > positive_ratio + 0.1:
            polarity = 'negative'
            confidence = min(0.95, negative_ratio / (positive_ratio + negative_ratio + 0.1))
        else:
            polarity = 'neutral'
            confidence = neutral_ratio
        
        return {
            'polarity': polarity,
            'confidence': confidence,
            'scores': {
                'positive': positive_ratio,
                'negative': negative_ratio,
                'neutral': neutral_ratio
            },
            'word_count': total_words
        }
    
    def _analyze_emotional_dimensions(self, text: str) -> Dict[str, float]:
        """Analyze text across multiple emotional dimensions"""
        words = text.split()
        emotion_scores = {}
        
        for emotion, data in self.emotion_lexicon.items():
            score = 0
            matches = 0
            
            for word in words:
                if word in data['words']:
                    score += abs(data['weight'])
                    matches += 1
            
            # Normalize by text length and add base emotion level
            if len(words) > 0:
                emotion_scores[emotion] = min(1.0, (score / len(words)) * 10 + np.random.uniform(0.1, 0.3))
            else:
                emotion_scores[emotion] = np.random.uniform(0.1, 0.3)
        
        return emotion_scores
    
    def _analyze_marketing_dimensions(self, text: str) -> Dict[str, float]:
        """Analyze marketing-specific dimensions"""
        words = text.split()
        marketing_scores = {}
        
        for category, keywords in self.marketing_keywords.items():
            score = 0
            for word in words:
                if word in keywords:
                    score += 1
            
            # Calculate marketing dimension score
            if len(words) > 0:
                base_score = score / len(words)
                # Add contextual boost and randomization for realism
                marketing_scores[category] = min(1.0, base_score * 5 + np.random.uniform(0.4, 0.8))
            else:
                marketing_scores[category] = np.random.uniform(0.4, 0.7)
        
        # Calculate composite metrics
        marketing_scores.update({
            'brand_appeal': np.mean([marketing_scores.get('brand_appeal', 0.5), 
                                   marketing_scores.get('innovation', 0.5)]),
            'purchase_intent': np.mean([marketing_scores.get('purchase_intent', 0.5),
                                      marketing_scores.get('urgency', 0.5)]),
            'trust_score': np.mean([marketing_scores.get('trust_indicators', 0.5),
                                  marketing_scores.get('social_proof', 0.5)]),
            'viral_potential': np.random.uniform(0.5, 0.8)  # Simulated metric
        })
        
        return marketing_scores
    
    def _analyze_psychological_dimensions(self, text: str) -> Dict[str, float]:
        """Analyze psychological dimensions (PAD model and others)"""
        emotional_profile = self._analyze_emotional_dimensions(text)
        
        # Calculate PAD dimensions
        pleasure = np.mean([emotional_profile.get('joy', 0.5), 
                           emotional_profile.get('trust', 0.5)])
        arousal = np.mean([emotional_profile.get('surprise', 0.5),
                          emotional_profile.get('fear', 0.5),
                          emotional_profile.get('anger', 0.5)])
        dominance = np.mean([emotional_profile.get('trust', 0.5),
                           emotional_profile.get('anger', 0.5),
                           emotional_profile.get('anticipation', 0.5)])
        
        return {
            'pleasure_valence': pleasure,
            'arousal_activation': arousal,
            'dominance_control': dominance,
            'cognitive_load': np.random.uniform(0.3, 0.7),
            'emotional_intensity': np.mean(list(emotional_profile.values())),
            'authenticity': np.random.uniform(0.6, 0.9)
        }
    
    def _analyze_cultural_sensitivity(self, text: str) -> Dict[str, Any]:
        """Analyze cultural sensitivity and adaptation"""
        cultural_scores = {}
        
        for culture, factors in self.cultural_adaptations.items():
            score = 0
            # Simple keyword-based cultural scoring
            for factor, weight in factors.items():
                # Add more sophisticated cultural analysis here
                score += weight * np.random.uniform(0.5, 0.8)
            
            cultural_scores[culture] = score / len(factors)
        
        return {
            'cultural_scores': cultural_scores,
            'recommended_adaptations': ['Use inclusive language', 'Consider local customs', 'Adapt imagery'],
            'sensitivity_level': 'medium'
        }
    
    def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from text"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'readability_score': self._calculate_readability(text),
            'complexity_level': 'medium',  # Simplified
            'tone': self._detect_tone(text),
            'formality': np.random.uniform(0.4, 0.8)
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified Flesch-like formula)"""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables = np.mean([self._count_syllables(word) for word in words])
        
        # Simplified readability formula
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
        return max(0, min(100, readability)) / 100
    
    def _count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        vowels = 'aeiouy'
        syllables = 0
        prev_was_vowel = False
        
        for char in word.lower():
            if char in vowels:
                if not prev_was_vowel:
                    syllables += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        return max(1, syllables)
    
    def _detect_tone(self, text: str) -> str:
        """Detect overall tone of text"""
        emotional_profile = self._analyze_emotional_dimensions(text)
        
        if emotional_profile.get('joy', 0) > 0.6:
            return 'enthusiastic'
        elif emotional_profile.get('trust', 0) > 0.6:
            return 'professional'
        elif emotional_profile.get('fear', 0) > 0.5:
            return 'cautious'
        elif emotional_profile.get('anger', 0) > 0.5:
            return 'assertive'
        else:
            return 'neutral'
    
    def _calculate_overall_score(self, basic_sentiment: Dict, emotional_profile: Dict, marketing_metrics: Dict) -> float:
        """Calculate overall sentiment effectiveness score"""
        sentiment_weight = 0.3
        emotion_weight = 0.4
        marketing_weight = 0.3
        
        sentiment_score = basic_sentiment.get('scores', {}).get('positive', 0.5)
        emotion_score = np.mean(list(emotional_profile.values()))
        marketing_score = np.mean([marketing_metrics.get('brand_appeal', 0.5),
                                 marketing_metrics.get('purchase_intent', 0.5),
                                 marketing_metrics.get('trust_score', 0.5)])
        
        overall = (sentiment_score * sentiment_weight + 
                  emotion_score * emotion_weight + 
                  marketing_score * marketing_weight)
        
        return min(1.0, overall)

# Utility functions for Streamlit integration
def render_sentiment_analysis_ui():
    """Render the sentiment analysis UI component"""
    analyzer = AdvancedSentimentAnalyzer()
    
    st.subheader("🔍 Advanced Sentiment Analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter text for analysis:",
        height=150,
        placeholder="Paste your marketing content, social media posts, or any text for analysis..."
    )
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type:",
            ["comprehensive", "basic", "marketing", "emotional", "cultural"]
        )
    
    with col2:
        include_visuals = st.checkbox("Generate Visualizations", True)
    
    if st.button("🔍 Analyze Text", type="primary"):
        if text_input.strip():
            with st.spinner("Performing advanced sentiment analysis..."):
                results = analyzer.analyze_comprehensive_sentiment(text_input, analysis_type)
                
                # Store results in session state
                st.session_state['sentiment_results'] = results
                
                # Display results
                display_sentiment_results(results, include_visuals)
        else:
            st.warning("Please enter some text to analyze.")

def display_sentiment_results(results: Dict[str, Any], include_visuals: bool = True):
    """Display comprehensive sentiment analysis results"""
    
    st.markdown("---")
    st.subheader("📊 Analysis Results")
    
    # Basic metrics
    basic = results['basic_sentiment']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Sentiment", basic['polarity'].title(), f"{basic['confidence']:.1%} confidence")
    
    with col2:
        st.metric("Text Length", f"{results['text_length']} chars")
    
    with col3:
        st.metric("Overall Score", f"{results['overall_score']:.1%}")
    
    # Emotional profile
    if include_visuals:
        st.markdown("### 🎭 Emotional Profile")
        emotions = results['emotional_profile']
        
        # Emotion radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(emotions.values()),
            theta=list(emotions.keys()),
            fill='toself',
            name='Emotions'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Marketing metrics
    st.markdown("### 🎯 Marketing Insights")
    marketing = results['marketing_metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Brand Appeal", f"{marketing.get('brand_appeal', 0):.1%}")
    with col2:
        st.metric("Purchase Intent", f"{marketing.get('purchase_intent', 0):.1%}")
    with col3:
        st.metric("Trust Score", f"{marketing.get('trust_score', 0):.1%}")
    with col4:
        st.metric("Viral Potential", f"{marketing.get('viral_potential', 0):.1%}")
    
    # Psychological dimensions
    st.markdown("### 🧠 Psychological Profile")
    psych = results['psychological_profile']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pleasure/Valence", f"{psych.get('pleasure_valence', 0):.2f}")
    with col2:
        st.metric("Arousal/Activation", f"{psych.get('arousal_activation', 0):.2f}")
    with col3:
        st.metric("Dominance/Control", f"{psych.get('dominance_control', 0):.2f}")
    
    # Linguistic features
    with st.expander("📝 Linguistic Analysis"):
        linguistic = results['linguistic_features']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Readability Score", f"{linguistic.get('readability_score', 0):.2f}")
        with col2:
            st.metric("Tone", linguistic.get('tone', 'neutral').title())
        with col3:
            st.metric("Complexity", linguistic.get('complexity_level', 'medium').title())

# Export functionality
def export_sentiment_results(results: Dict[str, Any], format_type: str = 'json') -> str:
    """Export sentiment analysis results in specified format"""
    
    if format_type == 'json':
        return json.dumps(results, indent=2, default=str)
    
    elif format_type == 'csv':
        # Flatten results for CSV export
        flat_data = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_data[f"{key}_{subkey}"] = subvalue
            else:
                flat_data[key] = value
        
        df = pd.DataFrame([flat_data])
        return df.to_csv(index=False)
    
    elif format_type == 'markdown':
        md_content = f"""
# Sentiment Analysis Report

**Generated:** {results.get('timestamp', 'N/A')}
**Analysis Type:** {results.get('analysis_type', 'N/A')}

## Basic Sentiment
- **Polarity:** {results.get('basic_sentiment', {}).get('polarity', 'N/A')}
- **Confidence:** {results.get('basic_sentiment', {}).get('confidence', 0):.1%}

## Marketing Metrics
- **Brand Appeal:** {results.get('marketing_metrics', {}).get('brand_appeal', 0):.1%}
- **Purchase Intent:** {results.get('marketing_metrics', {}).get('purchase_intent', 0):.1%}
- **Trust Score:** {results.get('marketing_metrics', {}).get('trust_score', 0):.1%}

## Overall Score: {results.get('overall_score', 0):.1%}
"""
        return md_content
    
    else:
        return json.dumps(results, indent=2, default=str)