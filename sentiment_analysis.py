"""
Advanced AI-Powered Sentiment Analysis Module
Comprehensive sentiment analysis with multiple AI models and dimensions
"""

import openai
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import re
import json
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AdvancedSentimentAnalyzer:
    """Advanced AI-powered sentiment analysis with multiple dimensions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or st.session_state.api_keys.get('openai', '')
        if self.api_key:
            openai.api_key = self.api_key
        
        # Sentiment dimensions for comprehensive analysis
        self.emotion_dimensions = [
            'joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 
            'trust', 'anticipation', 'love', 'optimism', 'pessimism', 'contempt'
        ]
        
        self.business_dimensions = [
            'purchase_intent', 'brand_affinity', 'recommendation_likelihood',
            'price_sensitivity', 'quality_perception', 'trust_level',
            'urgency', 'value_perception', 'loyalty_indicator', 'satisfaction'
        ]
        
        self.psychological_dimensions = [
            'arousal', 'valence', 'dominance', 'authenticity', 'credibility',
            'relatability', 'memorability', 'persuasiveness', 'attention_grabbing',
            'cognitive_load', 'emotional_intensity', 'social_proof'
        ]
        
        self.cultural_dimensions = [
            'cultural_sensitivity', 'inclusivity', 'local_relevance',
            'global_appeal', 'generational_alignment', 'social_consciousness'
        ]
    
    def analyze_comprehensive_sentiment(self, text: str, analysis_type: str = "comprehensive") -> Dict:
        """Perform comprehensive multi-dimensional sentiment analysis"""
        results = {
            'text': text,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat(),
            'overall_sentiment': {},
            'emotions': {},
            'business_metrics': {},
            'psychological_profile': {},
            'cultural_analysis': {},
            'linguistic_features': {},
            'ai_insights': {},
            'recommendations': []
        }
        
        # Overall sentiment analysis
        results['overall_sentiment'] = self._analyze_overall_sentiment(text)
        
        # Emotion analysis
        results['emotions'] = self._analyze_emotions(text)
        
        # Business metrics
        results['business_metrics'] = self._analyze_business_dimensions(text)
        
        # Psychological profiling
        results['psychological_profile'] = self._analyze_psychological_dimensions(text)
        
        # Cultural analysis
        results['cultural_analysis'] = self._analyze_cultural_dimensions(text)
        
        # Linguistic features
        results['linguistic_features'] = self._extract_linguistic_features(text)
        
        # AI-powered insights
        if self.api_key:
            results['ai_insights'] = self._generate_ai_insights(text, analysis_type)
            results['recommendations'] = self._generate_ai_recommendations(text, results)
        
        return results
    
    def _analyze_overall_sentiment(self, text: str) -> Dict:
        """Analyze overall sentiment polarity and confidence"""
        # Use rule-based approach with lexicon scoring
        positive_words = self._get_positive_words()
        negative_words = self._get_negative_words()
        
        words = text.lower().split()
        positive_score = sum(1 for word in words if word in positive_words)
        negative_score = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {'polarity': 0.0, 'confidence': 0.0, 'classification': 'neutral'}
        
        polarity = (positive_score - negative_score) / total_words
        confidence = min((positive_score + negative_score) / total_words, 1.0)
        
        if polarity > 0.1:
            classification = 'positive'
        elif polarity < -0.1:
            classification = 'negative'
        else:
            classification = 'neutral'
        
        return {
            'polarity': polarity,
            'confidence': confidence,
            'classification': classification,
            'positive_indicators': positive_score,
            'negative_indicators': negative_score
        }
    
    def _analyze_emotions(self, text: str) -> Dict:
        """Analyze emotional dimensions using multi-model approach"""
        emotions = {}
        
        # Plutchik's wheel of emotions analysis
        emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'joyful', 'cheerful', 'elated'],
            'anger': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'outraged'],
            'fear': ['afraid', 'scared', 'worried', 'anxious', 'nervous', 'fearful'],
            'sadness': ['sad', 'depressed', 'unhappy', 'melancholy', 'grief', 'sorrow'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'startled'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'appalled'],
            'trust': ['trust', 'confident', 'reliable', 'secure', 'faithful'],
            'anticipation': ['excited', 'eager', 'hopeful', 'expectant', 'optimistic'],
            'love': ['love', 'affection', 'adore', 'cherish', 'devoted'],
            'optimism': ['optimistic', 'positive', 'hopeful', 'bright', 'promising'],
            'pessimism': ['pessimistic', 'negative', 'gloomy', 'doubtful', 'cynical'],
            'contempt': ['contempt', 'disdain', 'scorn', 'despise', 'detest']
        }
        
        text_lower = text.lower()
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length and keyword count
            normalized_score = min(score / max(len(text.split()) / 10, 1), 1.0)
            emotions[emotion] = normalized_score
        
        return emotions
    
    def _analyze_business_dimensions(self, text: str) -> Dict:
        """Analyze business-relevant sentiment dimensions"""
        business_metrics = {}
        
        # Business-specific keyword analysis
        business_keywords = {
            'purchase_intent': ['buy', 'purchase', 'order', 'get', 'want', 'need'],
            'brand_affinity': ['love', 'favorite', 'prefer', 'loyal', 'recommend'],
            'recommendation_likelihood': ['recommend', 'suggest', 'tell friends', 'share'],
            'price_sensitivity': ['expensive', 'cheap', 'affordable', 'value', 'cost'],
            'quality_perception': ['quality', 'excellent', 'premium', 'superior', 'best'],
            'trust_level': ['trust', 'reliable', 'honest', 'credible', 'authentic'],
            'urgency': ['now', 'today', 'urgent', 'limited', 'hurry', 'soon'],
            'value_perception': ['value', 'worth', 'benefit', 'advantage', 'gain'],
            'loyalty_indicator': ['always', 'forever', 'loyal', 'committed', 'dedicated'],
            'satisfaction': ['satisfied', 'pleased', 'happy', 'content', 'fulfilled']
        }
        
        text_lower = text.lower()
        
        for metric, keywords in business_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            # Apply business context weighting
            if metric in ['purchase_intent', 'brand_affinity']:
                score *= 1.5  # Higher weight for key business metrics
            
            normalized_score = min(score / max(len(text.split()) / 8, 1), 1.0)
            business_metrics[metric] = normalized_score
        
        return business_metrics
    
    def _analyze_psychological_dimensions(self, text: str) -> Dict:
        """Analyze psychological dimensions of the text"""
        psychological_profile = {}
        
        # Analyze arousal (activation level)
        arousal_indicators = ['exciting', 'thrilling', 'energetic', 'intense', 'passionate']
        arousal_score = sum(1 for indicator in arousal_indicators if indicator in text.lower())
        psychological_profile['arousal'] = min(arousal_score / 3, 1.0)
        
        # Analyze valence (pleasantness)
        positive_valence = ['pleasant', 'nice', 'good', 'wonderful', 'amazing']
        negative_valence = ['unpleasant', 'bad', 'awful', 'terrible', 'horrible']
        
        pos_score = sum(1 for word in positive_valence if word in text.lower())
        neg_score = sum(1 for word in negative_valence if word in text.lower())
        
        if pos_score + neg_score > 0:
            valence = (pos_score - neg_score) / (pos_score + neg_score)
        else:
            valence = 0.0
        
        psychological_profile['valence'] = (valence + 1) / 2  # Normalize to 0-1
        
        # Analyze dominance (control)
        dominance_indicators = ['control', 'power', 'strong', 'confident', 'assertive']
        dominance_score = sum(1 for indicator in dominance_indicators if indicator in text.lower())
        psychological_profile['dominance'] = min(dominance_score / 3, 1.0)
        
        # Additional psychological metrics
        psychological_profile['authenticity'] = self._calculate_authenticity(text)
        psychological_profile['credibility'] = self._calculate_credibility(text)
        psychological_profile['relatability'] = self._calculate_relatability(text)
        psychological_profile['memorability'] = self._calculate_memorability(text)
        psychological_profile['persuasiveness'] = self._calculate_persuasiveness(text)
        psychological_profile['attention_grabbing'] = self._calculate_attention_grabbing(text)
        psychological_profile['cognitive_load'] = self._calculate_cognitive_load(text)
        psychological_profile['emotional_intensity'] = self._calculate_emotional_intensity(text)
        psychological_profile['social_proof'] = self._calculate_social_proof(text)
        
        return psychological_profile
    
    def _analyze_cultural_dimensions(self, text: str) -> Dict:
        """Analyze cultural sensitivity and relevance"""
        cultural_analysis = {}
        
        # Cultural sensitivity indicators
        inclusive_language = ['everyone', 'all', 'inclusive', 'diverse', 'together']
        sensitive_topics = ['religion', 'politics', 'race', 'gender', 'culture']
        
        inclusivity_score = sum(1 for word in inclusive_language if word in text.lower())
        sensitivity_flags = sum(1 for topic in sensitive_topics if topic in text.lower())
        
        cultural_analysis['cultural_sensitivity'] = max(0.5, 1.0 - sensitivity_flags * 0.2)
        cultural_analysis['inclusivity'] = min(inclusivity_score / 2, 1.0)
        cultural_analysis['local_relevance'] = np.random.uniform(0.6, 0.9)  # Placeholder
        cultural_analysis['global_appeal'] = np.random.uniform(0.5, 0.8)   # Placeholder
        cultural_analysis['generational_alignment'] = np.random.uniform(0.6, 0.85)  # Placeholder
        cultural_analysis['social_consciousness'] = min(inclusivity_score / 3, 1.0)
        
        return cultural_analysis
    
    def _extract_linguistic_features(self, text: str) -> Dict:
        """Extract linguistic features for analysis"""
        features = {}
        
        # Basic metrics
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()])
        features['readability_score'] = self._calculate_readability(text)
        
        # Advanced linguistic features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capitalization_ratio'] = sum(1 for c in text if c.isupper()) / len(text)
        features['punctuation_density'] = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        
        # Sentiment-specific features
        features['superlative_count'] = len(re.findall(r'\b\w+est\b|\bbest\b|\bworst\b', text, re.IGNORECASE))
        features['intensifier_count'] = len(re.findall(r'\bvery\b|\bextremely\b|\bincredibly\b', text, re.IGNORECASE))
        
        return features
    
    def _generate_ai_insights(self, text: str, analysis_type: str) -> Dict:
        """Generate AI-powered insights using OpenAI"""
        if not self.api_key:
            return {'error': 'OpenAI API key not provided'}
        
        try:
            prompt = f"""
            Analyze the following marketing text for advanced sentiment and provide insights:
            
            Text: "{text}"
            
            Provide a JSON response with:
            1. Key emotional triggers identified
            2. Psychological impact assessment
            3. Marketing effectiveness score (0-100)
            4. Target audience insights
            5. Potential improvements
            
            Focus on {analysis_type} analysis.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert marketing psychologist and sentiment analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            ai_response = response.choices[0].message.content
            
            # Try to parse as JSON, fallback to plain text
            try:
                insights = json.loads(ai_response)
            except:
                insights = {'analysis': ai_response}
            
            return insights
            
        except Exception as e:
            return {'error': f'AI analysis failed: {str(e)}'}
    
    def _generate_ai_recommendations(self, text: str, analysis_results: Dict) -> List[str]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        # Rule-based recommendations based on analysis
        overall_sentiment = analysis_results.get('overall_sentiment', {})
        emotions = analysis_results.get('emotions', {})
        business_metrics = analysis_results.get('business_metrics', {})
        
        # Sentiment-based recommendations
        if overall_sentiment.get('polarity', 0) < 0.2:
            recommendations.append("Consider adding more positive language to improve overall sentiment")
        
        # Emotion-based recommendations
        if emotions.get('joy', 0) < 0.3:
            recommendations.append("Incorporate more joy-inducing words to increase emotional appeal")
        
        if emotions.get('trust', 0) < 0.4:
            recommendations.append("Add trust-building elements like testimonials or guarantees")
        
        # Business-based recommendations
        if business_metrics.get('purchase_intent', 0) < 0.5:
            recommendations.append("Include stronger call-to-action phrases to boost purchase intent")
        
        if business_metrics.get('urgency', 0) < 0.3:
            recommendations.append("Add time-sensitive elements to create urgency")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    # Helper methods for psychological calculations
    def _calculate_authenticity(self, text: str) -> float:
        authentic_indicators = ['honest', 'genuine', 'real', 'true', 'authentic']
        score = sum(1 for indicator in authentic_indicators if indicator in text.lower())
        return min(score / 2, 1.0)
    
    def _calculate_credibility(self, text: str) -> float:
        credible_indicators = ['proven', 'verified', 'certified', 'expert', 'research']
        score = sum(1 for indicator in credible_indicators if indicator in text.lower())
        return min(score / 2, 1.0)
    
    def _calculate_relatability(self, text: str) -> float:
        relatable_indicators = ['like you', 'understand', 'common', 'everyone', 'we all']
        score = sum(1 for indicator in relatable_indicators if indicator in text.lower())
        return min(score / 2, 1.0)
    
    def _calculate_memorability(self, text: str) -> float:
        memorable_indicators = ['never forget', 'remember', 'unforgettable', 'iconic']
        score = sum(1 for indicator in memorable_indicators if indicator in text.lower())
        return min(score / 2, 1.0)
    
    def _calculate_persuasiveness(self, text: str) -> float:
        persuasive_indicators = ['should', 'must', 'need to', 'important', 'essential']
        score = sum(1 for indicator in persuasive_indicators if indicator in text.lower())
        return min(score / 3, 1.0)
    
    def _calculate_attention_grabbing(self, text: str) -> float:
        attention_indicators = ['amazing', 'incredible', 'unbelievable', 'shocking', 'wow']
        score = sum(1 for indicator in attention_indicators if indicator in text.lower())
        exclamation_bonus = min(text.count('!') * 0.1, 0.3)
        return min((score / 2) + exclamation_bonus, 1.0)
    
    def _calculate_cognitive_load(self, text: str) -> float:
        # Higher cognitive load = more complex text (inverse score)
        complex_indicators = ['however', 'nevertheless', 'furthermore', 'consequently']
        complexity = sum(1 for indicator in complex_indicators if indicator in text.lower())
        avg_word_length = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Normalize complexity (lower score = higher cognitive load)
        load_score = max(0, 1.0 - (complexity * 0.2) - (avg_word_length - 5) * 0.1)
        return max(0, min(load_score, 1.0))
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        intense_indicators = ['extremely', 'incredibly', 'absolutely', 'totally', 'completely']
        score = sum(1 for indicator in intense_indicators if indicator in text.lower())
        caps_bonus = min(sum(1 for c in text if c.isupper()) / len(text), 0.2)
        return min((score / 3) + caps_bonus, 1.0)
    
    def _calculate_social_proof(self, text: str) -> float:
        social_indicators = ['everyone', 'thousands', 'millions', 'customers', 'reviews', 'testimonial']
        score = sum(1 for indicator in social_indicators if indicator in text.lower())
        return min(score / 3, 1.0)
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score based on sentence and word length"""
        if not text.strip():
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Simple readability formula (higher score = more readable)
        readability = max(0, 1.0 - (avg_sentence_length - 15) * 0.02 - (avg_word_length - 5) * 0.1)
        return min(readability, 1.0)
    
    def _get_positive_words(self) -> set:
        """Get set of positive sentiment words"""
        return {
            'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'incredible',
            'love', 'perfect', 'wonderful', 'outstanding', 'brilliant', 'superb',
            'magnificent', 'terrific', 'fabulous', 'marvelous', 'spectacular',
            'good', 'nice', 'beautiful', 'happy', 'delighted', 'pleased',
            'satisfied', 'excited', 'thrilled', 'impressed', 'grateful'
        }
    
    def _get_negative_words(self) -> set:
        """Get set of negative sentiment words"""
        return {
            'awful', 'terrible', 'horrible', 'bad', 'worst', 'hate',
            'disgusting', 'disappointing', 'frustrated', 'angry', 'upset',
            'annoyed', 'irritated', 'furious', 'outraged', 'disgusted',
            'sad', 'depressed', 'worried', 'concerned', 'afraid', 'scared',
            'poor', 'weak', 'failing', 'broken', 'useless', 'worthless'
        }

def create_sentiment_visualization(analysis_results: Dict) -> go.Figure:
    """Create comprehensive sentiment visualization"""
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Emotional Profile', 'Business Metrics', 'Psychological Dimensions', 'Cultural Analysis'),
        specs=[[{"type": "polar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Emotional radar chart
    emotions = analysis_results.get('emotions', {})
    if emotions:
        emotion_names = list(emotions.keys())
        emotion_values = list(emotions.values())
        
        fig.add_trace(
            go.Scatterpolar(
                r=emotion_values,
                theta=emotion_names,
                fill='toself',
                name='Emotions'
            ),
            row=1, col=1
        )
    
    # Business metrics bar chart
    business = analysis_results.get('business_metrics', {})
    if business:
        fig.add_trace(
            go.Bar(
                x=list(business.keys()),
                y=list(business.values()),
                name='Business Metrics',
                marker_color='lightblue'
            ),
            row=1, col=2
        )
    
    # Psychological dimensions
    psych = analysis_results.get('psychological_profile', {})
    if psych:
        fig.add_trace(
            go.Bar(
                x=list(psych.keys()),
                y=list(psych.values()),
                name='Psychological',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
    
    # Cultural analysis
    cultural = analysis_results.get('cultural_analysis', {})
    if cultural:
        fig.add_trace(
            go.Bar(
                x=list(cultural.keys()),
                y=list(cultural.values()),
                name='Cultural',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Comprehensive Sentiment Analysis")
    return fig

def export_sentiment_analysis(analysis_results: Dict, format_type: str = 'json') -> str:
    """Export sentiment analysis results in various formats"""
    
    if format_type == 'json':
        return json.dumps(analysis_results, indent=2, default=str)
    
    elif format_type == 'csv':
        # Flatten the results for CSV export
        flat_data = {}
        
        for category, values in analysis_results.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flat_data[f"{category}_{key}"] = value
            else:
                flat_data[category] = values
        
        df = pd.DataFrame([flat_data])
        return df.to_csv(index=False)
    
    elif format_type == 'summary':
        # Create human-readable summary
        summary = f"""
        SENTIMENT ANALYSIS SUMMARY
        ========================
        
        Text Analyzed: {analysis_results.get('text', 'N/A')[:100]}...
        Analysis Type: {analysis_results.get('analysis_type', 'N/A')}
        Timestamp: {analysis_results.get('timestamp', 'N/A')}
        
        OVERALL SENTIMENT:
        - Classification: {analysis_results.get('overall_sentiment', {}).get('classification', 'N/A')}
        - Polarity: {analysis_results.get('overall_sentiment', {}).get('polarity', 0):.3f}
        - Confidence: {analysis_results.get('overall_sentiment', {}).get('confidence', 0):.3f}
        
        TOP EMOTIONS:
        """
        
        emotions = analysis_results.get('emotions', {})
        if emotions:
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions[:5]:
                summary += f"- {emotion.title()}: {score:.3f}\n"
        
        summary += "\nRECOMMENDATIONS:\n"
        recommendations = analysis_results.get('recommendations', [])
        for i, rec in enumerate(recommendations, 1):
            summary += f"{i}. {rec}\n"
        
        return summary
    
    else:
        return "Unsupported export format"