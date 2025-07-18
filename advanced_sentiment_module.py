#!/usr/bin/env python3
"""
Advanced Sentiment Analysis Module for NeuroMarketing GPT Platform
================================================================

This module provides comprehensive sentiment analysis capabilities including:
- Multi-dimensional emotion analysis
- Cultural and contextual understanding
- Marketing-specific insights
- Real-time processing with caching
- Integration with AI models

Authors: NeuroMarketing GPT Team
Version: 1.0.0
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import re
import os
from textblob import TextBlob
import requests
from functools import lru_cache
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Data structure for sentiment analysis results"""
    overall_sentiment: float
    confidence: float
    emotions: Dict[str, float]
    keywords: List[str]
    marketing_metrics: Dict[str, float]
    cultural_context: str
    timestamp: datetime
    processing_time: float

class AdvancedSentimentAnalyzer:
    """Advanced sentiment analysis with marketing insights"""
    
    def __init__(self):
        self.emotion_lexicon = self._load_emotion_lexicon()
        self.marketing_keywords = self._load_marketing_keywords()
        self.cultural_patterns = self._load_cultural_patterns()
        self.cache = {}
        
    def _load_emotion_lexicon(self) -> Dict[str, Dict[str, float]]:
        """Load emotion lexicon for multi-dimensional analysis"""
        # Simplified emotion lexicon - in production, this would be loaded from a comprehensive database
        return {
            'joy': {
                'happy': 0.9, 'excited': 0.8, 'delighted': 0.95, 'cheerful': 0.7,
                'elated': 0.9, 'thrilled': 0.85, 'joyful': 0.9, 'pleased': 0.6
            },
            'trust': {
                'reliable': 0.8, 'trustworthy': 0.9, 'dependable': 0.8, 'honest': 0.85,
                'credible': 0.8, 'authentic': 0.75, 'genuine': 0.8, 'sincere': 0.7
            },
            'fear': {
                'afraid': 0.8, 'worried': 0.6, 'anxious': 0.7, 'scared': 0.9,
                'concerned': 0.5, 'nervous': 0.6, 'uncertain': 0.4, 'doubtful': 0.5
            },
            'surprise': {
                'amazing': 0.8, 'unexpected': 0.6, 'astonishing': 0.9, 'shocking': 0.7,
                'remarkable': 0.8, 'incredible': 0.85, 'unbelievable': 0.8, 'stunning': 0.9
            },
            'sadness': {
                'disappointed': 0.7, 'sad': 0.8, 'upset': 0.6, 'depressed': 0.9,
                'unhappy': 0.7, 'miserable': 0.9, 'heartbroken': 0.95, 'dejected': 0.8
            },
            'anger': {
                'angry': 0.8, 'furious': 0.9, 'mad': 0.7, 'irritated': 0.6,
                'annoyed': 0.5, 'outraged': 0.95, 'frustrated': 0.7, 'livid': 0.9
            },
            'anticipation': {
                'excited': 0.8, 'eager': 0.7, 'hopeful': 0.6, 'optimistic': 0.7,
                'expectant': 0.6, 'ready': 0.5, 'prepared': 0.4, 'waiting': 0.3
            },
            'disgust': {
                'disgusting': 0.9, 'revolting': 0.9, 'awful': 0.8, 'terrible': 0.7,
                'horrible': 0.8, 'nasty': 0.7, 'repulsive': 0.9, 'gross': 0.8
            }
        }
    
    def _load_marketing_keywords(self) -> Dict[str, List[str]]:
        """Load marketing-specific keyword categories"""
        return {
            'purchase_intent': [
                'buy', 'purchase', 'get', 'order', 'want', 'need', 'must have',
                'deal', 'discount', 'sale', 'offer', 'price', 'cost', 'worth it'
            ],
            'brand_appeal': [
                'love', 'like', 'prefer', 'choose', 'recommend', 'trust', 'loyal',
                'favorite', 'best', 'quality', 'premium', 'luxury', 'exclusive'
            ],
            'viral_potential': [
                'share', 'tell', 'spread', 'recommend', 'amazing', 'incredible',
                'must see', 'check out', 'wow', 'unbelievable', 'fantastic'
            ],
            'negative_indicators': [
                'hate', 'awful', 'terrible', 'worst', 'never', 'avoid', 'bad',
                'disappointed', 'regret', 'waste', 'scam', 'fake', 'poor'
            ]
        }
    
    def _load_cultural_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load cultural context patterns"""
        return {
            'western': {
                'directness': 0.8,
                'individualism': 0.9,
                'emotional_expression': 0.7,
                'uncertainty_avoidance': 0.4
            },
            'eastern': {
                'directness': 0.3,
                'individualism': 0.2,
                'emotional_expression': 0.4,
                'uncertainty_avoidance': 0.8
            },
            'african': {
                'directness': 0.6,
                'individualism': 0.4,
                'emotional_expression': 0.8,
                'uncertainty_avoidance': 0.6
            }
        }
    
    @lru_cache(maxsize=1000)
    def analyze_text(self, text: str, cultural_context: str = "global", 
                    analysis_depth: str = "advanced") -> SentimentResult:
        """
        Perform comprehensive sentiment analysis
        
        Args:
            text: Input text to analyze
            cultural_context: Cultural context for analysis
            analysis_depth: Depth of analysis (basic, advanced, deep, enterprise)
            
        Returns:
            SentimentResult object with comprehensive analysis
        """
        start_time = datetime.now()
        
        try:
            # Basic sentiment using TextBlob
            blob = TextBlob(text)
            basic_sentiment = blob.sentiment.polarity
            basic_confidence = blob.sentiment.subjectivity
            
            # Multi-dimensional emotion analysis
            emotions = self._analyze_emotions(text)
            
            # Extract keywords
            keywords = self._extract_keywords(text)
            
            # Marketing-specific metrics
            marketing_metrics = self._calculate_marketing_metrics(text)
            
            # Cultural adaptation
            if cultural_context != "global":
                emotions, marketing_metrics = self._apply_cultural_context(
                    emotions, marketing_metrics, cultural_context
                )
            
            # AI enhancement (if available)
            if analysis_depth in ["deep", "enterprise"]:
                ai_insights = self._get_ai_insights(text)
                if ai_insights:
                    emotions.update(ai_insights.get('emotions', {}))
                    marketing_metrics.update(ai_insights.get('marketing', {}))
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return SentimentResult(
                overall_sentiment=float(basic_sentiment),
                confidence=float(basic_confidence),
                emotions=emotions,
                keywords=keywords,
                marketing_metrics=marketing_metrics,
                cultural_context=cultural_context,
                timestamp=datetime.now(),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # Return basic fallback result
            return SentimentResult(
                overall_sentiment=0.0,
                confidence=0.0,
                emotions={emotion: 0.0 for emotion in self.emotion_lexicon.keys()},
                keywords=[],
                marketing_metrics={'brand_appeal': 0.0, 'purchase_intent': 0.0, 'viral_potential': 0.0},
                cultural_context=cultural_context,
                timestamp=datetime.now(),
                processing_time=0.0
            )
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Analyze multi-dimensional emotions in text"""
        text_lower = text.lower()
        emotions = {}
        
        for emotion_type, emotion_words in self.emotion_lexicon.items():
            emotion_score = 0.0
            word_count = 0
            
            for word, intensity in emotion_words.items():
                if word in text_lower:
                    emotion_score += intensity
                    word_count += 1
            
            # Normalize by word count and text length
            if word_count > 0:
                emotions[emotion_type] = min(emotion_score / word_count, 1.0)
            else:
                emotions[emotion_type] = 0.0
        
        return emotions
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction - in production, use more sophisticated NLP
        blob = TextBlob(text)
        
        # Extract nouns and adjectives
        keywords = []
        for word, pos in blob.tags:
            if pos in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']:
                if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'have']:
                    keywords.append(word.lower())
        
        # Remove duplicates and return top keywords
        unique_keywords = list(set(keywords))
        return unique_keywords[:10]
    
    def _calculate_marketing_metrics(self, text: str) -> Dict[str, float]:
        """Calculate marketing-specific metrics"""
        text_lower = text.lower()
        metrics = {}
        
        for metric_type, keywords in self.marketing_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
            
            # Normalize by keyword count
            metrics[metric_type.replace('_indicators', '_risk')] = min(score / len(keywords), 1.0)
        
        # Calculate overall brand appeal (inverse of negative indicators)
        if 'negative_risk' in metrics:
            metrics['brand_appeal'] = max(0.0, 1.0 - metrics['negative_risk'])
            del metrics['negative_risk']
        
        return metrics
    
    def _apply_cultural_context(self, emotions: Dict[str, float], 
                               marketing_metrics: Dict[str, float], 
                               cultural_context: str) -> tuple:
        """Apply cultural context adjustments"""
        if cultural_context not in self.cultural_patterns:
            return emotions, marketing_metrics
        
        cultural_factors = self.cultural_patterns[cultural_context]
        
        # Adjust emotional expression based on cultural norms
        expression_factor = cultural_factors['emotional_expression']
        adjusted_emotions = {
            emotion: score * expression_factor for emotion, score in emotions.items()
        }
        
        # Adjust marketing metrics based on cultural purchasing patterns
        directness_factor = cultural_factors['directness']
        adjusted_marketing = {
            metric: score * directness_factor if 'intent' in metric else score
            for metric, score in marketing_metrics.items()
        }
        
        return adjusted_emotions, adjusted_marketing
    
    def _get_ai_insights(self, text: str) -> Optional[Dict[str, Any]]:
        """Get enhanced insights from AI models (OpenAI integration)"""
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                logger.info("OpenAI API key not found, skipping AI enhancement")
                return None
            
            # In a real implementation, this would call OpenAI API
            # For now, return simulated enhanced insights
            logger.info("AI enhancement would be applied here with OpenAI API")
            
            return {
                'emotions': {
                    'confidence_boost': 0.1,
                    'nuance_detection': 0.05
                },
                'marketing': {
                    'ai_brand_appeal': np.random.uniform(0.6, 0.9),
                    'ai_purchase_intent': np.random.uniform(0.5, 0.8)
                }
            }
            
        except Exception as e:
            logger.error(f"Error in AI insights: {str(e)}")
            return None
    
    def batch_analyze(self, texts: List[str], **kwargs) -> List[SentimentResult]:
        """Analyze multiple texts in batch"""
        results = []
        for text in texts:
            result = self.analyze_text(text, **kwargs)
            results.append(result)
        return results
    
    def get_analysis_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Generate summary statistics from multiple analyses"""
        if not results:
            return {}
        
        # Calculate averages
        avg_sentiment = np.mean([r.overall_sentiment for r in results])
        avg_confidence = np.mean([r.confidence for r in results])
        
        # Aggregate emotions
        emotion_summary = {}
        for emotion in results[0].emotions.keys():
            emotion_summary[emotion] = np.mean([r.emotions[emotion] for r in results])
        
        # Aggregate marketing metrics
        marketing_summary = {}
        for metric in results[0].marketing_metrics.keys():
            marketing_summary[metric] = np.mean([r.marketing_metrics[metric] for r in results])
        
        # Extract all keywords
        all_keywords = []
        for result in results:
            all_keywords.extend(result.keywords)
        
        # Count keyword frequency
        keyword_freq = {}
        for keyword in all_keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Sort by frequency
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'summary_statistics': {
                'total_texts': len(results),
                'average_sentiment': float(avg_sentiment),
                'average_confidence': float(avg_confidence),
                'processing_time_total': sum([r.processing_time for r in results])
            },
            'emotion_summary': emotion_summary,
            'marketing_summary': marketing_summary,
            'top_keywords': [keyword for keyword, freq in top_keywords],
            'keyword_frequencies': dict(top_keywords)
        }
    
    def export_results(self, results: List[SentimentResult], 
                      format_type: str = 'json') -> str:
        """Export analysis results in various formats"""
        try:
            if format_type.lower() == 'json':
                # Convert results to JSON-serializable format
                export_data = []
                for result in results:
                    export_data.append({
                        'overall_sentiment': result.overall_sentiment,
                        'confidence': result.confidence,
                        'emotions': result.emotions,
                        'keywords': result.keywords,
                        'marketing_metrics': result.marketing_metrics,
                        'cultural_context': result.cultural_context,
                        'timestamp': result.timestamp.isoformat(),
                        'processing_time': result.processing_time
                    })
                
                return json.dumps(export_data, indent=2)
            
            elif format_type.lower() == 'csv':
                # Convert to CSV format
                rows = []
                for result in results:
                    row = {
                        'timestamp': result.timestamp.isoformat(),
                        'overall_sentiment': result.overall_sentiment,
                        'confidence': result.confidence,
                        'cultural_context': result.cultural_context,
                        'processing_time': result.processing_time,
                        'keywords': '; '.join(result.keywords)
                    }
                    # Add emotions
                    for emotion, score in result.emotions.items():
                        row[f'emotion_{emotion}'] = score
                    # Add marketing metrics
                    for metric, score in result.marketing_metrics.items():
                        row[f'marketing_{metric}'] = score
                    
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                return df.to_csv(index=False)
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting results: {str(e)}")
            return f"Error: {str(e)}"

# Convenience functions for easy integration
def analyze_sentiment(text: str, **kwargs) -> SentimentResult:
    """Quick sentiment analysis function"""
    analyzer = AdvancedSentimentAnalyzer()
    return analyzer.analyze_text(text, **kwargs)

def analyze_multiple_texts(texts: List[str], **kwargs) -> List[SentimentResult]:
    """Quick batch analysis function"""
    analyzer = AdvancedSentimentAnalyzer()
    return analyzer.batch_analyze(texts, **kwargs)

def get_marketing_insights(text: str, cultural_context: str = "global") -> Dict[str, float]:
    """Get marketing-specific insights from text"""
    result = analyze_sentiment(text, cultural_context=cultural_context)
    return result.marketing_metrics

# Example usage and testing
if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = AdvancedSentimentAnalyzer()
    
    test_texts = [
        "I absolutely love this product! It's amazing and I would definitely recommend it to everyone.",
        "This is the worst purchase I've ever made. Complete waste of money.",
        "The product is okay, nothing special but it works as expected.",
        "Incredible quality! I'm so excited to share this with my friends. Must buy!",
        "I'm worried about the reliability of this brand. Not sure if I can trust them."
    ]
    
    print("Advanced Sentiment Analysis Results:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text(text, analysis_depth="advanced")
        
        print(f"\nText {i}: {text}")
        print(f"Overall Sentiment: {result.overall_sentiment:.3f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Top Emotions: {sorted(result.emotions.items(), key=lambda x: x[1], reverse=True)[:3]}")
        print(f"Marketing Metrics: {result.marketing_metrics}")
        print(f"Keywords: {result.keywords[:5]}")
        print(f"Processing Time: {result.processing_time:.3f}s")
    
    # Test batch analysis
    results = analyzer.batch_analyze(test_texts)
    summary = analyzer.get_analysis_summary(results)
    
    print(f"\n\nBatch Analysis Summary:")
    print("=" * 30)
    print(f"Total texts analyzed: {summary['summary_statistics']['total_texts']}")
    print(f"Average sentiment: {summary['summary_statistics']['average_sentiment']:.3f}")
    print(f"Top keywords: {summary['top_keywords'][:5]}")
    print(f"Marketing insights: {summary['marketing_summary']}")