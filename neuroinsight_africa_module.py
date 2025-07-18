"""
NeuroInsight-Africa Platform - Regional Intelligence Module
Implements African market neural patterns and cultural adaptation as specified in the problem statement
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Union
import json
from datetime import datetime
import re

class AfricanMarketModels:
    """African Market Neural Patterns - region-specific consumer behavior modeling"""
    
    def __init__(self):
        self.regional_patterns = self._initialize_regional_patterns()
        self.economic_factors = self._initialize_economic_factors()
        self.brand_loyalty_models = self._initialize_brand_loyalty_models()
        
    def _initialize_regional_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize region-specific neural patterns for African markets"""
        return {
            'west_africa': {
                'countries': ['Nigeria', 'Ghana', 'Senegal', 'Mali', 'Burkina Faso', 'Ivory Coast'],
                'primary_languages': ['English', 'French', 'Yoruba', 'Hausa', 'Twi'],
                'neural_patterns': {
                    'community_influence': 0.92,  # Very high community-based decision making
                    'price_sensitivity': 0.88,   # High price consciousness
                    'brand_trust': 0.75,         # Moderate brand trust, building over time
                    'mobile_preference': 0.95,   # Extremely high mobile usage
                    'social_proof_reliance': 0.90,  # Very high reliance on social proof
                    'family_input_weight': 0.85  # High family influence on purchases
                },
                'purchasing_patterns': {
                    'bulk_buying_tendency': 0.78,
                    'seasonal_variation': 0.82,
                    'credit_preference': 0.45,
                    'cash_preference': 0.75
                }
            },
            'east_africa': {
                'countries': ['Kenya', 'Tanzania', 'Uganda', 'Ethiopia', 'Rwanda'],
                'primary_languages': ['English', 'Swahili', 'Amharic', 'Kinyarwanda'],
                'neural_patterns': {
                    'community_influence': 0.89,
                    'price_sensitivity': 0.85,
                    'brand_trust': 0.72,
                    'mobile_preference': 0.93,
                    'social_proof_reliance': 0.87,
                    'family_input_weight': 0.82
                },
                'purchasing_patterns': {
                    'bulk_buying_tendency': 0.75,
                    'seasonal_variation': 0.88,
                    'credit_preference': 0.52,
                    'cash_preference': 0.68
                }
            },
            'southern_africa': {
                'countries': ['South Africa', 'Zimbabwe', 'Botswana', 'Zambia', 'Namibia'],
                'primary_languages': ['English', 'Afrikaans', 'Zulu', 'Xhosa', 'Tswana'],
                'neural_patterns': {
                    'community_influence': 0.85,
                    'price_sensitivity': 0.82,
                    'brand_trust': 0.78,
                    'mobile_preference': 0.91,
                    'social_proof_reliance': 0.83,
                    'family_input_weight': 0.79
                },
                'purchasing_patterns': {
                    'bulk_buying_tendency': 0.72,
                    'seasonal_variation': 0.75,
                    'credit_preference': 0.65,
                    'cash_preference': 0.55
                }
            },
            'north_africa': {
                'countries': ['Egypt', 'Morocco', 'Tunisia', 'Algeria', 'Libya'],
                'primary_languages': ['Arabic', 'French', 'Berber'],
                'neural_patterns': {
                    'community_influence': 0.87,
                    'price_sensitivity': 0.84,
                    'brand_trust': 0.76,
                    'mobile_preference': 0.89,
                    'social_proof_reliance': 0.85,
                    'family_input_weight': 0.88
                },
                'purchasing_patterns': {
                    'bulk_buying_tendency': 0.74,
                    'seasonal_variation': 0.79,
                    'credit_preference': 0.58,
                    'cash_preference': 0.62
                }
            }
        }
    
    def _initialize_economic_factors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regional economic integration factors"""
        return {
            'west_africa': {
                'purchasing_power_index': 0.45,
                'inflation_sensitivity': 0.88,
                'currency_stability': 0.65,
                'economic_growth_impact': 0.72,
                'disposable_income_factor': 0.48
            },
            'east_africa': {
                'purchasing_power_index': 0.42,
                'inflation_sensitivity': 0.85,
                'currency_stability': 0.68,
                'economic_growth_impact': 0.75,
                'disposable_income_factor': 0.46
            },
            'southern_africa': {
                'purchasing_power_index': 0.58,
                'inflation_sensitivity': 0.78,
                'currency_stability': 0.72,
                'economic_growth_impact': 0.68,
                'disposable_income_factor': 0.62
            },
            'north_africa': {
                'purchasing_power_index': 0.52,
                'inflation_sensitivity': 0.82,
                'currency_stability': 0.69,
                'economic_growth_impact': 0.70,
                'disposable_income_factor': 0.55
            }
        }
    
    def _initialize_brand_loyalty_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize African Brand Loyalty Models with community-based and family-influence factors"""
        return {
            'community_based_loyalty': {
                'weight': 0.35,
                'factors': {
                    'local_endorsement': 0.85,
                    'community_leader_influence': 0.82,
                    'peer_recommendations': 0.88,
                    'local_business_support': 0.79
                }
            },
            'family_influence_loyalty': {
                'weight': 0.30,
                'factors': {
                    'elder_approval': 0.83,
                    'family_tradition': 0.87,
                    'generational_preference': 0.75,
                    'family_budget_input': 0.90
                }
            },
            'personal_experience_loyalty': {
                'weight': 0.25,
                'factors': {
                    'product_satisfaction': 0.92,
                    'service_quality': 0.89,
                    'value_for_money': 0.94,
                    'accessibility': 0.86
                }
            },
            'cultural_alignment_loyalty': {
                'weight': 0.10,
                'factors': {
                    'cultural_sensitivity': 0.88,
                    'local_adaptation': 0.85,
                    'traditional_values_respect': 0.91,
                    'modern_integration': 0.72
                }
            }
        }
    
    def analyze_regional_market_patterns(self, region: str, product_category: str = "general") -> Dict[str, Any]:
        """Analyze market patterns for specific African region"""
        if region not in self.regional_patterns:
            return {'error': f'Region {region} not supported'}
        
        pattern = self.regional_patterns[region]
        economic = self.economic_factors[region]
        
        # Calculate market readiness score
        market_readiness = (
            pattern['neural_patterns']['brand_trust'] * 0.3 +
            pattern['neural_patterns']['mobile_preference'] * 0.25 +
            economic['purchasing_power_index'] * 0.25 +
            economic['currency_stability'] * 0.2
        )
        
        # Calculate community influence index
        community_index = (
            pattern['neural_patterns']['community_influence'] * 0.4 +
            pattern['neural_patterns']['social_proof_reliance'] * 0.3 +
            pattern['neural_patterns']['family_input_weight'] * 0.3
        )
        
        return {
            'region': region,
            'market_readiness_score': market_readiness,
            'community_influence_index': community_index,
            'recommended_strategies': self._generate_market_strategies(pattern, economic),
            'neural_patterns': pattern['neural_patterns'],
            'economic_factors': economic,
            'risk_factors': self._identify_risk_factors(pattern, economic)
        }
    
    def _generate_market_strategies(self, pattern: Dict, economic: Dict) -> List[str]:
        """Generate market strategies based on regional patterns"""
        strategies = []
        
        # High community influence strategies
        if pattern['neural_patterns']['community_influence'] > 0.85:
            strategies.append("Implement community ambassador programs")
            strategies.append("Focus on grassroots marketing and local influencers")
        
        # High price sensitivity strategies
        if pattern['neural_patterns']['price_sensitivity'] > 0.85:
            strategies.append("Develop tiered pricing with entry-level options")
            strategies.append("Offer flexible payment plans and micro-financing")
        
        # High mobile preference strategies
        if pattern['neural_patterns']['mobile_preference'] > 0.90:
            strategies.append("Prioritize mobile-first marketing and commerce")
            strategies.append("Integrate mobile payment solutions")
        
        # High family influence strategies
        if pattern['neural_patterns']['family_input_weight'] > 0.80:
            strategies.append("Create family-oriented marketing campaigns")
            strategies.append("Develop products that appeal to multiple generations")
        
        return strategies
    
    def _identify_risk_factors(self, pattern: Dict, economic: Dict) -> List[str]:
        """Identify potential risk factors for market entry"""
        risks = []
        
        if economic['currency_stability'] < 0.70:
            risks.append("Currency volatility may affect pricing stability")
        
        if economic['inflation_sensitivity'] > 0.85:
            risks.append("High inflation sensitivity requires flexible pricing")
        
        if pattern['neural_patterns']['brand_trust'] < 0.75:
            risks.append("Lower brand trust requires longer relationship building")
        
        if economic['purchasing_power_index'] < 0.50:
            risks.append("Limited purchasing power requires value-focused positioning")
        
        return risks

class CulturalSentimentAdapters:
    """Cultural Sentiment Adaptation - local language processing and cultural context algorithms"""
    
    def __init__(self):
        self.language_processors = self._initialize_language_processors()
        self.cultural_contexts = self._initialize_cultural_contexts()
        self.sentiment_adaptations = self._initialize_sentiment_adaptations()
    
    def _initialize_language_processors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize multi-language Africa processing capabilities"""
        return {
            'swahili': {
                'positive_indicators': ['nzuri', 'poa', 'safi', 'vizuri', 'bora', 'furaha'],
                'negative_indicators': ['mbaya', 'vibaya', 'hasira', 'huzuni', 'aibu'],
                'cultural_modifiers': {
                    'respect_terms': ['bwana', 'mama', 'mzee', 'dada'],
                    'community_terms': ['jamii', 'ukoo', 'umoja', 'pamoja'],
                    'trust_terms': ['amini', 'tumaini', 'uhakika']
                },
                'language_confidence': 0.85
            },
            'yoruba': {
                'positive_indicators': ['dara', 'gidi', 'pe', 'yoyi', 'dun'],
                'negative_indicators': ['buru', 'buburu', 'ko dara', 'iku'],
                'cultural_modifiers': {
                    'respect_terms': ['baba', 'mama', 'egbon', 'oko', 'iyawo'],
                    'community_terms': ['ebi', 'agbegbe', 'awujo'],
                    'trust_terms': ['gbagbo', 'gbekele', 'ife']
                },
                'language_confidence': 0.82
            },
            'amharic': {
                'positive_indicators': ['tiru', 'melek', 'des', 'konjo', 'gobez'],
                'negative_indicators': ['metfo', 'ayichilim', 'keftu', 'dehina'],
                'cultural_modifiers': {
                    'respect_terms': ['ato', 'weyzero', 'abat', 'ayzosh'],
                    'community_terms': ['bahil', 'zemed', 'mahber'],
                    'trust_terms': ['amene', 'segenet', 'tikem']
                },
                'language_confidence': 0.78
            },
            'zulu': {
                'positive_indicators': ['kuhle', 'mnandi', 'siyabonga', 'ngiyajabula'],
                'negative_indicators': ['kubi', 'ngiyakuzwa', 'ngicasukile'],
                'cultural_modifiers': {
                    'respect_terms': ['baba', 'mama', 'sawubona', 'ngiyabonga'],
                    'community_terms': ['umndeni', 'isizwe', 'ubuntu'],
                    'trust_terms': ['ukuthemba', 'ukwethemba', 'iqiniso']
                },
                'language_confidence': 0.80
            }
        }
    
    def _initialize_cultural_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cultural context analysis patterns"""
        return {
            'ubuntu_philosophy': {
                'weight': 0.25,
                'indicators': ['together', 'community', 'sharing', 'ubuntu', 'collective'],
                'sentiment_modifier': 1.2,
                'description': 'Ubuntu - I am because we are philosophy'
            },
            'respect_hierarchy': {
                'weight': 0.20,
                'indicators': ['elder', 'respect', 'wisdom', 'authority', 'tradition'],
                'sentiment_modifier': 1.15,
                'description': 'Respect for elders and traditional authority'
            },
            'family_centricity': {
                'weight': 0.22,
                'indicators': ['family', 'children', 'home', 'marriage', 'relatives'],
                'sentiment_modifier': 1.18,
                'description': 'Family as the central unit of society'
            },
            'spiritual_connection': {
                'weight': 0.18,
                'indicators': ['faith', 'prayer', 'blessing', 'spiritual', 'divine'],
                'sentiment_modifier': 1.1,
                'description': 'Strong spiritual and religious connections'
            },
            'economic_resilience': {
                'weight': 0.15,
                'indicators': ['hustle', 'business', 'opportunity', 'survival', 'improvement'],
                'sentiment_modifier': 1.05,
                'description': 'Economic resilience and entrepreneurial spirit'
            }
        }
    
    def _initialize_sentiment_adaptations(self) -> Dict[str, Dict[str, float]]:
        """Initialize sentiment adaptations for African cultural contexts"""
        return {
            'traditional_vs_modern': {
                'traditional_weight': 0.65,
                'modern_weight': 0.35,
                'balance_factor': 0.80  # Preference for balancing both
            },
            'individual_vs_collective': {
                'individual_weight': 0.25,
                'collective_weight': 0.75,
                'community_amplifier': 1.3
            },
            'direct_vs_contextual': {
                'direct_weight': 0.40,
                'contextual_weight': 0.60,
                'politeness_factor': 1.2
            }
        }
    
    def analyze_cultural_sentiment(self, text: str, target_culture: str = "pan_african") -> Dict[str, Any]:
        """Analyze sentiment with African cultural context adaptation"""
        
        # Detect language if possible
        detected_language = self._detect_african_language(text)
        
        # Apply cultural context analysis
        cultural_scores = self._analyze_cultural_contexts(text)
        
        # Apply Ubuntu philosophy weighting
        ubuntu_score = self._calculate_ubuntu_influence(text, cultural_scores)
        
        # Calculate traditional vs modern balance
        traditional_modern_balance = self._analyze_traditional_modern_balance(text)
        
        # Generate cultural adaptation recommendations
        adaptations = self._generate_cultural_adaptations(cultural_scores, traditional_modern_balance)
        
        return {
            'detected_language': detected_language,
            'cultural_context_scores': cultural_scores,
            'ubuntu_philosophy_score': ubuntu_score,
            'traditional_modern_balance': traditional_modern_balance,
            'cultural_sentiment_score': self._calculate_cultural_sentiment(cultural_scores, ubuntu_score),
            'adaptation_recommendations': adaptations,
            'confidence_level': self._calculate_confidence(detected_language, cultural_scores)
        }
    
    def _detect_african_language(self, text: str) -> Dict[str, Any]:
        """Detect African languages in text"""
        text_lower = text.lower()
        language_scores = {}
        
        for language, data in self.language_processors.items():
            score = 0
            word_count = 0
            
            # Check positive indicators
            for word in data['positive_indicators']:
                if word in text_lower:
                    score += 2
                    word_count += 1
            
            # Check negative indicators
            for word in data['negative_indicators']:
                if word in text_lower:
                    score += 2
                    word_count += 1
            
            # Check cultural modifiers
            for category, words in data['cultural_modifiers'].items():
                for word in words:
                    if word in text_lower:
                        score += 1
                        word_count += 1
            
            if word_count > 0:
                language_scores[language] = {
                    'score': score,
                    'confidence': min(score / 10, data['language_confidence']),
                    'word_matches': word_count
                }
        
        # Determine primary language
        if language_scores:
            primary_language = max(language_scores.keys(), key=lambda x: language_scores[x]['confidence'])
            return {
                'primary_language': primary_language,
                'confidence': language_scores[primary_language]['confidence'],
                'all_detected': language_scores
            }
        else:
            return {
                'primary_language': 'english',
                'confidence': 0.5,
                'all_detected': {}
            }
    
    def _analyze_cultural_contexts(self, text: str) -> Dict[str, float]:
        """Analyze cultural context indicators in text"""
        text_lower = text.lower()
        context_scores = {}
        
        for context, data in self.cultural_contexts.items():
            score = 0
            for indicator in data['indicators']:
                if indicator in text_lower:
                    score += data['weight'] * data['sentiment_modifier']
            
            context_scores[context] = min(score, 1.0)  # Cap at 1.0
        
        return context_scores
    
    def _calculate_ubuntu_influence(self, text: str, cultural_scores: Dict[str, float]) -> float:
        """Calculate Ubuntu philosophy influence on sentiment"""
        ubuntu_keywords = ['ubuntu', 'together', 'community', 'we', 'us', 'collective', 'shared']
        text_lower = text.lower()
        
        ubuntu_presence = sum(1 for keyword in ubuntu_keywords if keyword in text_lower)
        community_score = cultural_scores.get('ubuntu_philosophy', 0)
        
        return min((ubuntu_presence * 0.2 + community_score * 0.8), 1.0)
    
    def _analyze_traditional_modern_balance(self, text: str) -> Dict[str, float]:
        """Analyze traditional vs modern orientation"""
        traditional_keywords = ['tradition', 'culture', 'heritage', 'ancestors', 'custom', 'elder', 'wisdom']
        modern_keywords = ['technology', 'innovation', 'new', 'modern', 'digital', 'contemporary', 'progressive']
        
        text_lower = text.lower()
        
        traditional_score = sum(1 for keyword in traditional_keywords if keyword in text_lower) / len(traditional_keywords)
        modern_score = sum(1 for keyword in modern_keywords if keyword in text_lower) / len(modern_keywords)
        
        return {
            'traditional_orientation': min(traditional_score, 1.0),
            'modern_orientation': min(modern_score, 1.0),
            'balance_ratio': traditional_score / (traditional_score + modern_score + 0.01)  # Avoid division by zero
        }
    
    def _calculate_cultural_sentiment(self, cultural_scores: Dict[str, float], ubuntu_score: float) -> float:
        """Calculate overall cultural sentiment score"""
        base_score = sum(cultural_scores.values()) / len(cultural_scores) if cultural_scores else 0.5
        ubuntu_weight = 0.3
        
        return min((base_score * (1 - ubuntu_weight) + ubuntu_score * ubuntu_weight), 1.0)
    
    def _generate_cultural_adaptations(self, cultural_scores: Dict[str, float], traditional_modern: Dict[str, float]) -> List[str]:
        """Generate cultural adaptation recommendations"""
        adaptations = []
        
        # Ubuntu-based adaptations
        if cultural_scores.get('ubuntu_philosophy', 0) > 0.5:
            adaptations.append("Emphasize community benefits and collective value")
            adaptations.append("Use 'we' language instead of 'you' language")
        
        # Respect hierarchy adaptations
        if cultural_scores.get('respect_hierarchy', 0) > 0.5:
            adaptations.append("Include elder endorsements and testimonials")
            adaptations.append("Show respect for traditional authority figures")
        
        # Family-centric adaptations
        if cultural_scores.get('family_centricity', 0) > 0.5:
            adaptations.append("Focus on family benefits and multi-generational value")
            adaptations.append("Include family imagery and scenarios")
        
        # Traditional vs modern balance
        if traditional_modern['balance_ratio'] > 0.6:
            adaptations.append("Emphasize respect for tradition while showing innovation")
            adaptations.append("Bridge traditional values with modern solutions")
        
        return adaptations
    
    def _calculate_confidence(self, language_detection: Dict, cultural_scores: Dict[str, float]) -> float:
        """Calculate overall confidence in cultural analysis"""
        language_confidence = language_detection.get('confidence', 0.5)
        cultural_confidence = sum(cultural_scores.values()) / len(cultural_scores) if cultural_scores else 0.5
        
        return (language_confidence * 0.4 + cultural_confidence * 0.6)

class MultiLanguageAfrica:
    """Multi-language processing for African languages"""
    
    def __init__(self):
        self.supported_languages = ['swahili', 'yoruba', 'amharic', 'zulu', 'hausa', 'igbo']
        self.translation_mappings = self._initialize_translation_mappings()
        
    def _initialize_translation_mappings(self) -> Dict[str, Dict[str, str]]:
        """Initialize basic translation mappings for key marketing terms"""
        return {
            'swahili': {
                'welcome': 'karibu',
                'thank_you': 'asante',
                'good': 'nzuri',
                'bad': 'mbaya',
                'buy': 'nunua',
                'price': 'bei',
                'quality': 'ubora',
                'family': 'familia',
                'community': 'jamii'
            },
            'yoruba': {
                'welcome': 'kaabo',
                'thank_you': 'ese',
                'good': 'dara',
                'bad': 'buburu',
                'buy': 'ra',
                'price': 'owo',
                'quality': 'didara',
                'family': 'ebi',
                'community': 'agbegbe'
            },
            'amharic': {
                'welcome': 'enqan des aleh',
                'thank_you': 'ameseginalehu',
                'good': 'tiru',
                'bad': 'metfo',
                'buy': 'megzat',
                'price': 'waga',
                'quality': 'mekem',
                'family': 'beteseb',
                'community': 'bahil'
            },
            'zulu': {
                'welcome': 'sawubona',
                'thank_you': 'ngiyabonga',
                'good': 'kuhle',
                'bad': 'kubi',
                'buy': 'thenga',
                'price': 'intengo',
                'quality': 'ikhwalithi',
                'family': 'umndeni',
                'community': 'isizwe'
            }
        }
    
    def process_multilingual_content(self, text: str, target_language: str = 'english') -> Dict[str, Any]:
        """Process content for multi-language African markets"""
        
        # Detect current languages
        detected_languages = self._detect_languages(text)
        
        # Generate localized versions
        localized_versions = {}
        for lang in self.supported_languages:
            localized_versions[lang] = self._generate_localized_version(text, lang)
        
        # Cultural appropriateness check
        cultural_appropriateness = self._check_cultural_appropriateness(text)
        
        return {
            'original_text': text,
            'detected_languages': detected_languages,
            'localized_versions': localized_versions,
            'cultural_appropriateness': cultural_appropriateness,
            'recommendations': self._generate_localization_recommendations(text, detected_languages)
        }
    
    def _detect_languages(self, text: str) -> List[str]:
        """Simple language detection for African languages"""
        detected = []
        text_lower = text.lower()
        
        for language, translations in self.translation_mappings.items():
            matches = sum(1 for term in translations.values() if term in text_lower)
            if matches > 0:
                detected.append(language)
        
        return detected if detected else ['english']
    
    def _generate_localized_version(self, text: str, target_language: str) -> str:
        """Generate a culturally localized version of the text"""
        if target_language not in self.translation_mappings:
            return text
        
        # This is a simplified version - in reality, this would use proper translation services
        localized = text
        translations = self.translation_mappings[target_language]
        
        # Replace key terms with local equivalents
        for english_term, local_term in translations.items():
            localized = localized.replace(english_term, f"{english_term} ({local_term})")
        
        return localized
    
    def _check_cultural_appropriateness(self, text: str) -> Dict[str, Any]:
        """Check cultural appropriateness of content"""
        
        # Cultural sensitivity indicators
        sensitive_topics = ['religion', 'politics', 'gender roles', 'traditional practices']
        inappropriate_terms = ['primitive', 'backward', 'underdeveloped', 'uncivilized']
        
        text_lower = text.lower()
        
        issues = []
        for term in inappropriate_terms:
            if term in text_lower:
                issues.append(f"Potentially inappropriate term: '{term}'")
        
        return {
            'appropriate': len(issues) == 0,
            'issues_found': issues,
            'recommendation': "Review content for cultural sensitivity" if issues else "Content appears culturally appropriate"
        }
    
    def _generate_localization_recommendations(self, text: str, detected_languages: List[str]) -> List[str]:
        """Generate recommendations for content localization"""
        recommendations = []
        
        if 'english' in detected_languages and len(detected_languages) == 1:
            recommendations.append("Consider adding local language elements for better engagement")
            recommendations.append("Include culturally relevant examples and references")
        
        if len(detected_languages) > 1:
            recommendations.append("Maintain consistency across language versions")
            recommendations.append("Test content with native speakers")
        
        recommendations.append("Consider local payment methods and pricing strategies")
        recommendations.append("Adapt imagery to reflect local context and diversity")
        
        return recommendations

class RegionalEconomicFactors:
    """Regional Economic Integration - purchasing power, currency factors, inflation impact"""
    
    def __init__(self):
        self.economic_data = self._initialize_economic_data()
        self.currency_factors = self._initialize_currency_factors()
        self.inflation_models = self._initialize_inflation_models()
    
    def _initialize_economic_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize economic data for African regions"""
        return {
            'west_africa': {
                'gdp_per_capita_avg': 2150,
                'purchasing_power_parity': 0.42,
                'inflation_rate_avg': 8.5,
                'currency_volatility': 0.15,
                'mobile_money_penetration': 0.68,
                'internet_penetration': 0.54,
                'urban_population_ratio': 0.48
            },
            'east_africa': {
                'gdp_per_capita_avg': 1890,
                'purchasing_power_parity': 0.38,
                'inflation_rate_avg': 7.2,
                'currency_volatility': 0.12,
                'mobile_money_penetration': 0.75,
                'internet_penetration': 0.51,
                'urban_population_ratio': 0.34
            },
            'southern_africa': {
                'gdp_per_capita_avg': 4250,
                'purchasing_power_parity': 0.65,
                'inflation_rate_avg': 6.8,
                'currency_volatility': 0.18,
                'mobile_money_penetration': 0.45,
                'internet_penetration': 0.62,
                'urban_population_ratio': 0.67
            },
            'north_africa': {
                'gdp_per_capita_avg': 3680,
                'purchasing_power_parity': 0.58,
                'inflation_rate_avg': 9.2,
                'currency_volatility': 0.22,
                'mobile_money_penetration': 0.35,
                'internet_penetration': 0.71,
                'urban_population_ratio': 0.59
            }
        }
    
    def _initialize_currency_factors(self) -> Dict[str, Dict[str, Any]]:
        """Initialize currency stability and exchange rate factors"""
        return {
            'west_africa': {
                'primary_currencies': ['XOF', 'NGN', 'GHS', 'XAF'],
                'usd_exchange_volatility': 0.15,
                'regional_stability_index': 0.68,
                'inflation_hedge_factor': 0.12
            },
            'east_africa': {
                'primary_currencies': ['KES', 'TZS', 'UGX', 'ETB'],
                'usd_exchange_volatility': 0.12,
                'regional_stability_index': 0.72,
                'inflation_hedge_factor': 0.08
            },
            'southern_africa': {
                'primary_currencies': ['ZAR', 'BWP', 'ZMW', 'NAD'],
                'usd_exchange_volatility': 0.18,
                'regional_stability_index': 0.75,
                'inflation_hedge_factor': 0.15
            },
            'north_africa': {
                'primary_currencies': ['EGP', 'MAD', 'TND', 'DZD'],
                'usd_exchange_volatility': 0.22,
                'regional_stability_index': 0.63,
                'inflation_hedge_factor': 0.18
            }
        }
    
    def _initialize_inflation_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize inflation impact models"""
        return {
            'food_inflation_impact': {
                'weight': 0.35,
                'sensitivity_factor': 1.2,
                'regional_variation': 0.15
            },
            'fuel_inflation_impact': {
                'weight': 0.25,
                'sensitivity_factor': 1.1,
                'regional_variation': 0.20
            },
            'housing_inflation_impact': {
                'weight': 0.20,
                'sensitivity_factor': 0.9,
                'regional_variation': 0.25
            },
            'consumer_goods_impact': {
                'weight': 0.20,
                'sensitivity_factor': 1.0,
                'regional_variation': 0.10
            }
        }
    
    def analyze_economic_impact(self, region: str, product_price: float, product_category: str = "consumer_goods") -> Dict[str, Any]:
        """Analyze economic impact on purchasing decisions"""
        
        if region not in self.economic_data:
            return {'error': f'Economic data not available for region: {region}'}
        
        economic = self.economic_data[region]
        currency = self.currency_factors[region]
        
        # Calculate affordability index
        affordability_index = self._calculate_affordability_index(product_price, economic)
        
        # Calculate price sensitivity
        price_sensitivity = self._calculate_price_sensitivity(region, economic, product_category)
        
        # Calculate currency risk
        currency_risk = self._calculate_currency_risk(currency)
        
        # Calculate inflation impact
        inflation_impact = self._calculate_inflation_impact(region, product_category)
        
        # Generate economic recommendations
        recommendations = self._generate_economic_recommendations(
            affordability_index, price_sensitivity, currency_risk, inflation_impact
        )
        
        return {
            'region': region,
            'product_price': product_price,
            'affordability_index': affordability_index,
            'price_sensitivity_score': price_sensitivity,
            'currency_risk_score': currency_risk,
            'inflation_impact_score': inflation_impact,
            'economic_readiness': self._calculate_economic_readiness(affordability_index, price_sensitivity, currency_risk),
            'payment_recommendations': self._generate_payment_recommendations(economic, currency),
            'pricing_recommendations': recommendations,
            'market_entry_risk': self._assess_market_entry_risk(currency_risk, inflation_impact, price_sensitivity)
        }
    
    def _calculate_affordability_index(self, price: float, economic_data: Dict[str, Any]) -> float:
        """Calculate product affordability index"""
        gdp_per_capita = economic_data['gdp_per_capita_avg']
        ppp = economic_data['purchasing_power_parity']
        
        # Adjust for purchasing power parity
        adjusted_income = gdp_per_capita * ppp
        
        # Calculate price as percentage of monthly income (assuming gdp_per_capita is annual)
        monthly_income = adjusted_income / 12
        price_percentage = price / monthly_income if monthly_income > 0 else 1.0
        
        # Convert to affordability score (inverse relationship)
        affordability = max(0, 1 - min(price_percentage, 1))
        
        return affordability
    
    def _calculate_price_sensitivity(self, region: str, economic_data: Dict[str, Any], category: str) -> float:
        """Calculate price sensitivity for the region and category"""
        base_sensitivity = 1 - economic_data['purchasing_power_parity']
        
        # Category-specific adjustments
        category_adjustments = {
            'consumer_goods': 1.0,
            'food': 1.3,
            'technology': 0.8,
            'luxury': 0.6,
            'healthcare': 1.2
        }
        
        category_factor = category_adjustments.get(category, 1.0)
        return min(base_sensitivity * category_factor, 1.0)
    
    def _calculate_currency_risk(self, currency_data: Dict[str, Any]) -> float:
        """Calculate currency-related risk"""
        volatility = currency_data['usd_exchange_volatility']
        stability = currency_data['regional_stability_index']
        
        # Higher volatility and lower stability = higher risk
        risk_score = (volatility * 0.6) + ((1 - stability) * 0.4)
        return min(risk_score, 1.0)
    
    def _calculate_inflation_impact(self, region: str, category: str) -> float:
        """Calculate inflation impact on purchasing power"""
        economic = self.economic_data[region]
        inflation_rate = economic['inflation_rate_avg'] / 100  # Convert percentage to decimal
        
        # Category-specific inflation sensitivity
        if category in self.inflation_models:
            model = self.inflation_models[category]
            impact = inflation_rate * model['sensitivity_factor'] * model['weight']
        else:
            # Default consumer goods impact
            model = self.inflation_models['consumer_goods_impact']
            impact = inflation_rate * model['sensitivity_factor'] * model['weight']
        
        return min(impact, 1.0)
    
    def _calculate_economic_readiness(self, affordability: float, price_sensitivity: float, currency_risk: float) -> float:
        """Calculate overall economic readiness score"""
        # Weight the factors
        readiness = (
            affordability * 0.4 +
            (1 - price_sensitivity) * 0.3 +
            (1 - currency_risk) * 0.3
        )
        return readiness
    
    def _generate_payment_recommendations(self, economic: Dict[str, Any], currency: Dict[str, Any]) -> List[str]:
        """Generate payment method recommendations"""
        recommendations = []
        
        if economic['mobile_money_penetration'] > 0.6:
            recommendations.append("Prioritize mobile money payment options")
            recommendations.append("Integrate with popular mobile money providers")
        
        if currency['usd_exchange_volatility'] > 0.15:
            recommendations.append("Consider local currency pricing with regular updates")
            recommendations.append("Offer price lock-in options for volatile currency periods")
        
        if economic['purchasing_power_parity'] < 0.5:
            recommendations.append("Implement flexible payment plans and installments")
            recommendations.append("Consider micro-financing partnerships")
        
        recommendations.append("Accept multiple payment methods including cash")
        
        return recommendations
    
    def _generate_economic_recommendations(self, affordability: float, price_sensitivity: float, currency_risk: float, inflation_impact: float) -> List[str]:
        """Generate economic-based pricing recommendations"""
        recommendations = []
        
        if affordability < 0.3:
            recommendations.append("Consider significant price reduction or value engineering")
            recommendations.append("Develop entry-level product variants")
        
        if price_sensitivity > 0.7:
            recommendations.append("Implement tiered pricing strategy")
            recommendations.append("Emphasize value proposition and cost savings")
        
        if currency_risk > 0.2:
            recommendations.append("Use local currency pricing with hedging strategies")
            recommendations.append("Monitor exchange rates and adjust pricing regularly")
        
        if inflation_impact > 0.1:
            recommendations.append("Build inflation adjustment mechanisms into pricing")
            recommendations.append("Consider subscription or contract models to lock in prices")
        
        return recommendations
    
    def _assess_market_entry_risk(self, currency_risk: float, inflation_impact: float, price_sensitivity: float) -> Dict[str, Any]:
        """Assess overall market entry risk"""
        
        # Calculate composite risk score
        risk_score = (currency_risk * 0.4 + inflation_impact * 0.3 + price_sensitivity * 0.3)
        
        if risk_score < 0.3:
            risk_level = "Low"
            recommendation = "Favorable conditions for market entry"
        elif risk_score < 0.6:
            risk_level = "Medium"
            recommendation = "Proceed with careful market strategy and monitoring"
        else:
            risk_level = "High"
            recommendation = "Consider delayed entry or significant strategy modifications"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'recommendation': recommendation,
            'key_risk_factors': self._identify_key_risk_factors(currency_risk, inflation_impact, price_sensitivity)
        }
    
    def _identify_key_risk_factors(self, currency_risk: float, inflation_impact: float, price_sensitivity: float) -> List[str]:
        """Identify the key risk factors"""
        factors = []
        
        if currency_risk > 0.2:
            factors.append("High currency volatility")
        if inflation_impact > 0.15:
            factors.append("Significant inflation impact")
        if price_sensitivity > 0.7:
            factors.append("High price sensitivity in target market")
        
        return factors if factors else ["Low risk environment identified"]

# Main NeuroInsight-Africa Platform Integration Class
class NeuroInsightAfricaPlatform:
    """
    Main class integrating all African market intelligence components
    as specified in the problem statement
    """
    
    def __init__(self):
        self.regional_models = AfricanMarketModels()
        self.cultural_adapters = CulturalSentimentAdapters()
        self.language_processors = MultiLanguageAfrica()
        self.economic_integrators = RegionalEconomicFactors()
    
    def analyze_african_market_opportunity(self, 
                                         text_content: str, 
                                         target_region: str = "west_africa",
                                         product_price: float = 100.0,
                                         product_category: str = "consumer_goods") -> Dict[str, Any]:
        """
        Comprehensive African market analysis combining all components
        """
        
        # Regional market pattern analysis
        regional_analysis = self.regional_models.analyze_regional_market_patterns(target_region, product_category)
        
        # Cultural sentiment analysis
        cultural_analysis = self.cultural_adapters.analyze_cultural_sentiment(text_content, "pan_african")
        
        # Multi-language processing
        language_analysis = self.language_processors.process_multilingual_content(text_content)
        
        # Economic impact analysis
        economic_analysis = self.economic_integrators.analyze_economic_impact(target_region, product_price, product_category)
        
        # Generate integrated recommendations
        integrated_recommendations = self._generate_integrated_recommendations(
            regional_analysis, cultural_analysis, language_analysis, economic_analysis
        )
        
        # Calculate overall market opportunity score
        opportunity_score = self._calculate_market_opportunity_score(
            regional_analysis, cultural_analysis, economic_analysis
        )
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'target_region': target_region,
            'product_category': product_category,
            'product_price': product_price,
            'regional_analysis': regional_analysis,
            'cultural_analysis': cultural_analysis,
            'language_analysis': language_analysis,
            'economic_analysis': economic_analysis,
            'market_opportunity_score': opportunity_score,
            'integrated_recommendations': integrated_recommendations,
            'success_probability': self._calculate_success_probability(opportunity_score, regional_analysis, economic_analysis)
        }
    
    def _generate_integrated_recommendations(self, regional: Dict, cultural: Dict, language: Dict, economic: Dict) -> List[str]:
        """Generate integrated recommendations from all analyses"""
        recommendations = []
        
        # Regional recommendations
        if 'recommended_strategies' in regional:
            recommendations.extend(regional['recommended_strategies'])
        
        # Cultural recommendations  
        if 'adaptation_recommendations' in cultural:
            recommendations.extend(cultural['adaptation_recommendations'])
        
        # Language recommendations
        if 'recommendations' in language:
            recommendations.extend(language['recommendations'])
        
        # Economic recommendations
        if 'pricing_recommendations' in economic:
            recommendations.extend(economic['pricing_recommendations'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _calculate_market_opportunity_score(self, regional: Dict, cultural: Dict, economic: Dict) -> float:
        """Calculate overall market opportunity score"""
        
        # Regional readiness (30% weight)
        regional_score = regional.get('market_readiness_score', 0.5)
        
        # Cultural alignment (25% weight)  
        cultural_score = cultural.get('cultural_sentiment_score', 0.5)
        
        # Economic viability (35% weight)
        economic_score = economic.get('economic_readiness', 0.5)
        
        # Language accessibility (10% weight)
        language_score = cultural.get('confidence_level', 0.5)
        
        opportunity_score = (
            regional_score * 0.30 +
            cultural_score * 0.25 +
            economic_score * 0.35 +
            language_score * 0.10
        )
        
        return opportunity_score
    
    def _calculate_success_probability(self, opportunity_score: float, regional: Dict, economic: Dict) -> Dict[str, Any]:
        """Calculate success probability and timeline"""
        
        # Base success probability from opportunity score
        base_probability = opportunity_score
        
        # Adjust for risk factors
        risk_factors = economic.get('market_entry_risk', {})
        risk_adjustment = 1.0
        
        if risk_factors.get('risk_level') == 'High':
            risk_adjustment = 0.7
        elif risk_factors.get('risk_level') == 'Medium':
            risk_adjustment = 0.85
        
        final_probability = base_probability * risk_adjustment
        
        # Estimate timeline based on market readiness
        market_readiness = regional.get('market_readiness_score', 0.5)
        if market_readiness > 0.8:
            timeline = "3-6 months"
        elif market_readiness > 0.6:
            timeline = "6-12 months"
        else:
            timeline = "12-18 months"
        
        return {
            'success_probability': final_probability,
            'confidence_level': 'High' if final_probability > 0.7 else 'Medium' if final_probability > 0.5 else 'Low',
            'estimated_timeline': timeline,
            'key_success_factors': self._identify_success_factors(regional, economic)
        }
    
    def _identify_success_factors(self, regional: Dict, economic: Dict) -> List[str]:
        """Identify key success factors for market entry"""
        factors = []
        
        # From regional analysis
        if regional.get('community_influence_index', 0) > 0.8:
            factors.append("Strong community engagement strategy")
        
        # From economic analysis
        if economic.get('affordability_index', 0) > 0.6:
            factors.append("Product is well-positioned for target market purchasing power")
        
        # Payment infrastructure
        payment_recs = economic.get('payment_recommendations', [])
        if any('mobile money' in rec.lower() for rec in payment_recs):
            factors.append("Mobile payment integration critical")
        
        factors.extend([
            "Cultural sensitivity and local adaptation",
            "Community-based marketing approach",
            "Flexible pricing and payment options"
        ])
        
        return factors