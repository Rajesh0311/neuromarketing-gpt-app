"""
South African Cultural Intelligence Module
Enhanced cultural analysis for sarcasm & irony detection in South African context
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re


@dataclass
class SAulturalContext:
    """South African cultural context parameters"""
    race_group: str
    language_group: str
    region: str
    urban_rural: str
    age_group: str = "adult"
    education_level: str = "secondary"


@dataclass
class CulturalAnalysisResult:
    """Results of South African cultural analysis"""
    sarcasm_probability: float
    irony_probability: float
    ubuntu_compatibility: float
    cross_cultural_sensitivity: float
    regional_appropriateness: float
    language_respect_index: float
    cultural_markers: List[str]
    risk_assessment: Dict[str, float]
    recommendations: List[str]


class SouthAfricanCulturalAnalyzer:
    """
    Advanced cultural intelligence engine for South African sarcasm and irony detection
    Considers race groups, languages, regions, and cultural nuances
    """
    
    def __init__(self):
        self.race_groups = self._init_race_groups()
        self.languages = self._init_languages()
        self.regions = self._init_regions()
        self.urban_rural_contexts = self._init_urban_rural()
        self.ubuntu_principles = self._init_ubuntu_principles()
        self.cultural_patterns = self._init_cultural_patterns()
        
    def _init_race_groups(self) -> Dict[str, Dict[str, Any]]:
        """Initialize South African race group cultural characteristics"""
        return {
            "Black African": {
                "communication_style": "indirect",
                "hierarchy_respect": 0.9,
                "community_orientation": 0.95,
                "ubuntu_alignment": 0.9,
                "humor_patterns": ["storytelling", "metaphorical", "contextual"],
                "sarcasm_tolerance": 0.6,
                "language_code_switching": 0.8
            },
            "Coloured": {
                "communication_style": "mixed",
                "hierarchy_respect": 0.7,
                "community_orientation": 0.8,
                "ubuntu_alignment": 0.75,
                "humor_patterns": ["wordplay", "social_commentary", "ironic"],
                "sarcasm_tolerance": 0.8,
                "language_code_switching": 0.9
            },
            "Indian": {
                "communication_style": "formal_respectful",
                "hierarchy_respect": 0.85,
                "community_orientation": 0.8,
                "ubuntu_alignment": 0.7,
                "humor_patterns": ["wit", "cultural_references", "subtle"],
                "sarcasm_tolerance": 0.7,
                "language_code_switching": 0.6
            },
            "White": {
                "communication_style": "direct",
                "hierarchy_respect": 0.6,
                "community_orientation": 0.6,
                "ubuntu_alignment": 0.5,
                "humor_patterns": ["direct_sarcasm", "irony", "banter"],
                "sarcasm_tolerance": 0.9,
                "language_code_switching": 0.7
            }
        }
    
    def _init_languages(self) -> Dict[str, Dict[str, Any]]:
        """Initialize South African language group characteristics"""
        return {
            "English": {
                "sarcasm_markers": ["really", "obviously", "sure", "right"],
                "irony_patterns": ["contrast", "understatement"],
                "cultural_weight": 0.8,
                "formality_level": 0.7
            },
            "Afrikaans": {
                "sarcasm_markers": ["seker", "natuurlik", "ag man"],
                "irony_patterns": ["exaggeration", "dry_humor"],
                "cultural_weight": 0.8,
                "formality_level": 0.6
            },
            "isiZulu": {
                "sarcasm_markers": ["yebo", "ngempela", "kanti"],
                "irony_patterns": ["indirect_reference", "proverbs"],
                "cultural_weight": 0.9,
                "formality_level": 0.8
            },
            "isiXhosa": {
                "sarcasm_markers": ["ewe", "ngenyani", "kodwa"],
                "irony_patterns": ["storytelling", "metaphors"],
                "cultural_weight": 0.9,
                "formality_level": 0.8
            },
            "Setswana": {
                "sarcasm_markers": ["ee", "ruri", "mme"],
                "irony_patterns": ["wisdom_sayings", "contextual"],
                "cultural_weight": 0.85,
                "formality_level": 0.8
            },
            "Sepedi": {
                "sarcasm_markers": ["ee", "nnete", "le gona"],
                "irony_patterns": ["traditional_wisdom", "indirect"],
                "cultural_weight": 0.85,
                "formality_level": 0.8
            },
            "Sesotho": {
                "sarcasm_markers": ["ee", "nnete", "empa"],
                "irony_patterns": ["proverbs", "moral_lessons"],
                "cultural_weight": 0.85,
                "formality_level": 0.8
            },
            "siSwati": {
                "sarcasm_markers": ["yebo", "ngempela", "kodvwa"],
                "irony_patterns": ["cultural_references", "respect_based"],
                "cultural_weight": 0.8,
                "formality_level": 0.85
            },
            "Tshivenda": {
                "sarcasm_markers": ["ee", "nditambudziko", "fhedzi"],
                "irony_patterns": ["ancestral_wisdom", "nature_metaphors"],
                "cultural_weight": 0.8,
                "formality_level": 0.85
            },
            "Xitsonga": {
                "sarcasm_markers": ["ina", "hakunene", "kambe"],
                "irony_patterns": ["traditional_stories", "community_wisdom"],
                "cultural_weight": 0.8,
                "formality_level": 0.85
            },
            "isiNdebele": {
                "sarcasm_markers": ["yebo", "ngempela", "kodwa"],
                "irony_patterns": ["cultural_patterns", "traditional_context"],
                "cultural_weight": 0.75,
                "formality_level": 0.85
            }
        }
    
    def _init_regions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize South African regional characteristics"""
        return {
            "Gauteng": {
                "urbanization": 0.95,
                "diversity_index": 0.9,
                "economic_context": "high",
                "communication_style": "fast_paced",
                "humor_style": "cosmopolitan",
                "sarcasm_prevalence": 0.8
            },
            "Western Cape": {
                "urbanization": 0.8,
                "diversity_index": 0.85,
                "economic_context": "high",
                "communication_style": "relaxed",
                "humor_style": "witty",
                "sarcasm_prevalence": 0.85
            },
            "KwaZulu-Natal": {
                "urbanization": 0.6,
                "diversity_index": 0.8,
                "economic_context": "medium",
                "communication_style": "respectful",
                "humor_style": "traditional_modern",
                "sarcasm_prevalence": 0.6
            },
            "Eastern Cape": {
                "urbanization": 0.4,
                "diversity_index": 0.7,
                "economic_context": "low",
                "communication_style": "traditional",
                "humor_style": "storytelling",
                "sarcasm_prevalence": 0.5
            },
            "Limpopo": {
                "urbanization": 0.3,
                "diversity_index": 0.6,
                "economic_context": "low",
                "communication_style": "traditional",
                "humor_style": "community_based",
                "sarcasm_prevalence": 0.4
            },
            "Mpumalanga": {
                "urbanization": 0.5,
                "diversity_index": 0.7,
                "economic_context": "medium",
                "communication_style": "mixed",
                "humor_style": "practical",
                "sarcasm_prevalence": 0.6
            },
            "North West": {
                "urbanization": 0.4,
                "diversity_index": 0.6,
                "economic_context": "medium",
                "communication_style": "straightforward",
                "humor_style": "down_to_earth",
                "sarcasm_prevalence": 0.5
            },
            "Free State": {
                "urbanization": 0.5,
                "diversity_index": 0.7,
                "economic_context": "medium",
                "communication_style": "friendly",
                "humor_style": "warm",
                "sarcasm_prevalence": 0.6
            },
            "Northern Cape": {
                "urbanization": 0.4,
                "diversity_index": 0.6,
                "economic_context": "low",
                "communication_style": "simple",
                "humor_style": "dry",
                "sarcasm_prevalence": 0.7
            }
        }
    
    def _init_urban_rural(self) -> Dict[str, Dict[str, Any]]:
        """Initialize urban/rural context characteristics"""
        return {
            "Metropolitan": {
                "media_exposure": 0.9,
                "cultural_mixing": 0.9,
                "sarcasm_sophistication": 0.8,
                "irony_understanding": 0.85,
                "ubuntu_practice": 0.6
            },
            "Urban": {
                "media_exposure": 0.8,
                "cultural_mixing": 0.7,
                "sarcasm_sophistication": 0.7,
                "irony_understanding": 0.75,
                "ubuntu_practice": 0.7
            },
            "Rural": {
                "media_exposure": 0.5,
                "cultural_mixing": 0.4,
                "sarcasm_sophistication": 0.5,
                "irony_understanding": 0.6,
                "ubuntu_practice": 0.9
            },
            "Traditional": {
                "media_exposure": 0.3,
                "cultural_mixing": 0.2,
                "sarcasm_sophistication": 0.3,
                "irony_understanding": 0.5,
                "ubuntu_practice": 0.95
            }
        }
    
    def _init_ubuntu_principles(self) -> Dict[str, float]:
        """Initialize Ubuntu philosophy principles"""
        return {
            "interconnectedness": 0.9,
            "collective_responsibility": 0.85,
            "compassion": 0.9,
            "respect_for_others": 0.95,
            "sharing": 0.8,
            "harmony": 0.85,
            "humanity": 0.9
        }
    
    def _init_cultural_patterns(self) -> Dict[str, List[str]]:
        """Initialize cultural communication patterns"""
        return {
            "ubuntu_phrases": [
                "umuntu ngumuntu ngabantu",
                "together we are stronger",
                "community first",
                "shared humanity",
                "collective wisdom"
            ],
            "respect_markers": [
                "uncle", "auntie", "mama", "papa", "tata",
                "sir", "madam", "madala", "gogo"
            ],
            "cultural_sensitivity_triggers": [
                "apartheid", "racial", "tribal", "township",
                "struggle", "liberation", "rainbow nation"
            ],
            "code_switching_indicators": [
                "eish", "shame", "now now", "just now",
                "lekker", "braai", "robots", "bakkie"
            ]
        }
    
    def analyze_cultural_context(self, text: str, context: SAulturalContext) -> CulturalAnalysisResult:
        """
        Perform comprehensive South African cultural analysis
        """
        # Get cultural characteristics
        race_profile = self.race_groups.get(context.race_group, {})
        language_profile = self.languages.get(context.language_group, {})
        region_profile = self.regions.get(context.region, {})
        urban_rural_profile = self.urban_rural_contexts.get(context.urban_rural, {})
        
        # Analyze text for cultural markers
        cultural_markers = self._detect_cultural_markers(text)
        
        # Calculate sarcasm probability with cultural adjustments
        base_sarcasm = self._detect_base_sarcasm(text)
        cultural_sarcasm_modifier = self._calculate_cultural_sarcasm_modifier(
            race_profile, language_profile, region_profile, urban_rural_profile
        )
        sarcasm_probability = base_sarcasm * cultural_sarcasm_modifier
        
        # Calculate irony probability
        base_irony = self._detect_base_irony(text)
        cultural_irony_modifier = self._calculate_cultural_irony_modifier(
            race_profile, language_profile, region_profile, urban_rural_profile
        )
        irony_probability = base_irony * cultural_irony_modifier
        
        # Ubuntu compatibility analysis
        ubuntu_compatibility = self._analyze_ubuntu_compatibility(text, race_profile)
        
        # Cross-cultural sensitivity
        cross_cultural_sensitivity = self._analyze_cross_cultural_sensitivity(
            text, context, cultural_markers
        )
        
        # Regional appropriateness
        regional_appropriateness = self._analyze_regional_appropriateness(
            text, region_profile, urban_rural_profile
        )
        
        # Language respect index
        language_respect_index = self._analyze_language_respect(
            text, language_profile, context
        )
        
        # Risk assessment
        risk_assessment = self._assess_cultural_risks(
            text, context, cultural_markers
        )
        
        # Recommendations
        recommendations = self._generate_recommendations(
            text, context, ubuntu_compatibility, cross_cultural_sensitivity
        )
        
        return CulturalAnalysisResult(
            sarcasm_probability=min(1.0, max(0.0, sarcasm_probability)),
            irony_probability=min(1.0, max(0.0, irony_probability)),
            ubuntu_compatibility=ubuntu_compatibility,
            cross_cultural_sensitivity=cross_cultural_sensitivity,
            regional_appropriateness=regional_appropriateness,
            language_respect_index=language_respect_index,
            cultural_markers=cultural_markers,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
    
    def _detect_cultural_markers(self, text: str) -> List[str]:
        """Detect South African cultural markers in text"""
        markers = []
        text_lower = text.lower()
        
        # Check for Ubuntu principles
        for phrase in self.cultural_patterns["ubuntu_phrases"]:
            if phrase.lower() in text_lower:
                markers.append(f"Ubuntu: {phrase}")
        
        # Check for respect markers
        for marker in self.cultural_patterns["respect_markers"]:
            if marker.lower() in text_lower:
                markers.append(f"Respect: {marker}")
        
        # Check for code-switching indicators
        for indicator in self.cultural_patterns["code_switching_indicators"]:
            if indicator.lower() in text_lower:
                markers.append(f"Code-switching: {indicator}")
        
        # Check for sensitive topics
        for trigger in self.cultural_patterns["cultural_sensitivity_triggers"]:
            if trigger.lower() in text_lower:
                markers.append(f"Sensitive: {trigger}")
        
        return markers
    
    def _detect_base_sarcasm(self, text: str) -> float:
        """Detect base sarcasm probability"""
        sarcasm_indicators = [
            "really", "obviously", "sure", "right", "great",
            "wonderful", "fantastic", "perfect", "brilliant"
        ]
        
        # Simple pattern matching for demo
        words = text.lower().split()
        sarcasm_score = 0
        
        for word in words:
            if word in sarcasm_indicators:
                sarcasm_score += 0.2
        
        # Check for contrast patterns
        if "but" in text.lower() or "however" in text.lower():
            sarcasm_score += 0.3
        
        return min(1.0, sarcasm_score)
    
    def _detect_base_irony(self, text: str) -> float:
        """Detect base irony probability"""
        irony_patterns = [
            ("expect", "get"), ("should", "actually"), ("hope", "reality")
        ]
        
        irony_score = 0
        text_lower = text.lower()
        
        for pattern in irony_patterns:
            if pattern[0] in text_lower and pattern[1] in text_lower:
                irony_score += 0.4
        
        return min(1.0, irony_score)
    
    def _calculate_cultural_sarcasm_modifier(self, race_profile, language_profile, 
                                           region_profile, urban_rural_profile) -> float:
        """Calculate cultural modifier for sarcasm detection"""
        modifiers = []
        
        if race_profile:
            modifiers.append(race_profile.get("sarcasm_tolerance", 0.5))
        if region_profile:
            modifiers.append(region_profile.get("sarcasm_prevalence", 0.5))
        if urban_rural_profile:
            modifiers.append(urban_rural_profile.get("sarcasm_sophistication", 0.5))
        
        return np.mean(modifiers) if modifiers else 0.5
    
    def _calculate_cultural_irony_modifier(self, race_profile, language_profile,
                                         region_profile, urban_rural_profile) -> float:
        """Calculate cultural modifier for irony detection"""
        modifiers = []
        
        if urban_rural_profile:
            modifiers.append(urban_rural_profile.get("irony_understanding", 0.5))
        if language_profile:
            modifiers.append(language_profile.get("cultural_weight", 0.5))
        
        return np.mean(modifiers) if modifiers else 0.5
    
    def _analyze_ubuntu_compatibility(self, text: str, race_profile: Dict) -> float:
        """Analyze compatibility with Ubuntu philosophy"""
        ubuntu_score = 0.5
        text_lower = text.lower()
        
        # Positive Ubuntu indicators
        positive_indicators = ["together", "community", "share", "help", "support"]
        for indicator in positive_indicators:
            if indicator in text_lower:
                ubuntu_score += 0.1
        
        # Negative Ubuntu indicators
        negative_indicators = ["selfish", "individual", "alone", "compete", "beat"]
        for indicator in negative_indicators:
            if indicator in text_lower:
                ubuntu_score -= 0.1
        
        # Adjust based on race group Ubuntu alignment
        if race_profile:
            ubuntu_alignment = race_profile.get("ubuntu_alignment", 0.5)
            ubuntu_score = ubuntu_score * ubuntu_alignment
        
        return min(1.0, max(0.0, ubuntu_score))
    
    def _analyze_cross_cultural_sensitivity(self, text: str, context: SAulturalContext,
                                          cultural_markers: List[str]) -> float:
        """Analyze cross-cultural sensitivity"""
        sensitivity_score = 0.7
        
        # Check for inclusive language
        inclusive_words = ["everyone", "all", "together", "unity", "diverse"]
        for word in inclusive_words:
            if word.lower() in text.lower():
                sensitivity_score += 0.05
        
        # Check for exclusive or divisive language
        exclusive_words = ["them", "those people", "others", "different", "strange"]
        for word in exclusive_words:
            if word.lower() in text.lower():
                sensitivity_score -= 0.1
        
        # Penalize sensitive topic mentions without context
        sensitive_markers = [m for m in cultural_markers if m.startswith("Sensitive:")]
        sensitivity_score -= len(sensitive_markers) * 0.15
        
        return min(1.0, max(0.0, sensitivity_score))
    
    def _analyze_regional_appropriateness(self, text: str, region_profile: Dict,
                                        urban_rural_profile: Dict) -> float:
        """Analyze regional appropriateness"""
        appropriateness = 0.7
        
        if region_profile and urban_rural_profile:
            # Consider regional communication style
            comm_style = region_profile.get("communication_style", "mixed")
            
            if comm_style == "traditional" and "modern" in text.lower():
                appropriateness -= 0.1
            elif comm_style == "fast_paced" and len(text.split()) > 50:
                appropriateness -= 0.1
        
        return min(1.0, max(0.0, appropriateness))
    
    def _analyze_language_respect(self, text: str, language_profile: Dict,
                                context: SAulturalContext) -> float:
        """Analyze respect for linguistic diversity"""
        respect_score = 0.8
        
        # Check for language-specific markers
        if language_profile:
            markers = language_profile.get("sarcasm_markers", [])
            for marker in markers:
                if marker.lower() in text.lower():
                    respect_score += 0.05
        
        # Check for code-switching respect
        if context.language_group != "English":
            # Bonus for maintaining cultural authenticity
            respect_score += 0.1
        
        return min(1.0, max(0.0, respect_score))
    
    def _assess_cultural_risks(self, text: str, context: SAulturalContext,
                             cultural_markers: List[str]) -> Dict[str, float]:
        """Assess cultural risks in the content"""
        risks = {
            "offense_risk": 0.1,
            "misunderstanding_risk": 0.2,
            "exclusion_risk": 0.1,
            "stereotyping_risk": 0.1
        }
        
        # Increase risks based on sensitive markers
        sensitive_count = len([m for m in cultural_markers if m.startswith("Sensitive:")])
        if sensitive_count > 0:
            risks["offense_risk"] += sensitive_count * 0.2
            risks["misunderstanding_risk"] += sensitive_count * 0.15
        
        # Consider context
        if context.urban_rural in ["Rural", "Traditional"]:
            risks["misunderstanding_risk"] += 0.1
        
        return {k: min(1.0, v) for k, v in risks.items()}
    
    def _generate_recommendations(self, text: str, context: SAulturalContext,
                                ubuntu_compatibility: float,
                                cross_cultural_sensitivity: float) -> List[str]:
        """Generate cultural adaptation recommendations"""
        recommendations = []
        
        if ubuntu_compatibility < 0.5:
            recommendations.append("Consider emphasizing community values and collective benefit")
        
        if cross_cultural_sensitivity < 0.6:
            recommendations.append("Review content for inclusive language across all race groups")
        
        if context.urban_rural in ["Rural", "Traditional"]:
            recommendations.append("Simplify language and avoid complex sarcasm for rural audiences")
        
        if context.language_group != "English":
            recommendations.append(f"Consider adding {context.language_group} language elements for authenticity")
        
        return recommendations


def get_sa_cultural_options() -> Dict[str, List[str]]:
    """Get South African cultural selection options for UI"""
    return {
        "race_groups": ["Black African", "Coloured", "Indian", "White"],
        "languages": [
            "English", "Afrikaans", "isiZulu", "isiXhosa", "Setswana",
            "Sepedi", "Sesotho", "siSwati", "Tshivenda", "Xitsonga", "isiNdebele"
        ],
        "regions": [
            "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
            "Limpopo", "Mpumalanga", "North West", "Free State", "Northern Cape"
        ],
        "urban_rural": ["Metropolitan", "Urban", "Rural", "Traditional"]
    }