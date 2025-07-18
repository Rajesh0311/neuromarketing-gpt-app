#!/usr/bin/env python3
"""
Test Suite for South African Cultural Analyzer
=============================================

Tests the core functionality of the SA cultural intelligence module.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from south_african_cultural_analyzer import SouthAfricanCulturalAnalyzer, SAulturalContext, get_sa_cultural_options

def test_sa_cultural_options():
    """Test that SA cultural options are properly loaded"""
    print("🔍 Testing SA cultural options...")
    
    options = get_sa_cultural_options()
    
    # Test structure
    assert "race_groups" in options
    assert "languages" in options
    assert "regions" in options
    assert "urban_rural" in options
    
    # Test content
    assert "Black African" in options["race_groups"]
    assert "Coloured" in options["race_groups"]
    assert "Indian" in options["race_groups"]
    assert "White" in options["race_groups"]
    
    assert "English" in options["languages"]
    assert "Afrikaans" in options["languages"]
    assert "isiZulu" in options["languages"]
    
    assert "Gauteng" in options["regions"]
    assert "Western Cape" in options["regions"]
    
    assert "Metropolitan" in options["urban_rural"]
    assert "Rural" in options["urban_rural"]
    
    print(f"✅ SA cultural options working - {len(options['race_groups'])} race groups, {len(options['languages'])} languages")
    return True

def test_cultural_analyzer_initialization():
    """Test SA cultural analyzer initialization"""
    print("🔍 Testing SA cultural analyzer initialization...")
    
    analyzer = SouthAfricanCulturalAnalyzer()
    
    # Test that all components are initialized
    assert hasattr(analyzer, 'race_groups')
    assert hasattr(analyzer, 'languages')
    assert hasattr(analyzer, 'regions')
    assert hasattr(analyzer, 'ubuntu_principles')
    
    # Test data integrity
    assert "Black African" in analyzer.race_groups
    assert "ubuntu_alignment" in analyzer.race_groups["Black African"]
    
    assert "English" in analyzer.languages
    assert "sarcasm_markers" in analyzer.languages["English"]
    
    assert "Gauteng" in analyzer.regions
    assert "sarcasm_prevalence" in analyzer.regions["Gauteng"]
    
    print("✅ SA cultural analyzer initialization working")
    return True

def test_cultural_context_creation():
    """Test SA cultural context creation"""
    print("🔍 Testing SA cultural context creation...")
    
    context = SAulturalContext(
        race_group="Black African",
        language_group="English",
        region="Gauteng",
        urban_rural="Metropolitan",
        age_group="adult",
        education_level="tertiary"
    )
    
    assert context.race_group == "Black African"
    assert context.language_group == "English"
    assert context.region == "Gauteng"
    assert context.urban_rural == "Metropolitan"
    assert context.age_group == "adult"
    assert context.education_level == "tertiary"
    
    print("✅ SA cultural context creation working")
    return True

def test_sarcasm_analysis():
    """Test sarcasm analysis with SA cultural context"""
    print("🔍 Testing sarcasm analysis with SA cultural context...")
    
    analyzer = SouthAfricanCulturalAnalyzer()
    
    # Test context 1: Black African + English
    context1 = SAulturalContext(
        race_group="Black African",
        language_group="English",
        region="Gauteng", 
        urban_rural="Metropolitan"
    )
    
    result1 = analyzer.analyze_cultural_context(
        "Sure, that's a fantastic idea, really brilliant thinking there!",
        context1
    )
    
    # Test that analysis returns proper structure
    assert hasattr(result1, 'sarcasm_probability')
    assert hasattr(result1, 'irony_probability')
    assert hasattr(result1, 'ubuntu_compatibility')
    assert hasattr(result1, 'cross_cultural_sensitivity')
    assert hasattr(result1, 'regional_appropriateness')
    assert hasattr(result1, 'language_respect_index')
    assert hasattr(result1, 'cultural_markers')
    assert hasattr(result1, 'risk_assessment')
    assert hasattr(result1, 'recommendations')
    
    # Test value ranges
    assert 0 <= result1.sarcasm_probability <= 1
    assert 0 <= result1.ubuntu_compatibility <= 1
    assert 0 <= result1.cross_cultural_sensitivity <= 1
    assert isinstance(result1.cultural_markers, list)
    assert isinstance(result1.risk_assessment, dict)
    assert isinstance(result1.recommendations, list)
    
    # Test context 2: Coloured + Afrikaans
    context2 = SAulturalContext(
        race_group="Coloured",
        language_group="Afrikaans", 
        region="Western Cape",
        urban_rural="Urban"
    )
    
    result2 = analyzer.analyze_cultural_context(
        "Sure, that's a fantastic idea, really brilliant thinking there!",
        context2
    )
    
    # Results should be different for different cultural contexts
    # (due to different cultural modifiers)
    print(f"   Black African context - Sarcasm: {result1.sarcasm_probability:.1%}, Ubuntu: {result1.ubuntu_compatibility:.1%}")
    print(f"   Coloured context - Sarcasm: {result2.sarcasm_probability:.1%}, Ubuntu: {result2.ubuntu_compatibility:.1%}")
    
    print("✅ SA cultural sarcasm analysis working")
    return True

def test_ubuntu_analysis():
    """Test Ubuntu philosophy analysis"""
    print("🔍 Testing Ubuntu philosophy analysis...")
    
    analyzer = SouthAfricanCulturalAnalyzer()
    context = SAulturalContext(
        race_group="Black African",
        language_group="English",
        region="Gauteng",
        urban_rural="Rural"
    )
    
    # Test community-oriented text
    community_text = "Together we can build a better community for everyone"
    result_community = analyzer.analyze_cultural_context(community_text, context)
    
    # Test individual-oriented text  
    individual_text = "I need to win and beat everyone else to be successful"
    result_individual = analyzer.analyze_cultural_context(individual_text, context)
    
    # Community text should have higher Ubuntu compatibility
    assert result_community.ubuntu_compatibility >= result_individual.ubuntu_compatibility
    
    print(f"   Community text Ubuntu: {result_community.ubuntu_compatibility:.1%}")
    print(f"   Individual text Ubuntu: {result_individual.ubuntu_compatibility:.1%}")
    print("✅ Ubuntu philosophy analysis working")
    return True

def test_language_specific_analysis():
    """Test language-specific cultural analysis"""
    print("🔍 Testing language-specific analysis...")
    
    analyzer = SouthAfricanCulturalAnalyzer()
    
    # Test different language contexts
    contexts = [
        SAulturalContext("Coloured", "English", "Western Cape", "Urban"),
        SAulturalContext("Coloured", "Afrikaans", "Western Cape", "Urban"),
        SAulturalContext("Black African", "isiZulu", "KwaZulu-Natal", "Rural")
    ]
    
    text = "This is really great, absolutely wonderful"
    results = []
    
    for context in contexts:
        result = analyzer.analyze_cultural_context(text, context)
        results.append((context.language_group, result.language_respect_index))
    
    print("   Language respect indices:")
    for lang, respect in results:
        print(f"     {lang}: {respect:.1%}")
    
    print("✅ Language-specific analysis working")
    return True

def run_sa_cultural_tests():
    """Run the complete SA cultural test suite"""
    print("🚀 Starting South African Cultural Intelligence Test")
    print("=" * 55)
    
    tests = [
        ("SA Cultural Options", test_sa_cultural_options),
        ("Analyzer Initialization", test_cultural_analyzer_initialization),
        ("Cultural Context Creation", test_cultural_context_creation),
        ("Sarcasm Analysis", test_sarcasm_analysis),
        ("Ubuntu Analysis", test_ubuntu_analysis),
        ("Language-Specific Analysis", test_language_specific_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 35)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ FAILED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n\n📊 SA Cultural Test Results")
    print("=" * 35)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"\n🎉 ALL SA CULTURAL TESTS PASSED! 🇿🇦")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
        return False

if __name__ == "__main__":
    success = run_sa_cultural_tests()
    sys.exit(0 if success else 1)