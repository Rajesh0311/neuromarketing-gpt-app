"""
Test Enhanced Neuromarketing Platform Features
==============================================

Tests for the newly implemented comprehensive features:
1. Environmental Simulation Complete
2. Advanced Neural Processing 
3. NeuroInsight-Africa Advanced Features
"""

import numpy as np
from datetime import datetime

def test_environmental_simulation_import():
    """Test that environmental simulation module imports correctly"""
    from environmental_simulation_complete import EnvironmentalSimulationComplete
    
    env_sim = EnvironmentalSimulationComplete()
    assert env_sim is not None
    assert hasattr(env_sim, 'simulation_types')
    assert hasattr(env_sim, 'automotive_suite')
    assert hasattr(env_sim, 'mobile_recording')
    assert hasattr(env_sim, 'multisensory_integration')
    print("✅ Environmental simulation module imports correctly")

def test_advanced_neural_processor_import():
    """Test that advanced neural processor imports correctly"""
    from neural_simulation import AdvancedNeuralProcessor
    
    processor = AdvancedNeuralProcessor()
    assert processor is not None
    assert hasattr(processor, 'eeg_channels')
    assert hasattr(processor, 'frequency_bands')
    assert hasattr(processor, 'connectivity_matrix')
    assert hasattr(processor, 'cultural_modulation')
    print("✅ Advanced neural processor imports correctly")

def test_neuro_africa_features_import():
    """Test that NeuroInsight-Africa features import correctly"""
    from neuroinsight_africa_complete import AdvancedNeuroAfricaFeatures
    
    africa_features = AdvancedNeuroAfricaFeatures()
    assert africa_features is not None
    assert hasattr(africa_features, 'ubuntu_philosophy')
    assert hasattr(africa_features, 'african_neural_patterns')
    assert hasattr(africa_features, 'neurotechnology_suite')
    assert hasattr(africa_features, 'global_research_databases')
    print("✅ NeuroInsight-Africa advanced features import correctly")

def test_environmental_simulation_basic_functionality():
    """Test basic environmental simulation functionality"""
    from environmental_simulation_complete import EnvironmentalSimulationComplete
    
    env_sim = EnvironmentalSimulationComplete()
    
    # Test retail store simulation
    params = {
        "store_type": "grocery",
        "layout_style": "grid",
        "lighting": 70,
        "noise_level": 30,
        "crowd_density": 50,
        "temperature": 72
    }
    
    results = env_sim.run_environmental_simulation("retail_store", params)
    
    assert "simulation_type" in results
    assert "environment_analysis" in results
    assert "neural_response" in results
    assert "behavioral_predictions" in results
    assert "optimization_recommendations" in results
    assert results["simulation_type"] == "retail_store"
    print("✅ Environmental simulation basic functionality works")

def test_advanced_neural_eeg_generation():
    """Test advanced neural EEG generation"""
    from neural_simulation import AdvancedNeuralProcessor
    
    processor = AdvancedNeuralProcessor()
    
    # Test white noise EEG baseline generation
    eeg_baseline = processor.generate_white_noise_eeg_baseline(
        duration=5.0,
        cultural_context="ubuntu"
    )
    
    assert "eeg_channels" in eeg_baseline
    assert "time_axis" in eeg_baseline
    assert "cultural_context" in eeg_baseline
    assert "metadata" in eeg_baseline
    assert eeg_baseline["cultural_context"] == "ubuntu"
    assert len(eeg_baseline["eeg_channels"]) > 0
    print("✅ Advanced neural EEG generation works")

def test_dark_matter_neural_patterns():
    """Test dark matter neural pattern simulation"""
    from neural_simulation import AdvancedNeuralProcessor
    
    processor = AdvancedNeuralProcessor()
    
    # Generate baseline first
    eeg_baseline = processor.generate_white_noise_eeg_baseline(duration=5.0)
    
    # Test dark matter patterns
    dark_matter = processor.simulate_dark_matter_neural_patterns(
        eeg_baseline, unconscious_stimulus="brand_logo"
    )
    
    assert "dark_matter_patterns" in dark_matter
    assert "unconscious_stimulus" in dark_matter
    assert "overall_influence" in dark_matter
    assert "unconscious_processing_score" in dark_matter
    assert dark_matter["unconscious_stimulus"] == "brand_logo"
    print("✅ Dark matter neural patterns work")

def test_neuro_africa_advanced_neurodata():
    """Test NeuroInsight-Africa advanced neurodata simulation"""
    from neuroinsight_africa_complete import AdvancedNeuroAfricaFeatures
    
    africa_features = AdvancedNeuroAfricaFeatures()
    
    # Test advanced neurodata simulation
    results = africa_features.run_advanced_neurodata_simulation(
        cultural_context="ubuntu",
        stimulus_type="brand_message"
    )
    
    assert "baseline_eeg" in results
    assert "dark_matter_patterns" in results
    assert "african_neural_patterns" in results
    assert "ubuntu_coherence_index" in results
    assert "cultural_authenticity_score" in results
    assert "market_relevance" in results
    print("✅ NeuroInsight-Africa advanced neurodata simulation works")

def test_automotive_analysis():
    """Test automotive neuromarketing analysis"""
    from environmental_simulation_complete import EnvironmentalSimulationComplete
    
    env_sim = EnvironmentalSimulationComplete()
    
    vehicle_data = {
        "brand": "Test Auto",
        "model": "Test SUV",
        "target_demographic": "luxury_buyers",
        "price_range": "luxury"
    }
    
    results = env_sim.run_automotive_analysis("exterior_design", vehicle_data)
    
    assert "component" in results
    assert "analysis_results" in results
    assert "recommendations" in results
    assert "market_insights" in results
    assert results["component"] == "exterior_design"
    print("✅ Automotive analysis works")

def test_cognitive_load_measurement():
    """Test cognitive load measurement"""
    from neural_simulation import AdvancedNeuralProcessor
    
    processor = AdvancedNeuralProcessor()
    
    # Generate baseline EEG
    eeg_data = processor.generate_white_noise_eeg_baseline(duration=5.0)
    
    # Test cognitive load measurement
    cog_load = processor.measure_cognitive_load(eeg_data, task_complexity="medium")
    
    assert "channel_load_indicators" in cog_load
    assert "global_cognitive_load" in cog_load
    assert "adjusted_cognitive_load" in cog_load
    assert "load_category" in cog_load
    assert "recommendations" in cog_load
    print("✅ Cognitive load measurement works")

def test_eri_nec_analysis():
    """Test ERI & NEC analysis with cultural weights"""
    from neuroinsight_africa_complete import AdvancedNeuroAfricaFeatures
    
    africa_features = AdvancedNeuroAfricaFeatures()
    
    stimulus_data = {
        "text": "Test marketing message",
        "community_focus": 0.7,
        "individual_appeal": 0.3,
        "traditional_values": 0.6,
        "modern_innovation": 0.4
    }
    
    results = africa_features.calculate_eri_nec_with_cultural_weights(
        stimulus_data=stimulus_data,
        cultural_context="ubuntu"
    )
    
    assert "emotional_resonance_index" in results
    assert "neural_engagement_coefficient" in results
    assert "combined_metrics" in results
    print("✅ ERI & NEC analysis with cultural weights works")

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Enhanced Neuromarketing Platform Tests")
    print("=" * 60)
    
    test_functions = [
        test_environmental_simulation_import,
        test_advanced_neural_processor_import,
        test_neuro_africa_features_import,
        test_environmental_simulation_basic_functionality,
        test_advanced_neural_eeg_generation,
        test_dark_matter_neural_patterns,
        test_neuro_africa_advanced_neurodata,
        test_automotive_analysis,
        test_cognitive_load_measurement,
        test_eri_nec_analysis
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {str(e)}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print(f"✅ Success Rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All tests passed! Enhanced neuromarketing platform is ready!")
    else:
        print("⚠️ Some tests failed. Review the implementation.")

if __name__ == "__main__":
    run_all_tests()