# Quick test script - save as test_sa_viz.py
from sa_cultural_visualizations import sa_cultural_viz

print("🧪 Testing SA Cultural Visualizations...")

try:
    # Test Ubuntu gauge
    ubuntu_gauge = sa_cultural_viz.create_ubuntu_compatibility_gauge(0.75)
    print("✅ Ubuntu Compatibility Gauge: SUCCESS")
    
    # Test cultural radar
    cultural_radar = sa_cultural_viz.create_cultural_sensitivity_radar({})
    print("✅ Cultural Sensitivity Radar: SUCCESS")
    
    # Test language meter
    language_meter = sa_cultural_viz.create_language_respect_meter({})
    print("✅ Language Respect Meter: SUCCESS")
    
    # Test regional chart
    regional_chart = sa_cultural_viz.create_regional_appropriateness_chart({})
    print("✅ Regional Appropriateness Chart: SUCCESS")
    
    print("\n🎉 ALL SA CULTURAL VISUALIZATION TESTS PASSED!")
    print("Ready to integrate into Tab 2...")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("Please check dependencies...")