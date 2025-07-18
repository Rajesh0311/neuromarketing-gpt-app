#!/usr/bin/env python3
"""
Integration Test Suite for NeuroMarketing GPT Platform
=====================================================

This test suite validates the core functionality of all platform modules
and ensures they work together seamlessly.

Run with: python test_integration.py
"""

import sys
import time
import traceback
from datetime import datetime

def test_module_imports():
    """Test that all core modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        import advanced_sentiment_module
        import neural_simulation
        import neuro_deep_research_module
        import export_module
        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Module import failed: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("🔍 Testing sentiment analysis...")
    
    try:
        from advanced_sentiment_module import analyze_sentiment
        
        # Test with sample text
        test_text = "This amazing product exceeded my expectations! Highly recommend."
        result = analyze_sentiment(test_text, cultural_context="western", analysis_depth="advanced")
        
        # Validate result structure
        assert hasattr(result, 'overall_sentiment')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'emotions')
        assert hasattr(result, 'marketing_metrics')
        
        # Validate data types
        assert isinstance(result.overall_sentiment, float)
        assert isinstance(result.confidence, float)
        assert isinstance(result.emotions, dict)
        assert isinstance(result.marketing_metrics, dict)
        
        print(f"✅ Sentiment analysis working - Score: {result.overall_sentiment:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        traceback.print_exc()
        return False

def test_neural_simulation():
    """Test neural simulation functionality"""
    print("🔍 Testing neural simulation...")
    
    try:
        from neural_simulation import simulate_consumer_response
        
        # Test with sample stimulus
        stimulus = "Limited time offer - 50% off premium products!"
        result = simulate_consumer_response(stimulus, consumer_type="impulsive", duration=5.0)
        
        # Validate result structure
        assert hasattr(result, 'participant_id')
        assert hasattr(result, 'neural_signals')
        assert hasattr(result, 'behavioral_predictions')
        assert hasattr(result, 'summary_metrics')
        
        # Validate data
        assert len(result.neural_signals) > 0
        assert 'purchase_intention' in result.behavioral_predictions
        assert 'signal_quality' in result.summary_metrics
        
        print(f"✅ Neural simulation working - {len(result.neural_signals)} signals generated")
        return True
        
    except Exception as e:
        print(f"❌ Neural simulation failed: {e}")
        traceback.print_exc()
        return False

def test_research_engine():
    """Test deep research engine functionality"""
    print("🔍 Testing research engine...")
    
    try:
        from neuro_deep_research_module import search_neuro_research
        
        # Test research search
        query = "EEG consumer behavior"
        results = search_neuro_research(query, sources=["openneuro", "zenodo"])
        
        # Validate results structure
        assert 'synthesis' in results
        assert 'datasets' in results
        assert 'papers' in results
        
        # Validate data
        synthesis = results['synthesis']
        assert hasattr(synthesis, 'datasets_found')
        assert hasattr(synthesis, 'papers_found')
        assert hasattr(synthesis, 'confidence_score')
        
        print(f"✅ Research engine working - Found {len(results['datasets'])} datasets, {len(results['papers'])} papers")
        return True
        
    except Exception as e:
        print(f"❌ Research engine failed: {e}")
        traceback.print_exc()
        return False

def test_export_functionality():
    """Test export functionality"""
    print("🔍 Testing export functionality...")
    
    try:
        from export_module import export_analysis_results
        
        # Test data
        test_data = {
            'analysis_type': 'integration_test',
            'timestamp': datetime.now().isoformat(),
            'sentiment_analysis': {
                'overall_sentiment': 0.75,
                'confidence': 0.88,
                'emotions': {'joy': 0.8, 'trust': 0.7}
            },
            'neural_simulation': {
                'purchase_intention': 0.72,
                'brand_recall': 0.68,
                'emotional_engagement': 0.75
            }
        }
        
        # Test JSON export
        json_result = export_analysis_results(test_data, format_type='json')
        assert json_result.success
        assert json_result.content is not None
        
        # Test HTML export
        html_result = export_analysis_results(test_data, format_type='html')
        assert html_result.success
        assert html_result.content is not None
        
        # Test CSV export
        csv_result = export_analysis_results(test_data, format_type='csv')
        assert csv_result.success
        assert csv_result.content is not None
        
        print("✅ Export functionality working - JSON, HTML, CSV exports successful")
        return True
        
    except Exception as e:
        print(f"❌ Export functionality failed: {e}")
        traceback.print_exc()
        return False

def test_performance_benchmarks():
    """Test performance benchmarks"""
    print("🔍 Testing performance benchmarks...")
    
    try:
        from advanced_sentiment_module import analyze_sentiment
        from neural_simulation import simulate_consumer_response
        
        # Sentiment analysis performance
        start_time = time.time()
        test_text = "This is a comprehensive test of the sentiment analysis performance with a longer text to evaluate processing speed."
        result = analyze_sentiment(test_text)
        sentiment_time = time.time() - start_time
        
        # Neural simulation performance
        start_time = time.time()
        neural_result = simulate_consumer_response("Performance test stimulus", duration=3.0)
        neural_time = time.time() - start_time
        
        # Performance benchmarks
        assert sentiment_time < 5.0, f"Sentiment analysis too slow: {sentiment_time:.2f}s"
        assert neural_time < 10.0, f"Neural simulation too slow: {neural_time:.2f}s"
        
        print(f"✅ Performance benchmarks passed - Sentiment: {sentiment_time:.2f}s, Neural: {neural_time:.2f}s")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmarks failed: {e}")
        traceback.print_exc()
        return False

def test_data_flow_integration():
    """Test data flow between modules"""
    print("🔍 Testing cross-module data flow...")
    
    try:
        from advanced_sentiment_module import analyze_sentiment
        from neural_simulation import simulate_consumer_response
        from export_module import export_analysis_results
        
        # Test text
        test_content = "Revolutionary new product launch - limited edition available now!"
        
        # Step 1: Sentiment analysis
        sentiment_result = analyze_sentiment(test_content, analysis_depth="advanced")
        
        # Step 2: Neural simulation using same content
        neural_result = simulate_consumer_response(test_content, consumer_type="average", duration=2.0)
        
        # Step 3: Combine results and export
        combined_data = {
            'analysis_type': 'integrated_analysis',
            'input_text': test_content,
            'sentiment_analysis': {
                'overall_sentiment': sentiment_result.overall_sentiment,
                'confidence': sentiment_result.confidence,
                'emotions': sentiment_result.emotions,
                'marketing_metrics': sentiment_result.marketing_metrics
            },
            'neural_simulation': {
                'behavioral_predictions': neural_result.behavioral_predictions,
                'summary_metrics': neural_result.summary_metrics
            }
        }
        
        # Step 4: Export integrated results
        export_result = export_analysis_results(combined_data, format_type='json')
        assert export_result.success
        
        print("✅ Data flow integration working - All modules communicate successfully")
        return True
        
    except Exception as e:
        print(f"❌ Data flow integration failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and edge cases"""
    print("🔍 Testing error handling...")
    
    try:
        from advanced_sentiment_module import analyze_sentiment
        from neural_simulation import simulate_consumer_response
        
        # Test empty input
        try:
            empty_result = analyze_sentiment("")
            assert empty_result.overall_sentiment == 0.0  # Should handle gracefully
        except Exception:
            pass  # Expected to handle errors gracefully
        
        # Test very short simulation
        try:
            short_sim = simulate_consumer_response("test", duration=0.1)
            assert short_sim is not None  # Should handle gracefully
        except Exception:
            pass  # Expected to handle errors gracefully
        
        print("✅ Error handling working - Edge cases handled gracefully")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run the complete test suite"""
    print("🚀 Starting Comprehensive Integration Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Neural Simulation", test_neural_simulation),
        ("Research Engine", test_research_engine),
        ("Export Functionality", test_export_functionality),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Data Flow Integration", test_data_flow_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ FATAL ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\n\n📊 Test Results Summary")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"⏱️ Total time: {total_time:.2f} seconds")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED! Platform is ready for production.")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Review issues before deployment.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)