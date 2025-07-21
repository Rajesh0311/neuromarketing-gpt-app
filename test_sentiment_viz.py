# Quick test script - save this as test_sentiment_viz.py
from sentiment_visualizations import sentiment_viz
import streamlit as st

# Test data
test_sentiment_score = 0.75
test_sentiment_label = "positive"
test_emotions = {
    'Joy': 0.8,
    'Trust': 0.6, 
    'Anticipation': 0.7,
    'Surprise': 0.3,
    'Fear': 0.1,
    'Sadness': 0.1,
    'Disgust': 0.2,
    'Anger': 0.1
}

print("🧪 Testing Sentiment Visualizations...")

try:
    # Test gauge creation
    gauge_fig = sentiment_viz.create_sentiment_gauge(test_sentiment_score, test_sentiment_label)
    print("✅ Sentiment Gauge: SUCCESS")
    
    # Test radar chart
    radar_fig = sentiment_viz.create_emotion_radar_chart(test_emotions)
    print("✅ Emotion Radar: SUCCESS")
    
    # Test heatmap
    heatmap_fig = sentiment_viz.create_confidence_heatmap({})
    print("✅ Confidence Heatmap: SUCCESS")
    
    print("\n🎉 ALL VISUALIZATION TESTS PASSED!")
    print("Ready to integrate into main app...")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    print("Please check dependencies...")