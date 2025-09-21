#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
Run this before deploying to Streamlit Cloud
"""

def test_imports():
    """Test all required imports"""
    try:
        print("Testing imports...")
        
        # Core imports
        import streamlit as st
        print("✅ streamlit")
        
        import pandas as pd
        print("✅ pandas")
        
        import numpy as np
        print("✅ numpy")
        
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ plotly")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
        
        import seaborn as sns
        print("✅ seaborn")
        
        from wordcloud import WordCloud
        print("✅ wordcloud")
        
        import nltk
        print("✅ nltk")
        
        import textblob
        print("✅ textblob")
        
        import spacy
        print("✅ spacy")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ scikit-learn")
        
        import PyPDF2
        print("✅ PyPDF2")
        
        import docx2txt
        print("✅ docx2txt")
        
        import requests
        print("✅ requests")
        
        import yfinance as yf
        print("✅ yfinance")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    if success:
        print("\n✅ Ready for deployment!")
    else:
        print("\n❌ Fix import errors before deploying")
