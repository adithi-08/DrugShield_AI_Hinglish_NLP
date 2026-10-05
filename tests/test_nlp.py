import pytest
from src.ai_pipeline.nlp_classifier import HinglishSlangDetector

def test_hinglish_slang_normalization():
    detector = HinglishSlangDetector()
    raw_text = "bhaaiii 2 pudya score karna hai"
    normalized = detector.normalize_text(raw_text)
    
    assert "drug_packet" in normalized
    assert "buy_drugs" in normalized

def test_suspicious_message_detection():
    detector = HinglishSlangDetector()
    suspicious_msg = "chitta aur meow meow ka setting kara de"
    res = detector.analyze(suspicious_msg)
    
    assert res["is_suspicious"] is True
    assert res["risk_score"] >= 0.5
    assert "heroin" in res["mapped_categories"]