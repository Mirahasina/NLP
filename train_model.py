import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from collections import Counter
import heapq
import re

# We define the class here so it can be pickled
class SummarizerFinal:
    def __init__(self):
        self.idf_weights = {
            'en': {},
            'fr': {}
        }

    def train(self, texts, lang='en'):
        print(f"Training TF-IDF for {lang} on {len(texts)} articles...")
        vectorizer = TfidfVectorizer(max_features=20000, stop_words='english' if lang=='en' else None)
        vectorizer.fit(texts)
        self.idf_weights[lang] = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

    def summarize(self, text, n=3, nlp=None, stop_words=None, threshold=0.8):
        if not text or len(text) < 10: return ""
        if not nlp: return ""
        
        lang_code = 'en' if nlp.meta['lang'] == 'en' else 'fr'
        idfs = self.idf_weights.get(lang_code, {})
        
        doc = nlp(text)
        sentences = [sent for sent in doc.sents if len(sent.text.split()) >= 8] # Minimum 8 words for quality
        if not sentences: return ""
        
        # Scoring logic
        scored_sentences = []
        for sent in sentences:
            # Filter structural headers
            text_s = sent.text.strip()
            # Penalize if starts with bullet/number or follows "I. " pattern
            if re.match(r'^(\d+|[IVX]+)[\.\)]', text_s) or text_s.endswith(':') or '%' in text_s:
                score = 0
            else:
                # Use IDF if available, otherwise 1.0 (frequency)
                score = sum(idfs.get(t.lemma_.lower(), 1.0) for t in sent if not t.is_stop and t.pos_ in ['NOUN', 'ADJ', 'VERB'])
            
            # Length normalization (optional)
            score = score / (len(sent)**0.8 + 1)
            scored_sentences.append((sent, score))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        final_sentences = []
        selected_texts = []
        
        for sent, score in scored_sentences:
            if len(final_sentences) >= n: break
            if score <= 0: continue
            
            is_redundant = False
            for prev_text in selected_texts:
                s1 = set(sent.text.lower().split())
                s2 = set(prev_text.lower().split())
                if not s1 or not s2: continue
                overlap = len(s1 & s2) / min(len(s1), len(s2))
                if overlap > threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                final_sentences.append(sent)
                selected_texts.append(sent.text)
        
        return "\n- " + "\n- ".join([s.text.strip() for s in sorted(final_sentences, key=lambda x: x.start)])

if __name__ == "__main__":
    model = SummarizerFinal()
    
    # Load EN data
    en_path = 'cnn_dailymail/test.csv'
    if os.path.exists(en_path):
        df_en = pd.read_csv(en_path)
        model.train(df_en['article'].dropna().tolist(), lang='en')
    
    # Load FR data
    fr_path = 'orangesum/orangesum_test.csv'
    if os.path.exists(fr_path):
        df_fr = pd.read_csv(fr_path)
        # In OrangeSum, content is usually in 'text' column, check after dir list
        col = 'text' if 'text' in df_fr.columns else df_fr.columns[1]
        model.train(df_fr[col].dropna().tolist(), lang='fr')
        
    with open('modele_final.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved to modele_final.pkl")
