# === Imports ===
import os
import logging
from datetime import datetime
import uuid
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
from collections import defaultdict
from urllib.parse import urlparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import textstat
from bert_score import score as bert_scorer
from sentence_transformers.util import cos_sim
from newsapi import NewsApiClient
import trafilatura
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import chromadb
import torch
import re
import spacy
from operator import itemgetter
import nltk
from sklearn.cluster import AgglomerativeClustering
import difflib
import config
from database import SessionLocal, MonitoredTopic, NewsArticle, AnalysisResult
from ingestion import fetch_and_store_news, fetch_and_store_tweets

# === Initial Setup ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nltk.download('punkt', quiet=True)

# === Model Loading ===
# Load summarizer, embeddings, zero-shot classifier, and spaCy NER
summarizer_model_path = os.path.join(os.path.dirname(__file__), "model", "t5-news-summarizer-ft")
summarize_with_ft = None
try:
    device = 0 if torch.cuda.is_available() else -1
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': device_str})
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    if os.path.exists(summarizer_model_path):
        logging.info(f"Loading fine-tuned summarization model from: {summarizer_model_path}")
        tokenizer_ft = AutoTokenizer.from_pretrained(summarizer_model_path)
        model_ft = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_path).to(device_str)
        def summarize_with_ft_loaded(text, max_length=180, min_length=40):
            input_ids = tokenizer_ft("summarize: " + text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device_str)
            output_ids = model_ft.generate(input_ids, max_length=max_length, min_length=min_length, length_penalty=2.0)
            return tokenizer_ft.decode(output_ids[0], skip_special_tokens=True)
        summarize_with_ft = summarize_with_ft_loaded
    else:
        raise FileNotFoundError("Fine-tuned model directory not found.")
except Exception as e:
    logging.warning(f"Could not load fine-tuned model due to: {e}. Falling back to default summarizer.")
    try:
        summarize_with_ft = lambda x, **kwargs: pipeline('summarization', model='facebook/bart-large-cnn', device=device)(x, max_length=180, min_length=40, do_sample=False)[0]['summary_text']
    except Exception as e2:
        logging.error(f"Could not load fallback summarizer pipeline: {e2}")
        summarize_with_ft = lambda x, **kwargs: "[Summarization unavailable: model loading failed.]"

# Load spaCy model for NER
try:
    nlp_spacy = spacy.load('en_core_web_sm')
except Exception as e:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp_spacy = spacy.load('en_core_web_sm')

# === Utility Functions ===
# Article fetching, text cleaning, deduplication, summary quality checks, event extraction

def fetch_articles_newsapi(topic: str, num_articles: int = 8) -> list:
    """Fetches and cleans news articles using NewsAPI and trafilatura. Hard cap at 8 articles for speed."""
    num_articles = min(num_articles, 8)  # Hard cap for on-demand analysis
    print(f"[Progress] Fetching articles for topic: {topic} (max {num_articles} articles)")
    newsapi = NewsApiClient(api_key=config.NEWS_API_KEY)
    articles_raw = newsapi.get_everything(q=topic, language='en', sort_by='relevancy', page_size=num_articles)
    articles = []
    for a in articles_raw.get("articles", []):
        if a.get("url") and a.get("title") != "[Removed]":
            try:
                cleaned_text = trafilatura.extract(trafilatura.fetch_url(a["url"]), include_comments=False, include_tables=False, no_fallback=True)
                if cleaned_text: articles.append({**a, "text": cleaned_text})
            except Exception as e: logging.error(f"Failed to process URL {a['url']}: {e}")
    print(f"[Progress] Successfully processed and cleaned {len(articles)} articles.")
    logging.info(f"Successfully processed and cleaned {len(articles)} articles.")
    return articles

def clean_and_capitalize(text):
    # Remove duplicate lines and capitalize sentences
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    seen = set()
    unique_lines = []
    for l in lines:
        if l.lower() not in seen:
            unique_lines.append(l)
            seen.add(l.lower())
    text = ' '.join(unique_lines)
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s[:1].upper() + s[1:] if s else '' for s in sentences]
    cleaned = ' '.join(sentences)
    if cleaned and cleaned[-1] not in '.!?':
        cleaned += '.'
    return cleaned

def is_summary_acceptable(summary):
    # Check for repetition, shortness, or placeholder text in the summary
    lines = [l.strip() for l in summary.split('.') if l.strip()]
    if len(lines) < 2:
        return False
    if len(set(lines)) <= 1:
        return False
    if any('---' in l for l in lines):
        return False
    if any('paralysis' in l and lines.count(l) > 1 for l in lines):
        return False
    if summary.lower().startswith('not enough'):
        return False
    return True

def is_repetitive(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    return len(set(lines)) <= 2 or all(lines[0] == l for l in lines)

def filter_repetitive_lines(text):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    seen = []
    filtered = []
    for l in lines:
        if all(difflib.SequenceMatcher(None, l.lower(), s.lower()).ratio() < 0.85 for s in seen) and len(l.split()) > 5:
            filtered.append(l)
            seen.append(l)
    return '\n'.join(filtered)

def is_summary_repetitive(summary):
    lines = [l.strip() for l in summary.split('.') if l.strip()]
    if not lines:
        return True
    first = lines[0]
    return all(difflib.SequenceMatcher(None, first, l).ratio() > 0.85 for l in lines)

def extract_events(text, pub_date=None, source=None):
    # Extracts events from text using spaCy NER
    events = []
    doc = nlp_spacy(text)
    for sent in doc.sents:
        verb = None
        subj = None
        obj = None
        for token in sent:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                verb = token
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subj = token
            if token.dep_ in ('dobj', 'pobj', 'attr', 'oprd'):
                obj = token
        if verb and subj:
            event_text = sent.text.strip()
            events.append({
                "date": pub_date,
                "event": event_text,
                "source": source
            })
    return events

# === Core Analysis Logic ===
# Main analysis function, perspective extraction, evaluation, bias detection, etc.

def _perform_core_analysis(topic: str, articles: list) -> dict:
    import time
    step_times = {}
    t_start = time.time()
    step_times['start'] = t_start
    logging.info(f"[Timing] Core analysis started at {datetime.now().isoformat()}")
    if summarize_with_ft is None:
        raise RuntimeError("Summarizer function is not loaded. Summarization is unavailable.")
    logging.info(f"[Step] Starting core analysis for topic: {topic}")
    if not articles:
        logging.info("[Step] No articles to analyze. Returning default response.")
        t_end = time.time()
        logging.info(f"[Timing] Core analysis finished at {datetime.now().isoformat()} (Duration: {t_end-t_start:.2f}s)")
        return {
            "executive_summary": "No summary generated.",
            "detailed_bias_report": "No bias report generated.",
            "perspectives": [{"label": "Neutral Perspective", "summary": "No perspectives generated.", "evidence": []}],
            "summary_evaluation": {},
            "visualizations": {},
            "entities": [],
            "timeline": []
        }
    # --- Document Preparation ---
    t_docs = time.time()
    logging.info("[Timing] Preparing documents and splitting text...")
    docs = [Document(page_content=f"Title: {a['title']}\n{a['text']}", metadata={'source': a['url']}) for a in articles]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    client = chromadb.Client()
    collection = client.create_collection(name=f"news_{topic.replace(' ', '_')}_{uuid.uuid4().hex[:8]}")
    if splits:
        collection.add(documents=[s.page_content for s in splits], ids=[f"doc_{i}" for i in range(len(splits))], metadatas=[s.metadata for s in splits])
    t_docs_end = time.time()
    logging.info(f"[Timing] Document prep/splitting took {t_docs_end-t_docs:.2f}s")
    step_times['doc_prep'] = t_docs_end - t_docs
    # --- Context Query ---
    t_context = time.time()
    query = f"Provide a comprehensive overview of the key events and perspectives related to {topic}"
    context_result = collection.query(query_texts=[query], n_results=10)
    context_chunks = None
    if context_result and 'documents' in context_result and context_result['documents'] and context_result['documents'][0]:
        context_chunks = context_result['documents'][0]
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant information found."
    t_context_end = time.time()
    logging.info(f"[Timing] Context query took {t_context_end-t_context:.2f}s")
    step_times['context_query'] = t_context_end - t_context
    # --- Executive Summary ---
    t_summary = time.time()
    logging.info("[Step] Generating executive summary...")
    fallback_text = filter_repetitive_lines('\n'.join([a['text'] for a in articles[:10] if a.get('text')]))
    fallback_text = '\n'.join(sorted(set(fallback_text.split('\n'))))  # Remove near-duplicates strictly
    logging.info(f"[Debug] Input to summarizer (length {len(fallback_text)}): {repr(fallback_text[:500])} ...")
    if not context_chunks or is_repetitive(context):
        if not fallback_text or len(fallback_text.split()) < 50:
            summary_report = "Not enough diverse information found to generate a meaningful summary."
        else:
            prompt = f"Summarize the following news articles about '{topic}':\n{fallback_text}"
            summary_report = summarize_with_ft(prompt, max_length=350, min_length=100)
            if is_summary_repetitive(summary_report):
                summary_report = "Not enough diverse information found to generate a meaningful summary."
    else:
        prompt = f"As an expert news analyst, synthesize the following context about '{topic}' into a detailed, structured summary with these sections: Overall Situation, Key Driving Factors, Diverging Viewpoints, Political, Economic, Regional, and Social Perspectives, and Outlook.\n\nCONTEXT:\n---\n{context}\n---\nANALYST SUMMARY:"
        logging.info(f"[Debug] Input to summarizer (length {len(context)}): {repr(context[:500])} ...")
        summary_report = summarize_with_ft(prompt, max_length=350, min_length=100)
    summary_report = clean_and_capitalize(summary_report)
    if not is_summary_acceptable(summary_report):
        logging.warning(f"[Warning] Generated summary was not acceptable. Replacing with fallback message. Original: {repr(summary_report[:300])}")
        summary_report = "Not enough diverse or high-quality information found to generate a meaningful summary."
    t_summary_end = time.time()
    logging.info(f"[Timing] Executive summary generation took {t_summary_end-t_summary:.2f}s")
    step_times['summary'] = t_summary_end - t_summary
    # --- Perspective Extraction ---
    t_persp = time.time()
    logging.info("[Step] Extracting and clustering perspectives...")
    candidate_labels = ["political", "economic", "regional", "social", "environmental", "technological"]
    all_chunks = []
    for a in articles:
        text = a.get('text', '')
        if not text:
            continue
        doc = nlp_spacy(text)
        for sent in doc.sents:
            chunk = sent.text.strip()
            if chunk and len(chunk) > 30:
                all_chunks.append(chunk)
    # Deduplicate chunks
    seen_chunks = set()
    unique_chunks = []
    for c in all_chunks:
        c_norm = c.lower().strip()
        if c_norm not in seen_chunks:
            unique_chunks.append(c)
            seen_chunks.add(c_norm)
    # Embed all unique chunks
    logging.info(f"[Step] Embedding {len(unique_chunks)} sentences for clustering...")
    t_embed = time.time()
    chunk_embeddings = embedding_model.embed_documents(unique_chunks)
    t_embed_end = time.time()
    logging.info(f"[Timing] Embedding took {t_embed_end-t_embed:.2f}s")
    step_times['embedding'] = t_embed_end - t_embed
    # Cluster embeddings
    n_clusters = min(6, len(unique_chunks))
    if n_clusters < 2:
        n_clusters = 2
    t_cluster = time.time()
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clustering.fit_predict(chunk_embeddings)
    t_cluster_end = time.time()
    logging.info(f"[Timing] Clustering took {t_cluster_end-t_cluster:.2f}s")
    step_times['clustering'] = t_cluster_end - t_cluster
    # Group chunks by cluster
    clusters = [[] for _ in range(n_clusters)]
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(unique_chunks[idx])
    # For each cluster, assign a perspective label by majority vote (using zero-shot classification)
    perspectives = []
    t_zs = time.time()
    for cluster_chunks in clusters:
        if not cluster_chunks:
            continue
        label_scores = {lbl: 0.0 for lbl in candidate_labels}
        for chunk in cluster_chunks:
            result = zero_shot_classifier(chunk, candidate_labels=candidate_labels, multi_label=True)
            if isinstance(result, dict) and 'labels' in result and 'scores' in result:
                for lbl, score in zip(result['labels'], result['scores']):
                    if lbl in label_scores:
                        label_scores[lbl] += score
        best_label = max(label_scores, key=lambda k: label_scores[k])
        combined = '\n'.join(cluster_chunks[:15])
        perspective_prompt = f"As a news analyst, summarize the {best_label} perspective on '{topic}' from the following news coverage:\n{combined}"
        try:
            perspective_summary = summarize_with_ft(perspective_prompt, max_length=180, min_length=40)
        except Exception:
            perspective_summary = f"Not enough data for {best_label} perspective."
        perspective_summary = clean_and_capitalize(perspective_summary)
        evidence_texts = sorted(cluster_chunks, key=len, reverse=True)[:5]
        perspectives.append({
            "label": f"{best_label.capitalize()} Perspective",
            "summary": perspective_summary,
            "evidence": evidence_texts
        })
    t_zs_end = time.time()
    logging.info(f"[Timing] Zero-shot classification and perspective summarization took {t_zs_end-t_zs:.2f}s")
    step_times['zero_shot'] = t_zs_end - t_zs
    t_persp_end = time.time()
    logging.info(f"[Timing] Perspective extraction (total) took {t_persp_end-t_persp:.2f}s")
    step_times['perspective'] = t_persp_end - t_persp
    if not perspectives:
        perspectives = [{"label": "Neutral Perspective", "summary": "A summary of the neutral perspective.", "evidence": []}]
    logging.info("[Step] Perspective extraction complete.")
    # --- LLM Output Evaluation Metrics ---
    t_eval = time.time()
    logging.info("[Step] Calculating summary evaluation metrics...")
    try:
        ref_texts = [a['text'] for a in articles[:3] if a['text']]
        P, R, F1 = bert_scorer([summary_report], ref_texts, lang="en")
        faithfulness = float(F1.mean())
    except Exception:
        faithfulness = 0.0
    try:
        readability = float(textstat.textstat.flesch_reading_ease(summary_report))
    except Exception:
        readability = 0.0
    try:
        topic_emb = embedding_model.embed_query(topic)
        summary_emb = embedding_model.embed_query(summary_report)
        relevance = float(cos_sim(np.array([topic_emb]), np.array([summary_emb]))[0][0])
    except Exception:
        relevance = 0.0
    try:
        sentences = summary_report.split('.')
        if len(sentences) > 1:
            emb = [embedding_model.embed_query(s) for s in sentences if s.strip()]
            coherence = float(np.mean([cos_sim(np.array([emb[i]]), np.array([emb[i+1]]))[0][0] for i in range(len(emb)-1)]))
        else:
            coherence = 1.0
    except Exception:
        coherence = 0.0
    evaluation_results = {
        "Faithfulness (BERTScore)": round(faithfulness, 3),
        "Readability (Flesch)": round(readability, 2),
        "Relevance": round(relevance, 3),
        "Coherence": round(coherence, 3)
    }
    t_eval_end = time.time()
    logging.info(f"[Timing] Evaluation metrics took {t_eval_end-t_eval:.2f}s")
    step_times['evaluation'] = t_eval_end - t_eval
    logging.info("[Step] Summary evaluation metrics complete.")
    # --- Graph: Source Bias Comparison ---
    t_vis = time.time()
    logging.info("[Step] Generating source bias chart...")
    sources = [urlparse(a['url']).netloc for a in articles if a.get('url')]
    source_counts = pd.Series(sources).value_counts()
    source_bias = [float(getattr(TextBlob(str(a['text'])).sentiment, 'polarity', 0.0)) for a in articles if a.get('text')]
    source_bias_dict = defaultdict(list)
    for a in articles:
        if a.get('url') and a.get('text'):
            source_bias_dict[urlparse(a['url']).netloc].append(float(getattr(TextBlob(str(a['text'])).sentiment, 'polarity', 0.0)))
    avg_source_bias = {k: np.mean(v) for k, v in source_bias_dict.items()}
    os.makedirs(config.VISUALS_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_chart_path = os.path.join(config.VISUALS_PATH, f"source_bias_{timestamp}.png")
    if avg_source_bias:
        plt.figure(figsize=(8,4))
        plt.bar(list(avg_source_bias.keys()), list(avg_source_bias.values()), color='#3498db')
        plt.xlabel('Source')
        plt.ylabel('Average Sentiment (Polarity)')
        plt.title('Source Bias Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(source_chart_path)
        plt.close()
        source_chart_url = f"/static/visuals/{os.path.basename(source_chart_path)}"
    else:
        source_chart_url = None
    t_vis_end = time.time()
    logging.info(f"[Timing] Source bias chart generation took {t_vis_end-t_vis:.2f}s")
    step_times['visualization'] = t_vis_end - t_vis
    logging.info("[Step] Source bias chart complete.")
    # --- Richer Bias Detection ---
    t_bias = time.time()
    logging.info("[Step] Running richer bias detection (stance, subjectivity, framing, claims)...")
    stance_labels = ["pro", "anti", "neutral"]
    stance_by_source = defaultdict(list)
    subjectivity_by_source = defaultdict(list)
    framing_by_source = defaultdict(int)
    claim_by_source = defaultdict(list)
    hedging_words = ["allegedly", "reportedly", "claims", "suggests", "may", "might", "could", "possibly", "uncertain", "experts say", "according to"]
    factual_patterns = ["is", "are", "was", "were", "has", "have", "will", "can", "must", "should"]
    for a in articles:
        text = a.get('text', '')
        source = urlparse(a.get('url', '')).netloc if a.get('url') else 'unknown'
        if not text:
            continue
        stance_result = zero_shot_classifier(text, candidate_labels=stance_labels, hypothesis_template=f"This text is {{}} towards {topic}.")
        if isinstance(stance_result, dict) and 'labels' in stance_result and 'scores' in stance_result:
            label = stance_result['labels'][0]
            score = stance_result['scores'][0]
            stance_by_source[source].append((label, score))
        subj = float(getattr(TextBlob(str(text)).sentiment, 'subjectivity', 0.0))
        subjectivity_by_source[source].append(subj)
        hedges = sum(1 for w in hedging_words if w in text.lower())
        framing_by_source[source] += hedges
        doc = nlp_spacy(text)
        for sent in doc.sents:
            s = sent.text.strip()
            if any(fp in s for fp in factual_patterns) and len(s.split()) > 6:
                claim_by_source[source].append(s)
    stance_dist = {src: {lbl: 0 for lbl in stance_labels} for src in stance_by_source}
    for src, stances in stance_by_source.items():
        for label, _ in stances:
            stance_dist[src][label] += 1
    avg_subjectivity = {src: round(np.mean(vals), 3) for src, vals in subjectivity_by_source.items() if vals}
    bias_report = "\nStance Distribution by Source:\n"
    for src, dist in stance_dist.items():
        bias_report += f"  {src}: {dist}\n"
    bias_report += "\nAverage Subjectivity by Source:\n"
    for src, val in avg_subjectivity.items():
        bias_report += f"  {src}: {val}\n"
    bias_report += "\nFraming/Hedging Count by Source:\n"
    for src, val in framing_by_source.items():
        bias_report += f"  {src}: {val}\n"
    bias_report += "\nSample Claims by Source:\n"
    for src, claims in claim_by_source.items():
        bias_report += f"  {src}:\n"
        for c in claims[:3]:
            bias_report += f"    - {c}\n"
    t_bias_end = time.time()
    logging.info(f"[Timing] Richer bias detection took {t_bias_end-t_bias:.2f}s")
    step_times['bias_detection'] = t_bias_end - t_bias
    logging.info("[Step] Richer bias detection complete.")
    # --- Logical Fallacy Detection (Rule-Based) ---
    t_fallacy = time.time()
    logging.info("[Step] Detecting logical fallacies (rule-based)...")
    fallacy_patterns = {
        "ad hominem": ["attack the person", "personal attack", "you are just", "your character"],
        "strawman": ["misrepresent", "distort the argument", "that's not what I said"],
        "slippery slope": ["will lead to", "inevitably result in", "domino effect"],
        "appeal to authority": ["experts say", "according to authority", "as an expert"],
        "false dilemma": ["either or", "no other option", "only two choices"],
        "bandwagon": ["everyone is", "most people", "the majority"],
        "red herring": ["irrelevant", "off topic", "beside the point"],
        "appeal to emotion": ["feel sorry", "pity", "emotional response", "heartbreaking"],
    }
    detected_fallacies = []
    for a in articles:
        text = a.get('text', '')
        source = urlparse(a.get('url', '')).netloc if a.get('url') else 'unknown'
        if not text:
            continue
        doc = nlp_spacy(text)
        for sent in doc.sents:
            s = sent.text.strip().lower()
            for fallacy, patterns in fallacy_patterns.items():
                for pat in patterns:
                    if pat in s:
                        detected_fallacies.append({
                            "source": source,
                            "fallacy": fallacy,
                            "sentence": sent.text.strip()
                        })
    bias_report += "\nLogical Fallacies Detected (Rule-Based):\n"
    if detected_fallacies:
        for f in detected_fallacies[:10]:
            bias_report += f"  [{f['fallacy']}] {f['sentence']} (Source: {f['source']})\n"
        if len(detected_fallacies) > 10:
            bias_report += f"  ...and {len(detected_fallacies)-10} more.\n"
    else:
        bias_report += "  None detected.\n"
    t_fallacy_end = time.time()
    logging.info(f"[Timing] Logical fallacy detection took {t_fallacy_end-t_fallacy:.2f}s")
    step_times['fallacy_detection'] = t_fallacy_end - t_fallacy
    logging.info("[Step] Logical fallacy detection complete.")
    # --- Visualization: stance and subjectivity by source ---
    t_vis2 = time.time()
    logging.info("[Step] Generating stance and subjectivity charts...")
    stance_chart_path = None
    stance_chart_data = None
    try:
        if stance_dist:
            stance_df = pd.DataFrame(stance_dist).fillna(0)
            stance_df = stance_df.T
            stance_df.plot(kind='bar', stacked=True, figsize=(8,4))
            plt.xlabel('Source')
            plt.ylabel('Article Count')
            plt.title('Stance Distribution by Source')
            plt.tight_layout()
            stance_chart_path = os.path.join(config.VISUALS_PATH, f"stance_dist_{timestamp}.png")
            plt.savefig(stance_chart_path)
            plt.close()
            stance_chart_data = {
                "labels": list(stance_df.index),
                "pro": stance_df["pro"].tolist() if "pro" in stance_df else [],
                "anti": stance_df["anti"].tolist() if "anti" in stance_df else [],
                "neutral": stance_df["neutral"].tolist() if "neutral" in stance_df else [],
            }
    except Exception:
        stance_chart_path = None
        stance_chart_data = None
    subjectivity_chart_path = None
    try:
        if avg_subjectivity:
            plt.figure(figsize=(8,4))
            plt.bar(list(avg_subjectivity.keys()), list(avg_subjectivity.values()), color='#8e44ad')
            plt.xlabel('Source')
            plt.ylabel('Avg. Subjectivity')
            plt.title('Average Subjectivity by Source')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            subjectivity_chart_path = os.path.join(config.VISUALS_PATH, f"subjectivity_{timestamp}.png")
            plt.savefig(subjectivity_chart_path)
            plt.close()
    except Exception:
        subjectivity_chart_path = None
    t_vis2_end = time.time()
    logging.info(f"[Timing] Stance/subjectivity chart generation took {t_vis2_end-t_vis2:.2f}s")
    step_times['stance_subjectivity_charts'] = t_vis2_end - t_vis2
    logging.info("[Step] Stance and subjectivity charts complete.")
    visualizations = {}
    if source_chart_url:
        visualizations["source_bias_chart_url"] = source_chart_url
    else:
        visualizations["source_bias_chart_url"] = None
    if stance_chart_path:
        visualizations["stance_dist_chart_url"] = f"/static/visuals/{os.path.basename(stance_chart_path)}"
    if stance_chart_data:
        visualizations["stance_dist_chart_data"] = stance_chart_data
    if subjectivity_chart_path:
        visualizations["subjectivity_chart_url"] = f"/static/visuals/{os.path.basename(subjectivity_chart_path)}"
    # --- Entity Extraction & Timeline ---
    t_entity = time.time()
    logging.info("[Step] Generating named entities and timeline...")
    entity_counter = {}
    timeline_events = []
    for a in articles:
        text = a.get('text', '')
        if not text:
            continue
        doc = nlp_spacy(text)
        for ent in doc.ents:
            key = (ent.text.strip(), ent.label_)
            entity_counter[key] = entity_counter.get(key, 0) + 1
        pub_date = a.get('publishedAt')
        source = a.get('url')
        events = extract_events(text, pub_date, source)
        timeline_events.extend(events)
    sorted_entities = sorted([
        {"text": k[0], "type": k[1], "count": v}
        for k, v in entity_counter.items()
    ], key=lambda x: x['count'], reverse=True)
    timeline_sorted = sorted([e for e in timeline_events if e['date']], key=lambda x: x['date'])
    t_entity_end = time.time()
    logging.info(f"[Timing] Entity extraction and timeline took {t_entity_end-t_entity:.2f}s")
    step_times['entity_timeline'] = t_entity_end - t_entity
    t_end = time.time()
    logging.info(f"[Timing] Core analysis finished at {datetime.now().isoformat()} (Total duration: {t_end-t_start:.2f}s)")
    logging.info(f"[Timing] Step durations: {step_times}")
    return {
        "executive_summary": summary_report,
        "detailed_bias_report": bias_report,
        "perspectives": perspectives,
        "summary_evaluation": evaluation_results,
        "visualizations": visualizations,
        "entities": sorted_entities,
        "timeline": timeline_sorted
    }

# === Workflows ===
# On-demand and background analysis entry points

def run_on_demand_analysis(topic_name: str) -> dict:
    import time
    t_start = time.time()
    print(f"[Progress] ===== STARTING ON-DEMAND ANALYSIS FOR: '{topic_name}' =====")
    logging.info(f"===== STARTING ON-DEMAND ANALYSIS FOR: '{topic_name}' =====")
    articles_as_dict = [{ "url": a.get("url"), "title": a.get("title"), "text": a.get("text"), "sentiment": float(getattr(TextBlob(str(a.get("text",""))).sentiment, 'polarity', 0.0)), "publishedAt": a.get("publishedAt")} for a in fetch_articles_newsapi(topic_name, num_articles=8)]
    t_fetch_end = time.time()
    logging.info(f"[Timing] Article fetch/clean took {t_fetch_end-t_start:.2f}s")
    if not articles_as_dict: raise ValueError("No articles found for this topic. Please try a different query.")
    t_analysis_start = time.time()
    analysis_results = _perform_core_analysis(str(topic_name), articles_as_dict)
    t_analysis_end = time.time()
    logging.info(f"[Timing] Core analysis for topic '{topic_name}' took {t_analysis_end-t_analysis_start:.2f}s")
    t_end = time.time()
    print(f"[Progress] ✅ Successfully completed on-demand analysis for '{topic_name}'.")
    logging.info(f"✅ Successfully completed on-demand analysis for '{topic_name}'.")
    logging.info(f"[Timing] Total on-demand analysis duration: {t_end-t_start:.2f}s")
    return {"topic": topic_name, **analysis_results}

def run_full_analysis_for_topic(topic: MonitoredTopic):
    print(f"[Progress] ===== STARTING BACKGROUND ANALYSIS FOR TOPIC: '{topic.name}' =====")
    logging.info(f"===== STARTING BACKGROUND ANALYSIS FOR TOPIC: '{topic.name}' =====")
    with SessionLocal() as db:
        fetch_and_store_news(topic, db)
        fetch_and_store_tweets(topic, db)
        articles_to_analyze = db.query(NewsArticle).filter_by(topic_id=topic.id, is_analyzed=False).all()
        if not articles_to_analyze:
            logging.info(f"No new articles for '{topic.name}' to analyze. Skipping report generation.")
            return
        articles_as_dict = [{"url": a.url, "title": a.title, "text": a.text, "sentiment": float(getattr(TextBlob(str(a.text)).sentiment, 'polarity', 0.0)), "publishedAt": a.published_at} for a in articles_to_analyze]
        analysis_results = _perform_core_analysis(str(topic.name), articles_as_dict)
        filtered_results = {k: analysis_results[k] for k in [
            "executive_summary", "detailed_bias_report", "perspectives", "summary_evaluation", "visualizations"
        ] if k in analysis_results}
        new_report = AnalysisResult(topic_id=topic.id, **filtered_results)
        db.add(new_report)
        for article in articles_to_analyze:
            setattr(article, 'is_analyzed', True)
        db.commit()
    print(f"[Progress] ✅ Successfully saved a new background analysis report for '{topic.name}'.")
    logging.info(f"✅ Successfully saved a new background analysis report for '{topic.name}'.")