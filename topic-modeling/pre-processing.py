import re
import numpy as np
import pandas as pd
import spacy
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
from joblib import Parallel, delayed
import multiprocessing

nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

# Load and clean data
df = pd.read_excel('./output_wos_file.xlsx', index_col=0)
df = df.drop_duplicates(subset='Abstract', keep='first').reset_index(drop=True)

# Extend stop words
stop_words = stopwords.words('english')
stop_words.extend(['method', 'system', 'say', 'claim', 'wherein', 'device', 'apparatus', 'one', 'id', 'least',
                         'infomration', 'models', 'derivative', 'thereon', 'using', 'two', 'based', 'add', 'additional',
                         'also', 'well', 'include', 'including', 'however', 'first', 'second', 'method', 'methodology',
                         'level', 'datum', 'model', 'models', 'modeling', 'energy', 'power', 'respectively', 'process',
                         'base', 'used', 'uses', 'well', 'datum', 'novelty', 'use', 'draw', 'drawing', 'high', 'end',
                         'provides', 'provider', 'providers', 'unit', 'data', 'study', 'problem', 'base', 'examines',
                         'relevant', 'application', 'propose', 'survey', 'practice', 'theoretical', 'project', 'journal',
                         'challenge', 'factor', 'concept', 'method', 'methodology', 'increase', 'decrease', 'increasing',
                         'decreasing', 'support', 'influence', 'field', 'from', 'active', 'question', 'present', 'highly',
                         'issue', 'challenge', 'challenges', 'method', 'variable', 'investigate', 'impact', 'provide',
                         'find', 'finding', 'determination', 'determine', 'understand', 'subject', 'provide', 're',
                         'address', 'edu', 'study', 'implement', 'implementation', 'studies', 'however', 'improve',
                         'improvement', 'identify', 'theory', 'role', 'identification', 'identifying', 'identified',
                         'different', 'result', 'results', 'examine', 'find', 'use', 'paper', 'article', 'main', 'purpose',
                         'review', 'application', 'discussion', 'including', 'includs', 'paper', 'specify', 'focus',
                         'approach', 'literature', 'elsevier', 'analysis', 'framework', 'conclusion', 'analyze', 'rate',
                         'increasing', 'increase', 'indicator', 'individual', 'increased', 'measuring', 'measure',
                         'right', 'reserved', 'associate', 'measuring', 'differ', 'different', 'compare', 'comparing',
                         'variable', 'year', 'time', 'characteristic', 'test', 'china', 'chines', 'contribution', 'insight',
                         'identifi', 'includ', 'design', 'perspective', 'context', 'different', 'discuss', 'initial',
                         'discussion', 'author', 'authors', 'including', 'includes', 'consideration', 'show', 'shows',
                         'found', 'effect', 'affect', 'effects', 'impact', 'aim', 'reserch', 'study', 'analysis',
                         'literature review', 'theoretical framework', 'research question', 'methodology', 'findings',
                         'discussion', 'implications', 'theoretical contributions', 'empirical evidence', 'research gap',
                         'research agenda', 'quantitative', 'qualitative', 'case study', 'survey', 'empirical', 'analysis',
                         'modeling', 'framework', 'methodology', 'measurement', 'evaluation', 'data collection',
                         'research design', 'hypothesis', 'statistical analysis', 'implement', 'numerous', 'lessons',
                         'involve', 'make', 'agenda', 'highlight', 'nevertheless', 'play', 'influence', 'create', 'use',
                         'maker', 'utilise', 'follow', 'change', 'futur', 'related', 'area', 'research', 'science',
                         'systematic', 'academic', 'practitioner', 'topic', 'trend', 'consider', 'object', 'current',
                         'content', 'analytical', 'key', 'public', 'important', 'exist', 'work', 'map', 'direct', 'set',
                         'resulting', 'obtained', 'designed', 'obtaining', 'gain', 'suggesting', 'suggested', 'certain',
                         'reveal', 'revealing', 'determined', 'determining', 'achieve', 'achieving', 'establish',
                         'established', 'determine', 'examined', 'examining', 'considered', 'considering', 'importantly',
                         'indicate', 'indicated', 'showed', 'showing', 'evaluated', 'evaluating', 'proposed', 'proposing',
                         'implemented', 'implementing', 'various', 'especially', 'including', 'include', 'examines',
                         'examined', 'example', 'particular', 'according', 'accordance', 'follows', 'followed',
                         'illustrated', 'illustrate', 'represented', 'represent', 'represents', 'representing',
                         'highlighted', 'highlight', 'briefly', 'important', 'importance', 'significantly', 'significant',
                         'common', 'comparatively', 'generally', 'general', 'specific', 'especially', 'particularly',
                         'particular', 'distinct', 'differentiate', 'different', 'known', 'well-known', 'acknowledge',
                         'acknowledged', 'acknowledging', 'consideration', 'considering', 'consider', 'examining',
                         'examines', 'considered', 'focus', 'focusing', 'focused', 'highlight', 'highlighting',
                         'highlighted', 'analyze', 'analyzing', 'analyzed', 'analyze', 'clarified', 'clarify',
                         'clarifying', 'elaborate', 'elaborating', 'elaborated', 'discussed', 'discussing', 'discusses',
                         'discussion', 'discuss', 'evaluate', 'evaluating', 'evaluated', 'evaluate', 'examine', 'examining',
                         'examined', 'examine', 'explore', 'exploring', 'explored', 'explore', 'identify', 'identifying',
                         'identified', 'identify', 'illustrate', 'illustrating', 'illustrated', 'illustrate',
                         'investigate', 'investigating', 'investigated', 'investigate', 'observe', 'observing', 'observed',
                         'observe', 'outline', 'outlining', 'outlined', 'outline', 'present', 'presenting', 'presented',
                         'present', 'propose', 'proposing', 'proposed', 'propose', 'recognize', 'recognizing',
                         'recognized', 'recognize', 'show', 'showing', 'shown', 'show', 'speculate', 'speculating',
                         'speculated', 'speculate', 'suggest', 'suggesting', 'suggested', 'suggest'])
  # Keeping the original stop words here for brevity

# Precompile regex patterns for faster reuse
email_pattern = re.compile(r'\S*@\S*\s?')
digit_pattern = re.compile(r'\d+')
punctuation_pattern = re.compile(r'[^\w\s]')
newline_pattern = re.compile(r'\s+')
single_quote_pattern = re.compile(r"\'")

def clean_text(text):
    text = email_pattern.sub('', text)
    text = digit_pattern.sub('', text)
    text = punctuation_pattern.sub('', text)
    text = text.lower()
    text = newline_pattern.sub(' ', text)
    return single_quote_pattern.sub('', text)

def tokenize(text):
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    return [token for token in tokens if token not in stop_words]

# Parallel processing for batch text cleaning
def parallel_apply(func, data, n_jobs=None):
    n_jobs = n_jobs or multiprocessing.cpu_count()
    with Parallel(n_jobs=n_jobs) as parallel:
        return parallel(delayed(func)(item) for item in data)

# Clean text and tokenize in parallel
def safe_clean_text(text):
    if pd.isna(text) or text == '':
        return ''
    return clean_text(text)

df['text'] = df['Abstract'].apply(safe_clean_text)
df['tokens'] = parallel_apply(tokenize, df['text'])

# Bigrams and trigrams
bigram = Phrases(df['tokens'], min_count=10, threshold=50)
bigram_phraser = Phraser(bigram)
trigram = Phrases(bigram_phraser[df['tokens']], min_count=10, threshold=50)
trigram_phraser = Phraser(trigram)

def apply_phraser(doc):
    return trigram_phraser[bigram_phraser[doc]]

# Apply bigrams in parallel
df['bigrams'] = parallel_apply(apply_phraser, df['tokens'])

# Batch lemmatization with SpaCy
def batch_lemmatize(texts, allowed_postags=['NOUN', 'VERB', 'ADJ', 'ADV']):
    docs = nlp.pipe(texts, batch_size=50, disable=["parser", "ner"])
    return [[token.lemma_ for token in doc if token.pos_ in allowed_postags and token.lemma_ not in stop_words] for doc in docs]

df['lemmatized'] = batch_lemmatize(df['bigrams'].apply(' '.join))

# Parallel stemming
p = PorterStemmer()
def stem(text):
    return [p.stem(token) for token in text]

df['stemmed'] = parallel_apply(stem, df['lemmatized'])
