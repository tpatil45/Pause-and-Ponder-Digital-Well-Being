from app import app

import nltk
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.decomposition import LatentDirichletAllocation, NMF

nltk.download('punkt')
nltk.download('wordnet')
wnl = nltk.WordNetLemmatizer()

@app.route('/')
def index():
	return render_template('public/index.html')

@app.route('/', methods=['POST'])
def upload_file():
	file_type = request.form.get('file_type')
	if file_type == "Browsing":
	    uploaded_file = request.files['file']
	    if uploaded_file.filename != '':
	        uploaded_file.save(uploaded_file.filename)
	        #Importing Data
	        input_total = pd.read_json(uploaded_file.filename)
	        #Data with Time in Docs
	        input = pd.json_normalize(input_total['Browser History'])
	        input['time_usec'] = pd.to_datetime(input['time_usec'], unit='us')
	        input['dates'] = input['time_usec'].dt.date
	        input['months'] = input['time_usec'].dt.month
	        input['years'] = input['time_usec'].dt.year
	        input['month-year'] = input['time_usec'].dt.to_period('M')
	        input = input.sort_values(by='dates', ascending=False)

	else:
	    uploaded_file = request.files['file']
	    if uploaded_file.filename != '':
	        uploaded_file.save(uploaded_file.filename)
	        #Importing Data
	        input = pd.read_json(uploaded_file.filename)
	        #Data with Time in Docs
	        input['time_usec'] = pd.to_datetime(input['time'])
	        input['dates'] = input['time_usec'].dt.date
	        input['month-year'] = input['time_usec'].dt.to_period('M')
	        input = input.sort_values(by='dates', ascending=False)

	docs  = input['title'].values

	added_stopwords = frozenset(['searched','search','searched for'])
	my_stopwords = stop_words.ENGLISH_STOP_WORDS.union(added_stopwords)
	lemma_count_vectorizer = TfidfVectorizer(encoding = 'latin-1', lowercase = True, min_df = 10, token_pattern='[a-z][a-z][a-z]+',
	                                        max_df = 0.5, stop_words = my_stopwords, ngram_range = (1, 2))

	vecs = lemma_count_vectorizer.fit_transform(docs)
	# check the size of the constructed vocabulary
	#print(len(lemma_count_vectorizer.vocabulary_))
	vecs_feature_names = lemma_count_vectorizer.get_feature_names()

	# Run NMF
	no_topics = 20
	nmf = NMF(n_components = no_topics, random_state = 1, alpha = 0.1, l1_ratio = .5, init = 'nndsvd').fit(vecs)

	#Getting top words for each topic
	no_top_words = 10
	Topic_Words = {}
	for topic_idx, topic in enumerate(nmf.components_):
	    Topic_Words.__setitem__(topic_idx,
	                    [vecs_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
	    


	# Create Month - Topic Matrix
	nmf_output = nmf.transform(vecs)
	tvtv = nmf_output / nmf_output.sum(axis=1, keepdims=True)
	Doc_Top_pd = pd.DataFrame(tvtv)
	Doc_Top_pd['month-year'] = input['month-year'].values
	df = Doc_Top_pd.groupby("month-year").sum()
	df = df.transpose()



	#Getting top 10 words for each month
	max_index = df.idxmax()
	significant_words = []
	for index in max_index:
	    for topic in Topic_Words:
	        if index == topic:
	            significant_words.append(Topic_Words[index])



	#Final Dataframe
	year_month = df.columns.astype(str).tolist()
	output = pd.DataFrame(significant_words, columns = ['Word 1', 'Word 2', 'Word 3', 'Word 4', 'Word 5',
	                                            'Word 6', 'Word 7', 'Word 8', 'Word 9', 'Word 10'],index=year_month)
	#output.insert(loc=0, column='Year-Month', value=year_month)
	
	output = output.to_html()
	args = True

	return render_template('public/index.html', args=args, data = output)#redirect(url_for('output'))#render_template('output.html')#, output = output)
