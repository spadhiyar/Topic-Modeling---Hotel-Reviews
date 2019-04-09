
# coding: utf-8

# # Business Impact 
# 
# Summarizing and categorizing reviews by their subject matter can save hundreds of reading hours. Instead, one can train a model on historical reviews and extract main topics mentioned. Then these topics can be used to quickly filter reviews and flag the ones requiring action. For example, the output of these insights can be used to address concerns about concierge service or to develop an approach for reaching out to unhappy customers. New incoming reviews can be automatically processed by the model, tagged with a topic, and passed along to appropriate channels for action. This also enables addressing issues quickly, and potentially retaining customers who otherwise would have churned.

# # Method: 
# 
# We will use a technique called topic modeling, more specifically Latent Dirichlet Allocation LDA. LDA is one of the most widely used techniques for topic modeling. It is a generative probability model, meaning there is an assumed underlying probability distribution which generates the set of documents on hand. This makes it an appealing modeling option, as the generative model allows for scoring unseen documents.
# 
# 

# In[ ]:


import pandas as pd
import numpy as np
import spacy
import re
import gensim
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from nltk import FreqDist

# libraries for visualization
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[336]:


df = pd.read_csv('central-bookingcom-review.csv',header=None)


# In[337]:


rev=list([])
for data in df[1]:
    regex = re.compile('[^a-zA-Z ]')
    review=regex.sub('', str(data))
    rev.append(review)
df_rev=pd.DataFrame(rev,columns={'cust_review'})


# In[338]:


df_rev.head(20)


# # function to plot most frequent terms

# In[297]:



def freq_words(x, terms = 35):
  all_words = ' '.join([text for text in x])
  all_words = all_words.split()

  fdist = FreqDist(all_words)
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

  # selecting top 20 most frequent words
  d = words_df.nlargest(columns="count", n = terms) 
  plt.figure(figsize=(20,5))
  ax = sns.barplot(data=d, x= "word", y = "count")
  ax.set(ylabel = 'Count')
  plt.show()


# In[298]:


freq_words(df_rev['cust_review'])


# Most common words are ‘the’, ‘and’, ‘was’, 'I','to, so on and so forth. These words are not so important for our task and they do not tell any story. We’ need to get rid of these kinds of words. 

# # Preprocessing Task :

# - Let’s try to remove the stopwords and short words (<2 letters) from the reviews

# In[299]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[300]:


# Function to remove stop words

def remove_stopword(reve):
    review_new = ' '.join([i for i in reve if i not in stop_words])
    return review_new

# removing short words (<3)

df_rev['cust_review'] = df_rev['cust_review'].apply(lambda x: ' '.join([i for i in x.split() if len(i)>2]))

#removing stop words

#reviews = [remove_stopword(r.split()) for r in reviews['customer_reviews']]
reviews = [remove_stopword(r.split()) for r in df_rev['cust_review']]

# convert entire text in to lower case

reviews = [r.lower() for r in reviews]


# In[301]:


len(reviews)


# In[302]:


freq_words(reviews, 35)


# In[303]:


import en_core_web_sm

nlp = en_core_web_sm.load()


# In[304]:


def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags ])
       return output


# Let’s tokenize the reviews and then lemmatize the tokenized reviews.

# In[305]:


tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
print(tokenized_reviews[5])


# In[306]:


len(tokenized_reviews)


# In[307]:


reviews_2 = lemmatization(tokenized_reviews)
print(reviews_2[5]) # print lemmatized review


# In[308]:


len(reviews_2)


# # Create Biagram & Trigram Models

# In[309]:


from gensim.models import Phrases

# Add bigrams and trigrams to reviews_2, minimum count 5 means only that appear 5 times or more.

bigram = Phrases(reviews_2, min_count=5)
trigram = Phrases(bigram[reviews_2])

for idx in range(len(reviews_2)):
    for token in bigram[reviews_2[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            reviews_2[idx].append(token)
    for token in trigram[reviews_2[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            reviews_2[idx].append(token)


# As you can see, we have not just lemmatized the words but also filtered only nouns and adjectives. Let’s de-tokenize the lemmatized reviews and plot the most common words.

# In[310]:


reviews_3 = []
for i in range(len(reviews_2)):
    reviews_3.append(' '.join(reviews_2[i]))

df_rev['reviews'] = reviews_3

freq_words(df_rev['reviews'], 35)


# It seems that now most frequent terms in our data are relevant. We can now go ahead and start building our topic model.

# # Building LDA model

# We will start by creating the term dictionary of our corpus, where every unique term is assigned an index & then we will Remove rare & common tokens 

# In[312]:


dictionary = corpora.Dictionary(reviews_2)

print('size of dictionary before filter: %d' % len(dictionary))


# In[313]:


dictionary.filter_extremes(no_below=2, no_above=0.2)

print('size of dictionary after filter: %d' % len(dictionary))


# Then we will convert the list of reviews (reviews_2) into a Document Term Matrix using the dictionary prepared above.

# In[314]:


doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]


# In[315]:


len(doc_term_matrix)


# # Training the Model

# In[316]:


# Creating the object for LDA model using gensim library

LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=2,random_state=0,
                chunksize=160, passes=50)


# In[317]:


lda_model.print_topics()


# In[318]:



ldatopics=lda_model.show_topics(formatted=False)


# In[319]:


ldatopics = [[word for word, prob in topic] for topicid, topic in ldatopics]


# In[320]:


lda_coherence = CoherenceModel(topics=ldatopics, texts=reviews_2, dictionary=dictionary, window_size=10, coherence='c_v').get_coherence()


# In[321]:


lda_coherence


# # Finding out the optimal number of topics

# Topic coherence in essence measures the human interpretability of a topic model. Traditionally perplexity has been used to evaluate topic models however this does not correlate with human annotations at times. Topic coherence is another way to evaluate topic models with a much higher guarantee on human interpretability. Thus this can be used to compare different topic models among many other use-cases.

# In[322]:


from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LDA(corpus=corpus, num_topics=num_topics, id2word=dictionary, random_state=0)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()
    
    return lm_list, c_v


# In[323]:


get_ipython().run_cell_magic('time', '', 'lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=doc_term_matrix, texts=reviews_2, limit=10)')


# # Topic coherence is a metric of topic quality, found to be correlated with human judgement. Higher topic coherence is better. We see that in this case, K = 2 has the highest coherence.
# 
# # For this reason we have trained our model choosing number of topic= 2 

# # Topics Visualization

# To visualize our topics in a 2-dimensional space we will use the pyLDAvis library. This visualization is interactive in nature and displays topics along with the most relevant words.

# pyLDAvis can be used to calculate term relevancy, with parameter  λ  controlling the order of terms for a selected topic. For example, topic 1 can be characterized by terms such as honeymoon, amazing, perfect. Decreasing  λ  brings up other positive terms such as romantic, incredible, etc. For brevity, we can assign label Happy Honeymooners to this topics. Similarly, other topics can also be assigned a label based on what terms customers tend to mention. Looks like historical reviews fall into one of the 2 main topics:
# 
# 1. Happy customers
# 2. Customer complaints

# In[324]:


# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, doc_term_matrix, dictionary)
vis


# In[325]:


name_dict = {   0: "Happy Customer", # 1 on the chart
                1: "Customer complaints",    # 2 on the chart
            }

scored = lda_model[doc_term_matrix]
#topic_prob = map(lambda x: max(x, key=lambda item: item[1]), scored)
#scored_reviews = pd.DataFrame(list(zip(reviews_2, topic_prob)), columns=['Review', 'Main_Topic'])
#scored_reviews[['Topic', 'Prob']] = scored_reviews['Main_Topic'].apply(pd.Series)
#scored_reviews['Topic Name'] = scored_reviews['Topic'].map(name_dict)


# In[326]:


topic_prob = map(lambda x: max(x, key=lambda item: item[1]), scored)
scored_reviews = pd.DataFrame(list(zip(reviews_2, topic_prob)), columns=['Review', 'Main_Topic'])


# In[327]:


scored_reviews[:10]


# In[328]:


len(scored_reviews)


# In[329]:


scored_reviews[['Topic', 'Prob']] = scored_reviews['Main_Topic'].apply(pd.Series)


# In[330]:


scored_reviews[:10]


# In[331]:


scored_reviews['Topic Name'] = scored_reviews['Topic'].map(name_dict)


# In[332]:


df_2 = scored_reviews['Topic Name'].value_counts(normalize=True)

plt.rcParams['axes.facecolor'] = 'white'
ax = df_2.plot(kind='barh', figsize=[8,6], title='Reviews Per Category', color='#33A5D3')
pos = df_2.index.get_loc('Customer Complains')
ax.patches[pos].set_facecolor('#aa3333')


# In[333]:


df_2


# # The goal of this analysis is twofold:
# 
# 1. Reduce number of human hours spent on review reading and categorizing
# 2. Increase customer retention by addressing concernes in a timely manner
# 
# # Saving human hours of reading through reviews will help to improve other customer satisfaction metrics such as:
# 
# 1. Response time
# 2. Number of review replies and complaints addressed
# 3. Customer likelihood to recommend (net promoter score)
# 4. Customer ratings
# 5. Customer retention

# # Conclusion:
# 
# - Now that we have trained the model and labeled review categories, the final and perhaps most important step is to utilize model results. In     this particular use case, one could consider:
# 
# - Feeding new review summaries along with labels based on found topics into a BI tool to be presented to customer service representatives
#   Problematic reviews can be filtered and brought to attention
# 
# - Topic categories can be tracked through time to monitor trends in customer feedback
# 
# - This application of topic modeling is meant to show how unsupervised models can save time by summarizing text into managable categories.       Every data set is different, and often topic modeling is just the first step in making sense of text data. 
# 
# - However, even with a simple model as demonstrated here, one can start gaining valuable insights about customers by quickly summarizing and     visualizing reviews.
