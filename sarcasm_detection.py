import gensim
import numpy as np 
import pandas as pd
import os
import pickle
from collections import Counter

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.stem.wordnet import WordNetLemmatizer

#libaries for topic modeling
import gensim
from gensim import corpora
from pprintpp import pprint # pretty print 

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,15)

from wordcloud import WordCloud 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


import re
from bs4 import BeautifulSoup

# output dir and create dir
print("creating folder for output")
output_dir = os.getcwd()+"/output"
os.makedirs(output_dir, exist_ok=True)
output_gensim = output_dir + "/topic_model/"
os.makedirs(output_gensim, exist_ok=True)

cache_dir = os.path.join(os.getcwd()+"/cache", "sarcasm detection")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)

os.chmod(output_dir, int('660', base=8))

def preprocess_data(df, cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):

    cache_data = None

    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass

    if cache_data is None:
        #df = df.drop(columns=['article_link'], axis=0) # drop a article_link column
        #print(df['is_sarcastic'].value_counts())
        print("....convert from sentences to words.....")
        df['words'] = df['headline'].apply(headline_to_words)
        if cache_file is not None:
            cache_data = dict(cache_df = df)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)

    
    cache_df = (cache_data['cache_df'])
    return cache_df

def headline_to_words(headline):
    
    stemmer = WordNetLemmatizer()
    
    text = BeautifulSoup(headline, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [stemmer.lemmatize(w) for w in words] # stem
    
    return words

# return Top words
def get_top_n_words(words, n=50):
    counter = Counter()
    for i in words:
        counter.update(i)
    return counter.most_common(n)


# Topic Extractor Function
def topic_extractor(corpus, NUM_TOPICS):
    ldamodel = None
    if not os.path.isfile(output_gensim + "/model6.gensim"):
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15, alpha='auto',
                                                per_word_topics=True)
        ldamodel.save(output_gensim + "/model6.gensim")
        return ldamodel
    else:
        return gensim.models.LdaModel.load(output_gensim+ "/model6.gensim")

# WordCloud Ploting
def WordCloud_Save(worddict, save_loc):
    wordcloud = WordCloud(min_word_length = 3,
                      background_color='white', width=900, height=300)

    wordcloud.generate_from_frequencies(worddict)
    wordcloud.to_file(save_loc)
    

def cnf_mtx(ytest, ypred, filename):
    import seaborn as sns
    import matplotlib.pyplot as plt 
    from sklearn.metrics import confusion_matrix
    plt.clf()
    cm = confusion_matrix(ytest,ypred)
    print("----confusion matrix-----")
    print(cm)
    plt.title(filename.split(".")[0]) # title with fontsize 20

    plot = sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted labels') # x-axis label with fontsize 15
    plt.ylabel('True labels') # y-axis label with fontsize 15
    plot.get_figure().savefig(filename)


if __name__ == "__main__":
    #dataset examining..
    df = pd.read_json("dataset/Sarcasm_Headlines_Dataset.json", lines=True)
    
    cache_df = preprocess_data(df)
    features = cache_df['words']
    labels = cache_df['is_sarcastic']


    # Topic Modeling... to get 6 topics related to news headline dataset.
    print("....Started Topic Modelling with genism .....")
    dictionary = corpora.Dictionary(features.values)
    corpus = [dictionary.doc2bow(text) for text in features.values]

    topics = topic_extractor(corpus, 6).print_topics(num_words=10) # topic modeling for to find 6 topics related to dataset

    for each_topic in topics:
        pprint(each_topic)

    # EDA for top 50 words and worcloud for 200 frequently sarcastic words.

    top_50_words = get_top_n_words(features, 50)
    
    # ploting a bar graph for top 50 words
    names, values = zip(*top_50_words) 
    plt.clf() # clear and new graph
    plt.barh(range(len(top_50_words)), values)
    plt.yticks(range(len(names)),names)
    plt.xlabel("Frequency of Words")
    plt.ylabel("Words")
    plt.title("Top 50 Words")
    plt.savefig("top_50_words.png")


    #worldcloud 
    sarcastic_200_words = get_top_n_words(cache_df[cache_df['is_sarcastic']== 1]['words'], 200) #sarcastic 200 most common words  
    non_sarcastic_200_words = get_top_n_words(cache_df[cache_df['is_sarcastic']== 0]['words'], 200) #non sarcastic 200 most common words 

    print("-------- wordcloud for sarcastic most common 200 words----")
    WordCloud_Save(dict(sarcastic_200_words), "sarcastic_200_words.png" )
    print("-------- wordcloud for non sarcastic most common 200 words----")
    WordCloud_Save(dict(non_sarcastic_200_words), "non_sarcastic_200_words.png")


    #Machine Learning model buliding.

    #Transforming from words to numeric form by TFID Vectorizer
    print("-------- Transforming from words to numeric form ----")
    #
    features = features.apply(lambda x: ' '.join(x))
    tfid = TfidfVectorizer(ngram_range=(1,1))
    features = tfid.fit_transform(features).toarray()
    print(features.shape)

    

    #train test split the dataset for training and testing of ML model.
    print("-------- Spliting Dataset into train and test ----")
    Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.1, random_state=0)


    #Logisitic Regression Model
    print("-------- Started Logistic Regression Modeling ----")
    log_r = LogisticRegression()
    log_r.fit(Xtrain, ytrain)
    log_r_predict = log_r.predict(Xtest)

    cnf_mtx(ytest, log_r_predict, "Logistic-Confusion Matrix.png")
 

    print("ROC AUC score: %.3f " % roc_auc_score(ytest, log_r_predict)) # ROC
    print(classification_report(ytest, log_r_predict))
    print("Accuracy: %.3f" % accuracy_score(ytest, log_r_predict))

    ns_probs = [0 for _ in range(len(ytest))]
    lr_probs = log_r.predict_proba(Xtest)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(ytest, ns_probs)
    lr_auc = roc_auc_score(ytest, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs,)
    lr_fpr, lr_tpr, _ = roc_curve(ytest, lr_probs)
    # plot the roc curve for the model
    plt.clf() # clear and new graph
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.title('False Positive Rate vs True Positive Rate')
    plt.legend()
    # show the plot
    plt.savefig("Logistic ROC.png")

    #SVC Model 
    print("-------- Started SVC Modeling ----")
    svc = LinearSVC()
    svc.fit(Xtrain, ytrain)
    svc_predict = svc.predict(Xtest)
    cnf_mtx(ytest, svc_predict, "SVC-Confusion Matrix.png")
  

    print("ROC AUC score: %.3f " % roc_auc_score(ytest, svc_predict)) #ROC
    print(classification_report(ytest, svc_predict))
    print("Accuracy: %.3f" % accuracy_score(ytest, svc_predict))
    ns_probs = [0 for _ in range(len(ytest))]
    svc_probs = svc._predict_proba_lr(Xtest)
    # keep probabilities for the positive outcome only
    svc_probs = svc_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(ytest, ns_probs)
    svc_auc = roc_auc_score(ytest, svc_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('SVC: ROC AUC=%.3f' % (svc_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs,)
    svc_fpr, svc_tpr, _ = roc_curve(ytest, svc_probs)
    # plot the roc curve for the model
    plt.clf() # clear and new graph
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(svc_fpr, svc_tpr, marker='.', label='SVC')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.title('False Positive Rate vs True Positive Rate')
    plt.legend()
    # show the plot
    plt.savefig("SVC ROC.png")

    # Navies Basis Model
    print("-------- Started GaussianNB Modeling ----")

    nb = GaussianNB()
    nb.fit(Xtrain, ytrain)
    nb_predict = nb.predict(Xtest)

    cnf_mtx(ytest, nb_predict, "GaussianNB-Confusion Matrix.png")
 

    print("ROC AUC score: %.3f" % roc_auc_score(ytest, nb_predict)) #ROC
    print(classification_report(ytest, nb_predict))
    print("Accuracy: %.3f " % accuracy_score(ytest, nb_predict))

    ns_probs = [0 for _ in range(len(ytest))]
    nb_probs = nb.predict_proba(Xtest)
    # keep probabilities for the positive outcome only
    nb_probs = nb_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(ytest, ns_probs)
    nb_auc = roc_auc_score(ytest, nb_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('NB : ROC AUC=%.3f' % (nb_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs,)
    nb_fpr, nb_tpr, _ = roc_curve(ytest, nb_probs)
    # plot the roc curve for the model
    plt.clf() # clear and new graph
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(nb_fpr, nb_tpr, marker='.', label='NB')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.title('False Positive Rate vs True Positive Rate')
    plt.legend()
    # show the plot
    plt.savefig("GaussianNB ROC.png")

    # RandomForest Model 
    print("-------- Started RandomForest Classifer Modeling ----")
    rnf = RandomForestClassifier(n_estimators=10)
    rnf.fit(Xtrain, ytrain)
    rnf_predict = rnf.predict(Xtest)

    cnf_mtx(ytest, rnf_predict, "RandomForest-Confusion Matrix.png")

    print("ROC AUC score: %.3f" % roc_auc_score(ytest, rnf_predict))
    print(classification_report(ytest, log_r_predict)) #Classification Report
    print("Accuracy: %.3f" % accuracy_score(ytest, rnf_predict))

    ns_probs = [0 for _ in range(len(ytest))]
    rnf_probs = rnf.predict_proba(Xtest)
    # keep probabilities for the positive outcome only
    rnf_probs = rnf_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(ytest, ns_probs)
    rnf_auc = roc_auc_score(ytest, rnf_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('RandomForest : ROC AUC=%.3f' % (nb_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(ytest, ns_probs,)
    rnf_fpr, rnf_tpr, _ = roc_curve(ytest, rnf_probs)
    # plot the roc curve for the model
    plt.clf() # clear and new graph
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(rnf_fpr, rnf_tpr, marker='.', label='RandomForest')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.title('False Positive Rate vs True Positive Rate')
    plt.legend()
    # show the plot
    plt.savefig("RandomForest ROC.png")

    print("-------- Ended Modeling ----")



