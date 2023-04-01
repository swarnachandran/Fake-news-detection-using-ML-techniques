import urllib.request,sys,time
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re


filename = "Politifact.csv"
f = open(filename,"w", encoding = 'utf-8')
headers = "Date, Link, Statement, Tags, Source, Article_Body, Label, References\n"
f.write(headers)

upperframe = [] 


for page in range(92, 154):
    print('processing page :', page)
    url = 'https://www.politifact.com/factchecks/list/?page='+str(page)
    print(url)
    
    #an exception might be thrown, so the code should be in a try-except block
    try:
        #use the browser to get the url. This is suspicious command that might blow up.
        page=requests.get(url)                             # this might throw an exception if something goes wrong.
    
    except Exception as e:                                   # this describes what to do if an exception is thrown
        error_type, error_obj, error_info = sys.exc_info()      # get the exception information
        print ('ERROR FOR LINK:',url)                          #print the link that cause the problem
        print (error_type, 'Line:', error_info.tb_lineno)     #print error info and line that threw the exception
        continue                                              #ignore this page. Abandon this and go back.
    
    time.sleep(2)   
    soup = BeautifulSoup(page.text,'html.parser')
    
    frame = []
    
    links = soup.find_all('li',attrs={'class':'o-listicle__item'})
    
    for j in links:
        
        Statement = j.find("div",attrs={'class':'m-statement__quote'}).text.strip()
        
        Link = "https://www.politifact.com"
        Link += j.find("div",attrs={'class':'m-statement__quote'}).find('a')['href'].strip()
        
        Source = j.find('div', attrs={'class':'m-statement__meta'}).find('a').text.strip()
        Label = j.find('div', attrs ={'class':'m-statement__content'}).find('img',attrs={'class':'c-image__original'}).get('alt').strip()
        
        # get the article
        article = requests.get(Link) 
        article_soup = BeautifulSoup(article.text, "html.parser")
        
        # article's date         
        # extract from the article
        #Date = article_soup.find('span', attrs ={'class':'m-author__date'}).text.strip()
        
        ## in case of error, extract from the list - NOTE: inspect the TEXT INDEXES before proceeding
        Date = j.find('div',attrs={'class':'m-statement__body'}).find('footer').text[-18:-1].strip()

        # tags
        tag_links = article_soup.find_all('li', attrs={'class':'m-list__item'})
        tags = ''
        for j in tag_links:
            tag = j.text.strip()
            tags += (tag + ', ')
        Tags = tags[:-2]
        
        # article body
        Article_Body = article_soup.find('article', attrs={'class':'m-textblock'}).text.strip()
        
        #references
        ref_links = article_soup.find_all('div',attrs={'class':'t-row__center'})
        References = ref_links[-1].find("article",attrs={'class':'m-superbox__content'}).text.strip()
        
        # add all elements in a table
        frame.append((Date, Link, Statement, Tags, Source, Article_Body, Label, References))
        f = open(filename,"w", encoding = 'utf-8')
        f.write(Date.replace(",","^") + "," + Link + "," + Statement.replace(",","^") + "," + Tags \
                + Source.replace(",","^") + "," + Article_Body + Label.replace(",","^") + References + "\n")
        f.close()
        
    upperframe.extend(frame)
   
data = pd.DataFrame(upperframe, columns=['Date','Link','Statement','Tags','Source',
                                         'Article_Body','Label','References'])

data.shape
data.tail()


data.to_excel('Politifact_page_55_to_153.xlsx')

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics 
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import itertools
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


read_file = pd.read_excel (r'C:/Users/Dell/Downloads/project/Politifact_page_55_to_153.xlsx')
read_file.to_csv (r'C:/Users/Dell/Downloads/project/Politifact_page_55_to_153.csv', index = None, header=True)

df = pd.read_csv('C:/Users/Dell/Downloads/project/Politifact_page_55_to_153.csv')
df.shape
df.head()
#DataFlair - Get the labels
labels=df.Label
labels.head()

df.shape
df.head()
#DataFlair - Get the labels
labels=df.Label
labels.head()
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['Statement'], labels, test_size=0.2, random_state=7)
#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
#print(f'Accuracy: {round(score*100)}%')
#DataFlair - Build confusion matrix
cm=confusion_matrix(y_test,y_pred, labels=['barely-true','pants-fire','false'])

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="red" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
              horizontalalignment="center",
              color="red" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('RISK')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
plot_confusion_matrix(cm,target_names=['barely-true','pants-fire','false'])

