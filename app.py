from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.naive_bayes import MultinomialNB

# Loading Model

filename='model.pkl'
clf=pickle.load(open(filename,'rb'))
cv=pickle.load(open('transform.pkl','rb'))

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
#    message=pd.read_csv('clean_text.csv')


#x=message['message']
#y=message['label']

#lm=WordNetLemmatizer()
#corpus=[]
#for i in range(0,len(message)):
#    words=re.sub('[^a-zA-Z]',' ',message['message'][i])
#    words=words.lower()
#    words=words.split()
#    words=[lm.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
#    words=' '.join(words)
#    corpus.append(words)
#    cv=TfidfVectorizer('english')
#    x=cv.fit_transform(corpus).toarray()
#    y=pd.get_dummies(message['label'])
#    y=y.iloc[:,1].values
#    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=30)
#    mnb = MultinomialNB(alpha=0.2)
#    model=mnb.fit(x_train,y_train)
#    predict=model.predict(x_test)
#    score=accuracy_score(y_test,predict)
#

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect) 
    return render_template('result.html', prediction = my_prediction)

if __name__ == "__main__":
    app.run(debug=True)


