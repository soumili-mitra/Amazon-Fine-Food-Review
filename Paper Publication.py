# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:38:26 2023

@author: Soumili Mitra
"""

import pickle
import os
def classify_utterance(utt):
    # load the vectorizer
    loaded_vectorizer1 = pickle.load(open('vectorizer1.pickle', 'rb'))

    # load the model
    loaded_model1 = pickle.load(open('lr_classification1.model', 'rb'))

    loaded_vectorizer2 = pickle.load(open('vectorizer2.pickle', 'rb'))

    # load the model
    loaded_model2 = pickle.load(open('lr_classification2.model', 'rb'))
    loaded_vectorizer3 = pickle.load(open('vectorizer3.pickle', 'rb'))

    # load the model
    loaded_model3 = pickle.load(open('lr_classification3.model', 'rb'))
    loaded_vectorizer4 = pickle.load(open('vectorizer4.pickle', 'rb'))

    # load the model
    loaded_model4 = pickle.load(open('lr_classification4.model', 'rb'))
    loaded_vectorizer5 = pickle.load(open('vectorizer5.pickle', 'rb'))

    # load the model
    loaded_model5 = pickle.load(open('lr_classification5.model', 'rb'))
    loaded_vectorizer6 = pickle.load(open('vectorizer6.pickle', 'rb'))

    # load the model
    loaded_model6 = pickle.load(open('lr_classification6.model', 'rb'))
    # make a prediction
    va1=loaded_model1.predict(loaded_vectorizer1.transform([utt]))
    va2=loaded_model2.predict(loaded_vectorizer2.transform([utt]))
    va3=loaded_model3.predict(loaded_vectorizer3.transform([utt]))
    va4=loaded_model4.predict(loaded_vectorizer4.transform([utt]))
    va5=loaded_model5.predict(loaded_vectorizer5.transform([utt]))
    va6=loaded_model6.predict(loaded_vectorizer6.transform([utt]))
    
    total=(va1+va2+va3+va4+va5+va6)
    print(total)
    print(va1,va2,va3,va4,va5,va6)
    if(total>=5):
        print("Positive review")
    else:
        print("Negative review")
    
review = "Product was previously opened prior to shipping. Seal on jar was broken. Bubble wrap dirty and torn. This item should have never been shipped."
classify_utterance(review)