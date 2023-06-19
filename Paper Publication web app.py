# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:38:26 2023

@author: Soumili Mitra
"""

import pickle
import os
import streamlit as st
from PIL import Image
def classify_utterance(utt):
    # load the vectorizer
    loaded_vectorizer1 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/vectorizer1.pickle', 'rb'))

    # load the model
    loaded_model1 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/lr_classification1.model', 'rb'))

    loaded_vectorizer2 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/vectorizer2.pickle', 'rb'))

    # load the model
    loaded_model2 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/lr_classification2.model', 'rb'))
    loaded_vectorizer3 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/vectorizer3.pickle', 'rb'))

    # load the model
    loaded_model3 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/lr_classification3.model', 'rb'))
    loaded_vectorizer4 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/vectorizer4.pickle', 'rb'))

    # load the model
    loaded_model4 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/lr_classification4.model', 'rb'))
    loaded_vectorizer5 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/vectorizer5.pickle', 'rb'))

    # load the model
    loaded_model5 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/lr_classification5.model', 'rb'))
    loaded_vectorizer6 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/vectorizer6.pickle', 'rb'))

    # load the model
    loaded_model6 = pickle.load(open('E:/St.Xaviers notes/semester 4/Paper Publlication/lr_classification6.model', 'rb'))
    # make a prediction
    va1=loaded_model1.predict(loaded_vectorizer1.transform([utt]))
    va2=loaded_model2.predict(loaded_vectorizer2.transform([utt]))
    va3=loaded_model3.predict(loaded_vectorizer3.transform([utt]))
    va4=loaded_model4.predict(loaded_vectorizer4.transform([utt]))
    va5=loaded_model5.predict(loaded_vectorizer5.transform([utt]))
    va6=loaded_model6.predict(loaded_vectorizer6.transform([utt]))
    
    total=(va1+va2+va3+va4+va5+va6)
    col1, col2 = st.columns(2)
    if(total>=5):
        image = Image.open('E:/St.Xaviers notes/semester 4/Paper Publlication/5.jpg')
        col1.markdown("# Positive")
        col2.image(image)
    else:
        image = Image.open('E:/St.Xaviers notes/semester 4/Paper Publlication/1.jpg')
        col1.markdown("# Negative")
        col2.image(image) 
    
    
def main():
    st.title(":pink[Amazon Fine Food Review Analyser]")
    st.markdown("""---""")
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Amazon_logo.svg/2560px-Amazon_logo.svg.png'
    st.image(url,width=600)
    st.markdown("""---""")
    
    

    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    
    text_input = st.text_input("Enter the review👇",key="placeholder",
            label_visibility=st.session_state.visibility,
            disabled=st.session_state.disabled,placeholder="None",
        )
    
    
    if st.button('Review Predict'):
        import time
         
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)
        classify_utterance(text_input)
       
       
   
     
    
if __name__=='__main__':
    main()    