import streamlit as st
import spacy
import pickle
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
class BagOfWords():
    def __init__(self,df,no_of_labels,labels):
        self.df=df
        nlp=spacy.blank('en')
        self.nlp=nlp
        text_cat=self.nlp.create_pipe("textcat",config={"exclusive_classes": True,"architecture": "bow"})
        self.nlp.add_pipe(text_cat)
        for i in range(cno_of_labels):
            text_cat.add_label(labels[i])
    def train(self,feature,target):
        train_texts = self.df[feature].values
        train_labels = [{'cats': {'fake': label == 'fake','real': label == 'real'}} for label in self.df[target]]
        train_data = list(zip(train_texts, train_labels))
        import random
        from spacy.util import minibatch
        random.seed(1)
        spacy.util.fix_random_seed(1)
        optimizer = self.nlp.begin_training()
        losses = {}
        for epoch in range(5):
            random.shuffle(train_data)
            batches = minibatch(train_data, size=10000)
            for batch in batches:
                texts, labels = zip(*batch)
                self.nlp.update(texts, labels, sgd=optimizer, losses=losses)
        print('Done Training')
    def test(self,text_input):
        text=[text_input]
        docs=[self.nlp.tokenizer(token) for token in text]
        textcat = self.nlp.get_pipe('textcat')
        scores, _ = textcat.predict(docs)
        func=lambda x,y:st.write(f'''Confidence score that the news is FAKE: {x:.2f}   
                                   Confidence score that the news is REAL: {y:.2f}''')
        func(*scores[0])
        predicted_labels = scores.argmax(axis=1)
        return [textcat.labels[label] for label in predicted_labels]

st.title("News Validator")
st.write('''Fake news and hoaxes have been there since before the advent of the Internet. 
The widely accepted definition of Internet fake news is: fictitious articles deliberately 
fabricated to deceive readers. Social media and news outlets publish fake news to increase 
readership or as part of psychological warfare.To detect fake news articles,this application
uses NLP based models and tells you whether the news is fake or real with a confidence score.''')
st.write('NOTE: Currently,only bag of words model is used by default.Working on adding different models.')
model_name=st.sidebar.selectbox("Select the NLP model",("Bag of Words","Word Vectors"))
text=st.text_area('''Enter the article URL or paste the news headline.''',height=25)
button=st.button("Check")
if button:
    if text.startswith('http:') or text.startswith('https:'):
        try:
            with st.spinner('Please wait trying to fetch content from the URL'):
                options=Options()
                options.headless=True
                options.add_argument("--window-size=1920,1200")
                DRIVER_PATH= "C:/Users/acer/chromedriver_win32/chromedriver"
                driver=webdriver.Chrome(options=options,executable_path=DRIVER_PATH)
                driver.get(text)
                h1=driver.title
                if '|' in h1:
                    h1=h1[:h1.find('|')]
                if h1=="" or len(h1)<=3:
                    st.error("Oops! An unexpected error occured while trying to fetch the url.Try inputting normal text")
                    driver.quit()
                print(f'This is content in h1: {h1}')
                with open('Bag_Of_words.pkl', 'rb') as f: 
                    model = pickle.load(f)
                op1=model.test(h1)
                st.success(f'FINAL PREDICTION: {op1[0].upper()}')
                driver.quit()
        except:
            st.error("Oops! An unexpected error occured while trying to fetch the content from the URL.Try inputting normal text.")
            driver.quit()
    else:
        with open('Bag_Of_words.pkl', 'rb') as f:  
            model = pickle.load(f)
        op1=model.test(text)
        st.success(f'FINAL PREDICTION: {op1[0].upper()}')

st.title('Feedback')
st.write('''Humans are way more smarter than machines.As this application uses AI
            under the hood,you can help improve this AI model by giving your honest feedback.
            Press the button below when done.''')
feedback=st.radio("Was the prediction accurate?",('Yes','No'))
done=st.button('This is my feedback.')
if done:
    st.success('Thank you for giving the feedback.')





