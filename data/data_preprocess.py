# coding=utf-8
import pandas as pd
import numpy as np
import xml.etree.ElementTree
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import string
import preprocessor as p


def get_emo_txt_semeval():
    #trial data
    path1_gold='./emotion_text_data/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.emotions.gold'
    path1_xml='./emotion_text_data/AffectiveText.Semeval.2007/AffectiveText.trial/affectivetext_trial.xml'
    #test data
    path2_gold='./emotion_text_data/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.emotions.gold'
    path2_xml='./emotion_text-data/AffectiveText.Semeval.2007/AffectiveText.test/affectivetext_test.xml'

    #parse xml
    id_to_txt=dict()
    sentences=[]
    labels=[]
    classes=['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    

    e_path1_xml = xml.etree.ElementTree.parse(path1_xml).getroot()
    e_path2_xml = xml.etree.ElementTree.parse(path2_xml).getroot()

    for atype in e_path1_xml.findall('instance'):
        text = atype.text
        id_ = atype.get('id')
        id_to_txt[int(id_)]=text

    for atype in e_path2_xml.findall('instance'):
        text = atype.text
        id_ = atype.get('id')
        id_to_txt[int(id_)]=text

    
    with open(path1_gold, 'r') as f:
        for line in f:
            tokens=str(line).split()
            sen=id_to_txt[int(tokens[0])]
            tokens=tokens[1:]
            tokens=[int(i) for i in tokens]
            index=np.argmax(tokens)
            lab=classes[index]
            sentences.append(sen)
            labels.append(lab)

    with open(path2_gold, 'r') as f:
        for line in f:
            tokens=str(line).split()
            sen=id_to_txt[int(tokens[0])]
            tokens=tokens[1:]
            tokens=[int(i) for i in tokens]
            index=np.argmax(tokens)
            lab=classes[index]
            sentences.append(sen)
            labels.append(lab)
    
    df=pd.DataFrame({'text':sentences,'label':labels})
    df=df[['text','label']]
    df.to_csv('emo_txt_semeval.csv',index=False)

    
def get_emo_txt_tweets():
    path='./emotion_text_data/text_emotion.csv'
    data_tweets=pd.read_csv(path)
    df=pd.DataFrame({'text':data_tweets.content,'label':data_tweets.sentiment})
    df=df[['text','label']]
    df.to_csv('emo_txt_tweets.csv',index=False)

def get_emo_txt_isear():
    path='./emotion_text_data/ISEAR/ise_processed'
    sentences=[]
    labels=[]
    with open(path,'r') as f:
        for line in f:
            tokens=str(line).split('---')
            sen=tokens[2]
            lab=tokens[1]
            sentences.append(sen.replace('"','').strip())
            labels.append(lab)

    df=pd.DataFrame({'text':sentences,'label':labels})
    df=df[['text','label']]
    df.to_csv('emo_txt_isear.csv',index=False)


def get_sen_lab(type_, dir_):
    path_anger='./emotion_text_data/ei/{}/2018-EI-reg-En-anger-{}.txt'.format(dir_,type_)
    path_joy='./emotion_text_data/ei/{}/2018-EI-reg-En-joy-{}.txt'.format(dir_,type_)
    path_fear='./emotion_text_data/ei/{}/2018-EI-reg-En-fear-{}.txt'.format(dir_,type_)
    path_sadness='./emotion_text_data/ei/{}/2018-EI-reg-En-sadness-{}.txt'.format(dir_,type_)
    sentences=[]
    labels=[]
    with open(path_anger, 'r') as f:
        for line in f:
            line=str(line).split()
            sentences.append(' '.join(line[1:-2]))
            labels.append('anger')
    with open(path_joy, 'r') as f:
        for line in f:
            line=str(line).split()
            sentences.append(' '.join(line[1:-2]))
            labels.append('joy')
    with open(path_fear, 'r') as f:
        for line in f:
            line=str(line).split()
            sentences.append(' '.join(line[1:-2]))
            labels.append('fear')
    with open(path_sadness, 'r') as f:
        for line in f:
            line=str(line).split()
            sentences.append(' '.join(line[1:-2]))
            labels.append('sadness')

    return sentences, labels
            

def get_emo_txt_ei():
    sentences,labels = get_sen_lab('train', '2018-EI-reg-En-train')
    sentences2,labels2 = get_sen_lab('dev', '2018-EI-reg-En-dev')
    sentences3,labels3 = get_sen_lab('test', '2018-EI-reg-En-test')

    sentences.extend(sentences2)
    labels.extend(labels2)
    sentences.extend(sentences3)
    labels.extend(labels3)

    df=pd.DataFrame({'text':sentences,'label':labels})
    df=df[['text','label']]
    df.to_csv('emo_txt_ei_tweets.csv',index=False)

#--------------------------------------------------------------------------------------
#all cleaning methods
stop = set(stopwords.words('english'))  ## stores all the stopwords in the lexicon
exclude = set(string.punctuation)  ## stores all the punctuations
lemma = WordNetLemmatizer()
## lets create a list of all negative-words
negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                'even though', 'yet']


def clean_txt(txt):
    # convert to lower
    txt=str(txt).replace('#','').replace(" 's", '').replace("'s",'')
    txt=txt.lower()
    p.set_options(p.OPT.URL, p.OPT.EMOJI)
    txt=p.clean(txt)
    #tokenize
    nltk_tokens = nltk.word_tokenize(txt)
    #stop words removed
    stop_free=' '.join([i for i in nltk_tokens if i not in stop if i not in negative])
    #puncuation removed
    punc_free = ''.join([ch for ch in stop_free if ch not in exclude])
    #lemmatization
    normalized = ' '.join([lemma.lemmatize(word) for word in punc_free.split()])
    normalized = normalized.replace(' http ','').replace(' http','')
    
    return normalized.strip()




def get_clean_data():
    path='./emo_txt.csv'
    df=pd.read_csv(path)
    df['text']=df['text'].apply(lambda x: clean_txt(x))
    df.to_csv('emo_txt_clean.csv',index=False)
    

    
if __name__ == '__main__':
    print('Processing....')
    #get_emo_txt_semeval()
    #get_emo_txt_tweets()
    #get_emo_txt_isear()
    #get_emo_txt_ei()
    get_clean_data()
