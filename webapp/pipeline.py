import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
#import pysrt
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import string
import os
import subprocess
import argparse
#import preprocessor as p
#from pydub import AudioSegment
#from watson_developer_cloud import ToneAnalyzerV3
#from watson_developer_cloud.tone_analyzer_v3 import ToneInput


def get_emotions(utterances):
    service = ToneAnalyzerV3(
        version='2017-09-21',
        iam_apikey='<your_api_key>',
        url='https://gateway-lon.watsonplatform.net/tone-analyzer/api')
    tone_chat = service.tone_chat(utterances).get_result()
    # out=json.dumps(tone_chat, indent=2)
    # print(out)
    out=tone_chat["utterances_tone"]
    emotions=[]
    confidences=[]
    for obj in out:
        tones=obj['tones']
        best_score=0.
        best_emotion=''
        for tone in tones:
            score=tone['score']
            tone_name=tone['tone_name'].lower()
            if score>=best_score:
                best_score=score
                best_emotion=tone_name
        
        emotions.append(best_emotion)
        confidences.append(float(best_score))

    return emotions, confidences

def clean_txt(txt):
    #all cleaning methods
    stop = set(stopwords.words('english'))  ## stores all the stopwords in the lexicon
    exclude = set(string.punctuation)  ## stores all the punctuations
    lemma = WordNetLemmatizer()
    ## lets create a list of all negative-words
    negative = ['not', 'neither', 'nor', 'but', 'however', 'although', 'nonetheless', 'despite', 'except',
                    'even though', 'yet']
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


def create_emotion_csv(data):
    #create emotion.csv required for indexing
    starts=[]
    ends=[]
    texts=[]
    emotions=[]
    scores=[]
    for emo in data:
        start,end,text,emotion,score=emo
        pos_start=(start.hour*3600) + (start.minute*60) + (start.second)
        pos_end=(end.hour*3600) + (end.minute*60) + (end.second)
        starts.append(pos_start)
        ends.append(pos_end)
        texts.append(text)
        emotions.append(emotion)
        scores.append(score)

    df=pd.DataFrame({'start':starts,'end':ends, 'text': texts, 'emotion': emotions,
                      'score': scores})
    df.to_csv('emotion_index.csv', index=False)


def preprocess_video(video_path, data):
    emotions_main=[]
    threshold=0.5
    for d in data:
        start,end,text,emotion,score=d
        if score>=threshold:
            emotions_main.append([start,end,text,emotion])


    #---add sound effects
    #ffmpeg -i video.mp4 -f mp3 -ab 192000 -vn audio.mp3
    cmd1 = 'ffmpeg -i ' + video_path + ' -f wav -ab 192000 -vn audio.wav'      
    process = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE)
    process.wait()

    main_sound = AudioSegment.from_wav("./audio.wav")
    for emo in emotions_main:
        start,end,text,emotion=emo
        if emotion=='love' or emotion=='polite':
            sound=AudioSegment.from_wav("./sounds/aww.wav")
            # mix main with sound
            pos= ((end.hour*3600) + (end.minute*60) + (end.second)*1000)
            print('added sound-effect at',pos,'msec')
            main_sound = main_sound.overlay(sound, position=pos)
        elif emotion=='joy':
            sound=AudioSegment.from_wav("./sounds/joy.wav")
            # mix main with sound
            pos=((end.hour*3600) + (end.minute*60) + (end.second)*1000)
            print('added sound-effect at',pos,'msec')
            main_sound = main_sound.overlay(sound, position=pos)
        elif emotion=='impolite':
            sound=AudioSegment.from_wav("./sounds/scary.wav")
            # mix main with sound
            pos=((end.hour*3600) + (end.minute*60) + (end.second)*1000)
            print('added sound-effect at',pos,'msec')
            main_sound = main_sound.overlay(sound, position=pos)
        elif emotion=='sad':
            sound=AudioSegment.from_wav("./sounds/sad.wav")
            # mix main with sound
            pos=((end.hour*3600) + (end.minute*60) + (end.second)*1000)
            print('added sound-effect at',pos,'msec')
            main_sound = main_sound.overlay(sound, position=pos)
    
    # save the result
    main_sound.export("./mixed_sound.mp3", format="mp3")
    #ffmpeg -i video.mp4 -i mixed_sound.mp3 -c copy -map 0:v -map 1:a result.mp4
    cmd2 = 'ffmpeg -i ' + video_path + ' -i mixed_sound.mp3 -c copy -map 0:v -map 1:a ./static/out.mp4'      
    process = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE)
    process.wait()

    if os.path.exists('./audio.wav'):
        os.remove('./audio.wav')

    if os.path.exists('./mixed_sound.mp3'):
        os.remove('./mixed_sound.mp3')
    
def add_sound_effect(video_path, srt_path):
    try:
        subs = pysrt.open(srt_path,encoding='iso-8859-1')
        n=len(subs)
        data=[]
        utterances=[]
        for i in range(n):
            sub=subs[i]
            start=sub.start.to_time()
            end=sub.end.to_time()
            text=sub.text_without_tags
            #text=clean_txt(text)
            text=text.lower()
            data.append([start,end,text])
            utterances.append({'text': text, 'user': 'user'})
        #print(utterances)
        # utterances = [{
        #     'text': 'I am very happy.',
        #     'user': 'glenn'
        # }, {
        #     'text': 'It is a good day.',
        #     'user': 'glenn'
        # }]
        emotions, confidences = get_emotions(utterances)
        for i in range(n):
            data[i].append(emotions[i])
            data[i].append(confidences[i])
        #data->start,end,text,emotion,confidence
        preprocess_video(video_path,data)
        create_emotion_csv(data)
        return True, ''
    except Exception as e:
        print(e)
        return False, e



if __name__ == '__main__':
    print('Processing....')
    parser = argparse.ArgumentParser(description='pipeline for hackathon')
    parser.add_argument('--video', type=str, help='Input video.mp4 filepath', required=True)
    parser.add_argument('--srt', type=str, help='Input srt filepath',required=True)
   
    args = parser.parse_args()
    
    flag, err = add_sound_effect(args.video, args.srt)
