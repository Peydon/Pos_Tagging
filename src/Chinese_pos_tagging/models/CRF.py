from nltk.tag import CRFTagger
import pickle
from random import  shuffle
from src.data_process import datasets

PFR_file=open('../../../dataset/processed_data/PFR_data','rb')
PFR_data=pickle.load(PFR_file)
nltk_sentences=PFR_data['nltk_sentences']
num=int(len(nltk_sentences)*0.8)
shuffle(nltk_sentences)
train=nltk_sentences[0:num]
test=nltk_sentences[num+1:]

ct=CRFTagger()
ct.train(train_data=train,model_file='../../../result/crf.model')
print(ct.evaluate(test))
