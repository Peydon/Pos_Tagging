import gensim
import pickle
from src.pos_tags_dict import *
from keras.utils import to_categorical
import numpy as np
import random
class datasets:
    #单词向量化，保存到向量空间模型
    def embedding_PFR_data(self,):

        #开始处理原始数据
        PFR_file=open("../../../dataset/raw_data/PFR_data.txt",encoding='utf8')
        lines=PFR_file.readlines()
        PFR_file.close()

        #句子集合和单词-词性字典,最长句子长度
        sentences=[]
        sen_tags=[]
        word_tag_dict={}
        nltk_sentences=[]

        #开始处理每一行，对应的文本是每一段
        for line in lines:
            pairs=line.strip("\n").split()[1:]
            words=[]
            tags=[]
            nltk_sentence=[]

            #对数据集中的短语标记采取拼接单词成新单词(短语)加到原句子中
            #eg: [人民/n 大会堂/n]nt --> 人民/n 大会堂/n 人民大会堂/nt
            phrase_tag="" #短语词性
            phrase=""    #短语单词
            phrase_start=0
            phrase_end=0

            #处理单词词性结构 eg:人/n
            for pair in pairs:
                #短语标记开头
                if pair[0]=='[' and pair[1]!='/':
                    pair = pair[1:]
                    phrase_start=1
                #短语标记结尾
                if ']' in pair.split("/")[1]:
                    pair,phrase_tag=pair.split("]")
                    phrase_end=1
                #切分单词和词性
                word=tag=""
                try:
                    word,tag=pair.split("/")
                except:
                    print(pair)
                words.append(word)
                tags.append(tag)
                nltk_sentence.append((word,tag))
                word_tag_dict[word]=tag

                #特殊处理短语开头和结尾时的情况
                if phrase_start:
                    phrase+=word
                if phrase_end:
                    words.append(phrase)
                    tags.append(phrase_tag)
                    #nltk_sentence.append((phrase,phrase_tag))
                    word_tag_dict[phrase]=phrase_tag
                    phrase=phrase_tag=""
                    phrase_start=phrase_end=0

            if len(nltk_sentence)>0:
                nltk_sentences.append(nltk_sentence)
            sentence=[]
            temp_tags=[]
            #对每一段话再根据句子划分符号细分成每一句
            for word,tag in zip(words,tags):
                if (word in ['。','；','！','？','、','，','：']and len(sentence)>5):
                    sentences.append(sentence)
                    sen_tags.append(temp_tags)
                    sentence=[]
                    temp_tags=[]

                #去除标点符号
                elif tag != 'w' and word !='':
                    sentence.append(word)
                    temp_tags.append(tag)
            if len(sentence)>0:
                sentences.append(sentence)
                sen_tags.append(temp_tags)

        #处理后的数据由一个字典保存，包括句子集和单词-词性字典
        PFR_data={
            "sentences":sentences,
            "sen_tags":sen_tags,
            "word_tag_dict":word_tag_dict,
            "nltk_sentences":nltk_sentences,
        }
        print(len(sentences))

        #保存处理过后的数据和向量空间模型
        model=gensim.models.Word2Vec(sentences,size=200,min_count=10)
        model.save("../../../dataset/vector_models/PFR_model")
        PFR_file=open('../../../dataset/processed_data/PFR_data','wb')
        pickle.dump(PFR_data,file=PFR_file)
        PFR_file.close()


    #获取人民日报样本数据
    def load_PFR_data(self,modelname):
        #加载向量模型和处理后的数据
        model=gensim.models.Word2Vec.load("../../../dataset/vector_models/"+modelname+"_model")
        PFR_file=open('../../../dataset/processed_data/'+modelname+'_data','rb')
        PFR_data=pickle.load(PFR_file)




        #输入输出
        #x是句子集，每个句子由多个200维单词构成
        #y是词性，对应每个单词的词性
        #eg：x[[vector(200),vector(200),...],[],[],...]
        #eg:y[[[0,名词]，[1,动词]，...],[],[]...]
        x,y=[],[]
        sentences=PFR_data['sentences']
        sen_tags=PFR_data['sen_tags']
        word_tag_dict=PFR_data['word_tag_dict']

        embedding_matrix = np.zeros((len(word_tag_dict),200))
        index=0
        for word in word_tag_dict.keys():
            if word in model.wv.vocab:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[index] = model[word]
            index+=1

        word_index={}
        index=0
        for word in word_tag_dict.keys():
            word_index[word]=index
            index+=1

        #构造数据
        MIN_SEQ_LEN=1000
        MAX_SEQ_LEN=0
        num=0
        for sentence,origin_tags in zip(sentences,sen_tags):
            vectors=[]
            tags=[]
            for word,tag in zip(sentence,origin_tags):
                if word in model.wv.vocab:
                    vectors.append(word_index[word])
                    tags.append(PFR_tags_dict[tag][0])
                    #tags.append(PFR_tags_dict[word_tag_dict[word]][0])

            length=len(vectors)
            if length>5 and length<30:
                MAX_SEQ_LEN=max(MAX_SEQ_LEN,length)
                MIN_SEQ_LEN=min(MIN_SEQ_LEN,length)
                num+=len(vectors)
                x.append(vectors)
                y.append(tags)

        print(MIN_SEQ_LEN,MAX_SEQ_LEN)

        # 长度区间
        len_nums= MAX_SEQ_LEN - MIN_SEQ_LEN + 1
        output_dim=len(PFR_tags_dict)

        print(len(x),num,len(word_tag_dict),len_nums,output_dim)
        # 根据不同长度句子划分到不同的nd_array里
        len_x_samples = [[] for i in range(len_nums)]
        len_y_samples = [[] for j in range(len_nums)]
        for sentence, tags in zip(x, y):
            len_x_samples[len(sentence) - MIN_SEQ_LEN].append(sentence)
            len_y_samples[len(sentence) - MIN_SEQ_LEN].append(to_categorical(tags, output_dim))
        len_x_samples = [np.array(s) for s in len_x_samples]
        len_y_samples = [np.array(s) for s in len_y_samples]

        train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []

        # 划分测试集验证集和测试集
        for sample_x, sample_y in zip(len_x_samples, len_y_samples):
            l = len(sample_x)
            a = int(l * 8 / 10)
            b = int(l * 1 / 10)
            train_x.append(sample_x[:a])
            train_y.append(sample_y[:a])
            valid_x.append(sample_x[a + 1:a + b])
            valid_y.append(sample_y[a + 1:a + b])
            test_x.append(sample_x[a + b + 1:])
            test_y.append(sample_y[a + b + 1:])

        return embedding_matrix,train_x,train_y,valid_x,valid_y,test_x,test_y,len(x)

    # 获得训练集
    def generate_train_arrays(self,train_x,train_y,bacth_size):
        len_nums=len(train_x)
        while True:
            index = list(range(len_nums))
            len_left = {i: len(train_x[i]) for i in range(len_nums)}
            random.shuffle(index)
            while (True):
                find = 0
                for i in index:
                    left = len_left[i]
                    if left < bacth_size:
                        continue
                    L = len(train_x[i])
                    x = np.array(train_x[i][L - left:L - left + bacth_size])
                    y = np.array(train_y[i][L - left:L - left + bacth_size])
                    len_left[i] -= bacth_size
                    find = 1
                    yield (x, y)
                if find == 0: break

    # 获得验证集
    def generate_valid_arrays(self,valid_x,valid_y):
        while 1:
            for x, y in zip(valid_x, valid_y):
                yield (x, y)

    # 获得测试集
    def generate_test_arrays(self,test_x,test_y):
        while 1:
            for x, y in zip(test_x, test_y):
                yield (x, y)
