import gensim
import pickle
from src.pos_tags_dict import *
class datasets:
    #单词向量化，保存到向量空间模型
    def embedding_PFR_data(self,):

        #开始处理原始数据
        PFR_file=open("../../../dataset/raw_data/199801.txt")
        lines=PFR_file.readlines()
        PFR_file.close()

        #句子集合和单词-词性字典,最长句子长度
        setences=[]
        word_tag_dict={}
        MAX_SEQ_LEN=0
        MIN_SEQ_LEN=100000

        #开始处理每一行，对应的文本是每一段
        for line in lines:
            pairs=line.strip("\n").split()[1:-1]
            words=[]

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
                word,tag=pair.split("/")
                words.append(word)
                word_tag_dict[word]=tag

                #特殊处理短语开头和结尾时的情况
                if phrase_start:
                    phrase+=word
                if phrase_end:
                    words.append(phrase)
                    word_tag_dict[phrase]=phrase_tag
                    phrase=""
                    phrase_start=phrase_end=0

            setence=[]
            #对每一段话再根据句子划分符号细分成每一句
            for word in words:
                if (word in ['。','；','！','？','、','，']):
                    setences.append(setence)
                    setence=[]

                #去除标点符号
                elif word_tag_dict[word] != 'w' and word !='':
                    setence.append(word)
            setences.append(setence)

        #处理后的数据由一个字典保存，包括句子集和单词-词性字典
        PFR_data={
            "sentences":setences,
            "word_tag_dict":word_tag_dict,
        }

        #保存处理过后的数据和向量空间模型
        model=gensim.models.Word2Vec(setences,size=200,min_count=10)
        model.save("../../../dataset/vector_models/PFR_model")
        PFR_file=open('../../../dataset/processed_data/PFR_data','wb')
        pickle.dump(PFR_data,file=PFR_file)
        PFR_file.close()


    #获取人民日报样本数据
    def load_PFR_data(self):
        #加载向量模型和处理后的数据
        model=gensim.models.Word2Vec.load("../../../dataset/vector_models/PFR_model")
        PFR_file=open('../../../dataset/processed_data/PFR_data','rb')
        PFR_data=pickle.load(PFR_file)

        #输入输出
        #x是句子集，每个句子由多个200维单词构成
        #y是词性，对应每个单词的词性
        #eg：x[[vector(200),vector(200),...],[],[],...]
        #eg:y[[[0,名词]，[1,动词]，...],[],[]...]
        x,y=[],[]
        sentences=PFR_data['sentences']
        word_tag_dict=PFR_data['word_tag_dict']

        #构造数据
        MinLen=1000
        MaxLen=0
        for sentence in sentences:
            vectors=[]
            tags=[]
            for word in sentence:
                if word in model.wv.vocab:
                    vectors.append(model[word])
                    tags.append(PFR_tags_dict[word_tag_dict[word]][0])
            length=len(vectors)
            if length>7 and length<30:
                MaxLen=max(MaxLen,length)
                MinLen=min(MinLen,length)
                x.append(vectors)
                y.append(tags)
        print(MinLen,MaxLen)
        return x,y,MaxLen,MinLen,model