
from abc import ABCMeta, abstractmethod

import flask
import json
from flask import request
from flask_cors import CORS
from flask import  render_template

import random
import json
import pickle
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

ReStr = ""


class IAssistant(metaclass=ABCMeta):

    @abstractmethod
    def train_model(self):
        """ Implemented in child class """

    @abstractmethod
    def request_tag(self, message):
        """ Implemented in child class """

    @abstractmethod
    def get_tag_by_id(self, id):
        """ Implemented in child class """

    @abstractmethod
    def request_method(self, message):
        """ Implemented in child class """

    @abstractmethod
    def request(self, message):
        """ Implemented in child class """


class GenericAssistant(IAssistant):

    def __init__(self, intents, intent_methods={}, model_name="assistant_model"):
        self.intents = intents
        self.intent_methods = intent_methods
        self.model_name = model_name

        if intents.endswith(".json"):
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()

    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents, encoding='UTF-8').read())

    def train_model(self):

        self.words = []
        self.classes = []
        documents = []
        ignore_letters = ['!', '?', ',', '.']

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        self.model = Sequential()
        self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(train_y[0]), activation='softmax'))

        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"{self.model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{self.model_name}_classes.pkl', 'wb'))
        else:
            self.model.save(f"{model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'{model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{model_name}_classes.pkl', 'wb'))

    def load_model(self, model_name=None):
        if model_name is None:
            self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'{self.model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'{self.model_name}.h5')
        else:
            self.words = pickle.load(open(f'{model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'{model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'{model_name}.h5')

    def _clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence, words):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence):
        p = self._bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.1
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def _get_response(self, ints, intents_json):
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "对不起，我听不懂。"
        return result

    def request_tag(self, message):
        pass

    def get_tag_by_id(self, id):
        pass

    def request_method(self, message):
        pass

    def request(self, message):
        ints = self._predict_class(message)
        if ints[0]['intent'] in self.intent_methods.keys():
            self.intent_methods[ints[0]['intent']]()
        return self._get_response(ints, self.intents)


def function_for_none():
    global ReStr
    ReStr = "你好|名字|帮助菜单"


def function_for_greetings():
    global ReStr
    ReStr = "帮助菜单|你叫什么名字|哪里有疫情"


def function_for_goodbye():
    global ReStr
    ReStr = "再见|帮助菜单"


def function_for_help():
    global ReStr
    ReStr = "再见|你好"


def function_for_thanks():
    global ReStr
    ReStr = "再见|你好"

def function_for_name():
    global ReStr
    ReStr = "帮助菜单|你好|再见"


def function_for_what():
    global ReStr
    ReStr = "如何预防|再见|临床表现"


def function_for_characteristic():
    global ReStr
    ReStr = "如何预防|传播途径|再见"


def function_for_spread():
    global ReStr
    ReStr = "如何预防|临床表现|再见"

def function_for_performance():
    global ReStr
    ReStr = "如何预防|再见|疫情现状"

def function_for_prevention():
    global ReStr
    ReStr = "疫情现状|临床表现|再见"

def function_for_situation():
    global ReStr
    ReStr = "国内现状|全国中高风险地区|防疫政策"

def function_for_situation_nation():
    global ReStr
    ReStr = "中高风险地区|防疫政策|再见"

def function_for_risk():
    global ReStr
    ReStr = "防疫政策|国内现状|再见"

def function_for_policy():
    global ReStr
    ReStr = "北京|上海|广州|深圳|南京|杭州|重庆|成都|武汉|合肥"


def function_for_beijing():
    global ReStr
    ReStr = "上海|广州|深圳|南京|杭州|重庆|成都|武汉|合肥"


def function_for_shanghai():
    global ReStr
    ReStr = "北京|广州|深圳|南京|杭州|重庆|成都|武汉|合肥"


def function_for_guangzhou():
    global ReStr
    ReStr = "北京|上海|深圳|南京|杭州|重庆|成都|武汉|合肥"

def function_for_shenzhen():
    global ReStr
    ReStr = "北京|上海|广州|南京|杭州|重庆|成都|武汉|合肥"


def function_for_nanjing():
    global ReStr
    ReStr = "北京|上海|广州|深圳|杭州|重庆|成都|武汉|合肥"

def function_for_hangzhou():
    global ReStr
    ReStr = "北京|上海|广州|深圳|南京|重庆|成都|武汉|合肥"

def function_for_chongqing():
    global ReStr
    ReStr = "北京|上海|广州|深圳|南京|杭州|成都|武汉|合肥"

def function_for_chengdu():
    global ReStr
    ReStr = "北京|上海|广州|深圳|南京|杭州|重庆|武汉|合肥"

def function_for_wuhan():
    global ReStr
    ReStr = "北京|上海|广州|深圳|南京|杭州|重庆|成都|合肥"

def function_for_hefei():
    global ReStr
    ReStr = "北京|上海|广州|深圳|南京|杭州|重庆|成都|武汉"
mappings = {'': function_for_none,
            'greeting': function_for_greetings,
            'goodbye': function_for_goodbye,
            'help': function_for_help,
            'thanks': function_for_thanks,
            'what': function_for_what,
            'characteristic': function_for_characteristic,
            'spread': function_for_spread,
            'performance': function_for_performance,
            'prevention': function_for_prevention,
            'situation': function_for_situation,
            'situation_nation': function_for_situation_nation,
            'risk': function_for_risk,
            'policy': function_for_policy,
            'beijing': function_for_beijing,
            'name': function_for_name,
            'shanghai': function_for_shanghai,
            'guangzhou': function_for_guangzhou,
            'shenzhen': function_for_shenzhen,
            'nanjing': function_for_nanjing,
            'hangzhou': function_for_hangzhou,
            'chongqing': function_for_chongqing,
            'chengdu': function_for_chengdu,
            'wuhan': function_for_wuhan,
            'hefei': function_for_hefei
            }
assistant = GenericAssistant('intents.json', intent_methods=mappings, model_name="test_model")

assistant.train_model()
assistant.save_model()

# don't need train model everytime
#assistant.load_model("test_model")


server = flask.Flask(__name__)
CORS(server, resources=r'/*')

@server.route('/answer',methods=['POST','GET'])#路由
def answer():
    data = request.get_data().decode('utf-8')
    # data=request.json()
    # quest=data.get('quest')
    # print(data)
    strin = assistant.request(data)
    result = {'answers': strin, 'Restr': ReStr}
    return json.dumps(result,ensure_ascii=False)


@server.route('/bot',methods=['GET','POST'])
def serverrun():

    #msg = request.values.get('msg')

    return render_template("test3.html")
    #msg  = request.get_data().decode('utf-8')
    #answer = assistant.request(msg)
    #result = {'answers': answer, 'Restr': ReStr}
    #return render_template("test.html",**result)
    #return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    server.run(debug=True, port=8088, host='127.0.0.1')  # 指定端口、host,0.0.0.0代表不管几个网卡，任何ip都可以访问
