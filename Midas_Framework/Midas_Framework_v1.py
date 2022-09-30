import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, LSTM, Concatenate, Input, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate, InputLayer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
K.clear_session()
from tensorflow.keras.layers import BatchNormalization
import pickle


'''
Midas Generate 
'''
class Midas_Generate:
    def __init__(self, dataset, vect, y_var, x_var,
                 NLP_var, sc_var, t_var='n', LSTM_var = '128 64 256',
                 DL_var = '20 30 50', o_var = 'adam', s_var = 'MinMaxScaler',
                 NN_mode = "regression", bs_var = 128, e_var = 10, ver_var = 1, val_var = 0.2):
        self.dataset = dataset
        self.vect = vect
        self.y_var = y_var
        self.x_var = x_var
        self.NLP_var = NLP_var
        self.t_var = t_var
        self.LSTM_var = LSTM_var
        self.DL_var = DL_var
        self.o_var = o_var
        self.s_var = s_var
        self.sc_var = sc_var
        self.NN_mode = NN_mode
        self.bs_var = bs_var
        self.e_var = e_var
        self.ver_var = ver_var
        self.val_var = val_var
        self.y_variables = list(self.y_var.split())
        self.NLP_variables = list(self.NLP_var.split())
        self.X_text = self.dataset[self.NLP_variables]
        self.x_variables = list(self.x_var.split(', '))
        self.scaler_column_list = list(self.sc_var.split())
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_text_train = None
        self.X_text_test = None
        self.embeddings_matrix = None
        self.coefslen = None
        self.word_index = None
        self.vocab_size = None
        self.maxlen = None
        self.column_number = None
        self.output_shape = None
        self.trained_model = None
        self.tokenizer = None
        self.final_model = None
        self.external_validation = None
        a = 'y'
        if a in self.t_var:
            self.t_var = True
        else:
            self.t_var = False

    def NN_variables(self):
        self.dataset = self.dataset[self.NLP_variables + self.y_variables + self.x_variables]
        return self.dataset

    def split_func(self):
        i = 0
        for n in self.NLP_variables:
            self.x_variables = self.dataset.drop([self.NLP_variables[i]], axis=1)
            i = i + 1

        i= 0
        for y in self.y_variables:
            self.x_variables = self.x_variables.drop([self.y_variables[i]], axis=1)
            i = i + 1

        y = self.dataset[self.y_variables]
        self.column_number = len(self.x_variables.columns)
        self.output_shape = int(self.dataset[self.y_var].nunique())
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_text_train, self.X_text_test = train_test_split(self.x_variables, y, self.X_text, test_size=0.2, random_state=0)


    def scaler_func(self):
        a = 'MinMaxScaler'
        if a in self.s_var:
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        i = 0
        for c in self.scaler_column_list:
            y = self.scaler_column_list[i]
            self.X_train[y] = scaler.fit_transform(self.X_train[y].values.reshape(-1,1))
            self.X_test[y] = scaler.fit_transform(self.X_test[y].values.reshape(-1,1))
            self.x_variables[y] = scaler.fit_transform(self.x_variables[y].values.reshape(-1,1))
            i = i + 1


    def label_func(self):
        label = self.X_train.select_dtypes(include=[object])
        label_list = label.columns
        le = preprocessing.LabelEncoder()

        i = 0
        for l in label_list:
            x = label_list[i]
            fitted_le = le.fit(self.x_variables.loc[:,x])
            self.X_train.loc[:,x] = fitted_le.transform(self.X_train.loc[:,x])
            self.X_test.loc[:,x] = fitted_le.transform(self.X_test.loc[:,x])
            self.x_variables.loc[:,x] = fitted_le.transform(self.x_variables.loc[:,x])
            i = i + 1

        if self.NN_mode == 'categorical':
            fitted_le = le.fit(self.dataset[self.y_var])
            self.y_train = fitted_le.transform(self.y_train)
            self.y_test = fitted_le.transform(self.y_test)

        return self.x_variables

    def preprocess_text(self, sen):
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sen)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    def NLP_prep_func(self):
        X1_train = []
        sentences = list(self.X_text_train[self.NLP_var])
        for sen in sentences:
            X1_train.append(self.preprocess_text(sen))

        X1_test = []
        sentences = list(self.X_text_test[self.NLP_var])
        for sen in sentences:
            X1_test.append(self.preprocess_text(sen))

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(X1_train)

        X1_train = self.tokenizer.texts_to_sequences(X1_train)
        X1_test = self.tokenizer.texts_to_sequences(X1_test)
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.tokenizer.word_index) + 1

        a = 0
        b = 0
        for p in X1_train:
            list_len = len(X1_train[b])
            a = a + list_len
            b = b + 1

        self.maxlen = int(a / b)

        self.X_text_train = pad_sequences(X1_train, padding='post', maxlen=self.maxlen)
        self.X_text_test = pad_sequences(X1_test, padding='post', maxlen=self.maxlen)
        return self.tokenizer, self.maxlen

    def embeding_function(self):
        embeddings_index = {};
        with open(self.vect, encoding="utf-8") as f:
            for line in f:
                values = line.split();
                word = values[0];
                coefs = np.asarray(values[1:], dtype='float32');
                embeddings_index[word] = coefs;
        self.coefslen =len(coefs)

        self.embeddings_matrix = np.zeros((self.vocab_size+1, self.coefslen));
        for word, i in self.word_index.items():
            embedding_vector = embeddings_index.get(word);
            if embedding_vector is not None:
                self.embeddings_matrix[i] = embedding_vector;
        return self.embeddings_matrix

    def NN_func(self):

        model_text = keras.Sequential()
        model_text.add(keras.Input(shape=(self.maxlen,)))
        model_text.add(layers.Embedding(self.vocab_size+1, self.coefslen, weights=[self.embeddings_matrix], input_length=self.maxlen, trainable=self.t_var))

        # Call model on a test input
        units = list(int(n) for n in self.LSTM_var.split())
        counter = 0
        for i in units:
            counter =+ 1
            if counter<len(units):
                model_text.add(layers.LSTM(i, return_sequences=True))
            else:
                model_text.add(layers.LSTM(i))

        model_text.add(layers.Flatten())
        model_var = keras.Sequential()
        model_var.add(keras.Input(shape=(self.column_number,)))
        units_var = list(int(n) for n in self.DL_var.split())

        for i in units_var:
            model_var.add(tf.keras.layers.Dense(i , activation='relu' ))
            #Add Dropout layers fi parameter set + create parameter...
        merged = Concatenate([model_var.output, model_text.output])
        merged_layers = concatenate([model_var.output, model_text.output])
        x = BatchNormalization()(merged_layers)
        x = Dense(300)(x)
        x = Dropout(0.2)(x)
        x = Dense(300)(x)
        x = Dropout(0.2)(x)
        if self.NN_mode == 'regression':
            x = Dense(1)(x)
            out = BatchNormalization()(x)
            #out = Activation('relu')(x)
        elif self.NN_mode == 'binary':
            x = Dense(1)(x)
            x = BatchNormalization()(x)
            out = Activation('sigmoid')(x)
        elif self.NN_mode == 'categorical':
            x = Dense(self.output_shape)(x)
            x = BatchNormalization()(x)
            out = Activation('sigmoid')(x)
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)
        else:
            Print("Error in prediction mode, regression, categorical or binary")

        self.final_model = Model([model_var.input, model_text.input], [out])

        #make
        if self.NN_mode == "regression":
            self.final_model.compile(loss=keras.losses.MeanSquaredError(reduction='auto'), optimizer = self.o_var, metrics=['mse'])
        elif self.NN_mode == "binary":
            self.final_model.compile(loss=keras.losses.binary_crossentropy, optimizer = self.o_var, metrics=['acc'])
        elif self.NN_mode == "categorical":
            self.final_model.compile(loss=keras.losses.categorical_crossentropy, optimizer = self.o_var, metrics=['acc'])
        else:
            Print("Error in prediction mode, regression, categorical or binary")

        self.final_model.summary()
        self.trained_model = self.final_model.fit(x=[self.X_train, self.X_text_train], y=self.y_train, batch_size=self.bs_var, epochs=self.e_var, verbose=self.ver_var, validation_split=self.val_var)



        if self.NN_mode == "regression":
            preds  = self.final_model.predict([self.X_test, self.X_text_test])
            c=np.mean(preds)
            d=np.mean(self.y_test)
            z=c-d
            x=z/d
            self.external_validation = x*100
            print(self.external_validation)

        elif self.NN_mode == "binary":
            a = self.final_model.evaluate([self.X_test, self.X_text_test], self.y_test)
            self.external_validation = a
            print('loss: ' + str(a[0]) + ', acc: ' + str(a[1]))

        elif self.NN_mode == "categorical":
            a = self.final_model.evaluate([self.X_test, self.X_text_test], self.y_test)
            self.external_validation = a
            print('loss: ' + str(a[0]) + ', acc: ' + str(a[1]))


        else:
            Print("Error in prediction mode, regression, categorical or binary")

        return self.trained_model, self.final_model, self.external_validation, self.NN_mode
        print('END')

    def main(self, dataset, vect, y_var, x_var,
             NLP_var, sc_var, t_var, LSTM_var,
             DL_var , o_var, s_var,
             NN_mode, bs_var, e_var, ver_var, val_var):
        m = Midas_Generate(dataset, vect, y_var, x_var,
                           NLP_var, sc_var, t_var, LSTM_var,
                           DL_var , o_var, s_var,
                           NN_mode, bs_var, e_var, ver_var, val_var)
        m.NN_variables()
        m.split_func()
        m.scaler_func()
        m.label_func()
        NN_tokenizer, maxlen = m.NLP_prep_func()
        m.embeding_function()
        Trained_NN_model, model, external_validation, NN_mode = m.NN_func()

        return NN_tokenizer, maxlen, Trained_NN_model, model, external_validation, NN_mode
'''
Midas Class
'''
class Midas():
    def __init__(self):
        self.NN_tokenizer = None
        self.Trained_NN_model = None
        self.maxlen = None
        self.model = None
        self.pred_text = None
        self.pred_variables = None
        self.external_validation = None
        self.NN_mode = None

    def Generate(self, dataset, vect, y_var, x_var,
                 NLP_var, sc_var="", t_var='n', LSTM_var = '128 64 256',
                 DL_var = '20 30 50', o_var = 'adam', s_var = 'MinMaxScaler',
                 NN_mode = "regression", bs_var = 128, e_var = 10, ver_var = 1, val_var = 0.2):
        self.NN_tokenizer, self.maxlen, self.Trained_NN_model, self.model, self.external_validation, self.NN_mode = Midas_Generate.main(self, dataset, vect, y_var, x_var,
                                                                                                                                        NLP_var, sc_var, t_var, LSTM_var,
                                                                                                                                        DL_var , o_var, s_var,
                                                                                                                                        NN_mode, bs_var, e_var, ver_var, val_var)

    def preprocess_text(self, sen):
        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sen)
        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)
        return sentence

    def Data_prep_NLP(self, dataset, NLP_var):
        self.prep_text = []
        sentences = list(dataset[NLP_var])
        for sen in sentences:
            self.prep_text.append(Midas_Generate.preprocess_text(self, sen))

        self.prep_text = self.NN_tokenizer.texts_to_sequences(self.prep_text)

        self.pred_text = pad_sequences(self.prep_text, padding='post', maxlen=self.maxlen)
        return self.pred_text

    def Label(self,dataset):
        label = dataset.select_dtypes(include=[object])
        label_list = label.columns
        i = 0
        for l in label_list:
            x = label_list[i]
            le = preprocessing.LabelEncoder()
            fitted_le = le.fit(dataset.loc[:,x])
            dataset.loc[:,x] = fitted_le.transform(dataset.loc[:,x])
            i = i + 1
        return dataset

    def Scaler(self, dataset, scaler_columns, scaler_indput):
        scaler_columns_list = list(scaler_columns.split())
        a = 'MinMaxScaler'
        if a in scaler_indput:
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        i = 0
        for s in scaler_columns_list:
            y = scaler_columns_list[i]
            dataset[y] = scaler.fit_transform(dataset[y].values.reshape(-1,1))
            i = i + 1
        return dataset

    def SaveModel(self, name):
        l = [self.NN_tokenizer, self.maxlen]
        self.model.save(name + ".h5")
        with open(name+".pickel", 'wb') as handle:
            pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model have been saved Succesfully")


    def LoadModel(self, name):
        self.model = tf.keras.models.load_model(name + ".h5")
        l = pickle.load(open(name+".pickel", 'rb'))
        self.NN_tokenizer = l[0]
        self.maxlen = l[1]
        self.model.summary()
        print("Model has been loaded succesfully")

        def Predict(self, dataset, X_pred, NLP_pred, scaler_columns, scaler_input):
            variables_text = list(NLP_pred.split())
            variables_struc = list(X_pred.split())
            dataset = dataset[ variables_text + variables_struc ]

            self.pred_text = self.Data_prep_NLP(dataset, NLP_pred)
            dataset = dataset.drop([NLP_pred], axis = 1)
            self.Label(dataset)
            self.Scaler(dataset, scaler_columns, scaler_input)

            if self.NN_mode == "categorical":
                X_pred = to_categorical(X_pred)

            self.pred_variables = dataset.to_numpy()
            predictions = self.model.predict([self.pred_variables, self.pred_text])
            return predictions