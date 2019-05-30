import pywt
import pandas as pd
import numpy as np

import matplotlib.pyplot as  plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Conv1D
from keras.layers import BatchNormalization
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten, Dropout
from keras.layers.core import Dense
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import regularizers
import keras

def period_plot(x,y,title,color,picture,figure_no,sub_row,sub_col,sub_num):
    if(picture == True):
        plt.figure(figure_no)
        plt.subplot(sub_row, sub_col, sub_num)
        time = np.arange(0, x) * (1 / 500)
        plt.plot(time, y,color = color )
        plt.xlabel("time/s")
        plt.ylabel('amplitude')
        plt.title(title)
        plt.grid(True)
        #plt.show()
        #plt.axis('off')#关闭刻度
    return



class VGG16Net:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, height, width)

        #tcon
       # model.add(Conv2D(20, (1, 1), activation='linear',use_bias= False, input_shape=inputShape))

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(64, (3, 3), activation='relu',input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))


        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))


        # second (and only) set of FC => RELU layers
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))


        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))


        # third (and only) set of FC => RELU layers
        model.add(Dense(1000, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))


        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        # return the constructed network architecture
        return model

class U_Net_2d:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        #K.set_image_data_format('channels_first')
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, height, width)
        #structure
        inputs = Input(inputShape)
        print(np.shape(inputs))
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        print(np.shape(conv1))
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        print(np.shape(conv1))
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print(np.shape(pool1))
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print(np.shape(pool1))
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = UpSampling2D(size=(2, 2))(drop5)

        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 =  UpSampling2D(size=(2, 2))(conv7)
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
        conv9 = Conv2D(2, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        output = Reshape((80000,))(conv10)

        model = Model(input=inputs, output=output)

        print(model.summary())

        # return the constructed network architecture
        return model



class U_Net_1d:
    @staticmethod
    def build(width, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, width)
            # structure

        inputs = Input(inputShape)
        conv1 = Conv1D(64, 12, activation='relu', padding='same')(inputs)
        conv1 = Conv1D(64, 12, activation='relu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)

        pool1 = MaxPooling1D(pool_size=2)(conv1)

        conv2 = Conv1D(128, 12, activation='relu', padding='same')(pool1)
        conv2 = Conv1D(128, 12, activation='relu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)

        pool2 = MaxPooling1D(pool_size=2)(conv2)

        conv3 = Conv1D(256, 12, activation='relu', padding='same')(pool2)
        conv3 = Conv1D(256, 12, activation='relu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)

        pool3 = MaxPooling1D(pool_size=2)(conv3)

        conv4 = Conv1D(512, 12, activation='relu', padding='same')(pool3)
        conv4 = Conv1D(512, 12, activation='relu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        print(np.shape(conv4))
        pool4 = MaxPooling1D(pool_size=5)(conv4)

        conv5 = Conv1D(1024, 12, activation='relu', padding='same')(pool4)
        conv5 = Conv1D(1024, 12, activation='relu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)

        uppool3 = UpSampling1D(5)(conv5)
        merge3 = concatenate([conv4, uppool3], axis=-1)
        conv6 = Conv1D(512, 12, activation='relu', padding='same')(merge3)
        conv6= Conv1D(512, 12, activation='relu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)

        uppool2 = UpSampling1D(2)(conv6)
        merge2 = concatenate([conv3, uppool2], axis=-1)
        conv7 = Conv1D(256, 12, activation='relu', padding='same')(merge2)
        conv7 = Conv1D(256, 12, activation='relu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        uppool1 = UpSampling1D(2)(conv7)
        merge1 = concatenate([conv2, uppool1], axis=-1)
        conv8 = Conv1D(128, 12, activation='relu', padding='same')(merge1)
        conv8 = Conv1D(128, 12, activation='relu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        uppool0 = UpSampling1D(2)(conv8)
        merge0 = concatenate([conv1, uppool0], axis=-1)
        conv9 = Conv1D(64, 12, activation='relu', padding='same')(merge0)
        conv9 = Conv1D(64, 12, activation='relu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        conv10 = Conv1D(2, 1, activation='relu', padding='same')(conv9)


        conv11 = Conv1D(1, 1, activation='sigmoid', padding='same')(conv10)
        conv11 = BatchNormalization()(conv11)

        output = Conv1D(1, 1, activation='sigmoid', padding='same')(conv11)
        output = Reshape((80000,))(output)
        #output = Reshape((5000,))(conv11)
        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        # return the constructed network architecture
        return model

class LISTNet:
    @staticmethod
    def build(width, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, width)

        #tcon
     #   model.add(Conv1D(4, 1, activation='linear',use_bias= False, input_shape=inputShape))

        # first set of CONV => RELU => POOL layers
        model.add(Conv1D(8, 5, activation='relu',input_shape=inputShape))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(0.25))

        # second set of CONV => RELU => POOL layers
        model.add(Conv1D(4, 5, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))


        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        print(model.summary())

        # return the constructed network architecture
        return model


'''
def wavelet_transform(signal):#500hz以上的不要了

    wavelet = pywt.Wavelet('db5')#使用db5小波基

    coeffs = pywt.wavedec(data=signal,wavelet=wavelet,level=4)#小波分解得小波系数，后面越高越细节


    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[2] = np.zeros_like(coeffs[2])
    coeffs[3] = np.zeros_like(coeffs[3])
    coeffs[4] = np.zeros_like(coeffs[4])

    restruct1 = pywt.waverec(coeffs=coeffs,wavelet=wavelet)

    coeffs = pywt.wavedec(data=signal, wavelet=wavelet, level=4)  # 小波分解得小波系数，后面越高越细节

    coeffs[0] = np.zeros_like(coeffs[0])
    coeffs[2] = np.zeros_like(coeffs[2])
    coeffs[3] = np.zeros_like(coeffs[3])
    coeffs[4] = np.zeros_like(coeffs[4])

    restruct2 = pywt.waverec(coeffs=coeffs, wavelet=wavelet)

    coeffs = pywt.wavedec(data=signal, wavelet=wavelet, level=4)  # 小波分解得小波系数，后面越高越细节

    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[0] = np.zeros_like(coeffs[0])
    coeffs[3] = np.zeros_like(coeffs[3])
    coeffs[4] = np.zeros_like(coeffs[4])

    restruct3 = pywt.waverec(coeffs=coeffs, wavelet=wavelet)

    coeffs = pywt.wavedec(data=signal, wavelet=wavelet, level=4)  # 小波分解得小波系数，后面越高越细节

    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[2] = np.zeros_like(coeffs[2])
    coeffs[0] = np.zeros_like(coeffs[0])
    coeffs[4] = np.zeros_like(coeffs[4])

    restruct4 = pywt.waverec(coeffs=coeffs, wavelet=wavelet)

    coeffs = pywt.wavedec(data=signal,wavelet=wavelet,level=4)#小波分解得小波系数，后面越高越细节


    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[2] = np.zeros_like(coeffs[2])
    coeffs[3] = np.zeros_like(coeffs[3])
    coeffs[0] = np.zeros_like(coeffs[0])

    restruct5 = pywt.waverec(coeffs=coeffs,wavelet=wavelet)

    return restruct1,restruct2,restruct3,restruct4,restruct5
'''

def load_data(fold1path,fold1labelpath,fold2path,fold2labelpath):
    # load the image, pre-process it, and store it in the data list
    fold1 = pd.read_csv(fold1path, header=None)
    fold1 = np.array(fold1)
    fold1label = pd.read_csv(fold1labelpath, header=None)
    fold1label = np.array(fold1label)

    fold2 = pd.read_csv(fold2path, header=None)
    fold2 = np.array(fold2)
    fold2label = pd.read_csv(fold2labelpath, header=None)
    fold2label = np.array(fold2label)
    return fold1,fold1label,fold2,fold2label

'''
    fold3 = pd.read_csv(fold3path, header=None)
    fold3 = np.array(fold3)
    fold3label = pd.read_csv(fold3labelpath, header=None)
    fold3label = np.array(fold3label)

    fold4 = pd.read_csv(fold4path, header=None)
    fold4 = np.array(fold4)
    fold4label = pd.read_csv(fold4labelpath, header=None)
    fold4label = np.array(fold4label)

    fold5 = pd.read_csv(fold5path, header=None)
    fold5 = np.array(fold5)
    fold5label = pd.read_csv(fold5labelpath, header=None)
    fold5label = np.array(fold5label)
'''
    
def train(model,trainX, trainY, testX, testY,modelname):
    # initialize the model
    print("[INFO] compiling model...")
    # model = VGG16(include_top=True, weights=None, input_shape=(36,36,3), pooling='max', classes=2)
    #model = VGG16Net.build(width=25, height=25, depth=1, classes=2)
    opt = keras.optimizers.SGD(lr=0.025, nesterov=True)
   # model.compile(loss="mean_squared_logarithmic_error", optimizer=opt,
    #              metrics=["accuracy"])
    #model.compile(loss="binary_crossentropy", optimizer=opt,
      #            metrics=["accuracy"])
    #model.compile(loss="mean_squared_logarithmic_error", optimizer=opt)
    model.compile(loss="binary_crossentropy", optimizer=opt)

    # train the network
    print("[INFO] training network...")

    # keras.callbacks.EarlyStopping(monitor='val_acc', patience=20, verbose=0, mode='auto')
    show = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=False, write_images=True,
                                       embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')  # 20轮不变stop
    vary_rate = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=45, verbose=0, mode='auto',
                                                  min_delta=0.00001, cooldown=0,
                                                  min_lr=0)  # new learn_rate = learn_rate*factor

    #model.fit(x=trainX, y=trainY, batch_size=100, epochs=20000, validation_data=(testX, testY), callbacks=[show,stop,vary_rate])
    model.fit(x=trainX, y=trainY, batch_size=50, epochs=500, validation_data=(testX, testY), callbacks=[show,stop,vary_rate])

    #predictY = model.predict(x=testX, batch_size=71)

    #acc = accuracy_score(testY,predictY)
    #val_table = classification_report(testY, predictY)


    model.save('G:/testUnet/'+modelname)
   # #model.save('E:/zs/DECG_Challenge/right_train/model/1D/U_net_loss_bc/'+modelname)


    return

#main
'''
fold1,fold1label,fold2,fold2label,fold3,fold3label,fold4,fold4label,fold5,fold5label = load_data('E:/zs/DECG_Challenge/train/test_folded_data/data_fold1.csv','E:/zs/DECG_Challenge/train/test_folded_data/label_fold1.csv',
                                                                                                 'E:/zs/DECG_Challenge/train/test_folded_data/data_fold2.csv','E:/zs/DECG_Challenge/train/test_folded_data/label_fold2.csv',
                                                                                                 'E:/zs/DECG_Challenge/train/test_folded_data/data_fold3.csv','E:/zs/DECG_Challenge/train/test_folded_data/label_fold3.csv',
                                                                                                 'E:/zs/DECG_Challenge/train/test_folded_data/data_fold4.csv','E:/zs/DECG_Challenge/train/test_folded_data/label_fold4.csv',
                                                                                                 'E:/zs/DECG_Challenge/train/test_folded_data/data_fold5.csv','E:/zs/DECG_Challenge/train/test_folded_data/label_fold5.csv')


fold1,fold1label,fold2,fold2label,fold3,fold3label,fold4,fold4label,fold5,fold5label = load_data('E:/zs/DECG_Challenge/train/folded_data/data_fold1.csv','E:/zs/DECG_Challenge/train/folded_data/label_fold1.csv',
                                                                                                 'E:/zs/DECG_Challenge/train/folded_data/data_fold2.csv','E:/zs/DECG_Challenge/train/folded_data/label_fold2.csv',
                                                                                                 'E:/zs/DECG_Challenge/train/folded_data/data_fold3.csv','E:/zs/DECG_Challenge/train/folded_data/label_fold3.csv',
                                                                                                 'E:/zs/DECG_Challenge/train/folded_data/data_fold4.csv','E:/zs/DECG_Challenge/train/folded_data/label_fold4.csv',
                                                                                                 'E:/zs/DECG_Challenge/train/folded_data/data_fold5.csv','E:/zs/DECG_Challenge/train/folded_data/label_fold5.csv')
'''

fold1,fold1label,fold2,fold2label = load_data('fold1_data.csv','fold1.csv','fold2_data.csv','fold2.csv')

#print(np.shape(fold1label))
#print(np.shape(fold1))
#print(fold1[0])
#print(np.shape(fold1label))
#print(fold1label[0])

#test_list1,test_list2,test_list3,test_list4,test_list5 = wavelet_transform(fold1[1])

#period_plot(len(test_list1),test_list1,'DECG0-15.625Hz','blue',True,1,2,3,1)
#period_plot(len(test_list2),test_list2,'DECG15.625-31.25Hz','blue',True,1,2,3,2)
#period_plot(len(test_list3),test_list3,'DECG31.25-62.5Hz','blue',True,1,2,3,3)
#period_plot(len(test_list4),test_list4,'DECG62.5-125Hz','blue',True,1,2,3,4)
#period_plot(len(test_list5),test_list5,'DECG125-250Hz','blue',True,1,2,3,5)

#plt.show()
'''
#wavelet decomposing_2d
#fold1
whole_signal_fold1 = []
for i in range(len(fold1)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold1[i])
    whole_signal_fold1.append(np.transpose([np.reshape(signal1,(100,50)),np.reshape(signal2,(100,50)),np.reshape(signal3,(100,50))]))#提取前三层的有用信号拉成二维

whole_signal_fold1 = np.array(whole_signal_fold1)

print(np.shape(whole_signal_fold1))
print(np.shape(whole_signal_fold1[0,:,:,0]))
test_data = np.reshape(np.transpose(whole_signal_fold1[0,:,:,0]),5000)

#period_plot(len(test_data),test_data,'DECG0-15.625Hz','blue',True,1,2,1,2)
#plt.show()

#fold2
whole_signal_fold2 = []
for i in range(len(fold2)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold2[i])
    whole_signal_fold2.append(np.transpose([np.reshape(signal1, (100, 50)), np.reshape(signal2, (100, 50)),np.reshape(signal3, (100, 50))]))# 提取前三层的有用信号拉成二维

whole_signal_fold2 = np.array(whole_signal_fold2)

#fold3
whole_signal_fold3 = []
for i in range(len(fold3)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold3[i])
    whole_signal_fold3.append(np.transpose([np.reshape(signal1, (100, 50)), np.reshape(signal2, (100, 50)), np.reshape(signal3,(100,50))])) # 提取前三层的有用信号拉成二维

whole_signal_fold3 = np.array(whole_signal_fold3)

#fold4
whole_signal_fold4 = []
for i in range(len(fold4)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold4[i])
    whole_signal_fold4.append(np.transpose([np.reshape(signal1, (100, 50)), np.reshape(signal2, (100, 50)),np.reshape(signal3, (100, 50))])) # 提取前三层的有用信号拉成二维
whole_signal_fold4 = np.array(whole_signal_fold4)

#fold5
whole_signal_fold5 = []
for i in range(len(fold5)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold5[i])
    whole_signal_fold5.append(np.transpose([np.reshape(signal1, (100, 50)), np.reshape(signal2, (100, 50)),np.reshape(signal3, (100, 50))]))  # 提取前三层的有用信号拉成二维
whole_signal_fold5 = np.array(whole_signal_fold5)
'''

#wavelet decomposing_1d
#fold1
'''
whole_signal_fold1 = []
for i in range(len(fold1)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold1[i])
    whole_signal_fold1.append(np.transpose([signal1,signal2]))#提取前2层的有用信号
whole_signal_fold1 = np.array(whole_signal_fold1)

#fold2
whole_signal_fold2 = []
for i in range(len(fold2)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold2[i])
    whole_signal_fold2.append(np.transpose([signal1,signal2]))#提取前2层的有用信号
whole_signal_fold2 = np.array(whole_signal_fold2)

#fold3
whole_signal_fold3 = []
for i in range(len(fold3)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold3[i])
    whole_signal_fold3.append(np.transpose([signal1,signal2]))#提取前2层的有用信号
whole_signal_fold3 = np.array(whole_signal_fold3)

#fold4
whole_signal_fold4 = []
for i in range(len(fold4)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold4[i])
    whole_signal_fold4.append(np.transpose([signal1,signal2]))#提取前2层的有用信号
whole_signal_fold4 = np.array(whole_signal_fold4)

#fold5
whole_signal_fold5 = []
for i in range(len(fold5)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold5[i])
    whole_signal_fold5.append(np.transpose([signal1,signal2]))#提取前2层的有用信号
whole_signal_fold5 = np.array(whole_signal_fold5)
'''
'''
#wavelet decomposing_1d
#fold1
whole_signal_fold1 = []
for i in range(len(fold1)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold1[i])
    whole_signal_fold1.append(np.transpose([signal1,signal2,signal3]))#提取前三层的有用信号
whole_signal_fold1 = np.array(whole_signal_fold1)

#fold2
whole_signal_fold2 = []
for i in range(len(fold2)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold2[i])
    whole_signal_fold2.append(np.transpose([signal1,signal2,signal3]))#提取前三层的有用信号
whole_signal_fold2 = np.array(whole_signal_fold2)

#fold3
whole_signal_fold3 = []
for i in range(len(fold3)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold3[i])
    whole_signal_fold3.append(np.transpose([signal1,signal2,signal3]))#提取前三层的有用信号
whole_signal_fold3 = np.array(whole_signal_fold3)

#fold4
whole_signal_fold4 = []
for i in range(len(fold4)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold4[i])
    whole_signal_fold4.append(np.transpose([signal1,signal2,signal3]))#提取前三层的有用信号
whole_signal_fold4 = np.array(whole_signal_fold4)

#fold5
whole_signal_fold5 = []
for i in range(len(fold5)):
    signal1,signal2,signal3,signal4,signal5 = wavelet_transform(fold5[i])
    whole_signal_fold5.append(np.transpose([signal1,signal2,signal3]))#提取前三层的有用信号
whole_signal_fold5 = np.array(whole_signal_fold5)
#print(np.shape(whole_signal))
#print(np.shape(whole_signal[1,:,1]))

#period_plot(len(whole_signal[1,:,2]),whole_signal[1,:,2],'DECG0-15.625Hz','blue',True,2,2,3,1)
#plt.show()

'''
#train

#fold1 as validation set
#X_train = np.concatenate((whole_signal_fold2,whole_signal_fold3,whole_signal_fold4,whole_signal_fold5), axis=0)  # 合并训练集

#Y_train = np.concatenate((fold2label,fold3label,fold4label,fold5label), axis=0)  # 合并训练集标签
X_train = np.reshape(fold1,(160,80000,1))

Y_train = fold1label

X_test = np.reshape(fold2,(20,80000,1))  # 合并测试集

Y_test = fold2label  # 合并测试集标签

print('size of train_set')

print(np.shape(X_train))

print('size of train_label_set')
print(np.shape(Y_train))

print('size of test_set')

print(np.shape(X_test))

print('size of test_label_set')
print(np.shape(Y_test))

#1D
train(U_Net_1d.build(width=80000, depth=1, classes=80000),X_train, Y_train, X_test, Y_test,'core12_t500_earlystop50_fold1_2depth_standardmodel_2sig.h5')
#2D
#train(U_Net_2d.build(width=100, height=50, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold1_model.h5')


'''
#fold2 as validation set
X_train = np.concatenate((whole_signal_fold1,whole_signal_fold3,whole_signal_fold4,whole_signal_fold5), axis=0)  # 合并训练集

Y_train = np.concatenate((fold1label,fold3label,fold4label,fold5label), axis=0)  # 合并训练集标签

X_test = whole_signal_fold2  # 合并测试集

Y_test = fold2label  # 合并测试集标签

print('size of train_set')

print(np.shape(X_train))

print('size of train_label_set')
print(np.shape(Y_train))

print('size of test_set')

print(np.shape(X_test))

print('size of test_label_set')
print(np.shape(Y_test))

#1D
#train(U_Net_1d.build(width=5000, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold2_model.h5')
#2D
train(U_Net_2d.build(width=100, height=50, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold2_model.h5')


#fold3 as validation set
X_train = np.concatenate((whole_signal_fold1,whole_signal_fold2,whole_signal_fold4,whole_signal_fold5), axis=0)  # 合并训练集

Y_train = np.concatenate((fold1label,fold2label,fold4label,fold5label), axis=0)  # 合并训练集标签

X_test = whole_signal_fold3  # 合并测试集

Y_test = fold3label  # 合并测试集标签

print('size of train_set')

print(np.shape(X_train))

print('size of train_label_set')
print(np.shape(Y_train))

print('size of test_set')

print(np.shape(X_test))

print('size of test_label_set')
print(np.shape(Y_test))

#1D
#train(U_Net_1d.build(width=5000, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold3_model.h5')
#2D
train(U_Net_2d.build(width=100, height=50, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold3_model.h5')



#fold4 as validation set
X_train = np.concatenate((whole_signal_fold1,whole_signal_fold2,whole_signal_fold3,whole_signal_fold5), axis=0)  # 合并训练集

Y_train = np.concatenate((fold1label,fold2label,fold3label,fold5label), axis=0)  # 合并训练集标签

X_test = whole_signal_fold4  # 合并测试集

Y_test = fold4label  # 合并测试集标签

print('size of train_set')

print(np.shape(X_train))

print('size of train_label_set')
print(np.shape(Y_train))

print('size of test_set')

print(np.shape(X_test))

print('size of test_label_set')
print(np.shape(Y_test))

#1D
#train(U_Net_1d.build(width=5000, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold4_model.h5')
#2D
train(U_Net_2d.build(width=100, height=50, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold4_model.h5')



#fold5 as validation set
X_train = np.concatenate((whole_signal_fold1,whole_signal_fold2,whole_signal_fold3,whole_signal_fold4), axis=0)  # 合并训练集

Y_train = np.concatenate((fold1label,fold2label,fold3label,fold4label), axis=0)  # 合并训练集标签

X_test = whole_signal_fold5  # 合并测试集

Y_test = fold5label  # 合并测试集标签

print('size of train_set')

print(np.shape(X_train))

print('size of train_label_set')
print(np.shape(Y_train))

print('size of test_set')

print(np.shape(X_test))

print('size of test_label_set')
print(np.shape(Y_test))

#1D
#train(U_Net_1d.build(width=5000, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold5_model.h5')
#2D
train(U_Net_2d.build(width=100, height=50, depth=3, classes=5000),X_train, Y_train, X_test, Y_test,'fold5_model.h5')
'''