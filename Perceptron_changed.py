""""
Spyder Editor
This is a temporary script file.
"""
import numpy as np
import pandas as pd

## Taking in the training dataset from the specified location
## and working on it 
read_train=pd.read_csv('train.data',sep=",",header=None)
read_train=np.array(read_train)

## Test data 
read_test=pd.read_csv('test.data',sep=",",header=None)
read_test=np.array(read_test)

## Receving the last column without any conversion
label_train= np.delete(read_train,np.s_[0:4],1)
label_test= np.delete(read_test,np.s_[0:4],1)
#print ("Labels without conversion",label_train)


# Creation of the new_label array in order to 
# convert classes to the desired number
new_label=np.empty(label_train.shape[0],int)
new_label=np.zeros(label_train.shape[0],int)

new_testlabel=np.empty(label_test.shape[0],int)
new_testlabel=np.zeros(label_test.shape[0],int)
#print("Initial new label",new_label)


############# PRE- PROCESSING DATA ########################

def dataTrainPreprocess(traindata,labelSet):

    for i in range(0,label_train.shape[0]):
        ##Converting the last column of the array to desired numbers for working
        # class-1 vs class-2
        if (label_train[i]=='class-1'):
           new_label[i]=labelSet[0]
        elif (label_train[i]=='class-2'):
           new_label[i]=labelSet[1]
        else:
            new_label[i]=labelSet[2]
            
     ## The dataset to be used for 
    new_traindata=read_train[:,:4]
    new_traindata=np.c_[new_traindata,new_label]
    #print(new_traindata)
    return new_traindata
 
def dataTestPreprocess(testdata,labelSet):    
    for i in range(0,label_test.shape[0]):
        ##Converting the last column of the array to desired numbers for working
        # class-1 vs class-2
        if (label_test[i]=='class-1'):
            new_testlabel[i]=labelSet[0]
        elif (label_test[i]=='class-2'):
            new_testlabel[i]=labelSet[1]
        else:
            new_testlabel[i]=labelSet[2]
           
    
    #   ## The dataset to be used for creating the training data
    # new_traindata=read_train[:,:4]
    # new_traindata=np.c_[new_traindata,new_label]
    # #print(new_traindata)   
            
      ## The dataset to be used for creating the testing data
    new_testdata=read_test[:,:4]
    new_testdata=np.c_[new_testdata,new_testlabel]
    #print(new_testdata)
    return new_testdata
    
##################### BINARY PERCEPTRON ##################

def TrainCal(traindata,maxIter):
   
    sampleCounter=0
    w=np.array([0,0,0,0])
    bias=0
    wrongCounter=0
    accuracy=0
  
    
    np.random.shuffle(traindata)
    x_features=traindata[:,[0,1,2,3]]
    y=traindata[:,4]
    
    for j in range(0,maxIter):        
        for i in range(0,len(traindata)):
            #print(x_features[i])
            
            activation=np.inner(w,x_features[i])+bias
            sampleCounter+=1
            #print(activation)
            #print(y[i]*1)
            #print("Counter ", sampleCounter)
            if (y[i]*activation<=0 and y[i]!=0):
                bias=bias+y[i]
                w=w+(y[i]*x_features[i])
                # print("Updated weights ", w )
                # print("Updated Bias", bias)
                wrongCounter+=1
                
            accuracy=(1-wrongCounter/sampleCounter)*100
            accuracy=round(accuracy,2)
    print("Training Accuracy: ",accuracy)
    return w,bias

def TestCal(testdata,w,bias):
    
    wrongCounter=0    
    y=testdata[:,4]
    counter=0
    #print(y)
    
    for i in range(0,len(testdata)):
        counter+=1
        activation=np.dot(w,testdata[i][:4])+bias
              
              
        if activation > 0 :
            activation = 1
        else:
            activation= -1
        #print (activation)
        
        if (y[i]!=0 and y[i]!=activation):
            #print(y[i])
            wrongCounter+=1
            
    #print("No of wrong values ",wrongCounter)
    accuracy=(1-wrongCounter/counter)*100
    accuracy=round(accuracy,2)
    print("Testing Accuracy",accuracy)  
    #print(counter,wrongCounter)
    
######################### MULTI CLASS ##############################

def multiTrainCal(traindata,maxIter):
   
    sampleCounter=0
    w=np.array([0,0,0,0])
    bias=0
    wrongCounter=0
    accuracy=0
  
    
    np.random.shuffle(traindata)
    x_features=traindata[:,[0,1,2,3]]
    y=traindata[:,4]
    
    for j in range(0,maxIter):        
        for i in range(0,len(traindata)):
                        
            activation=np.inner(w,x_features[i])+bias
            sampleCounter+=1
            if (y[i]*activation<=0 and y[i]!=0):
                bias=bias+y[i]
                w=w+(y[i]*x_features[i])
                wrongCounter+=1
                
            accuracy=(1-wrongCounter/sampleCounter)*100
            accuracy=round(accuracy,2)
    #print("Training Accuracy: ",accuracy)
    return w,bias,accuracy

def multiTestCal(testdata,w1,b1,w2,b2,w3,b3):
    wrongCounter=0
    counter=0
    y=testdata[:,4]
   
    
    for i in range(0,len(testdata)):
        counter+=1
        activation1=np.dot(w1,testdata[i][:4])+b1
        activation2=np.dot(w2,testdata[i][:4])+b2
        activation3=np.dot(w3,testdata[i][:4])+b3
        activation=max(activation1,activation2,activation3)    
        #print(activation)
              
        if activation==activation1:
            activation=1
        elif activation==activation2:
            activation=2
        else:
            activation=3
               
        if (y[i]!=activation):
            #print(y[i])
            wrongCounter+=1
            
    #print("No of wrong values ",wrongCounter)
    accuracy=(1-wrongCounter/counter)*100
    accuracy=round(accuracy,2)
    #print("Testing Accuracy for Multi class",accuracy)  
    #print(counter,wrongCounter)
    return accuracy
        
######################## L2 Reguarisation #######################

def multiRegTrainCal(traindata,maxIter,l2reg):
   
    sampleCounter=0
    w=np.array([0,0,0,0])
    bias=0
    wrongCounter=0
    accuracy=0
  
    
    np.random.shuffle(traindata)
    x_features=traindata[:,[0,1,2,3]]
    y=traindata[:,4]
    
    for j in range(0,maxIter):        
        for i in range(0,len(traindata)):
                        
            activation=np.inner(w,x_features[i])
            sampleCounter+=1
            if (y[i]*activation<=0 and y[i]!=0):
                bias=bias+y[i]
                w=(1-(2*l2reg))*w+(y[i]*x_features[i])
                wrongCounter+=1
                
            accuracy=(1-wrongCounter/sampleCounter)*100
            accuracy=round(accuracy,2)
    #print("Training Accuracy: ",accuracy)
   
    return w,bias,accuracy




############### The different arrays ############################
labelSet=np.array([[1,-1,0],[1,0,-1],[0,1,-1]])
cases=[" Class 1 vs Class 2","Class 1 vs Class 3"," Class 2 vs Class 3"]

labelSet1=np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]])
#cases1=["Class 1 vs rest","Class 2 vs rest","Class 3 vs rest" ]

### QUESTION 3 & 2

for i in range(0,len(labelSet)):
    new_traindata=dataTrainPreprocess(label_train, labelSet[i])
    print("------------------------------------------------")
    print("Result of ",cases[i])
    #print("------------------------------------------------")
    (x,y)=TrainCal(new_traindata,20)   
    new_testdata=dataTestPreprocess(label_test, labelSet[i])
    #TestCal(new_traindata,x,y)
    TestCal(new_testdata,x,y)
    


###################### QUESTION 4 ##########################
 
new_traindata1=dataTrainPreprocess(label_train, labelSet1[0])
new_traindata2=dataTrainPreprocess(label_train, labelSet1[1])
new_traindata3=dataTrainPreprocess(label_train, labelSet1[2])
(w1,b1,a1)=multiTrainCal(new_traindata1,20)
(w2,b2,a2)=multiTrainCal(new_traindata2,20)
(w3,b3,a3)=multiTrainCal(new_traindata3,20)


## Set the label to three different integers to 
## differentiate which class the testing data would be closest to.
## No need to convert it to -1,1 or 0

label=np.array([1,2,3])
new_testdata=dataTestPreprocess(label_test, label)

print("------------------------------------------------")
print("Result of Q4 - MULTI CLASS \n")   

accuracy=(a1+a2+a3)/3
accuracy=round(accuracy,2)
print("Training accuracy for Multi class is: ", accuracy) 
accuracy=multiTestCal(new_testdata,w1,b1,w2,b2,w3,b3)
print("Testing Accuracy for Multi class: ",accuracy)  

    
####################### QUESTION 5 ##########################
l2reg=[0.01,0.1,1.0,10.0,100.0]
l2class=['0.01','0.1','1.0','10.0','100.0']


print("-------------------------------------------------")
print("                 L2 Reg   " )
new_traindata1=dataTrainPreprocess(label_train, labelSet1[0])
new_traindata2=dataTrainPreprocess(label_train, labelSet1[1])
new_traindata3=dataTrainPreprocess(label_train, labelSet1[2])

for i in range(0,len(l2reg)):
    
    (w1,b1,a1)=multiRegTrainCal(new_traindata1,20,l2reg[i])
    (w2,b2,a2)=multiRegTrainCal(new_traindata2,20,l2reg[i])
    (w3,b3,a3)=multiRegTrainCal(new_traindata3,20,l2reg[i])

    accuracy=(a1+a2+a3)/3
    accuracy=round(accuracy,2)
    print("Training accuracy for",l2class[i]," is ",accuracy)
 
    label=np.array([1,2,3])
    new_testdata=dataTestPreprocess(label_test, label)
    accuracy=multiTestCal(new_testdata,w1,b1,w2,b2,w3,b3)
    print("Testing accuracy for",l2class[i]," is: ",accuracy)
    
    print("-------------------------------------------------")