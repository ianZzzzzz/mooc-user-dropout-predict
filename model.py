'''
work flow:
    load dataset 
    -> split 
    -> feature,label, sample's enroll id 
    -> train model 
    ->show performance

'''
#%%

#typing 
from typing import List, Dict
from numpy import ndarray
from numpy import datetime64
from pandas import DataFrame

# I/O 
import json
import numpy as np
import pandas as pd
np.set_printoptions(suppress=True)

# model preprocess
from sklearn.impute import SimpleImputer
Padding = SimpleImputer(
    missing_values=np.nan, strategy='most_frequent') 
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
 
#%% 
def split_feature_id_label(dict_data)-> list:
    """
        dict_data : { e_id: {
                            'log_features':[] ,
                            'info_features':[] ,
                            'label' : 1 or 0
                            } 
                    }

    Returns:
        list_assemble_data : [[ *log_features , *info_features ],......]
        list_e_id : [ e_id_1 ,e_id_2,...... ]
        list_label : [ 0, 1, ..........]
    """    
    list_assemble_data = []
    list_e_id = []
    list_label = []
    for e_id,dict_ in dict_data.items():

        log_data  = dict_['features']

        label     = dict_['label']

        list_assemble_data.append(log_data)
        list_e_id.append(int(e_id))
        list_label.append(int(label))

    return list_assemble_data , list_e_id , list_label
def format_transform(
    dict_feature,
    matrix_type: str,
    ignore=True
    )-> dict:
    """[ dict format convert ]
       [ Abandon part : normallize each matrix by action series length]

    Args:
        dict_feature ([dict]): {
            eid1:{ 
                'info_features':[...],
                'log_features' :[...],
                'label:[0 or 1] }
            ,eidn{...}
            }
        matrix_type (str): [simple or complex ]. Defaults to 'simple',have better performance
        ignore (bool, optional): [description]. Defaults to False.

    Returns:
        dict: {
            eid1:{ 
                'features':[...],
                'label:[0 or 1] }
            ,eidn{...}
            }
    """    
    import numpy as np
    if matrix_type =='complex':
        for eid,dict_ in dict_feature.items():
            matrix_= dict_['features'][18:]
            sum_ = np.sum(matrix_)
            if (sum_ != 0):
                for i in range(18,502):
                    dict_feature[eid]['features'][i] =int(30000*dict_feature[eid]['features'][i]/sum_)
        return dict_feature
    if matrix_type =='simple':
        new_dict = {}
        for eid,dict_ in dict_feature.items():
            if ignore ==False:    
                matrix_= dict_['log_features'][8:]           
                sum_ = np.sum(matrix_)
                if sum_ != 0:
                    for i in range(8,24): 
                        # 8/24 head/end position of transfer matrix in list
                        try:
                            dict_feature[eid]['log_features'][i] =int(1000*dict_feature[eid]['log_features'][i]/sum_)
                        except:
                            print(dict_feature[eid]['log_features'][i],sum_)
                        
                    
            new_dict[eid]={}
            new_dict[eid]['features'] = [
                *dict_feature[eid]['info_features'],
                *dict_feature[eid]['log_features']
                ]
           
            new_dict[eid]['label'] = dict_feature[eid]['label']

        return new_dict
def transform_float_label_to_int(
    predict_label_list,
    threshold: float
    )->list:

    predict_label_int = []
    for i in predict_label_list:
        value= i
        if value >threshold:label_ = int(1)
        else:label_ = int(0)
        predict_label_int.append(label_)

    return predict_label_int

def measure(predict_label_int,list_label_test):
    #[ evaluate model]
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
     
    f1        = f1_score(predict_label_int,list_label_test)
    accuracy  = accuracy_score(predict_label_int,list_label_test) 
    precision = precision_score(predict_label_int,list_label_test)
    recall    = recall_score(predict_label_int,list_label_test)

    print(
        'F1',round(f1,4),
        'precision',round(precision,2),
        'recall',round(recall,2),
        'accuracy',round(accuracy,2),
        )
def plot_AUC(ori_label,predict_label):
    # [ plot roc_curve ]
    import pylab as plt
    import warnings;warnings.filterwarnings('ignore')
    from sklearn import metrics
    fpr, tpr, threshold = metrics.roc_curve(ori_label, predict_label)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6,6))
    plt.title('Validation ROC')
    plt.plot(fpr, tpr, 'b', label = ' AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
def xgb_model(
    train_data,
    test_data,
    mode ='predict',
    deep = 4
    )->list:
    import xgboost as xgb
    dtrain = xgb.DMatrix(train_data, list_label_train)
    dtest = xgb.DMatrix(test_data,list_label_test)
    
    
    if mode =='predict':  
        num_rounds = 10
        params = {}
        watchlist = [(dtrain,'train'),(dtest,'test')]  
        XGB = xgb.train(
            params, 
            dtrain, 
            num_rounds,
            watchlist,
            early_stopping_rounds=10)     

        #import pickle
        #SAVE MODEL
        #pickle.dump(XGB, open("XGB_simple_no_nom_f1_9015.pickle.dat", "wb"))

        XGB_predict_label = XGB.predict(dtest)
        #plot_AUC(list_label_test,XGB_predict_label)
        for i in [ 0.4 ]: #BEST threshold
            XGB_predict_label_int = transform_float_label_to_int(XGB_predict_label,threshold=i)
        #    print('XGB ',i,' : ')
        #    measure(
        #            XGB_predict_label_int,list_label_test)    
        return XGB_predict_label_int,XGB_predict_label
    if mode =='analy':
        num_rounds = 10
        params = {
            'objective': 'binary:logistic', 
            'max_depth':deep}
        watchlist = [(dtrain,'train'),(dtest,'test')]
        XGB = xgb.train(
            params, 
            dtrain, 
            num_rounds,
            watchlist,
            early_stopping_rounds=10) 
            
        return XGB
def rf_model(train_data_padding,test_data_padding)->list:
     
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier()
    rf.fit(train_data_padding, list_label_train)
    result = rf.predict(test_data_padding)
    return result,rf.predict_proba(test_data_padding)[:,1]
    #auc = roc_auc_score(list_label_test,rf.predict_proba(test_data_padding)[:,1])
    #print('AUC: ',auc)
    #measure(
    #    result,list_label_test)
    #return rf
    #return rf.predict_proba(test_data_padding)[:,1]
def lr_model(
    train_data_padding,
    test_data_padding,
    mode = 'predict')->list:
    
    from sklearn import linear_model    
    lr = linear_model.LinearRegression(
        normalize=True
        ,n_jobs=-1)
    lr.fit(train_data_padding, list_label_train)
    if mode == 'predict':
        result = lr.predict(test_data_padding)

        #plot_AUC(list_label_test,result)

        #for i in [0.001, 0.499,0.5,0.501 ]:
        for i in [0.5 ]:
            predict_LinearRegression = transform_float_label_to_int(
                result,
                threshold= i)
            #print('LR ',i,' : ')
        #    measure(
        #        predict_LinearRegression,list_label_test)

        return predict_LinearRegression,result
    if mode=='analy':
        return lr 
def show_model_performance(
    xgb,lr,rf,       # list ,int items inside  ,label
    xgb_p,lr_p,rf_p, # list ,float items inside,probability 
    weight=None,mode='vote',
    show=True):
    """[caculate integreated result and show all model's performance]

    Args:
        xgb ([list]): [xgboost int result]
        lr  ([list]): [linear regression int result]
        rf  ([list]): [random forest int result]
        xgb_p ([list]):  [probability result]
        lr_p  ([list]):  [probability result]
        rf_p  ([list]):  [probability result]
        weight ([float], optional): [ set for two model integreated ]. Defaults to None.
        mode (str, optional):  [ decide result by vote or avg_probability]. Defaults to 'vote'.
        show (bool, optional): [ show performance ]. Defaults to True.
    """    
    if mode =='avg':
        new_result = []
        for i in range(len(xgb_p)):
            new_result.append((xgb_p[i]*weight+lr_p[i]*(1-weight)))
        plot_AUC(list_label_test,new_result)
        for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7 ]:
        
            predict_LinearRegression = transform_float_label_to_int(
                new_result,
                threshold= i)
            print('emsenble threshold ',i,' : ')
            measure(
                predict_LinearRegression,list_label_test)

    if show==True:
        dict_ = {
            'xgb':[xgb,xgb_p],
            'lr' :[lr ,lr_p] ,
            'rf' :[rf ,rf_p] }

        for k,v in dict_.items():
            print(k,': ')
            plot_AUC(list_label_test,v[1])
            measure(v[0],list_label_test)
    if mode =='vote':
        result = []
        result_p = []
        for i in range(len(xgb)):
            result_p.append(
                np.mean(
                    [xgb_p[i],lr_p[i],rf_p[i]]
                    ))

            if xgb[i]+lr[i]+rf[i]<1:
                result.append(0)
            else:
                result.append(1)
        print('集成模型：')
        plot_AUC(list_label_test,result_p)
     
        measure(
                result,list_label_test)

#%% Load data
dataset_path ={
    'train' : 'after_processed_data_file\\feature\\train_dataset.json'
    ,'test' : 'after_processed_data_file\\feature\\test_dataset.json'}

dict_train_simple_matrix = json.load(open(dataset_path['train'],'r'))
dict_test_simple_matrix  = json.load(open(dataset_path['test'],'r'))

#%% data preprocess
dict_train_simple_matrix = format_transform(
    dict_train_simple_matrix,
    matrix_type = 'simple',
    ignore=True)
dict_test_simple_matrix = format_transform(
    dict_test_simple_matrix,
    matrix_type = 'simple',
    ignore= True)

list_data_train_simple,list_e_id_train,list_label_train = split_feature_id_label(dict_train_simple_matrix)
list_data_test_simple,list_e_id_test,list_label_test    = split_feature_id_label(dict_test_simple_matrix)

train_data_simple_padding = Padding.fit_transform(
    np.array( list_data_train_simple))
test_data_simple_padding  = Padding.fit_transform(
    np.array( list_data_test_simple))

train_PolynomialFeatures = poly.fit_transform(train_data_simple_padding) 
test_PolynomialFeatures = poly.fit_transform(test_data_simple_padding) 


#%% train and predict
lr_result,lr_prob=lr_model(
    train_PolynomialFeatures,test_PolynomialFeatures)
xgb_result,xgb_prob=xgb_model(
    train_PolynomialFeatures,test_PolynomialFeatures)
rf_result,rf_prob = rf_model(
    train_data_simple_padding,test_data_simple_padding)

show_model_performance(
    xgb=xgb_result,
    xgb_p=xgb_prob,
    lr=lr_result,
    lr_p=lr_prob,
    rf=rf_result,
    rf_p=rf_prob,
    mode='vote',weight=None)




#%% show tree of xgboost and coef of linear regression


col = [
    'gender', 'birth_year', 'edu_degree', 
    'course_category', 'course_type', 'course_duration', 
    'student_amount', 'course_amount',
    'dropout_rate_of_course', 'dropout_rate_of_user',
    'L_mean', 'L_var', 'L_skew', 'L_kurtosis', 
    'S_mean', 'S_var', 'S_skew', 'S_kurtosis',
    'video-video','video-answer','video-comment','video-courseware',
    'answer-video','answer—answer','answer-comment','answer-courseware',
    'comment-video','comment-answer','comment-comment','comment-courseware',
    'courseware-video','courseware-answer','courseware-comment','courseware-courseware']

df_train = pd.DataFrame(train_data_simple_padding,columns = col) 
df_test  = pd.DataFrame(test_data_simple_padding,columns = col)

xgb_plot_tree = xgb_model(
    train_data=df_train,
    test_data=df_test,
    mode='analy',
    deep=4)
import xgboost as xgb
import matplotlib.pyplot as plt
xgb.to_graphviz(xgb_plot_tree )

lr = lr_model(
    test_data_padding=test_data_simple_padding,
    train_data_padding=train_data_simple_padding,
    mode='analy')
lr_coeficient_df = pd.DataFrame(lr.coef_,index=col)






# %%
