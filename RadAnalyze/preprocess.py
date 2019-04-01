# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:16:44 2018

@author: Caizhengting
"""
import re
import os
from pandas import *
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.model_selection import StratifiedKFold

def merge_header(fname,encode='gbk',sheet=0):
    """
    Merge the headers and set the patientID as the index of raw files downloaded from Radcloud cloud
    """
    # input the file
    if re.findall('.csv',fname): # the format is csv
        df = read_csv(fname, header = None, encoding = encode)
    elif re.findall('.xls',fname) or re.findall('.xlsx',fname): # the format is xls or xlsx
        df = read_excel(fname, sheet, header = None)
    else: 
        raise AttributeError('the format of file is not "csv", "xls" or "xlsx".')
    # merge the header containing three rows
    cols = list(df.iloc[0:3,:].values);df.columns = cols 
    cols = ['_'.join(col[::-1]) for col in list(df.columns)];df.columns = cols # renew the columns'name, namely the header
    df = df.iloc[3:,:] # delete the top three rows , which actually are the headers
    df = df.apply(to_numeric, errors='ignore') # change to numeric as much as possible
    # set the patientID as the index
    try:
        df.custom_custom_patient_id = df.custom_custom_patient_id.str.strip() # delete the space
    except AttributeError:
        pass
    df.custom_custom_patient_id = df.custom_custom_patient_id.replace(regex=' +',value = '-') # replace all of Non-numeric characters as '-'
    df.index = df.custom_custom_patient_id;df.index = df.index.rename('MLID') # renew the index's name
    return(df.sort_index())

def get_features(df):
    """
    Get the columns representing radiomics features from the  processed files by 'merge_header' method 
    """
    for i in range(len(df.columns)) :
        if not('custom' in df.columns[i]) : break
    feature = df.iloc[:,i:]
    return(feature.sort_index())

def Std_features(feature,mean=0,std=1):
    """
    Standardization of feature values, converted into those with specific mean and standard deviation
    """
    scaler = StandardScaler().fit(feature)
    feature_std = scaler.transform(feature) # 
    feature_std = DataFrame(feature_std, columns = feature.columns, index=feature.index)
    feature_std = (feature_std+mean)*std
    return(feature_std.sort_index())

def DataFrame_label(label):
# seem not very useful
    if type(label) == type(Series()):
        label = DataFrame(label_binarize(label,classes=label.drop_duplicates()),index=label.index)
    elif type(label) == type(DataFrame()):
        label = label.iloc[:,0]
        label = DataFrame(label_binarize(label,classes=label.drop_duplicates()),index=label.index)
    else:
         raise AttributeError('Input structure of label error. Only can use Series and DataFrame!')
    label.columns = ['label']
    label.index = label.index.rename('MLID')
    return(label.sort_index())

def get_CVID(label,run=1000,n_splits=5,name='',fpath='./'):
    """
    Group the whole data using cross-validation 
    """
    if run == 1: 
        return(get_CVID_CSV(label,n_splits=n_splits,name=name,fpath=fpath))
    else:
        label = DataFrame_label(label)
        if name == '':
            writer_train = ExcelWriter(os.path.join(fpath,'ID_Train_CV.xlsx'))
            writer_test = ExcelWriter(os.path.join(fpath,'ID_Test_CV.xlsx'))
        else:
            writer_train = ExcelWriter(os.path.join(fpath,'_'.join([name,'ID_Train_CV.xlsx'])))
            writer_test = ExcelWriter(os.path.join(fpath,'_'.join([name,'ID_Test_CV.xlsx'])))

        for num in range(run):
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
            ID_train = DataFrame()
            ID_test = DataFrame()
            # Stratified k-fold
            i = 0
            for train, test in kf.split(label,label.label):
                train = DataFrame(label.iloc[train,:].index.rename(str(i)))
                test = DataFrame(label.iloc[test,:].index.rename(str(i)))
                ID_train = concat([ID_train, train], axis = 1)
                ID_test = concat([ID_test, test], axis = 1)
                i = i + 1
            ID_train.to_excel(writer_train, str(num))
            ID_test.to_excel(writer_test, str(num))
        writer_train.save()
        writer_test.save()

def get_CVID_CSV(label,n_splits=5,name='One',fpath='./'):
    """
    Group the whole data using cross-validation, just one run
    """
    label = DataFrame_label(label)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    ID_train = DataFrame()
    ID_test = DataFrame()
    i = 0
    for train, test in kf.split(label,label.label):
        train = DataFrame(label.iloc[train,:].index.rename(str(i)))
        test = DataFrame(label.iloc[test,:].index.rename(str(i)))
        ID_train = concat([ID_train, train], axis = 1)
        ID_test = concat([ID_test, test], axis = 1)
        i = i+1
    ID_train.to_csv(os.path.join(fpath,'_'.join([name,'ID_Train_CV.csv'])))
    ID_test.to_csv(os.path.join(fpath,'_'.join([name,'ID_Test_CV.csv'])))
    return(ID_train,ID_test)