# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:13:24 2018

# UPDATE IN 20180820

@author: Caizhengting
"""
import os
from numpy import *
from pandas import *
import matplotlib.pyplot as plt
from own_packages.SN import *
from scipy import interp
from copy import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,brier_score_loss,roc_auc_score

from sklearn.metrics import auc

def CV_learning(paras,n_splits,scoring='roc_auc'):
    """
    Machine learning using cross-validation, and use the class of paras
    Not store the ID of training set or test set
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    scores = []
    pred_proba_group = []
    try:
        scorer = check_scoring(paras.model, scoring=scoring)
    except:pass
    for train, test in kf.split(paras.X,paras.y):
        X_train = paras.X.iloc[train,:]
        y_train = paras.y.iloc[train]
        X_test  = paras.X.iloc[test,:]
        y_test  = paras.y.iloc[test]
        paras.model.fit(X_train,y_train)
        pred_proba_tmp = DataFrame(paras.model.predict_proba(X_test),index = X_test.index)
        pred_proba_group.append(pred_proba_tmp)
        try:
            scores.append(scorer(paras.model,X_test,y_test))
        except:
            if scoring == 'aic':
                scores.append(Criterion(y_test,pred_proba_tmp.iloc[:,1],len(X_test.columns)).AIC())
            elif scoring == 'bic':
                scores.append(Criterion(y_test,pred_proba_tmp.iloc[:,1],len(X_test.columns)).BIC())
            else:
                raise ValueError('%r is not a valid scoring value.' 
                                'Use sorted(sklearn.metrics.SCORERS.keys()' 
                                'to get valid options.' %(scoring))
    pred_proba_all = concat(pred_proba_group).sort_index()
    return(scores,pred_proba_group,pred_proba_all)

class permutation_test(paras):
    """
    permutation test, score is AUC
    置换检验
    """
    def __init__(self,X,y,name,model='',fpath='./',size=10):
       super().__init__(model=model,X_train=X,y_train=y,name=name,fpath=fpath,size=size)

    def pvalue(self,real_auc,n_splits=5,run=1000):
        self.__aucs = list()
       # kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        for num in range(run):
            Rself = copy(self)
            Rself.y = Series(np.random.permutation(self.y_train),index = self.y_train.index)
            _,_,pred_proba_all= CV_learning(Rself,n_splits)
            roc_auc = roc_auc_score(Rself.y, pred_proba_all.iloc[:,1])
            self.__aucs.append(roc_auc)
        self.__pvalue = (sum(array(self.__aucs) > real_auc)+1)/(run+1)
        return(self.__pvalue)
        
    def pvalue_plot(self,real_auc,n_splits=5,run=1000):
        try: #judge whether having pvalue and aucs or not
            self.__pvalue
        except AttributeError:
            self.__pvalue = self.pvalue(real_auc=real_auc,n_splits=n_splits,run=run)
            
        fig, ax = plt.subplots()
        ax.hist(self.__aucs,edgecolor="black")
        if self.__pvalue >= 0.001:
            ax.axvline(real_auc, linestyle='--', color='k',label='real AUC(pvalue = %0.3f)' %self.__pvalue)
        else:
            ax.axvline(real_auc, linestyle='--', color='k',label='real AUC(pvalue < 0.001)')
        xunit,yunit = plot_unit([ax.get_xlim(),ax.get_ylim()])
        ax.text(real_auc-xunit, ax.get_ylim()[1]+yunit, round(real_auc,3))
        ax.legend(loc="upper left")
        ax.set_xlabel('AUC')
        ax.set_ylabel('Count')
        ax.set_title('Permutation Test')
        fig.savefig(os.path.join(self.fpath,'_'.join([self.name,'Permutation.png'])))
        return(self.__pvalue)

def pred_indice(obs,pred_proba_pos,name,label,fpath,cutoff_typeof='Youden',CI_typeof='Binomial'):
    cutoff = Find_Optimal_Cutoff(obs,pred_proba_pos,typeof=cutoff_typeof)
    pred = DataFrame(index=obs.index,columns=['label'])
    pred[pred_proba_pos >= cutoff[1]] = 1; pred[pred_proba_pos < cutoff[1]] = 0

    indice= dict()
    model_report = classification_report(obs,pred,target_names=label)
    model_matrix = confusion_matrix(obs,pred)
    TN, FP, FN, TP = confusion_matrix(obs,pred).ravel()
    indice['cutoff'] = cutoff # cutoff = [typeof,value]
    indice['recall'] = recall_score(obs,pred) # sensitivity = recall
    indice['precision'] = precision_score(obs,pred)
    indice['sensitivity'] = recall_score(obs,pred) # sensitivity = recall
    indice['specificity'] = TN / (TN + FP)
    indice['accuracy'] = accuracy_score(obs,pred)
    indice['F1'] = f1_score(obs,pred)
    indice['brier'] = brier_score_loss(obs,pred_proba_pos,pos_label=obs.max())
    indice['AUC'] = roc_auc_score(obs,pred_proba_pos)
    indice['95%CI-AUC'] = CI_ROC(obs,pred_proba_pos,typeof=CI_typeof) # CI = [typeof,CI_lower,CI_upper]

    fname =os.path.join(fpath,'_'.join([name,'stat.txt']))
    with open(fname, 'w') as f:
        print(model_report,file=f)
        print('[Confusion Matrix]',file=f)
        print('TN\tFP\nFN\tTP',file=f)
        print(model_matrix,'\n',file=f)
        for key,value in zip(indice.keys(),indice.values()):
            print(key,'=',value,file=f)
    with open(fname, 'r') as f:
        print(''.join(f.readlines()))
    return(pred,indice)

def merge_indice_group(pred_indice_group,name,fpath='./'):
    indice_group = DataFrame()
    cv = 0
    for indice_tmp in pred_indice_group:
        try: del indice_tmp['cutoff']
        except: pass
        try: del indice_tmp['95%CI-AUC']
        except: pass
        indice_tmp = DataFrame.from_dict(indice_tmp,orient='index',columns=[str(cv)])
        indice_group = concat([indice_group,indice_tmp],axis=1)
        cv += 1
    Mean = indice_group.mean(axis=1); Mean.name = 'Mean'
    Std = indice_group.std(axis=1); Std.name = 'Std'
    indice_group = concat([indice_group,Mean,Std],axis=1)
    print(indice_group)
    indice_group.to_csv(os.path.join(fpath,'_'.join([name,'stat_CV.csv'])))
    return(indice_group)

def print_model_params(model_params,name,fpath):
    pname = os.path.join(fpath,'_'.join([name,'Params_model.txt']))
    with open(pname, 'w') as f:
        print('\nthe Params of model\n',file=f)
        for key in model_params:
            print('\t{0}:{1}\n'.format(key,model_params[key]),file=f)

def ROC_plot_One(obs,pred_proba_pos,name,fpath,size):
    """
    Plot the ROC curve for a specific class
    """
    plot_default(size=size)
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(obs,pred_proba_pos)
    roc_auc = auc(fpr,tpr)
    tpr[0] = 0.0
    ax.plot(fpr,tpr,color='navy',label='AUC = {0:.3f}'.format(roc_auc))
    ax.plot([0, 1],[0, 1],color='darkorange',linestyle='--',alpha=.8)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic(ROC)')
    fig.savefig(os.path.join(fpath,'_'.join([name,'ROC.png'])))

def ROC_plot_group(obs_group,pred_proba_group,name,fpath='./',size=10):
    
    plot_default(size=size)
    fig, ax = plt.subplots()
    tprs = []
    auc_group = []
    mean_fpr = linspace(0, 1, 100)
    for cv in range(len(obs_group)):
        obs = obs_group[cv]
        pred_proba = pred_proba_group[cv]
        # Compute ROC curve and area the curve
        fpr, tpr, _ = roc_curve(obs, pred_proba.iloc[:,1])
        tprs.append(interp(mean_fpr,fpr,tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        auc_group.append(roc_auc)
        ax.plot(fpr,tpr,alpha=0.6,label='ROC fold {0} (AUC = {1:.3f})'.format(cv,roc_auc))

    mean_tpr = mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr,mean_tpr)
    std_auc = std(auc_group)
    ax.plot(mean_fpr,mean_tpr,color='navy',label=r'Mean ROC (AUC = {0:.3f} $\pm$ {1:.3f})'.format(mean_auc, std_auc),lw=2)
    std_tpr = std(tprs, axis=0)
    tprs_upper = minimum(mean_tpr + std_tpr, 1)
    tprs_lower = maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr,tprs_lower,tprs_upper,color='grey',alpha=.3,label=r'$\pm$ 1 std. dev.')
    ax.plot([0, 1],[0, 1],color='darkorange',linestyle='--',alpha=.8)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic(ROC)')
    fig.savefig(os.path.join(fpath,'_'.join([name,'ROC_CV.png'])))

def ROC_compare(obs,pred_proba_pos_group,fpath='./',size=10):
    if type(pred_proba_pos_group) != type(DataFrame()):
        raise ValueError('Please transfer the input data into the DataFrame structure!')
    plot_default(size=size)
    fig, ax = plt.subplots()
    for pred_proba_pos,name in zip(pred_proba_pos_group,pred_proba_pos_group.columns):
        fpr, tpr, _ = roc_curve(obs, pred_proba_pos)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label='{0} (AUC = {1:.3f})' .format(name, roc_auc))
    ax.plot([0, 1], [0, 1],color='darkorange',linestyle='--',alpha=.8)
    ax.legend(loc="lower right")
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic(ROC)')
    fig.savefig(os.path.join(fpath,'Compare_ROC.png'))
    return(Delong_test(obs,pred_proba_pos_group))

def Output_One(paras,label,data_typeof,cutoff_typeof,CI_typeof):
    data_typeof = data_typeof.lower()
    name = '_'.join([paras.name,data_typeof.lower()])
    fpath = paras.fpath
    if data_typeof == 'test':
        obs = paras.y_test
        fea = paras.X_test
    elif data_typeof == 'train':
        obs = paras.y_train
        fea = paras.X_train
    else:
        obs = paras.y
        fea = paras.X

    print('------{0}------'.format(data_typeof))
    if data_typeof == 'test' or data_typeof == 'train':
        pred_proba = DataFrame(paras.model.predict_proba(fea),index=fea.index,columns=label)
    else:
        pred_proba = paras.pred_proba
    pred_proba_pos = pred_proba.iloc[:,1]
    pred,indice = pred_indice(obs,pred_proba_pos,name,label,fpath,cutoff_typeof,CI_typeof)
    ROC_plot_One(obs,pred_proba_pos,name,fpath,paras.size)
    pred_proba.to_csv(os.path.join(fpath,'_'.join([name,'pred_proba.csv'])))
    pred.to_csv(os.path.join(fpath,'_'.join([name,'pred.csv'])))
    if data_typeof == 'test':
        paras.pred_test = pred
        paras.pred_proba_test = pred_proba
        paras.indice_test = indice
    elif data_typeof == 'train':
        paras.pred_train = pred
        paras.pred_proba_train = pred_proba
        paras.indice_train = indice
    else:
        paras.pred = pred
        paras.indice = indice
    return(paras)

class ModelPred(paras):
    """
    """
    def __init__(self,model,X_train,y_train,X_test,y_test,name,fpath='./',size=10):
        super().__init__(model=model,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,name=name,fpath=fpath,size=size)
    
    def OneSplit(self,label,cutoff_typeof='Youden',CI_typeof='Binomial'):
        self.model.fit(self.X_train,self.y_train)

        # print the parameters of model
        print_model_params(self.model.get_params(),self.name,self.fpath)
        
        # pred of proba and label, indices, ROC
        self = Output_One(self,label,'test',cutoff_typeof,CI_typeof)
        self = Output_One(self,label,'train',cutoff_typeof,CI_typeof)         
            
        # output
        return(self)

    def OneCV(self,label,cutoff_typeof='Youden',CI_typeof='Binomial'):
        fpath_out = os.path.join(self.fpath,self.name)
        if not os.path.exists(fpath_out):os.makedirs(fpath_out)
        output_group = paras(y_test=list(),pred_proba_test=list(),indice_test=list(),
                            y_train=list(),pred_proba_train=list(),indice_train=list())
        output_all = paras(fpath=fpath_out,name=self.name)
        cv = 0
        print('-----Start Model-----')
        for X_test,X_train,y_test,y_train in zip(self.X_test,self.X_train,self.y_test,self.y_train):
            print('-----{0}-----'.format(cv))
            fpath_tmp = os.path.join(self.fpath,str(cv))
            name = '_'.join([self.name,str(cv)])
            self_tmp = paras(model=self.model,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,fpath=fpath_tmp,name=name)
            self_tmp = ModelPred.OneSplit(self_tmp,label,cutoff_typeof,CI_typeof)
            output_group.y_train.append(y_train)
            output_group.y_test.append(y_test)
            output_group.pred_proba_train.append(self_tmp.pred_proba_train)
            output_group.pred_proba_test.append(self_tmp.pred_proba_test)
            output_group.indice_train.append(self_tmp.indice_train)
            output_group.indice_test.append(self_tmp.indice_test)
            cv += 1
        output_all.y = concat(output_group.y_test).sort_index()
        output_all.pred_proba = concat(output_group.pred_proba_test).sort_index()
        # CV
        train_name = '_'.join([self.name,'train'])
        test_name = '_'.join([self.name,'test'])
        output_group.indice_train = merge_indice_group(output_group.indice_train,self.name,fpath_out)
        output_group.indice_test = merge_indice_group(output_group.indice_test,self.name,fpath_out)
        ROC_plot_group(output_group.y_train,output_group.pred_proba_train,train_name,fpath_out,self.size)
        ROC_plot_group(output_group.y_test,output_group.pred_proba_test,test_name,fpath_out,self.size)
        #all
        output_all = Output_One(output_all,label,'all',cutoff_typeof,CI_typeof)
        print('-----Finish!-----')
        return(output_group,output_all)