#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 01:55:35 2022

@author: fujita
"""
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch
import scipy.stats as sts
import statsmodels.stats as stsmodel
from statsmodels.stats.multitest import local_fdr



class Tensor_class:
    

    def __init__(self):

        self.timepointlist=[-10,0,10,20,30,45,60,75,90,120,150,180,210,240]#-10
        self.timepoint4list=[-5,0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240]
        
        
    def NormalizeEachMolEachSubjZscore(self,DF):
        NewDF = DF.copy()
        for i in self.SubjectName:
            NewDF.loc[i] =self.mkZscore(DF.loc[i], list(DF.index),list(DF.columns),'col')     
        return(NewDF)
        
    def mkFasting(self,Data):
        MolLabel=list(Data.index)
        ###FastingIdx=[i + 14*(j-1) for j in range(1,21) for i in [0,1] ]
        FastingDF=pd.DataFrame(data=None,index=range(20),columns=MolLabel)
        for j in range(1,21):    
                FastingDF.iloc[j-1,:]=list(np.nanmean(Data.iloc[:,[14*(j-1),14*(j-1)+1] ],axis=1))    
        return(FastingDF)
    def mkFasting4d(self,Data):
        MolLabel=list(Data.index)
        ###FastingIdx=[i + 14*(j-1) for j in range(1,21) for i in [0,1] ]
        FastingDF=pd.DataFrame(data=None,index=range(3),columns=MolLabel)
        for j in range(1,4):    
                FastingDF.iloc[j-1,:]=list(np.nanmean(Data.iloc[:,[26*(j-1),26*(j-1)+1] ],axis=1))    
        return(FastingDF)
    def mkZscore(self,CorrFastPropDF,label,labelProp,axis):
        if axis=='col':
            ax=0
        else:
            ax=1
        meanNP = np.array(CorrFastPropDF) -np.nanmean(np.array(CorrFastPropDF),axis=ax,keepdims=True)
        stdNP = np.nanstd(np.array(CorrFastPropDF),axis=ax)
        signDF = pd.DataFrame(index=label,columns=labelProp)
        
        for i in range(len(labelProp)): 
            signNP = meanNP[:,i]/stdNP[i]
            signDF[labelProp[i]] = signNP
        return(signDF)  
    
    def df_z(self,df, axis=0):
        if axis == 0:
            return (df - df.mean())/df.std()
        else:
            df = df.T
            df = (df - df.mean())/df.std()
            return df.T    
        
    def NaNtozero(self,DF):
        Col = list(DF.columns)      
        list(itertools.chain.from_iterable(DF[Col][DF[Col].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()].values))

    def ConstructionTucker(self,dim,S,Ulist):
        from tensorly import tucker_to_tensor
        return(tucker_to_tensor((S, Ulist)))
    
    def Construction(self,dim,S,Ulist):
        metaStr = self.transformationStrBuilder(dim)
        AA = torch.einsum(metaStr,S,*Ulist) 
        return(AA)        
    def shape2index(self,n):
        indexs = ""
        assert n < 8
        for i in range(n):
            indexs = indexs + chr(97+i)
        return indexs
    def transformationStrBuilder(self,n):
        metaStr = self.shape2index(n)
        metaStr += ","
        for i in range(n):
            metaStr += chr(97+i)+chr(97+n+i)
            if i != n-1:
                metaStr += ","
        metaStr += "->"
        for i in range(n):
            metaStr += chr(97+n+i)
        return metaStr
        



    def Analeachcomponents(self,comptimeDF,compmetaboDF,compsubjectDF,OptionDict):

        ColorList=['magenta','orange','purple','cyan']
        self.plotMultComp(comptimeDF[[0,1]],'Comptime12',ColorList[:2])
        self.plotMultComp(comptimeDF[[2,3]],'Comptime34',ColorList[2:])

        
        OptionDict['xlabel']='Component  1';OptionDict['ylabel']='Component 2';OptionDict['Annotate']=0;OptionDict['title']='Compindividuals';OptionDict['calcR'] =''
        ColorList = ['black' for i in range(len(compsubjectDF.index))]
        self.mkScatterWHist(compsubjectDF[0],compsubjectDF[1],ColorList,OptionDict)
        self.mkBar(list(compsubjectDF[0]), 'Compsubject1', 'Component 1', '', 'black', [str(i+1) for i in range(20)], 10, 10,10)
        self.mkBar(list(compsubjectDF[1]), 'Compsubject2', 'Component 2', '', 'black', [str(i+1) for i in range(20)], 10, 10,10)
        #self.mkBar(list(compsubjectDF[4]), 'Compsubject5', 'Component 5', '', 'black', [str(i+1) for i in range(20)], 10, 10,10,save_dir)
        #self.mkBar(list(compsubjectDF[5]), 'Compsubject6', 'Component 6', '', 'black', [str(i+1) for i in range(20)], 10, 10,10,save_dir)
 
        
        OptionDict['Label']=list(compmetaboDF.index);OptionDict['Annotate']=1;OptionDict['markersize'] = 30

        OptionDict['xlabel']='Component  1';OptionDict['ylabel']='Component 2';OptionDict['Annotate']=1;OptionDict['title']='Compmetabolites12_abs';OptionDict['calcR'] =''
        ColorList = ['black' for i in range(len(compmetaboDF.index))]
        self.mkScatterWHist(compmetaboDF[0].abs(),compmetaboDF[1].abs(),ColorList,OptionDict)
        OptionDict['title']='Compmetabolites12';
        self.mkScatterWHist(compmetaboDF[0],compmetaboDF[1],ColorList,OptionDict)
        
        OptionDict['xlabel']='Component  3';OptionDict['ylabel']='Component 4';OptionDict['Annotate']=1;OptionDict['title']='Compsubject34'
        self.mkScatterWHist(compmetaboDF[2],compmetaboDF[3],ColorList,OptionDict)
        OptionDict['xlabel']='Component  5';OptionDict['ylabel']='Component 4';OptionDict['Annotate']=1;OptionDict['title']='Compsubject56'
        #GH.mkScatterWHist(compmetaboDF_New['X5'],compmetaboDF_New['X6'],save_dir,ColorList,OptionDict)
        OptionDict['xlabel']='Component  7';OptionDict['ylabel']='Component 8';OptionDict['Annotate']=1;OptionDict['title']='Compsubject78'
        #GH.mkScatterWHist(compmetaboDF_New['X7'],compmetaboDF_New['X8'],save_dir,ColorList,OptionDict)
        
        modules = dict()

        for ii in [0,1,2,3] :   
            Pvalue = sts.chi2.sf(sts.zscore(compmetaboDF[ii])**2,1)        
            Pvalue = sts.chi2.sf(sts.zscore(compmetaboDF[ii])**2,1)
            _,QvalueBH,_,_ = stsmodel.multitest.multipletests(Pvalue, alpha=0.1,method='fdr_bh')

            QvalueBH=pd.DataFrame(QvalueBH)
            QvalueBH.index= list(compmetaboDF.index)
            mollist = list(QvalueBH[QvalueBH[0]<OptionDict['qvalcutoff']].index)
            print(mollist)  
            modules[ii] = mollist
        return(modules)
            
    def plotMultComp(self,DF,title,colorList=cm.jet):
        
        Col = list(DF.columns)
        plt.rcParams["font.size"] = 10
        plt.rcParams['lines.linewidth'] = 20
        fig=plt.figure(figsize=(7.5,7.5))
        ax=fig.add_subplot(111)
        for i in range(0,len(Col)): #cm.jet(i/len(Col))
            ax.plot(list(DF[Col[i]].index),list(DF[Col[i]]),lw=2, marker="o",markersize=10,color=colorList[i],label = Col[i])#axis=0   
            ax.set_xticks(list(DF[Col[0]].index))
            #ax.legend(fontsize=8,loc='upper right')
            
        axis=['top','bottom','left','right']
        line_width=[0.5, 0.5, 0.5, 0.5]
        for a,w in zip(axis, line_width):  # change axis width
            ax.spines[a].set_linewidth(w)
        ax.tick_params(labelleft="true",direction='out',labelsize=20,axis='both');
        ax.set_xticks([ 0, 60, 120, 180, 240]) #[0, 0.2, 0.4, 0.6]
        ax.set_xticklabels(['0', '60', '120', '180', '240'])#, rotation=30, fontsize='small') #'0.0', '0.2', '0.4', '0.6'
        ax.set_xlabel('Time(min))')
        ax.set_ylabel('zscore')
        ax.set_title(title)

    def plotComp(self,DF,save_dir,title):
        Col = list(DF.columns)
        plt.rcParams["font.size"] = 10
        plt.rcParams['lines.linewidth'] = 20
    
        for i in range(0,len(Col)):
            fig=plt.figure(figsize=(10,7.5))
            ax=fig.add_subplot(111)
            
            ax.plot(list(DF[Col[i]].index),list(DF[Col[i]]),color='black',lw=1, marker="o",)#axis=0   
            ax.set_xticks(list(DF[Col[0]].index))
            
            axis=['top','bottom','left','right']
            line_width=[0.5, 0.5, 0.5, 0.5]
            
            for a,w in zip(axis, line_width):  # change axis width
                ax.spines[a].set_linewidth(w)





    def mk3dData(self,Data, OptionDict):
        result_arr=[]
        for i in range(1,21):
            AddDF = Data.iloc[:,(14*(i-1)):(14+14*(i-1))].fillna(0).T     
            AddDF.index=self.timepointlist
            if i== 1:                 
                TmCsDF = Data.iloc[:,(14*(i-1)):(14+14*(i-1))].fillna(0).T
                TmCsDF.index=self.timepointlist
                TmCshDF = TmCsDF.copy()
            else:
                TmCshDF=pd.concat([TmCshDF,AddDF],axis=0) 
                TmCsDF = AddDF                
            if (OptionDict['DecompositionType']=='Tucker'):
                 TmCsDF= (( TmCsDF -  TmCsDF.mean()) /  TmCsDF.std(ddof=0))
                 if i== 1:                  
                     TmCsDFNmlzed = TmCsDF.fillna(0).copy()

            else:
                 TmCsDF= (( TmCsDF -  TmCsDF.min()) / ( TmCsDF.max() -  TmCsDF.min()))

                 if i== 1:                  
                     TmCsDFNmlzed = TmCsDF.fillna(0).copy()
            result_arr.append(TmCsDF.fillna(0))
            TmCsDFNmlzed=pd.concat([TmCsDFNmlzed,TmCsDF.fillna(0)],axis=0) 

            
        DFStack= np.stack(result_arr, axis=0)  
        return(DFStack,TmCshDF,TmCsDFNmlzed)       



    def mkBar(self,List1, Title, xlabel, ylabel, Color, xticks, size, xsize,Titlesize):  
        x = np.linspace( 1, len(List1), len(List1) )
        
        fig = plt.figure(figsize=(5,3))
        ax1 = fig.add_axes((0, 1, 1, 1))
    
        ax1.bar( [0+0.1*i for i in range(len(xticks))], List1,width=0.05, color=Color,tick_label=xticks,linewidth=0.5,ec=Color)
        
        xmin, xmax, ymin, ymax = ax1.axis() 
        
        p = ax1.hlines([0], xmin, xmax, "black", linestyles='solid',linewidth=1)     # hlines
        
        axis=['top','bottom','left','right']
        line_width=[1,1,1,1]
        
        for a,w in zip(axis, line_width):  # change axis width
            ax1.spines[a].set_linewidth(w)
    
        ax1.set_xlim(xmin, xmax)
        ax1.set_title(Title,fontsize=Titlesize)
        ax1.set_xlabel(xlabel,fontsize=xsize)
        ax1.set_ylabel(ylabel,fontsize=size)
        ax1.set_xticklabels(labels=xticks,rotation=270,fontsize=xsize)  


    def mkScatterWHist(self,list1,list2,ColorList,Optiondict):
        xs, ys = (6,6)
        fig = plt.figure(figsize=(xs,ys))
        
        try:
            if Optiondict['calcR']=='pearson':
                r, p = sts.pearsonr(list1,list2)
            elif Optiondict['calcR']=='spearman':
                r, p = sts.spearmanr(list1,list2)
        except:
                Optiondict['calcR']='pearson';r, p = sts.pearsonr(list1,list2)
            
        RankTimeVarDF = pd.DataFrame(data=None,columns=[Optiondict['xlabel'],Optiondict['ylabel']])
        RankTimeVarDF[Optiondict['xlabel']] = list1; RankTimeVarDF[Optiondict['ylabel']]= list2
        if (xs==12) and (ys==6):
            ax1 = fig.add_axes((0, 0.25, 0.75, 0.75))
            ax2 = fig.add_axes((0, 1, 0.75, 0.25), sharex=ax1)
            ax3 = fig.add_axes((0.75, 0.25, 0.17, 0.75), sharey=ax1)
        else:
            ax1 = fig.add_axes((0, 0.25, 0.75, 0.75))
            ax2 = fig.add_axes((0, 1, 0.75, 0.25), sharex=ax1)
            ax3 = fig.add_axes((0.75, 0.25, 0.25, 0.75), sharey=ax1)   

        try:
            markersize= Optiondict['markersize'] 
        except:
            markersize= 100 
            
        ax1.tick_params(labelsize=20,direction='out')
        ax3.tick_params(labelleft=False,labelsize=20,axis='y',color='white',left=False,direction='out')#;ax3.tick_params()
        ax3.tick_params(labelleft=False,labelsize=20,axis='x')#;ax3.tick_params()
        try:
            if Optiondict['errorbar']==1:
                x_errlist=Optiondict['x_err']
                y_errlist=Optiondict['y_err']
                for x, y, x_err,y_err,c in zip(list1,list2, x_errlist, y_errlist,ColorList):
                    ax1.errorbar(x, y, xerr = x_err, yerr = y_err,  fmt=c, elinewidth=0.5,capsize=1, ecolor='black')
                ax1.scatter(list1,list2,c=ColorList,s=markersize, facecolor='white')# = sns.jointplot(x=DFCol[i]+ str(num[jj]) , y='CV of log10(Param'+DF2Col[ii]+')',color=ColorList,space=0, data=SNSDF)
    
        except:
            ax1.scatter(list1,list2,c=ColorList,s=markersize,edgecolors='black',linewidths=0.1)# = sns.jointplot(x=DFCol[i]+ str(num[jj]) , y='CV of log10(Param'+DF2Col[ii]+')',color=ColorList,space=0, data=SNSDF) , alpha=0.4)
            #ax1.scatter(list1,list2,edgecolors=ColorList,s=markersize,facecolor='None',linewidth=0.2)# = sns.jointplot(x=DFCol[i]+ str(num[jj]) , y='CV of log10(Param'+DF2Col[ii]+')',color=ColorList,space=0, data=SNSDF)
    
        ax2.hist(list1, bins=16,ec='black')
        ax3.hist(list2, bins=16,orientation="horizontal",ec='black')
        ax2.tick_params(labelleft="true",labelbottom=False,direction='out',labelsize=5,axis='x',color='white');
        ax2.tick_params(labelleft="true",labelsize=20,axis='y');
        plot_axis = plt.axis()
        #ax1.set_xlim([ 0.0005,0.0025])
   
        axis=['top','bottom','left','right']
        line_width=[1,1,1,1]
        
        for a,w in zip(axis, line_width):  # change axis width
            ax1.spines[a].set_linewidth(w)
            ax2.spines[a].set_linewidth(w)        
            ax3.spines[a].set_linewidth(w)
            
        ax1.set_xlabel(Optiondict['xlabel'],fontsize=20)
        ax1.set_ylabel(Optiondict['ylabel'],fontsize=20)  
        ax1.set_title(Optiondict['title'])
        xmin, xmax, ymin, ymax = ax1.axis() 
        x = np.arange(xmin, 1, 0.1)
        ymin=min(list2); ymax=max(list2)
        xmin=min(list1); xmax=max(list1)
        try:
            if Optiondict['calcR'] in ['pearson','spearman']:
                #ax1.set_ylim([-0.1, 1.1])
                Roundp = round(p,100)
                Roundr = round(r,3)
                b = '%.2e'%Roundp
                TenRoundp = b[0:b.find('e')] + '$\it{×10^{-'+str(b[len(b)-1])+'}}$'
                p='{:e}'.format(p)
                r='{:e}'.format(r)
                TenRoundp = str(p)[0:4] + '$\it{×10^{-'+str(p)[str(p).find('e')+2:]+'}}$'
                TenRoundr = str(r)[0:4] + '$\it{×10^{-'+str(r)[str(r).find('e')+2:]+'}}$'

                ax2.set_title('$\it{R}$ = ' + TenRoundr + ', $\it{p}$ = ' + TenRoundp,fontsize=20)#, xy=(xmin+(xmin/100)+80,ymax-(ymax/100)+0.02))    
                
            else:
                r=0;p=0   
    
        except:
            print('44')
            r=0;p=0  
           
        try:
            if Optiondict['mkScatterWHist_drawEllipse']==1:
                NewDF = Optiondict['NewDF'];groups = Optiondict['groups']; MolLabel = Optiondict['MolLabel']
                drawEllipse(ax1,NewDF,groups, MolLabel,'TimeVar','AveCorr')
        except:
            pass
        if Optiondict['Annotate'] == 1:
            AnDict=dict({'Glucose':'Glc','Insulin':'Ins','C-peptide':'CRP','GIP(Active)':'GIP','Pyruvate':'Pyr','Total bile acid':'TBA',
                       'Citrate':'Cit','Cortisol':'Cor','Free fatty acid':'FFA','Total ketone body':'Ketone','Glutamic acid':'Glu',
                       'Citrulline':'Citr','Methionine':'Met','Isoleucine':'Ile','Leucine':'Leu','Tyrosine':'Tyr','4-Methyl-2-oxopentanoate':'4M2O','Glu + threo-beta-methylaspartate':'Glu+TBM','Growth hormone':'GH'})
            for i in range(len(list1)):
                if Optiondict['Label'][i] in ['Glucose','Insulin','C-peptide','GIP(Active)','Pyruvate','Total bile acid',
                       'Citrate','Cortisol','Free fatty acid','Total ketone body','Glutamic acid',
                       'Citrulline','Methionine','Isoleucine','Leucine','Tyrosine','4-Methyl-2-oxopentanoate','Glu + threo-beta-methylaspartate','Growth hormone']:
                    ax1.annotate(AnDict[Optiondict['Label'][i]],fontsize=5, xy=(list1[i],list2[i]))###0.1
    
                else:
                    try:
                        ax1.annotate(Optiondict['Label'][i],fontsize=Optiondict['LabelSize'], xy=(list1[i],list2[i]))
                    except:
                        ax1.annotate(Optiondict['Label'][i],fontsize=5, xy=(list1[i],list2[i]))
            RankTimeVarDF.index=Optiondict['Label']
        try:        
            if Optiondict['y=x'] == 1:
                    x = np.linspace(xmin,xmax)  
                    y = x
                    ax1.plot(x,y,"r-",linewidth=1)#,label='y=x') 
        except:
            pass

    
        return(r,p)          
 