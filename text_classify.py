# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:31:23 2021

@author: tt
"""


import sys
import os
import jieba
import jieba.posseg as  pseg
import importlib
import imp
import pickle
from sklearn.datasets.base import Bunch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer#TF-IDF向量生成类
from sklearn.svm import SVC
from sklearn.pipeline import  Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import  metrics
from sklearn.metrics import classification_report
#配置utf-8输出环境

imp.reload(sys)

def savefile(savepath,content): #保存至文件
	fp=open(savepath,"w",encoding="gb2312",errors='ignore')
	fp.write(content)
	fp.close()

def readfile(path): #读取文件
	fp=open(path,"r",encoding="gb2312",errors='ignore')
	content=fp.read()
	fp.close()
	return content

def corpus_segment(corpus_path, seg_path):
	catelist=os.listdir(corpus_path)#获取corpus_path下的所有子目录
	for mydir in catelist:
		class_path=corpus_path+mydir+"/" #拼出子目录的路径
		seg_dir=seg_path+mydir+"/" #拼出分词后语料分类目录
		if not os.path.exists(seg_dir): #是否存在目录
			os.makedirs(seg_dir) #没有，则创建
		file_list=os.listdir(class_path) #获取目录下的所有文件
		for file_path in file_list: #遍历目录下的文件
			fullname=class_path+file_path #拼出文件名全路径
			content=readfile(fullname).strip() #读取文件的内容
			#删除换行和多余的空格
			content=content.replace("\r\n","").strip()
# 			content_seg=jieba.cut(content)#为文件内容分词
			content_seg = pseg.cut(content)
			noun = ['n', 'ns', 'nt', 'nz', 'nx']  
			result_seg=""
			for word, flag in content_seg:
				if flag in noun:
					result_seg=result_seg+word+' '

			#将处理好的文件保存到分词后语料目录
			savefile(seg_dir+file_path,result_seg)

def split_word():
	#整个语料库的分词主程序
	corpus_path="C:/Users/tt/Downloads/corpus_fudan/train/"#未分次训练语料库路径
	seg_path="C:/Users/tt/Downloads/corpus_fudan/train_seg/" #分词后训练语料库的路径
	corpus_segment(corpus_path,seg_path)
	corpus_path="C:/Users/tt/Downloads/corpus_fudan/test/"#未分次测试语料库路径
	seg_path="C:/Users/tt/Downloads/corpus_fudan/test_seg/" #分词后测试语料库的路径
	corpus_segment(corpus_path,seg_path)

def create_bunch():
	bunch=Bunch(target_name=[],label=[],filenames=[],contents=[])
	wordbag_path="D:/downloads/corpus_fudan/train_test.dat"
	seg_path="D:/downloads/corpus_fudan/train_seg/"
	catelist=os.listdir(seg_path)
	bunch.target_name.extend(catelist)#将类别信息保存到Bunch对象
	for mydir in catelist:
	
	    class_path=seg_path+mydir+"/"	
	    file_list=os.listdir(class_path)	
	    for file_path in file_list:
	
	        fullname=class_path+file_path	
	        bunch.label.append(mydir)#保存当前文件的分类标签	
	        bunch.filenames.append(fullname)#保存当前文件的文件路径	
	        bunch.contents.append(readfile(fullname).strip())#保存文件词向量
	#Bunch对象持久化
	file_obj=open(wordbag_path,"wb")
	pickle.dump(bunch,file_obj)
	file_obj.close()

def readbunchobj(path):

    file_obj=open(path,"rb")

    bunch=pickle.load(file_obj)

    file_obj.close()

    return bunch
def writebunchobj(path,bunchobj):

    file_obj=open(path,"wb")

    pickle.dump(bunchobj,file_obj)

    file_obj.close()

def create_dict_and_train_svm():
	path="D:/downloads/corpus_fudan/train.dat"
	test_path="D:/downloads/corpus_fudan/test.dat"
	train_bunch=readbunchobj(path)
	test_bunch=readbunchobj(test_path)
	stopword_path="D:/downloads/corpus_fudan/stopword.txt"
	fp=open(stopword_path,"r",encoding="utf-8-sig",errors='ignore')
	content=fp.read()
	fp.close()
	stpwrdlst=content.splitlines()
	
	#构建TF-IDF词向量空间对象
	tfidfspace=Bunch(target_name=train_bunch.target_name,label=train_bunch.label,filenames=train_bunch.filenames,tdm=[],vocabulary={})
	#使用TfidVectorizer初始化向量空间模型
	vectorizer=TfidfVectorizer(max_df=0.6,max_features=10000,token_pattern=r"(?u)\b\w+\b",stop_words=stpwrdlst,sublinear_tf=True)
	#文本转为词频矩阵，单独保存字典文件
	tfidfspace.tdm=vectorizer.fit_transform(train_bunch.contents)
	tfidfspace.vocabulary=vectorizer.vocabulary_
	test_x=[]
	test_x=vectorizer.transform(test_bunch.contents)
	#创建词袋的持久化
# 	space_path="C:/Users/tt/Downloads/corpus_fudan/tfidfspace.dat"
# 	writebunchobj(space_path,tfidfspace)
	# 新建模型，传入chi2表示使用卡方分布选取特征
	model = SelectKBest(chi2)#选择k个最佳特征
	# fit和transform一步到位，注意这里需要将词频统计得到的稀疏矩阵转为完整矩阵表示
	# select_feature为经选取过滤后的词频矩阵，可用于文本分类
# 	select_feature = model.fit_transform(tfidfspace.tdm.toarray(), tfidfspace.label)
# 	feature_names=vectorizer.get_feature_names()
# 	result_temp = []
# 	# 获取所选特征，model.get_support(True)返回原始输入矩阵（方阵）中，选取的下标，
# 	# result_temp为经过选取后的分词特征
# 	for index in model.get_support(True):
# 		result_temp.append(feature_names[index])
# 	print(result_temp)
	svc=SVC()
	# pipe=make_pipeline(vect,SVC)
	pipe_svc=Pipeline([('selectkbest',model),("clf",svc)])
	k_range=[2000]
	C_range=[0.1,1.0,10]
	g_range=[1.0]#[0.01,0.1,1.0,10,100]
	para_grid=[
# 		{'clf__C':C_range,
#  	    'clf__kernel':['linear'],
# 		'selectkbest__k':k_range
# 		}
 	    {'clf__gamma':g_range,
 	    'clf__kernel':['rbf'],
		'selectkbest__k':k_range
		}
	]
	gs=GridSearchCV(estimator=pipe_svc,param_grid=para_grid,cv=10,n_jobs=-1)
	gs.fit(tfidfspace.tdm.toarray(),tfidfspace.label)
	gs.best_estimator_.fit(tfidfspace.tdm.toarray(),tfidfspace.label)
	y_pred = gs.best_estimator_.predict(test_x.toarray())
	print(gs.best_params_)
	print(metrics.accuracy_score(test_bunch.label,y_pred))# 打印该模型的性能报告
	# 直接使用sklearn打印精度，召回率和F1值
	target_names = tfidfspace.target_name
	print(classification_report(test_bunch.label, y_pred, target_names=target_names))
# split_word()
create_bunch()
create_dict_and_train_svm()
