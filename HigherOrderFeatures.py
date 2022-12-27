#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Md Azimul Haque
"""

import category_encoders as ce

import pandas as pd
import numpy as np
np.seterr(divide = 'ignore') 

from collections import Counter

from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer,LabelEncoder

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'


from ImportData_EDAFeatures import CreateDF_UntilEDA

def split_data(dataset,dependent_variable,problemtype='regression'):

    if problemtype=='regression':
        #if regression problem, divide feature into quantiles and then do stratified k fold
        dataset[dependent_variable[0]+'Quartile'] = pd.qcut(dataset[dependent_variable[0]], q=10,labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
    
        y = dataset[dependent_variable[0]+'Quartile']
    else:
        y = dataset[dependent_variable[0]]


    #train, and external test
    data_intermediate,data_external_test,y_intermediate,y_external_test = train_test_split(dataset,y,stratify=y,test_size=0.2)
    
    data_intermediate.reset_index(inplace=True,drop=True)
    data_external_test.reset_index(inplace=True,drop=True)
    y_intermediate.reset_index(inplace=True,drop=True)
    y_external_test.reset_index(inplace=True,drop=True)
    
    #train and validation data
    data_train,data_validation_test,y_train,y_validation_test = train_test_split(data_intermediate,y_intermediate,stratify=y_intermediate,test_size=0.2)
    data_train.reset_index(inplace=True,drop=True)
    data_validation_test.reset_index(inplace=True,drop=True)
    y_train.reset_index(inplace=True,drop=True)
    y_validation_test.reset_index(inplace=True,drop=True)
    
    
    #save results as dictionary    
    data_dict = {'data_external_test':data_external_test,'data_validation_test':data_validation_test}
    
    #stratified k fold cross validation    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    
    fold_dict = {}
    index = 0
    for train_index, test_index in skf.split(data_train, y_train):

        fold_training = data_train[data_train.index.isin(train_index)]
        fold_test = data_train[data_train.index.isin(test_index)]
        fold_dict[index] = {'fold_training':fold_training,'fold_test':fold_test}
        index += 1
    
    data_dict['fold_dict'] = fold_dict
    
    return data_dict

def check_corr(corr_df,feature,dependent_variable):
    
    #original_corr = round(np.corrcoef(corr_df[dependent_variable[0]].values,corr_df[feature].values)[0][1],3)

    best_column = ''
    best_corr = round(np.corrcoef(corr_df[dependent_variable[0]].values,corr_df[feature].values)[0][1],3)    
    
    for column in corr_df.columns:
        if column not in [dependent_variable[0],feature]:

            ## filter and remove inf rows for log transformed column
#            if '_Log' in column:
#                log_feature = corr_df[column][corr_df[column].values != -np.inf]
#                filtered_dependent = corr_df[dependent_variable[0]][corr_df[column].values != -np.inf]
#                new_corr = round(np.corrcoef(log_feature,filtered_dependent)[0][1],3)
#                
#            else:
            new_corr = round(np.corrcoef(corr_df[dependent_variable[0]].values,corr_df[column].values)[0][1],3)

            #find best correlation
            if abs(new_corr) > abs(best_corr):
                best_corr = new_corr
                best_column = column
    
    if best_column == '':
        #Single plot
        return feature
    else:
        #process plot
        return best_column.split('_')[-1]


def find_numerical_transformation_regression(dataset,feature,dependent_variable):
    standscal = StandardScaler()
    minmax = MinMaxScaler()
    boxcox = PowerTransformer(method='box-cox')
    yeojohnson = PowerTransformer(method='yeo-johnson')
    
    corr_df = pd.DataFrame({feature:dataset[feature].values,dependent_variable[0]:dataset[dependent_variable[0]].values})
    corr_df.reset_index(inplace=True,drop=True)

    if dependent_variable[0] == 'selling_price':
        corr_df[feature]=corr_df[feature].astype('int64')
     

    #square
    corr_df[feature+'_Square'] = np.power(corr_df[feature], 2)

    #cube
    corr_df[feature+'_Cube'] = np.power(corr_df[feature], 3)
    #sqrt
    corr_df[feature+'_Sqrt'] = np.sqrt(corr_df[feature])
    #cbrt
    corr_df[feature+'_Cbrt'] = np.cbrt(corr_df[feature])
    #log
    #corr_df[feature+'_Log'] = np.log(corr_df[feature])
    #standard scaler
    corr_df[feature+'_Stand'] = standscal.fit_transform(corr_df[[feature]])
    #minmax
    corr_df[feature+'_MinMax'] = minmax.fit_transform(corr_df[[feature]])
    #boxcox
    if 0 not in corr_df[feature].values and feature!='year':
        corr_df[feature+'_BoxCox'] = boxcox.fit_transform(corr_df[[feature]])
    #yeo-johnson
    # print(feature,corr_df[feature].max(),corr_df[feature].min())
    # print(feature,corr_df[feature].dtype)
    # print(feature,corr_df[feature].dtype)
    # print(feature,corr_df[feature].unique())
    # print(corr_df[feature].value_counts())
    if feature!='year':
        corr_df[feature+'_Yeo'] = yeojohnson.fit_transform(corr_df[[feature]])
    
    ##check correlation
    feature_transformed = check_corr(corr_df,feature,dependent_variable)
    
    if feature_transformed == feature:
        return 'None'
    else:
        return feature_transformed

def do_transformation(dataset,feature,transformation_name,new_numerical_features):
    
    #square
    if transformation_name == 'Square':
        dataset[feature+'_Square'] = np.power(dataset[feature], 2)
        new_numerical_features.append(feature+'_Square')
    #cube
    elif transformation_name == 'Cube':
        dataset[feature+'_Cube'] = np.power(dataset[feature], 3)
        new_numerical_features.append(feature+'_Cube')
    #sqrt
    elif transformation_name == 'Sqrt':
        dataset[feature+'_Sqrt'] = np.sqrt(dataset[feature])
        new_numerical_features.append(feature+'_Sqrt')
    #cbrt
    elif transformation_name == 'Cbrt':
        dataset[feature+'_Cbrt'] = np.cbrt(dataset[feature])
        new_numerical_features.append(feature+'_Cbrt')
#    #log
#    elif transformation_name == 'Log':
#        dataset[feature+'_Log'] = np.log(dataset[feature])
#        new_numerical_features.append(feature+'_Log')
    
    return dataset,new_numerical_features

def numericalHOFE_Identification(dataset,dependent_variable,numerical_features):
    decide_type = {}
    new_numerical_features = []
    for feature in numerical_features:
        # dataset[feature]=dataset[feature].astype('int64')
        # print(feature,dataset[feature].dtype)
        
        transformation_name = find_numerical_transformation_regression(dataset,feature,dependent_variable)
    
        if transformation_name in ['Square','Cube','Sqrt','Cbrt','Log']:
            dataset,new_numerical_features = do_transformation(dataset,feature,transformation_name,new_numerical_features)
        else:
            decide_type[feature] = transformation_name
    
    numerical_features += new_numerical_features
    
    return decide_type,dataset,numerical_features

def categoricalHOFE_All(dataset,dependent_variable,categorical_features,numerical_features):
    label_encoder = LabelEncoder()
    
    categorical_linear = []
    categorical_tree = []
    
    for feature in numerical_features:
        ##binning
        try:
            dataset[feature+'_Quartile'] = pd.qcut(dataset[feature], q=10,labels=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
            categorical_features.append(feature+'_Quartile')
        except:
            pass
    
    #for each feature
    for feature in categorical_features:
        
        if feature not in ['torque']:
            if feature in ['haschildren','toCouponGEQ15min','toCouponGEQ25min','directionsame','directionopp','acceptedcoupon','rejectedcoupon','acceptedcoupon','rejectedcoupon']:
                categorical_linear.append(feature)
            else:
                #dummy
                dataset2 = pd.get_dummies(dataset[feature],drop_first=True)
                for column in dataset2.columns:
                    dataset[str(feature)+'_'+str(column)] = dataset2[column]
                    categorical_linear.append(str(feature)+'_'+str(column))

        if feature in ['haschildren','toCouponGEQ15min','toCouponGEQ25min','directionsame','directionopp','acceptedcoupon','rejectedcoupon','acceptedcoupon','rejectedcoupon']:
            categorical_tree.append(feature)
        else:
            #label encoding
            dataset[feature+'_Encoded'] = label_encoder.fit_transform(dataset[feature])
            categorical_tree.append(feature+'_Encoded')
        
    
    return dataset,categorical_features,categorical_linear,categorical_tree


def OrdinalHOFE_All(dataset,ordinal_features):
    
    ordinal_features_engineered = []
    
    for feature in ordinal_features:
        #rank
        if feature == 'income':
            income_ranking_dictionary = {'Less than $12500':1,
                                        '$12500 - $24999':2,
                                        '$25000 - $37499':3,
                                        '$37500 - $49999':4,
                                        '$50000 - $62499':5,
                                        '$62500 - $74999':6,
                                        '$75000 - $87499':7,
                                        '$87500 - $99999':7, 
                                        '$100000 or More':8}
            dataset['income_Ranking'] = dataset['income'].replace(income_ranking_dictionary)
            ordinal_features_engineered.append('income_Ranking')
        elif feature == 'education':
            education_ranking_dictionary={'Some High School':1,
                                          'High School Graduate':2,
                                          'Associates degree':3,
                                          'Some college - no degree':4, 
                                          'Bachelors degree':5,
                                          'Graduate degree (Masters or Doctorate)':6}
            dataset['education_Ranking'] = dataset['education'].replace(education_ranking_dictionary)
            ordinal_features_engineered.append('education_Ranking')            
        
        ### polynomial
        temp_columns = list(dataset.columns)
        #save index for joining
        dataset['Saveindex'] = dataset.index
        
        encoder = ce.PolynomialEncoder(cols=[feature])
        data2 = encoder.fit_transform(dataset, verbose=1)
        
        new_columns = list(set(data2.columns).difference(set(temp_columns)))
        if 'Saveindex' in new_columns:
            new_columns.remove('Saveindex')
        
        if 'intercept' in new_columns:
            new_columns.remove('intercept')
        
        #print('new_columns:',new_columns)
        name_dict = {}
        new_names = []
        for name in new_columns:
            if feature in name:
                name_dict[name] = feature+'_'+'Polynomial'+''.join(name.split('_')[1])
                new_names.append(feature+'_'+'Polynomial'+''.join(name.split('_')[1]))
        #print(name_dict)
        data2.rename(columns=name_dict,inplace=True)
        ordinal_features_engineered += new_names
        data2 = data2[new_names+['Saveindex']]
        
        dataset=dataset.merge(data2)
        
        
        ### backward differencing
        temp_columns = list(dataset.columns)
        #save index for joining
        dataset['Saveindex'] = dataset.index
        
        encoder = ce.BackwardDifferenceEncoder(cols=[feature])
        data2 = encoder.fit_transform(dataset, verbose=1)
        
        new_columns = list(set(data2.columns).difference(set(temp_columns)))
        if 'Saveindex' in new_columns:
            new_columns.remove('Saveindex')
        
        if 'intercept' in new_columns:
            new_columns.remove('intercept')
        
        #print('new_columns:',new_columns)
        name_dict = {}
        new_names = []
        for name in new_columns:
            if feature in name:
                name_dict[name] = feature+'_'+'BackwardDifference'+''.join(name.split('_')[1])
                new_names.append(feature+'_'+'BackwardDifference'+''.join(name.split('_')[1]))
        #print(name_dict)
        data2.rename(columns=name_dict,inplace=True)
        ordinal_features_engineered += new_names
        
        data2 = data2[new_names+['Saveindex']]
        dataset=dataset.merge(data2)
        
    del dataset['Saveindex']
    return dataset,ordinal_features_engineered        
        #backward differencing
        

def get_encoding(calc_df,feature,dependent_variable):
    
    
    count_encoding_dictionary_missing = dict(calc_df[feature].value_counts())

    #percent encoding
    percent_encoding_dictionary_missing = {}
    for key in count_encoding_dictionary_missing:
        percent_encoding_dictionary_missing[key] = round((count_encoding_dictionary_missing[key]/calc_df.shape[0])*100,2)


    rank_list = sorted(count_encoding_dictionary_missing, key=count_encoding_dictionary_missing.get)

    #count rank encoding
    rank_count_encoding_dictionary_missing = {}
    counter = 1

    for category_secondary in rank_list:
        rank_count_encoding_dictionary_missing[category_secondary] = counter

        counter += 1

    #mean encoding
    mean_groupby = calc_df[[feature,dependent_variable[0]]].groupby(feature).mean()
    mean_encoding_missing = {}

    for index_value in mean_groupby.index:
        mean_encoding_missing[index_value] = mean_groupby[dependent_variable[0]][index_value]

    return count_encoding_dictionary_missing,percent_encoding_dictionary_missing,rank_count_encoding_dictionary_missing,mean_encoding_missing
    
def calc_missing_category(data_dict,feature,category,dependent_variable):

    count_encoding_dictionary_missing = {}
    percent_encoding_dictionary_missing = {}
    rank_count_encoding_dictionary_missing = {}
    mean_encoding_missing = {}
    
    #- loop and find another cross validation where it is present
    for fold_dict_index in data_dict['fold_dict'].keys():

        if category in data_dict['fold_dict'][fold_dict_index]['fold_training'][feature].unique():

            calc_df = data_dict['fold_dict'][fold_dict_index]['fold_training']

            #- calculate encoding
            count_encoding_dictionary_missing,percent_encoding_dictionary_missing,rank_count_encoding_dictionary_missing,mean_encoding_missing = get_encoding(calc_df,feature,dependent_variable)
            break
            #count encoding
        elif category in data_dict['fold_dict'][fold_dict_index]['fold_test'][feature].unique():

            calc_df = data_dict['fold_dict'][fold_dict_index]['fold_test']
            #- calculate encoding
            count_encoding_dictionary_missing,percent_encoding_dictionary_missing,rank_count_encoding_dictionary_missing,mean_encoding_missing = get_encoding(calc_df,feature,dependent_variable)
            break
        elif category in data_dict['data_validation_test'][feature].unique():

            calc_df = data_dict['data_validation_test']
            #- calculate encoding
            count_encoding_dictionary_missing,percent_encoding_dictionary_missing,rank_count_encoding_dictionary_missing,mean_encoding_missing = get_encoding(calc_df,feature,dependent_variable)
            break
        elif category in data_dict['data_external_test'][feature].unique():

            calc_df = data_dict['data_external_test']
            #- calculate encoding
            count_encoding_dictionary_missing,percent_encoding_dictionary_missing,rank_count_encoding_dictionary_missing,mean_encoding_missing = get_encoding(calc_df,feature,dependent_variable)
            break
        else:
            print('no logic')
    
    

    return count_encoding_dictionary_missing,percent_encoding_dictionary_missing,rank_count_encoding_dictionary_missing,mean_encoding_missing

def process_all(dataset,dependent_variable,categorical_features,numerical_features,ordinal_features,problemtype='regression'):

    standscal = StandardScaler()
    minmax = MinMaxScaler()
    boxcox = PowerTransformer(method='box-cox')
    yeojohnson = PowerTransformer(method='yeo-johnson')
    numerical_features_linear = []
    numerical_features_tree = []
    ordinal_features_engineered = []
    
    #get dummy and label encoded features for entire dataset
    dataset,categorical_features,categorical_linear,categorical_tree = categoricalHOFE_All(dataset,dependent_variable,categorical_features,numerical_features)
    
    #identify type of transformation needed for regression problem and numerical features
    if problemtype=='regression':
    
        decide_type,dataset,numerical_features = numericalHOFE_Identification(dataset,dependent_variable,numerical_features)
        
    #ordinal features
    if ordinal_features:
        #process all
        dataset,ordinal_features_engineered = OrdinalHOFE_All(dataset,ordinal_features)

    ##create data dictionary    
    data_dict = split_data(dataset,dependent_variable,problemtype=problemtype)
    
    
    categorical_features_include_list = []
        
    #### for each categorical feature, do transformation for train-test-external test
    for feature in categorical_features:
        category_counter = 0
        for fold_dict_index in data_dict['fold_dict'].keys():
            calc_df = data_dict['fold_dict'][fold_dict_index]['fold_training']        
            #for each unique_category_in_feature

            for category in dataset[feature].unique():
                  #check if any category not present in training data
                if category not in calc_df[feature].unique():
                    category_counter += 1
        if category_counter == 0:        
            categorical_features_include_list.append(feature)

    ##perform all transformation to obtain higher order features
    for fold_dict_index in data_dict['fold_dict'].keys():
                
        #get training datafarme for ease of calculation
        calc_df = data_dict['fold_dict'][fold_dict_index]['fold_training']        
        
        #### for each categorical feature, do transformation for train-test-external test
        for feature in categorical_features_include_list:
            
            #count encoding
            count_encoding_dictionary = dict(Counter(calc_df[feature]))
            
            #percent encoding
            percent_encoding_dictionary = {}
            for key in count_encoding_dictionary:
                percent_encoding_dictionary[key] = round((count_encoding_dictionary[key]/calc_df.shape[0])*100,2)
    
            rank_list = sorted(count_encoding_dictionary, key=count_encoding_dictionary.get)
            
            #count rank encoding
            rank_count_encoding_dictionary = {}
            counter = 1
            for category in rank_list:
                rank_count_encoding_dictionary[category] = counter
                counter += 1
            
            #mean encoding
            mean_groupby = calc_df[[feature,dependent_variable[0]]].groupby(feature).mean()
            mean_encoding = {}
            
            for index_value in mean_groupby.index:
                if dependent_variable[0] == 'sellingprice':
                    mean_encoding[index_value] =mean_groupby[dependent_variable[0]][index_value]
                else:
                    mean_encoding[index_value] = mean_groupby[dependent_variable[0]][index_value]


            #for each unique_category_in_feature
            category_counter = 0
            for category in dataset[feature].unique():
                  #check if any category not present in training data
                if category not in calc_df[feature].unique():
                    category_counter += 1
                    
            
                    #do not process
                    #in another function, give input data_dict, column name, unique categories, dependent_variable
                    # count_encoding_dictionary_missing,percent_encoding_dictionary_missing,rank_count_encoding_dictionary_missing,mean_encoding_missing = calc_missing_category(data_dict,feature,category,dependent_variable)

                    # #fetch encoding for category
                    # count_encoding_dictionary[category] = count_encoding_dictionary_missing[category]
                    # percent_encoding_dictionary[category] = percent_encoding_dictionary_missing[category]
                    # rank_count_encoding_dictionary[category] = rank_count_encoding_dictionary_missing[category]
                    # mean_encoding[category] = mean_encoding_missing[category]

            if category_counter == 0:
                #replace for train
                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_countEncoded']=calc_df[feature].replace(count_encoding_dictionary)
                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_percentEncoded']=calc_df[feature].replace(percent_encoding_dictionary)
                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_countRankEncoded']=calc_df[feature].replace(rank_count_encoding_dictionary)
                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_MeanEncoded']=calc_df[feature].replace(mean_encoding)
                
                #replace for fold test
                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_countEncoded']=data_dict['fold_dict'][fold_dict_index]['fold_test'][feature].replace(count_encoding_dictionary)
                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_percentEncoded']=data_dict['fold_dict'][fold_dict_index]['fold_test'][feature].replace(percent_encoding_dictionary)
                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_countRankEncoded']=data_dict['fold_dict'][fold_dict_index]['fold_test'][feature].replace(rank_count_encoding_dictionary)
                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_MeanEncoded']=data_dict['fold_dict'][fold_dict_index]['fold_test'][feature].replace(mean_encoding)
                
                #for data_external_test and data_validation_test, only do perprocess for once. 
                #it can be done randomly at any split, so we chose it at first index.
                if fold_dict_index == 0:
                    
                    #replace for data_external_test
                    data_dict['data_external_test'][feature+'_countEncoded']=data_dict['data_external_test'][feature].replace(count_encoding_dictionary)
                    data_dict['data_external_test'][feature+'_percentEncoded']=data_dict['data_external_test'][feature].replace(percent_encoding_dictionary)
                    data_dict['data_external_test'][feature+'_countRankEncoded']=data_dict['data_external_test'][feature].replace(rank_count_encoding_dictionary)
                    data_dict['data_external_test'][feature+'_MeanEncoded']=data_dict['data_external_test'][feature].replace(mean_encoding)
    
                    #replace for data_validation_test
                    data_dict['data_validation_test'][feature+'_countEncoded']=data_dict['data_external_test'][feature].replace(count_encoding_dictionary)
                    data_dict['data_validation_test'][feature+'_percentEncoded']=data_dict['data_external_test'][feature].replace(percent_encoding_dictionary)
                    data_dict['data_validation_test'][feature+'_countRankEncoded']=data_dict['data_external_test'][feature].replace(rank_count_encoding_dictionary)
                    data_dict['data_validation_test'][feature+'_MeanEncoded']=data_dict['data_external_test'][feature].replace(mean_encoding)
                    
                    #add new feature names
                    categorical_tree += [feature+'_countEncoded',feature+'_percentEncoded',feature+'_countRankEncoded',feature+'_MeanEncoded']
                    categorical_linear += [feature+'_countEncoded',feature+'_percentEncoded',feature+'_countRankEncoded',feature+'_MeanEncoded']
            
            
        #for each numerical feature, do transformation for train-test-external test
        for feature in numerical_features:

            if problemtype=='regression' and feature in decide_type.keys():

                #standard scaler
                if decide_type[feature] == 'Stand':
                    standscal.fit(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    
                    data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_Stand']=standscal.transform(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_Stand']=standscal.transform(data_dict['fold_dict'][fold_dict_index]['fold_test'][[feature]])
                    
                    if fold_dict_index == 0:
                        data_dict['data_external_test'][feature+'_Stand']=standscal.transform(data_dict['data_external_test'][[feature]])
                        data_dict['data_validation_test'][feature+'_Stand']=standscal.transform(data_dict['data_validation_test'][[feature]])
                        numerical_features_linear.append(feature+'_Stand')
                #minmax
                elif decide_type[feature] == 'MinMax':
                    minmax.fit(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    
                    data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_MinMax']=minmax.transform(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_MinMax']=minmax.transform(data_dict['fold_dict'][fold_dict_index]['fold_test'][[feature]])
                    
                    if fold_dict_index == 0:
                        data_dict['data_external_test'][feature+'_MinMax']=minmax.transform(data_dict['data_external_test'][[feature]])
                        data_dict['data_validation_test'][feature+'_MinMax']=minmax.transform(data_dict['data_validation_test'][[feature]])
                        numerical_features_linear.append(feature+'_MinMax')
                    
                #boxcox
                elif decide_type[feature] == 'BoxCox':

                    boxcox.fit(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    
                    data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_BoxCox']=boxcox.transform(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_BoxCox']=boxcox.transform(data_dict['fold_dict'][fold_dict_index]['fold_test'][[feature]])
                    
                    if fold_dict_index == 0:
                        data_dict['data_external_test'][feature+'_BoxCox']=boxcox.transform(data_dict['data_external_test'][[feature]])
                        data_dict['data_validation_test'][feature+'_BoxCox']=boxcox.transform(data_dict['data_validation_test'][[feature]])
                        numerical_features_linear.append(feature+'_BoxCox')
                #yeo-johnson
                elif decide_type[feature] == 'Yeo':

                    yeojohnson.fit(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    
                    data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_Yeo']=yeojohnson.transform(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_Yeo']=yeojohnson.transform(data_dict['fold_dict'][fold_dict_index]['fold_test'][[feature]])
                    
                    if fold_dict_index == 0:
                        data_dict['data_external_test'][feature+'_Yeo']=yeojohnson.transform(data_dict['data_external_test'][[feature]])
                        data_dict['data_validation_test'][feature+'_Yeo']=yeojohnson.transform(data_dict['data_validation_test'][[feature]])
                        numerical_features_linear.append(feature+'_Yeo')
                    
            
            #try all types of transformation
            elif problemtype!='regression':

                #standard scaler
                standscal.fit(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                
                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_Stand']=standscal.transform(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_Stand']=standscal.transform(data_dict['fold_dict'][fold_dict_index]['fold_test'][[feature]])
                
                if fold_dict_index == 0:
                    data_dict['data_external_test'][feature+'_Stand']=standscal.transform(data_dict['data_external_test'][[feature]])
                    data_dict['data_validation_test'][feature+'_Stand']=standscal.transform(data_dict['data_validation_test'][[feature]])
                    numerical_features_linear.append(feature+'_Stand')
                
                #minmax
                minmax.fit(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                
                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_MinMax']=minmax.transform(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_MinMax']=minmax.transform(data_dict['fold_dict'][fold_dict_index]['fold_test'][[feature]])
                
                if fold_dict_index == 0:
                    data_dict['data_external_test'][feature+'_MinMax']=minmax.transform(data_dict['data_external_test'][[feature]])
                    data_dict['data_validation_test'][feature+'_MinMax']=minmax.transform(data_dict['data_validation_test'][[feature]])
                    numerical_features_linear.append(feature+'_MinMax')
                    
                #boxcox
                
                if 0 not in data_dict['fold_dict'][fold_dict_index]['fold_training'][feature].values:
                
                    boxcox.fit(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    
                    data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_BoxCox']=boxcox.transform(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                    data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_BoxCox']=boxcox.transform(data_dict['fold_dict'][fold_dict_index]['fold_test'][[feature]])
                    
                    if fold_dict_index == 0:
                        data_dict['data_external_test'][feature+'_BoxCox']=boxcox.transform(data_dict['data_external_test'][[feature]])
                        data_dict['data_validation_test'][feature+'_BoxCox']=boxcox.transform(data_dict['data_validation_test'][[feature]])
                        numerical_features_linear.append(feature+'_BoxCox')
                
                #yeo-johnson
                yeojohnson.fit(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                
                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_Yeo']=yeojohnson.transform(data_dict['fold_dict'][fold_dict_index]['fold_training'][[feature]])
                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_Yeo']=yeojohnson.transform(data_dict['fold_dict'][fold_dict_index]['fold_test'][[feature]])
                
                if fold_dict_index == 0:
                    data_dict['data_external_test'][feature+'_Yeo']=yeojohnson.transform(data_dict['data_external_test'][[feature]])
                    data_dict['data_validation_test'][feature+'_Yeo']=yeojohnson.transform(data_dict['data_validation_test'][[feature]])
                    numerical_features_linear.append(feature+'_Yeo')
                    
                #sqrt
                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_Sqrt']=np.sqrt(data_dict['fold_dict'][fold_dict_index]['fold_training'][feature])
                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_Sqrt']=np.sqrt(data_dict['fold_dict'][fold_dict_index]['fold_test'][feature])
                
                if fold_dict_index == 0:
                    data_dict['data_external_test'][feature+'_Sqrt']=np.sqrt(data_dict['data_external_test'][feature])
                    data_dict['data_validation_test'][feature+'_Sqrt']=np.sqrt(data_dict['data_validation_test'][feature])
                    numerical_features_linear.append(feature+'_Sqrt')
                    numerical_features_tree.append(feature+'_Sqrt')
                    
#                #log
#                data_dict['fold_dict'][fold_dict_index]['fold_training'][feature+'_Log']=np.log(data_dict['fold_dict'][fold_dict_index]['fold_training'][feature])
#                data_dict['fold_dict'][fold_dict_index]['fold_test'][feature+'_Log']=np.log(data_dict['fold_dict'][fold_dict_index]['fold_test'][feature])
                
#                if fold_dict_index == 0:
#                    data_dict['data_external_test'][feature+'_Log']=np.log(data_dict['data_external_test'][feature])
#                    data_dict['data_validation_test'][feature+'_Log']=np.log(data_dict['data_validation_test'][feature])
#                    numerical_features_linear.append(feature+'_Log')


    return dataset,data_dict,categorical_linear,categorical_tree,categorical_features_include_list,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered



def createHotelCancellations(dataset,dependent_variable,numerical_features,categorical_features):
    
    dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = process_all(dataset,dependent_variable,categorical_features,numerical_features,problemtype='')
    
    return dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered

def createCouponRecommendation(dataset,dependent_variable,numerical_features,categorical_features):
    
    dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = process_all(dataset,dependent_variable,categorical_features,numerical_features,problemtype='')
    
    return dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered
    
def createPredictRoomBooking(dataset,dependent_variable,numerical_features,categorical_features):

    dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = process_all(dataset,dependent_variable,categorical_features,numerical_features,problemtype='regression')
    
    return dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered

def createCarSales(dataset,dependent_variable,numerical_features,categorical_features):
    
    dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = process_all(dataset,dependent_variable,categorical_features,numerical_features,problemtype='regression')
    
    return dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered

def CreateDF_UntilHOFE(dataset,dependent_variable,numerical_features,categorical_features,dataset_name=''):
    '''
    4 options. 1 for each dataset: 'HotelCancellations', 'CouponRecommendation', 'PredictRoomBooking','CarSales'
    '''
    
    ordinal_features = []
    
    if dataset_name == 'HotelCancellations':
        dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = process_all(dataset,dependent_variable,categorical_features,numerical_features,ordinal_features,problemtype='')
    elif dataset_name == 'CouponRecommendation':
        #delete income
        categorical_features.remove('income')
        categorical_features.remove('education')
        #add ordinal
        ordinal_features += ['income','education']
        dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = process_all(dataset,dependent_variable,categorical_features,numerical_features,ordinal_features,problemtype='')
    elif dataset_name == 'PredictRoomBooking':
        dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = process_all(dataset,dependent_variable,categorical_features,numerical_features,ordinal_features,problemtype='regression')
    elif dataset_name == 'CarSales':
        dataset,data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = process_all(dataset,dependent_variable,categorical_features,numerical_features,ordinal_features,problemtype='regression')
    
    return data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered

if __name__ == '__main__':
#    problem = 'CouponRecommendation'
#    dataset,dependent_variable,numerical_features,categorical_features = CreateDF_UntilEDA(problem)
#    data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = CreateDF_UntilHOFE(dataset,dependent_variable,numerical_features,categorical_features,dataset_name=problem)

    'HotelCancellations', 'CouponRecommendation', 'PredictRoomBooking','CarSales'
    problem = 'CarSales'
    dataset,dependent_variable,numerical_features,categorical_features = CreateDF_UntilEDA(problem)
    data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = CreateDF_UntilHOFE(dataset,dependent_variable,numerical_features,categorical_features,dataset_name=problem)
    