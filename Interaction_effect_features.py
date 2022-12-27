"""
@author: Md Azimul Haque
"""

import pandas as pd
import numpy as np

from ImportData_EDAFeatures import CreateDF_UntilEDA
from HigherOrderFeatures import CreateDF_UntilHOFE

np.seterr(divide = 'ignore')
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
# suppress warnings
warnings.filterwarnings('ignore')

def add_column(data_dict,extra_column,numerical_features_tree):
    
    extra_column2 = extra_column.replace(" ","").replace("_","")
    
    col1 = extra_column.split()[0]
    col2 = extra_column.split()[2]
    
    first = extra_column.split()[0]
    second = extra_column.split()[2]

    for data_fold_index in data_dict['fold_dict'].keys():
        
        # if 'transmissionandowner' not in data_dict['fold_dict'][data_fold_index]['fold_training'].columns:
        #     print('Not found')
        # if 'transmissionandowner' not in data_dict['fold_dict'][data_fold_index]['fold_test'].columns:
        #     print('Not found')
        # if 'transmissionandowner' not in data_dict['data_external_test'].columns:
        #     print('Not found')
        # if 'transmissionandowner' not in data_dict['data_validation_test'].columns:
        #     print('Not found')
        
        if data_dict['fold_dict'][data_fold_index]['fold_training'][first].dtype == 'category' or data_dict['fold_dict'][data_fold_index]['fold_test'][first].dtype == 'category' or data_dict['data_external_test'][first].dtype == 'category' or data_dict['data_validation_test'][first].dtype == 'category':
            data_dict['fold_dict'][data_fold_index]['fold_training'][first] = data_dict['fold_dict'][data_fold_index]['fold_training'][first].astype('int64')
            data_dict['fold_dict'][data_fold_index]['fold_test'][first] = data_dict['fold_dict'][data_fold_index]['fold_test'][first].astype('int64')

            data_dict['data_external_test'][first] = data_dict['data_external_test'][first].astype('int64')
            data_dict['data_validation_test'][first] = data_dict['data_validation_test'][first].astype('int64')
            
        elif data_dict['fold_dict'][data_fold_index]['fold_training'][second].dtype == 'category' or data_dict['fold_dict'][data_fold_index]['fold_test'][second].dtype == 'category' or data_dict['data_external_test'][second].dtype == 'category' or data_dict['data_validation_test'][second].dtype == 'category':
            
            data_dict['fold_dict'][data_fold_index]['fold_training'][second] = data_dict['fold_dict'][data_fold_index]['fold_training'][second].astype('int64')
            data_dict['fold_dict'][data_fold_index]['fold_test'][second] = data_dict['fold_dict'][data_fold_index]['fold_test'][second].astype('int64')

            data_dict['data_external_test'][second] = data_dict['data_external_test'][second].astype('int64')
            data_dict['data_validation_test'][second] = data_dict['data_validation_test'][second].astype('int64')
        
        if extra_column == 'brand_countRankEncoded and mileage_Quartile_MeanEncoded':
            data_dict['fold_dict'][data_fold_index]['fold_test'][col1] = data_dict['fold_dict'][data_fold_index]['fold_test'][col1].astype('int64')
        
        #print('Start the feature:',,data_dict['fold_dict'][0]['fold_training'][first].dtype,data_dict['fold_dict'][0]['fold_training'][second].dtype)
        
        data_dict['fold_dict'][data_fold_index]['fold_training'][extra_column2] = data_dict['fold_dict'][data_fold_index]['fold_training'][col1]*data_dict['fold_dict'][data_fold_index]['fold_training'][col2]
        data_dict['fold_dict'][data_fold_index]['fold_test'][extra_column2] = data_dict['fold_dict'][data_fold_index]['fold_test'][col1]*data_dict['fold_dict'][data_fold_index]['fold_test'][col2]
        data_dict['data_external_test'][extra_column2] = data_dict['data_external_test'][col1]*data_dict['data_external_test'][col2]
        data_dict['data_validation_test'][extra_column2] = data_dict['data_validation_test'][col1]*data_dict['data_validation_test'][col2]
        
        if data_fold_index == 0:
            numerical_features_tree.append(extra_column2)
            
    return data_dict,numerical_features_tree

    

#fetch data
def get_data(problem):
    
    dataset,dependent_variable,numerical_features,categorical_features = CreateDF_UntilEDA(problem)
    
    if problem == 'CarSales':
        
        #ANOVA interaction features
        dataset['fuelandowner'] = dataset['fuel'] + dataset['owner']
        dataset['fuelandtorque'] = dataset['fuel'] + dataset['torque']
        dataset['fuelandbrandAndModel'] = dataset['fuel'] + dataset['brandAndModel']
        dataset['sellertypeandtransmission'] = dataset['sellertype'] + dataset['transmission']
        dataset['sellertypeandowner'] = dataset['sellertype'] + dataset['owner']
        dataset['sellertypeandtorque'] = dataset['sellertype'] + dataset['torque']
        dataset['sellertypeandbrand'] = dataset['sellertype'] + dataset['brand']
        dataset['sellertypeandbrandAndModel'] = dataset['sellertype'] + dataset['brandAndModel']
        dataset['transmissionandowner'] = dataset['transmission'] + dataset['owner']
        dataset['transmissionandtorque'] = dataset['transmission'] + dataset['torque']
        dataset['transmissionandbrand'] = dataset['transmission'] + dataset['brand']
        dataset['transmissionandbrandAndModel'] = dataset['transmission'] + dataset['brandAndModel']
        dataset['ownerandtorque'] = dataset['owner'] + dataset['torque']
        dataset['ownerandbrand'] = dataset['owner'] + dataset['brand']
        dataset['ownerandbrandAndModel'] = dataset['owner'] + dataset['brandAndModel']
    
        categorical_features += ['fuelandowner','fuelandtorque','fuelandbrandAndModel','sellertypeandtransmission','sellertypeandowner','sellertypeandtorque','sellertypeandbrand','sellertypeandbrandAndModel','transmissionandowner','transmissionandtorque','transmissionandbrand','transmissionandbrandAndModel','ownerandtorque','ownerandbrand','ownerandbrandAndModel']
    
    
    data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = CreateDF_UntilHOFE(dataset,dependent_variable,numerical_features,categorical_features,dataset_name=problem)
    
    if problem == 'CarSales':
        #SHAP interaction features
        interactions = ['fuel_countEncoded and seats_countEncoded','brand_countRankEncoded and maxpower_Quartile_countEncoded','brand_countRankEncoded and mileage_Quartile_MeanEncoded','seats_countEncoded and kmdriven_Quartile_countRankEncoded','fuel_countEncoded and maxpower_Quartile_countRankEncoded','seats_countEncoded and brand_countEncoded','seats_countEncoded and maxpower_Quartile_MeanEncoded','torque_countRankEncoded and brand_countRankEncoded','seats_countEncoded and maxpower_Quartile_countRankEncoded','fuel_countEncoded and seats_MeanEncoded','seats_MeanEncoded and brand_countRankEncoded','brand_countRankEncoded and mileage_Quartile_countEncoded','owner_countEncoded and brand_countRankEncoded','owner_countEncoded and seats_countEncoded','owner_countEncoded and seats_MeanEncoded','owner_MeanEncoded and kmdriven_Quartile_countRankEncoded','owner_countEncoded and kmdriven_Quartile_MeanEncoded','seats_MeanEncoded and maxpower_Quartile_countRankEncoded','owner_countEncoded and maxpower_Quartile_countRankEncoded','sellertype_countEncoded and owner_countEncoded','brand_countRankEncoded and brandAndModel_countRankEncoded','fuel_countEncoded and owner_MeanEncoded','seats_MeanEncoded and mileage_Quartile_countRankEncoded','seats_countEncoded and brandAndModel_countEncoded','torque_countEncoded and seats_countEncoded','owner_countEncoded and kmdriven_Quartile_countRankEncoded','owner_MeanEncoded and seats_countEncoded','seats_countEncoded and brandAndModel_countRankEncoded','seats_MeanEncoded and maxpower_Quartile_MeanEncoded','torque_countRankEncoded and seats_countEncoded','owner_countEncoded and torque_countRankEncoded','owner_countEncoded and maxpower_Quartile_MeanEncoded','fuel_countEncoded and torque_countRankEncoded','fuel_countEncoded and brandAndModel_countRankEncoded','brand_countRankEncoded and brandAndModel_countEncoded','seats_countEncoded and maxpower_Quartile_countEncoded','sellertype_countEncoded and mileage_Quartile_countRankEncoded','owner_countEncoded and kmdriven_Quartile_countEncoded','seats_countEncoded and mileage_Quartile_MeanEncoded','owner_countEncoded and brand_MeanEncoded','brand_countRankEncoded and kmdriven_Quartile_countEncoded','brandAndModel_countRankEncoded and mileage_Quartile_countRankEncoded','sellertype_countEncoded and seats_MeanEncoded','owner_countEncoded and mileage_Quartile_MeanEncoded','owner_countEncoded and brand_countEncoded','sellertype_countEncoded and seats_countEncoded','seats_MeanEncoded and brandAndModel_countRankEncoded','seats_countEncoded and mileage_Quartile_countEncoded','owner_countEncoded and brandAndModel_countRankEncoded','owner_countEncoded and brandAndModel_countEncoded','brand_countEncoded and maxpower_Quartile_countRankEncoded','owner_countEncoded and mileage_Quartile_countEncoded','owner_MeanEncoded and maxpower_Quartile_MeanEncoded','torque_countEncoded and brand_countRankEncoded','brand_MeanEncoded and mileage_Quartile_countRankEncoded','kmdriven_Quartile_MeanEncoded and maxpower_Quartile_countRankEncoded','kmdriven_Quartile_countRankEncoded and mileage_Quartile_MeanEncoded','seats_MeanEncoded and kmdriven_Quartile_countRankEncoded','seats_MeanEncoded and mileage_Quartile_MeanEncoded','fuel_countEncoded and kmdriven_Quartile_MeanEncoded','brandAndModel_countEncoded and mileage_Quartile_countRankEncoded','torque_countRankEncoded and maxpower_Quartile_MeanEncoded','brand_countRankEncoded and brandAndModel_MeanEncoded','brand_MeanEncoded and kmdriven_Quartile_countRankEncoded','owner_MeanEncoded and mileage_Quartile_MeanEncoded','torque_MeanEncoded and brand_countRankEncoded','owner_MeanEncoded and seats_MeanEncoded','sellertype_countEncoded and maxpower_Quartile_countRankEncoded','torque_countRankEncoded and mileage_Quartile_countRankEncoded','brand_countEncoded and maxpower_Quartile_MeanEncoded','torque_countRankEncoded and seats_MeanEncoded','seats_MeanEncoded and kmdriven_Quartile_MeanEncoded','kmdriven_Quartile_countEncoded and mileage_Quartile_countRankEncoded','seats_MeanEncoded and brandAndModel_countEncoded','seats_MeanEncoded and brand_MeanEncoded','brand_MeanEncoded and maxpower_Quartile_MeanEncoded','sellertype_countEncoded and maxpower_Quartile_MeanEncoded','owner_countEncoded and torque_countEncoded','fuel_countEncoded and maxpower_Quartile_countEncoded','kmdriven_Quartile_countEncoded and maxpower_Quartile_MeanEncoded','owner_countEncoded and maxpower_Quartile_countEncoded','brandAndModel_countRankEncoded and maxpower_Quartile_countRankEncoded','seats_MeanEncoded and brand_countEncoded','torque_countRankEncoded and kmdriven_Quartile_countRankEncoded','sellertype_countEncoded and owner_MeanEncoded','brandAndModel_countRankEncoded and maxpower_Quartile_MeanEncoded','brandAndModel_countRankEncoded and kmdriven_Quartile_countRankEncoded','brand_countEncoded and kmdriven_Quartile_countRankEncoded','torque_MeanEncoded and seats_countEncoded','mileage_Quartile_countEncoded and maxpower_Quartile_MeanEncoded','seats_countEncoded and kmdriven_Quartile_MeanEncoded','brand_countEncoded and mileage_Quartile_countRankEncoded','fuel_countEncoded and kmdriven_Quartile_countEncoded','owner_MeanEncoded and mileage_Quartile_countRankEncoded','seats_countEncoded and kmdriven_Quartile_countEncoded','sellertype_countEncoded and mileage_Quartile_MeanEncoded','fuel_countEncoded and owner_countEncoded','owner_MeanEncoded and maxpower_Quartile_countRankEncoded','torque_countRankEncoded and maxpower_Quartile_countRankEncoded','torque_countEncoded and seats_MeanEncoded','seats_countEncoded and brand_MeanEncoded','owner_MeanEncoded and kmdriven_Quartile_MeanEncoded','sellertype_countEncoded and brand_countEncoded','fuel_countEncoded and torque_countEncoded','seats_MeanEncoded and maxpower_Quartile_countEncoded','seats_MeanEncoded and mileage_Quartile_countEncoded','mileage_Quartile_countRankEncoded and maxpower_Quartile_countEncoded','sellertype_countEncoded and torque_countRankEncoded','seats_MeanEncoded and kmdriven_Quartile_countEncoded','brand_MeanEncoded and maxpower_Quartile_countRankEncoded','kmdriven_Quartile_MeanEncoded and maxpower_Quartile_MeanEncoded','brand_MeanEncoded and maxpower_Quartile_countEncoded','fuel_countEncoded and brand_MeanEncoded','brand_countEncoded and brandAndModel_countRankEncoded','owner_MeanEncoded and brand_MeanEncoded','fuel_countEncoded and brandAndModel_countEncoded','mileage_Quartile_MeanEncoded and maxpower_Quartile_countRankEncoded','sellertype_countEncoded and brandAndModel_countRankEncoded','sellertype_countEncoded and mileage_Quartile_countEncoded','sellertype_countEncoded and maxpower_Quartile_countEncoded','kmdriven_Quartile_MeanEncoded and mileage_Quartile_countRankEncoded','kmdriven_Quartile_countEncoded and maxpower_Quartile_countRankEncoded','fuel_countEncoded and brand_countEncoded','owner_MeanEncoded and mileage_Quartile_countEncoded','torque_countRankEncoded and mileage_Quartile_countEncoded','brandAndModel_countEncoded and maxpower_Quartile_countRankEncoded','brandAndModel_countEncoded and maxpower_Quartile_MeanEncoded','owner_MeanEncoded and brandAndModel_countRankEncoded','fuel_countEncoded and sellertype_countEncoded','owner_MeanEncoded and torque_countRankEncoded','brandAndModel_countRankEncoded and mileage_Quartile_countEncoded','fuel_countEncoded and mileage_Quartile_MeanEncoded','brand_countEncoded and mileage_Quartile_MeanEncoded','mileage_Quartile_countEncoded and maxpower_Quartile_countRankEncoded','torque_MeanEncoded and mileage_Quartile_countRankEncoded','mileage_Quartile_MeanEncoded and maxpower_Quartile_MeanEncoded','fuel_countEncoded and mileage_Quartile_countEncoded','torque_countEncoded and mileage_Quartile_countRankEncoded','brand_countEncoded and maxpower_Quartile_countEncoded','torque_countEncoded and maxpower_Quartile_MeanEncoded','kmdriven_Quartile_MeanEncoded and maxpower_Quartile_countEncoded','brandAndModel_countRankEncoded and kmdriven_Quartile_MeanEncoded','sellertype_countEncoded and torque_countEncoded','brand_MeanEncoded and mileage_Quartile_countEncoded','brandAndModel_countRankEncoded and mileage_Quartile_MeanEncoded','owner_MeanEncoded and kmdriven_Quartile_countEncoded','sellertype_countEncoded and brand_MeanEncoded','brandAndModel_countEncoded and kmdriven_Quartile_countRankEncoded','brand_MeanEncoded and kmdriven_Quartile_MeanEncoded','sellertype_countEncoded and kmdriven_Quartile_MeanEncoded','owner_MeanEncoded and brand_countEncoded','torque_countRankEncoded and brand_countEncoded','torque_countEncoded and kmdriven_Quartile_countRankEncoded','torque_countRankEncoded and brand_MeanEncoded','sellertype_countEncoded and kmdriven_Quartile_countEncoded','brand_countEncoded and brandAndModel_countEncoded','mileage_Quartile_countEncoded and maxpower_Quartile_countEncoded','owner_MeanEncoded and torque_countEncoded','brand_MeanEncoded and brandAndModel_countRankEncoded','brandAndModel_countRankEncoded and maxpower_Quartile_countEncoded','kmdriven_Quartile_MeanEncoded and mileage_Quartile_countEncoded','torque_countRankEncoded and mileage_Quartile_MeanEncoded','kmdriven_Quartile_countEncoded and mileage_Quartile_countEncoded','kmdriven_Quartile_MeanEncoded and mileage_Quartile_MeanEncoded','seats_MeanEncoded and brandAndModel_MeanEncoded','kmdriven_Quartile_countRankEncoded and maxpower_Quartile_countEncoded','brand_countEncoded and mileage_Quartile_countEncoded','brand_countEncoded and kmdriven_Quartile_MeanEncoded','brandAndModel_countRankEncoded and kmdriven_Quartile_countEncoded','kmdriven_Quartile_countEncoded and mileage_Quartile_MeanEncoded','seats_countEncoded and brandAndModel_MeanEncoded','torque_countRankEncoded and kmdriven_Quartile_MeanEncoded','torque_countEncoded and brand_countEncoded','torque_MeanEncoded and seats_MeanEncoded','owner_MeanEncoded and brandAndModel_countEncoded','brandAndModel_MeanEncoded and kmdriven_Quartile_countRankEncoded','torque_countEncoded and maxpower_Quartile_countRankEncoded','brand_MeanEncoded and mileage_Quartile_MeanEncoded','sellertype_countEncoded and brandAndModel_countEncoded','brand_MeanEncoded and brandAndModel_countEncoded','torque_countEncoded and brand_MeanEncoded','torque_countRankEncoded and kmdriven_Quartile_countEncoded','brandAndModel_MeanEncoded and mileage_Quartile_countRankEncoded','torque_countRankEncoded and brandAndModel_countEncoded','brandAndModel_MeanEncoded and maxpower_Quartile_countRankEncoded','brandAndModel_countEncoded and kmdriven_Quartile_MeanEncoded','brand_countEncoded and kmdriven_Quartile_countEncoded','torque_countRankEncoded and maxpower_Quartile_countEncoded','owner_MeanEncoded and maxpower_Quartile_countEncoded','torque_countEncoded and kmdriven_Quartile_MeanEncoded','torque_countEncoded and mileage_Quartile_MeanEncoded','torque_countRankEncoded and brandAndModel_countRankEncoded','torque_countEncoded and kmdriven_Quartile_countEncoded','torque_countEncoded and mileage_Quartile_countEncoded','owner_countEncoded and brandAndModel_MeanEncoded','brand_MeanEncoded and kmdriven_Quartile_countEncoded','brandAndModel_countEncoded and kmdriven_Quartile_countEncoded','torque_countEncoded and brandAndModel_countRankEncoded','brandAndModel_countEncoded and maxpower_Quartile_countEncoded','mileage_Quartile_MeanEncoded and maxpower_Quartile_countEncoded','brandAndModel_countEncoded and mileage_Quartile_MeanEncoded','torque_MeanEncoded and kmdriven_Quartile_countRankEncoded','brandAndModel_MeanEncoded and maxpower_Quartile_MeanEncoded','torque_countEncoded and maxpower_Quartile_countEncoded','kmdriven_Quartile_countEncoded and maxpower_Quartile_countEncoded','owner_countEncoded and torque_MeanEncoded','brandAndModel_countEncoded and mileage_Quartile_countEncoded','torque_countEncoded and brandAndModel_countEncoded','torque_MeanEncoded and maxpower_Quartile_countRankEncoded','owner_MeanEncoded and brandAndModel_MeanEncoded','torque_MeanEncoded and brandAndModel_countRankEncoded','torque_MeanEncoded and maxpower_Quartile_MeanEncoded','owner_MeanEncoded and torque_MeanEncoded','brandAndModel_MeanEncoded and maxpower_Quartile_countEncoded','torque_countRankEncoded and brandAndModel_MeanEncoded','fuel_countEncoded and torque_MeanEncoded','fuel_countEncoded and brandAndModel_MeanEncoded','brand_MeanEncoded and brandAndModel_MeanEncoded','torque_MeanEncoded and brand_MeanEncoded','brand_countEncoded and brandAndModel_MeanEncoded','sellertype_countEncoded and torque_MeanEncoded','sellertype_countEncoded and brandAndModel_MeanEncoded','torque_MeanEncoded and brand_countEncoded','brandAndModel_MeanEncoded and mileage_Quartile_MeanEncoded','brandAndModel_MeanEncoded and kmdriven_Quartile_countEncoded','brandAndModel_MeanEncoded and mileage_Quartile_countEncoded','torque_MeanEncoded and kmdriven_Quartile_countEncoded','torque_MeanEncoded and mileage_Quartile_MeanEncoded','torque_MeanEncoded and mileage_Quartile_countEncoded','brandAndModel_MeanEncoded and kmdriven_Quartile_MeanEncoded','torque_countEncoded and brandAndModel_MeanEncoded','torque_MeanEncoded and brandAndModel_countEncoded','torque_MeanEncoded and maxpower_Quartile_countEncoded','torque_MeanEncoded and kmdriven_Quartile_MeanEncoded','torque_MeanEncoded and brandAndModel_MeanEncoded']
        
        for extra_column in interactions:
            first,_,second = extra_column.split()

            if first in data_dict['fold_dict'][0]['fold_training'].columns and second in data_dict['fold_dict'][0]['fold_training'].columns:
                #print('Start the feature:',extra_column,data_dict['fold_dict'][0]['fold_training'][first].dtype,data_dict['fold_dict'][0]['fold_training'][second].dtype)
                #print('first:',first,extra_column,data_dict['fold_dict'][0]['fold_training'][first].value_counts())
                #print('second:',second,extra_column,data_dict['fold_dict'][0]['fold_training'][second].value_counts())
                    

                # if extra_column not in ['fuel_countEncoded and maxpower_Quartile_countRankEncoded']:
                #print('Start the feature:',extra_column,data_dict['fold_dict'][0]['fold_training'][first].dtype,data_dict['fold_dict'][0]['fold_training'][second].dtype)
                data_dict,numerical_features_tree = add_column(data_dict,extra_column,numerical_features_tree)
            else:
                pass
                #print('Remove the feature:',extra_column)
            
    if problem == 'CouponRecommendation':
        interactions = ['time_MeanEncoded and coupon_MeanEncoded']
        for extra_column in interactions:
            data_dict,numerical_features_tree = add_column(data_dict,extra_column,numerical_features_tree)
    
    return data_dict,dependent_variable,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered

if __name__ == '__main__':
    problem = 'CarSales'
    data_dict,categorical_linear,categorical_tree,categorical_features,numerical_features_linear,numerical_features_tree,ordinal_features,ordinal_features_engineered = get_data(problem)
    