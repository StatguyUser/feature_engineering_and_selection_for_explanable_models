"""
@author: Md Azimul Haque
"""
import pandas as pd
import numpy as np

def recode_categories(dataset,feature_name):
    crosstab = np.round(pd.crosstab(dataset[feature_name],dataset['CancelStatus'], normalize='index')*100)
    zero = crosstab[crosstab['Not Canceled']>50].index.values
    one = crosstab[crosstab['Canceled']>50].index.values
    
    dataset[feature_name+'_Recoded'] = np.where(dataset[feature_name].isin(one),'One','Zero')
    
    return zero,one,dataset

def createHotelCancellations():
    dataset = pd.read_csv("D:\\Book\\H1.csv")
    
    #fill na with 0 for children
    dataset['Children'].fillna(0,inplace=True)
    monthmap = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
    dataset['ArrivalDateMonthNumber'] = dataset['ArrivalDateMonth'].map(monthmap)
    dataset['ArrivalDate'] = pd.to_datetime(dataset['ArrivalDateDayOfMonth'].astype(str) + '-' + dataset['ArrivalDateMonth'].astype(str) + '-'+ dataset['ArrivalDateYear'].astype(str))
    dataset['ReservationStatusDate']=pd.to_datetime(dataset['ReservationStatusDate'])
    
    dataset['CancelStatus'] = np.where(dataset['IsCanceled']==1,'Canceled','Not Canceled')
    
    
    #Remove cases where there are no guests. both adults and children is 0 and ADR less than or equal to 0.
    dataset=dataset[((dataset['Adults']>0) | (dataset['Children']>0)) &  (dataset['ADR']>0)]
    dataset.reset_index(inplace=True,drop=True)
    
    #country feature has few hundred missing values and hence we will remove from analysis.
    dataset.dropna(inplace=True)
    dataset.reset_index(inplace=True,drop=True)
    
    #create booking date
    dataset['BookingDate'] = dataset['ArrivalDate'] - pd.to_timedelta(dataset['LeadTime'], unit='d')

    ### Create EDA Features from categorical features
    dataset['LessCancellation'] = np.where(((dataset['Meal']=='BB       ') | (dataset['DepositType']=='No Deposit       ')),1,0)
    ## create binary features
    zero_Country,one_Country,dataset = recode_categories(dataset,feature_name='Country')
    zero_Agent,one_Agent,dataset = recode_categories(dataset,feature_name='Agent')
    zero_Company,one_Company,dataset = recode_categories(dataset,feature_name='Company')        
    ##dependent variable
    dependent_variable = ['IsCanceled']
    
    ### Create EDA feattures from numerical features
    #more than 5 cancellations by the same guest in the past
    dataset['PreviousCancellationsMoreThan5']=np.where(dataset['PreviousCancellations']>5,1,0)
    
    #did not cancel more than 14 times in the past
    dataset['PreviousBookingsNotCanceledMoreThan14']=np.where(dataset['PreviousBookingsNotCanceled']>14,1,0)
    
    #waiting list for more than 121 days
    dataset['DaysInWaitingListMoreThan121']=np.where(dataset['DaysInWaitingList']>121,1,0)


    ###### percentile+encoding features to be done in higher order numerical features
    #['LeadTime','ADR']
    
    ##list of numerical features
    numerical_features = ['LeadTime', 'StaysInWeekendNights', 'StaysInWeekNights', 'Adults', 'Children','Babies', 'PreviousCancellations','PreviousBookingsNotCanceled','BookingChanges','DaysInWaitingList','ADR', 'TotalOfSpecialRequests']

    ##list of categorical features
    categorical_features = ['Meal', 'Country', 'MarketSegment', 'DistributionChannel','ReservedRoomType', 'AssignedRoomType','DepositType', 'Agent', 'Company','CustomerType','RequiredCarParkingSpaces','ArrivalDateYear', 'ArrivalDateMonth','ArrivalDateWeekNumber', 'ArrivalDateDayOfMonth','IsRepeatedGuest','Country_Recoded', 'Agent_Recoded','Company_Recoded','LessCancellation','PreviousCancellationsMoreThan5','PreviousBookingsNotCanceledMoreThan14','DaysInWaitingListMoreThan121']
    
    return dataset,dependent_variable,numerical_features,categorical_features

def createCouponRecommendation():
    dataset = pd.read_csv("D:\\Book\\in-vehicle-coupon-recommendation.csv")
    #dependent variable is 'Y', where 1 being coupon was accepted and 0 being not accepted
    
    #feature 'car' has no description in the paper, nor in webpage: https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation#
    #hence we delete this, as it has maximum missing data
    del dataset['car']
    
    #we also drop the missing value rows in other remaining features, as we do not know the reason why they exist.
    #we will do our analysis on remaining data
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True,inplace=True)
    dataset.shape
    
    dataset['age'].replace(['50plus', 'below21'], [52, 18], inplace=True)

    dataset['age'] = dataset['age'].astype(int)
    
    #create new column for dependent variable for visualization purpose
    dataset['couponstatus'] = np.where(dataset['Y']==1,'accepted','not accepted')
    
    dataset['accepted_coupon']=np.where(((dataset['weather']=='Sunny') & (dataset['occupation']=='Business & Financial') & (dataset['CoffeeHouse'].isin(['1~3','gt8'])) & (dataset['coupon'].isin(['Carry out & Take away','Restaurant(<20)']))),1,0)
    dataset['rejected_coupon']=np.where(((dataset['CoffeeHouse']=='never') & (dataset['occupation']=='Community & Social Services') & (dataset['weather'].isin(['Rainy','Snowy'])) & (dataset['coupon'].isin(['Bar','Restaurant(20-50)']))),1,0)
    numerical_features = ['temperature','age']
    categorical_features = ['destination', 'passanger', 'weather', 'time', 'coupon','expiration', 'gender', 'maritalStatus', 'has_children','education', 'occupation', 'income', 'Bar', 'CoffeeHouse', 'CarryAway','RestaurantLessThan20', 'Restaurant20To50', 'toCoupon_GEQ5min','toCoupon_GEQ15min', 'toCoupon_GEQ25min', 'direction_same','direction_opp','accepted_coupon','rejected_coupon']
    dependent_variable = ['Y']

    return dataset,dependent_variable,numerical_features,categorical_features

def createPredictRoomBooking():
    dataset = pd.read_csv("D:\\Book\\H2_PredictingOccupancy_100daysPrior.csv")
    
    dataset['AdjustedLeadTime_CumulativeNumberOfRooms'] = dataset['AdjustedLeadTime']*dataset['CumulativeNumberOfRooms']
    dataset['AdjustedLeadTime_CumulativeNumberOfRooms_Substract'] = dataset['AdjustedLeadTime']*dataset['CumulativeNumberOfRooms_Substract']
    dataset['AdjustedLeadTime_CumulativeNumberOfRoomsNet'] = dataset['AdjustedLeadTime']*dataset['CumulativeNumberOfRoomsNet']
    
    dataset['AdjustedLeadTime_CumulativeRevenue'] = dataset['AdjustedLeadTime']*dataset['CumulativeRevenue']
    dataset['AdjustedLeadTime_CumulativeRevenue_Substract'] = dataset['AdjustedLeadTime']*dataset['CumulativeRevenue_Substract']
    dataset['AdjustedLeadTime_CumulativeRevenueNet'] = dataset['AdjustedLeadTime']*dataset['CumulativeRevenueNet']
    
    #EDA feature
    dataset['MonthOfYearPeakDemand'] = np.where(dataset['MonthOfYear'].isin([7,8]),1,0)
    
    numerical_features=['AdjustedLeadTime', 'NumberOfRooms','CumulativeNumberOfRooms', 'NumberOfRooms_Substract','CumulativeNumberOfRooms_Substract', 'CumulativeNumberOfRoomsNet', 'AdjustedLeadTime_CumulativeNumberOfRooms','AdjustedLeadTime_CumulativeNumberOfRooms_Substract','AdjustedLeadTime_CumulativeNumberOfRoomsNet', 'Revenue', 'CumulativeRevenue', 'Revenue_Substract','CumulativeRevenue_Substract', 'CumulativeRevenueNet', 'AdjustedLeadTime_CumulativeRevenue','AdjustedLeadTime_CumulativeRevenue_Substract','AdjustedLeadTime_CumulativeRevenueNet']
    
    dependent_variable = ['TotalRooms']
    
    categorical_features = ['DayOfWeek','Weekend','WeekOfYear', 'DayOfMonth', 'WeekOfMonth', 'MonthOfYear','MonthOfYearPeakDemand']
    
    #### For month of the year, we can create different type of higher order feature representation.
    return dataset,dependent_variable,numerical_features,categorical_features

def createCarSales():
    dataset = pd.read_csv("D:\\Book\\Car details v3.csv")
    dataset.dropna(inplace=True)
    dataset.reset_index(inplace=True,drop=True)
    
    #basic data cleaning based on common sense
    dataset = dataset[(dataset['km_driven']>=0) & (dataset['mileage']!='0.0kmpl') & (dataset['max_power']!='bhp')]
    dataset.reset_index(inplace=True,drop=True)
    
    dataset['mileage']=pd.to_numeric(dataset['mileage'].str.split(n=1, expand=True)[0])
    dataset['engine']=pd.to_numeric(dataset['engine'].str.split(n=1, expand=True)[0])
    dataset['max_power']=pd.to_numeric(dataset['max_power'].str.split(n=1, expand=True)[0])
    
    #create features for car brand, which is the first word 
    dataset['brand']=dataset['name'].str.split().str[0]
    #create features for car brand and model, which is the first and second word
    dataset['brandAndModel']=dataset['brand']+" "+dataset['name'].str.split().str[1]
    
    
    percentile=90
    nineteenth_percentile = np.percentile(dataset['selling_price'],percentile)
    
    dataset['Above90']=np.where(dataset['selling_price']>nineteenth_percentile,1,0)
    
    # ##delete
    # dataset=dataset[dataset['selling_price']>nineteenth_percentile]
    # dataset.reset_index(inplace=True,drop=True)
    # ##delete
    
    ###Higher order feature engineering for number of seats and price relationship
    # average selling price by Above90 groups
    # number of seats. Or consider it as ordinal feature
    
    dependent_variable = ['selling_price']
    numerical_features = ['year', 'km_driven', 'seats', 'mileage', 'engine', 'max_power']
    categorical_features = ['fuel', 'seller_type','transmission', 'owner', 'torque', 'seats', 'brand', 'brandAndModel', 'Above90']
    
    # for feature in numerical_features:
    #     dataset[feature]=dataset[feature].astype('int64')
    #     print(feature,dataset[feature].dtype)
    
    return dataset,dependent_variable,numerical_features,categorical_features


def CreateDF_UntilEDA(dataset_name=''):
    '''
    4 options. 1 for each dataset: 'HotelCancellations', 'CouponRecommendation', 'PredictRoomBooking','CarSales'
    '''
    if dataset_name == 'HotelCancellations':
        dataset,dependent_variable,numerical_features2,categorical_features2=createHotelCancellations()
    elif dataset_name == 'CouponRecommendation':
        dataset,dependent_variable,numerical_features2,categorical_features2=createCouponRecommendation()
    elif dataset_name == 'PredictRoomBooking':
        dataset,dependent_variable,numerical_features2,categorical_features2=createPredictRoomBooking()
    elif dataset_name == 'CarSales':
        dataset,dependent_variable,numerical_features2,categorical_features2=createCarSales()
    
    ##remove underscore from feature name
    rename_dict = {}
    for col in dataset.columns:
        rename_dict[col] = col.replace("_", "")
    dataset.rename(columns=rename_dict, inplace=True)

    dependent_variable = [dependent_variable[0].replace("_", "")]
    
    numerical_features = []
    for col in numerical_features2:
        numerical_features.append(col.replace("_", ""))

    categorical_features = []
    for col in categorical_features2:
        categorical_features.append(col.replace("_", ""))

    
    return dataset,dependent_variable,numerical_features,categorical_features

if __name__ == '__main__':
    #pass

    # counts =dataset['brandAndModel'].value_counts()
    # mean = dataset[['brandAndModel','selling_price']].groupby('brandAndModel').mean()
    # mean_df = pd.DataFrame({'brandAndModel':mean.index,'AveragePrice':mean.values.ravel()})
    # count_df = pd.DataFrame({'brandAndModel':counts.index,'NumberofRecords':counts.values})
    # result_df = mean_df.merge(count_df)
    
    # #number1 = dataset.shape[0]
    
    # result_df2=result_df[(result_df.NumberofRecords>30) & (result_df.AveragePrice<=887500)]
    # dataset2 = dataset[dataset['brand'].isin(result_df2['brand'].values)]
    # dataset2.reset_index(inplace=True,drop=True)
    dataset,dependent_variable,numerical_features,categorical_features = createPredictRoomBooking()
