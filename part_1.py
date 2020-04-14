#load data sets
train_df = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")
test_df = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")
submission = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

#check for missings
display(train_df.isnull().sum()/len(train_df)*100)
display(test_df.isnull().sum()/len(test_df)*100)
#Id                 0.000000
#Province_State    57.507987
#Country_Region     0.000000
#Date               0.000000
#ConfirmedCases     0.000000
#Fatalities         0.000000
#dtype: float64
#ForecastId         0.000000
#Province_State    57.507987
#Country_Region     0.000000
#Date               0.000000
#dtype: float64

#check for data range in both datasets
print("The lowest date in the train data set is {} and the highest {}.".format(train_df['Date'].min(),train_df['Date'].max()))
print("The lowest date in the test data set is {} and the highest {}.".format(test_df['Date'].min(),test_df['Date'].max()))
#The lowest date in the train data set is 2020-01-22 and the highest 2020-04-13.
#The lowest date in the test data set is 2020-04-02 and the highest 2020-05-14.

#just some cosmetic renaming
train_df.rename(columns={'Province_State':'State','Country_Region':'Country'}, inplace=True)
test_df.rename(columns={'Province_State':'State','Country_Region':'Country'}, inplace=True)

#function for replacing all the missings in the state column
def missings(state, country):
    return country if pd.isna(state) == True else state
    
#if there are no states specified for a country, the missing is replaced with the country´s name
train_df['State'] = train_df.apply(lambda x: missings(x['State'],x['Country']),axis=1)
test_df['State'] = test_df.apply(lambda x: missings(x['State'],x['Country']),axis=1)

print("In our data set are {} countries and {} states.".format(train_df['Country'].nunique(),train_df['State'].nunique()))
#In our data set are 184 countries and 312 states.

df_confirmedcases = train_df.groupby(['Country','State']).max().groupby('Country').sum().sort_values(by='ConfirmedCases', ascending=False).reset_index().drop(columns='Id')
df_confirmedcases[:20].set_index('Country').style.background_gradient(cmap='Oranges')
