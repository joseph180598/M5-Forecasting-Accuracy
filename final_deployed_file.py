
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request,render_template
from bs4 import BeautifulSoup
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import warnings
warnings.filterwarnings('ignore')
import datetime
import gc
gc.collect()
import lightgbm
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from tqdm import tqdm
import time


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


###################################################
main_dir=''

sales=pd.read_csv(main_dir+'sales_train_evaluation.csv')
print('Sales Loaded')

calendar=pd.read_csv(main_dir+'calendar.csv',dayfirst=True,keep_date_col=True)
print('Calendar Loaded')

sales_price=pd.read_csv(main_dir+'sell_prices.csv')
print('Sell Price Loaded')

def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i,t in (enumerate(types)):
        if 'int' in str(t):
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        elif t == object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df  

sales=downcast(sales)
print('sales Completed...')

calendar=downcast(calendar)
print('calendar Completed...')

sales_price=downcast(sales_price)
print('sales_price Completed...')

###################################################


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    a= to_predict_list['review_text']
    a=str(a)
    print(a)

    start=time.time()
    store_name=a.split('_')[3]+'_'+a.split('_')[4]
    from tqdm import tqdm

    id_cols=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    directory=''

    data=sales[sales['id']==a]

    calendar['date']=pd.to_datetime(calendar['date'], dayfirst=True)

    event_dates=calendar[(calendar['event_name_1']!='no_event') | (calendar['event_name_2']!='no_event')]['date']

    def days_to_event(i):
        j=(event_dates.values[np.absolute(event_dates-i).argmin()]-i).astype('timedelta64[D]')
        #print(j)
        return j

    calendar['days_to_next_event']=np.array(list(days_to_event(a) for a in tqdm(calendar['date'].values)))

    calendar['days_to_next_event']=calendar['days_to_next_event'].dt.days

    f=lambda x: 1 if x==7 or x==1 or x==2 else 0
    calendar['is_weekoff']=calendar['wday'].map(f) 
    calendar['is_weekoff']=calendar['is_weekoff']

    calendar['day_of_month']=calendar['date'].dt.day.astype(np.int16)

    calendar['week']=calendar['date'].dt.isocalendar().week
    calendar['week']=pd.Categorical(calendar['week'])

    #calendar['day_to_sun']=7-calendar['wday']

    calendar['day_of_month']=calendar['day_of_month'].astype('category')

    for d in range(1942,1942+28):
        col = 'd_' + str(d)
        data[col] = 0
        data[col] = data[col].astype(np.int16)
    
    model=pickle.load(open(directory+'model'+store_name+'.pkl','rb'))

    melted_data = pd.melt(data, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='sold').dropna()
    processed_data = pd.merge(melted_data, calendar, on='d', how='left')
    processed_data = pd.merge(processed_data, sales_price, on=['store_id','item_id','wm_yr_wk'], how='left')

    def day(a):
        n=a.split('_')[1]
        return n

    processed_data['d']=processed_data['d'].apply(day).astype(int)

    processed_data['item_sold_avg'] = processed_data.groupby('item_id')['sold'].transform('mean').astype(np.float16)    
    processed_data['state_sold_avg'] = processed_data.groupby('state_id')['sold'].transform('mean').astype(np.float16)    #total 3 unique values, 1 for each state
    processed_data['store_sold_avg'] = processed_data.groupby('store_id')['sold'].transform('mean').astype(np.float16)  #10 unique values
    processed_data['cat_sold_avg'] = processed_data.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
    processed_data['dept_sold_avg'] = processed_data.groupby('dept_id')['sold'].transform('mean').astype(np.float16)
    processed_data['cat_dept_sold_avg'] = processed_data.groupby(['cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
    processed_data['store_item_sold_avg'] = processed_data.groupby(['store_id','item_id'])['sold'].transform('mean').astype(np.float16)
    processed_data['cat_item_sold_avg'] = processed_data.groupby(['cat_id','item_id'])['sold'].transform('mean').astype(np.float16)
    processed_data['dept_item_sold_avg'] = processed_data.groupby(['dept_id','item_id'])['sold'].transform('mean').astype(np.float16)
    processed_data['state_store_sold_avg'] = processed_data.groupby(['state_id','store_id'])['sold'].transform('mean').astype(np.float16)
    processed_data['state_store_cat_sold_avg'] = processed_data.groupby(['state_id','store_id','cat_id'])['sold'].transform('mean').astype(np.float16)
    processed_data['store_cat_dept_sold_avg'] = processed_data.groupby(['store_id','cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)

    main_data=processed_data[processed_data['d']>1942-180]

#print(processed_data.head(2))
    [d_id,d_item_id,d_dept_id,d_cat_id,d_store_id,d_state_id]=pickle.load(open(directory+'dict.pkl','rb'))

    events=['event_name_1','event_type_1','event_name_2','event_type_2']

    from sklearn.preprocessing import LabelEncoder
    for i in events:
        main_data[i]=main_data[i].astype('category')
        main_data[i] = main_data[i].cat.add_categories("nan").fillna("nan")
        encoder= pickle.load(open(directory+'encoder_'+i+'.pkl','rb'))
        main_data[i]=encoder.transform(main_data[i]).astype(np.int8)
    
    cols = main_data.dtypes.index.tolist()
    types = main_data.dtypes.values.tolist()
    for i,type in enumerate(types):
        if type.name == 'category':
            encoder=pickle.load(open(directory+'cat_encoder_'+str(cols[i])+'.pkl','rb'))
            main_data[cols[i]] = encoder.transform(main_data[cols[i]])
    events=['event_name_1','event_type_1','event_name_2','event_type_2']

    for i in events:
        main_data[i]=main_data[i].astype('category')
    
    lags = [28,35,42,49]


    for lag in tqdm(lags):
        main_data['sold_lag_'+str(lag)] = main_data.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['sold'].shift(lag).astype(np.float16)

    lags2 = [-2,-1,1,2]
    for lag in lags2:
        main_data['event1_lag_'+str(lag)] = main_data.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['event_name_1'].shift(lag).astype(np.float16)
        main_data['event1_lag_'+str(lag)].fillna(100, inplace=True)
        main_data['event1_lag_'+str(lag)]=main_data['event1_lag_'+str(lag)].astype(np.int8)
        main_data['event1_lag_'+str(lag)]=main_data['event1_lag_'+str(lag)].astype('category')
    

    years=main_data['date'].dt.year.unique()

    main_data['is_christmas']=0

    for y in tqdm(years):
        christmas=pd.to_datetime('25-12-'+str(y),dayfirst=True)
        main_data['is_christmas'].loc[main_data['date'][main_data['date']==christmas].index]=1
    
    encoder=pickle.load(open(directory+'wm_yr_wk_linear_encoder.pkl','rb'))

    main_data['wm_yr_wk_linear']=encoder.transform(main_data['wm_yr_wk'].values).astype(np.int16)

    main_data['price_lag'] = main_data.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],as_index=False)['sell_price'].shift(7).astype(np.float16)
    main_data['price-diff']=(main_data['price_lag']-main_data['sell_price']).astype(np.float16)
    main_data.drop(['price_lag'], axis=1, inplace=True)

    main_data=main_data.loc[main_data['sell_price'].dropna().index]

    main_data['decimal']=main_data['sell_price'].apply(lambda x: 100*(x-int(x))).astype(np.int8)

    main_data['expanding_price_mean'] = main_data.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform(lambda x: x.expanding(2).mean()).astype(np.float16)
    main_data['diff_moving_mean']=main_data['expanding_price_mean']-main_data['sell_price']
    main_data.drop(['expanding_price_mean'], axis=1, inplace=True)

    main_data.drop(['wday'], axis=1, inplace=True)

    encoder=pickle.load(open(directory+'year_encoder.pkl','rb'))

    main_data['year']=encoder.transform(main_data['year']).astype(np.int8)

    main_data['daily_avg_price'] = main_data.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','d'])['sell_price'].transform('mean').astype(np.float16)
    main_data['avg_price'] = main_data.groupby(['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'])['sell_price'].transform('mean').astype(np.float16)
    main_data['selling_trend'] = (main_data['daily_avg_price'] - main_data['avg_price']).astype(np.float16)
    main_data.drop(['daily_avg_price','avg_price','cat_id','state_id','date'],axis=1,inplace=True)

    forward_data=main_data[main_data['d']>=1942].drop('sold', axis=1)

    d_file=forward_data[['d','id']]
    predictions=model.predict(forward_data)
    d_file['predicted_sales']=np.ceil(predictions).astype(int)

    d_file = pd.pivot(d_file, index='id', columns='d', values='predicted_sales').reset_index()
    d_file.columns=['id'] + ['F' + str(i + 1) for i in range(28)]
    d_file.id = d_file.id.map(d_id)
    print((time.time()-start)/60)
    d_file=d_file.to_html()
    return render_template('record.html', table=d_file)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

