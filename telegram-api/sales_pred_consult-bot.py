import os
import json
import requests
import pandas as pd
from flask import Flask, request, Response


# contants
TOKEN = ''


def parse_message(message):

    # get chat_id #
    chat_id = message['message']['chat']['id']
    # get text with store ids numbers
    store_ids_text = message['message']['text']
    # remove spaces and split text using commas
    store_ids = pd.Series(store_ids_text.strip().split(','))
    # extract the numbers and set type as int
    store_ids = list(store_ids.str.extractall("(\d+)")[0].astype(int))

    return chat_id, store_ids


def load_dataset(store_ids):

    # loading test dataset
    test = pd.read_csv('stores_info/test.csv')
    stores = pd.read_csv('stores_info/store.csv')
    # merge test dataset with store dataset
    df_test = pd.merge(test, stores, how='left', on='Store')
    # choose store
    df_test = df_test[df_test['Store'].isin(store_ids)]
    # test if dataframe is not empty
    if not df_test.empty:
        # remove closed day and null
        df_test = df_test[(df_test['Open'] != 0) & (df_test['Open'].notna())]
        # convert dataframe to json
        data = json.dumps(df_test.to_dict(orient='records'))
    else:
        data = {}

    return data


def predict(data):

    # API Call #
    # Heroku
    url    = 'https://sales-predictions-ds.herokuapp.com/sales/predict'
    header = {'Content-type': 'application/json'}
    data   = data

    # sending request to Handler API
    r = requests.post(url, data=data, headers=header)
    print(f'Status Code {r.status_code}')

    # converting received predictions of sales to dataframe
    d1 = pd.DataFrame(r.json(), columns=r.json()[0].keys())

    return d1


def compose_message(d2):

    # number of reports is limited to 50 stores ids
    if len(d2) > 50:
        d2 = d2.iloc[0:50]
    # message headline
    msg = 'Sales forecast for the next 7 weeks:\n\n'
    # concatenate each store sales prediction in the response message
    for i in range(len(d2)):
        msg += f"Store number:{d2.loc[i, 'store']:4} - ${d2.loc[i, 'prediction']:,.2f}\n\n"
    
    return msg


def send_message(chat_id, text):

    url = f'https://api.telegram.org/bot{TOKEN}/'
    url = url + f'sendMessage?chat_id={chat_id}'

    r = requests.post(url, json={'text': text})
    print(f'Status Code: {r.status_code}')

    return None


# Initialize API
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        message = request.get_json()
        # calling function that extracts chat and store Ids
        chat_id, store_ids = parse_message(message)
        # check if there is a store_id number in the list
        if len(store_ids) > 0:
            # loading data
            data = load_dataset(store_ids)
            # check if dataframe was loaded correctly
            if len(data) > 0:
                # prediction
                d1 = predict(data)
                # prediction for all sales in the next 7 weeks (by store)
                d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()
                # 
                msg = compose_message(d2)
                # send message
                send_message(chat_id, text=msg)
                return Response('OK', status=200)
            # response to a non existing store
            else:
                send_message(chat_id, text='Store not available.')
                return Response('OK', status=200)
        # response in case there is no valid number as store_id
        else:
            send_message(chat_id, text='Store ID is not valid.')
            return Response('OK', status=200)
    # answer to 'GET' method
    else:
        return '<h1> Sales Prediction Telegram Bot </h1>'


if __name__ == '__main__':

    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
