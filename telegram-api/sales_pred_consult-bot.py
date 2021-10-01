import os
import json
import requests
import pandas as pd
from flask import Flask, request, Response


# Bot info
# https://api.telegram.org/bot2035600574:AAEhFg2lJFdWFeG8I4f_ibV0PGKS9N3VPXk/getMe

# get updates
# https://api.telegram.org/bot2035600574:AAEhFg2lJFdWFeG8I4f_ibV0PGKS9N3VPXk/getUpdates

# Webhook updates
# https://api.telegram.org/bot2035600574:AAEhFg2lJFdWFeG8I4f_ibV0PGKS9N3VPXk/setWebhook?url=https://sales-pred-telegram.herokuapp.com/

# send message - chat_id and text are mandatory
# https://api.telegram.org/bot2035600574:AAEhFg2lJFdWFeG8I4f_ibV0PGKS9N3VPXk/sendMessage?chat_id=5555555&text=Hello

# contants
TOKEN = ''


def send_message(chat_id, text):
    url = f'https://api.telegram.org/bot{TOKEN}/'
    url = url + f'sendMessage?chat_id={chat_id}'

    r = requests.post(url, json={'text': text})
    print(f'Status Code: {r.status_code}')

    return None


def load_dataset(store_id=1):

    # loading test dataset
    test = pd.read_csv('stores_info/test.csv')
    stores = pd.read_csv('stores_info/store.csv')

    # merge test dataset with store dataset
    df_test = pd.merge(test, stores, how='left', on='Store')

    # choose store
    df_test = df_test[df_test['Store'] == store_id]

    # test if dataframe is not empty
    if not df_test.empty:
        # remove closed day and null
        df_test = df_test[(df_test['Open'] != 0) & (df_test['Open'].notna())]
        # convert dataframe to json
        data = json.dumps(df_test.to_dict(orient='records'))
    else:
        data = 'error'

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


def parse_message(message):

    # get chat_id #
    chat_id = message['message']['chat']['id']
    # get store_id # and try to convert as int
    store_id = message['message']['text']
    store_id = store_id.strip().replace('/', '')
    try:
        store_id = int(store_id)
    except ValueError:
        store_id = 'error'

    return chat_id, store_id


# Initialize API
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        message = request.get_json()

        # calling function that extracts chat and store Ids
        chat_id, store_id = parse_message(message)

        # check if store_id is a valid store number or something else
        if store_id != 'error':
            
            # loading data
            data = load_dataset(store_id)
            # check if dataframe was loaded correctly
            if data != 'error':
                # prediction
                d1 = predict(data)
                # prediction for all sales in the next 7 weeks (by store)
                d2 = d1[['store', 'prediction']].groupby('store').sum().reset_index()
                # send message
                msg = f"The sales forecast for the next 7 weeks for Store No. {d2.loc[0, 'store']} is ${d2.loc[0, 'prediction']:,.2f}."
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
    # answer to a 'GET' message
    else:
        return '<h1> Sales Prediction Telegram Bot </h1>'


if __name__ == '__main__':

    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)
