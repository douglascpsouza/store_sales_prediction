import os
import pandas as pd
import xgboost as xgb
from flask             import Flask, request, Response
from stores.StoreSales import StoreSales

# loading model
model = xgb.XGBRegressor()
model.load_model('model/xgb_model_tuned.json')

# initialize API
app = Flask(__name__)

@app.route('/sales/predict', methods=['POST'])
def sales_predict():
    test_json = request.get_json()
    
    # if not null
    if test_json:
        if isinstance(test_json, dict):
            # single row
            test_raw = pd.DataFrame(test_json, index=[0])
        else:
            # multiple rows
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        # instantiate StoreSales Class
        pipeline = StoreSales()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.get_prediction(model, test_raw, df3)
        
        return df_response
        
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    
    port = os.environ.get('PORT', 5000)
    
    app.run(host='0.0.0.0', port=port)
