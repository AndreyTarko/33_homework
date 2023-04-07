import dill
from pathlib import Path
import pandas as pd
import json
from datetime import datetime


def predict():
    with open('/Users/andrey/DataScience/airflow_hw/data/models/cars_pipe_202304071847.pkl', 'rb') as file:
        model = dill.load(file)

    dir = '/Users/andrey/DataScience/airflow_hw/data/test'
    predlist = Path(dir).glob('*.json')
    df_all_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for file in predlist:
        with open(file) as data:
            form = json.load(data)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df_pred = pd.DataFrame(x)
            df_all_pred = pd.concat([df_all_pred, df_pred], axis=0)

    df_all_pred.to_csv(
        f'/Users/andrey/DataScience/airflow_hw/data/predictions/cars_preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv')


if __name__ == '__main__':
    predict()
