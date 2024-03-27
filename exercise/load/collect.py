import urllib.parse
import urllib.request
import json
import csv 
import os
from prefect import flow, task


MAXDATASIZE = 100

def generate_params(token, month, limit):
    params = {'$$app_token' : token,
            '$where' : f"tpep_pickup_datetime between '2021-{month:02d}-01T00:00:00' and '2021-{(month+1):02d}-01T00:00:00'",
            '$limit' : limit }
    return params

@task
def build_url(url, params):
    query_string = urllib.parse.urlencode(params)
    url = f"{url}?{query_string}"
    return url


@task
def load_data(url):
    response = urllib.request.urlopen(url) 
    
    data_json = json.loads(response.read()) 

    return data_json


@task
def save_data(data, filename):
    os.makedirs('data', exist_ok=True)
    fields = data[0].keys()
    with open(filename, 'w', newline='') as file: 
        writer = csv.DictWriter(file, fieldnames = fields)
        writer.writeheader() 
        writer.writerows(data)

@flow
def collect_flow(raw_data_path: str):

    # build url
    url = 'https://data.cityofnewyork.us/resource/m6nq-qud6.json'

    paramsJan = generate_params('4DEF3tlQcOGifuw7lGrIDfhPd', 1, MAXDATASIZE)
    paramsFeb = generate_params('4DEF3tlQcOGifuw7lGrIDfhPd', 2, MAXDATASIZE)
    paramsMar = generate_params('4DEF3tlQcOGifuw7lGrIDfhPd', 3, MAXDATASIZE)

    urlJan = build_url(url, paramsJan)
    urlFeb = build_url(url, paramsFeb)
    urlMar = build_url(url, paramsMar)

    # load data from url
    dataJan = load_data(urlJan)
    dataFeb = load_data(urlFeb)
    dataMar = load_data(urlMar)

    # save data
    save_data(dataJan, os.path.join(raw_data_path, f'yellow-2021-{1:02d}.csv'))
    save_data(dataFeb, os.path.join(raw_data_path, f'yellow-2021-{2:02d}.csv'))
    save_data(dataMar, os.path.join(raw_data_path, f'yellow-2021-{3:02d}.csv'))