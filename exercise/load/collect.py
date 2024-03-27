import csv
import json
import os
import urllib.parse
import urllib.request
from prefect import task, flow

DATALIMIT = 100

@task
def generate_query_params(token, month, limit):
    params = {
        "$$app_token": token,
        "$where": f"tpep_pickup_datetime between '2021-{month:02d}-01T00:00:00' and '2021-{(month+1):02d}-01T00:00:00'",
        "$limit": limit
    }
    return params

@task
def build_query_url(url, params):
    query = urllib.parse.urlencode(params)
    url = f"{url}?{query}"
    return url 

@task(retries=4, retry_delay_seconds=2)
def load_data(url):
    response = urllib.request.urlopen(url)
    data = json.loads(response.read())
    return data

@task
def save_data(data, filename):
    fields = data[0].keys()
    with open(filename, 'w', newline='') as file:
        csv_writer = csv.DictWriter(file, fields)
        csv_writer.writeheader()
        csv_writer.writerows(data)

@flow
def collect_flow(data_path: str):
    os.makedirs(data_path, exist_ok=True)

    url = "https://data.cityofnewyork.us/resource/m6nq-qud6.json"

    paramsJan = generate_query_params("4DEF3tlQcOGifuw7lGrIDfhPd", 1, DATALIMIT)
    paramsFeb = generate_query_params("4DEF3tlQcOGifuw7lGrIDfhPd", 2, DATALIMIT)
    paramsMar = generate_query_params("4DEF3tlQcOGifuw7lGrIDfhPd", 3, DATALIMIT)

    urlJan = build_query_url(url, paramsJan)
    urlFeb = build_query_url(url, paramsFeb)
    urlMar = build_query_url(url, paramsMar)

    dataJan = load_data(urlJan)
    dataFeb = load_data(urlFeb)
    dataMar = load_data(urlMar)

    save_data(dataJan, os.path.join(data_path, "yellow-2021-01.csv"))
    save_data(dataFeb, os.path.join(data_path, "yellow-2021-02.csv"))
    save_data(dataMar, os.path.join(data_path, "yellow-2021-03.csv"))