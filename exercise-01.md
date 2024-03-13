# 💻 MLOps: Labo04 - Deploying a Web Service

## 🛠️ Assignment

*you can work alone or in pairs of 2. Just make sure to copy the changes to each team-members repository.*

 - Go to the [NYC OpenData website](https://opendata.cityofnewyork.us/data/)
 - Find the API endpoint for the Yellow Taxi Trip Data from 2022 **in csv format**.
 - Use ThunderClient/Postman to query the endpoint in order to retrieve only data from Januari
   - Look at [the SoQL documentation](https://dev.socrata.com/docs/endpoints) to find out how to query the data
 - *optional: setup a user account and create an app token, so the requests don't fail as often.*
 - setup a prefect and MLFlow server.
 - transform the Jupyter Notebook into Python scripts.
 - use prefect flows and tasks to setup a pipeline.
 - add a function to load the data from the API endpoint in CSV format (instead of reading a parquet file).
 - train a model using MLFlow (you can use the same training methods as were used in the Jupyter Notebook).
 - deploy the model as a web service.
 - create a Docker container and run it.
 - test your prediction endpoint.

Push your changes **and your end-result** to github.