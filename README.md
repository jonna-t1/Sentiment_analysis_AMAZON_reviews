# TITLE

The purpose of the project was to gain business intelligence and bolster profit margins through intelligence gained from large scale opinion mining also known as sentiment analysis. To solve a growing need to understand the increasing demands that “big data” trends present. The client uses sentiment analysis models on a regular basis and would like to gain actionable insights regarding data trends. Visualisations are presented to gather information quickly and efficiently on sentiment classification, these are portable visualisations represented in a web application format. In addition, the calibrated models will perform sentiment analysis without labelled data.

The project uses the Sci-kit learn machine learning library to address the research question: what the best performing machine learning classifier and evaluating it with the best suitable method. Experiments with different classifiers identified a Stochastic Gradient Descent iterative learner that regularises the best performing linear model which was Logistic Regression as the most suitable model for sentiment analysis, in agreement with literature. The project is then implemented within a traditional software setting – a web application. 

## Skills gathered

Many skills were adopted on the project some of which encompass; Literature review skills, Sci-learn, Pandas data structures, metrics to validate a classifier, Machine learning technicalities and vocabulary, Heuristic evaluations, Iterative learning model application architecture, Automation skills, Unit/integration testing, Database design, the Django web framework, python data structures, Pagination with search features, JavaScript and Chartjs.

The final project relied on drawing together concepts from machine learning, natural language processing and applying them to a software product. The results and outcomes of the experiment and were a highly accurate, precise machine learning classifier that enables simple and interpretable visualisations of results in a user-friendly display validated through client feedback, heuristic evaluations and user questionnaires. Future investigation would include real time feedback on sentiment this would cascade change to the training architecture. Deep Neural Network investigations for model improvement would also be considered.

## Requirements

python3 -m pip install --upgrade pip

Recommended: python3 -m venv /path/to/new/virtual/environment, source env/bin/active

pip install -r requirements.txt --upgrade

=== UBUNTU / DEBIAN ===</br>
sudo apt-get install python3-tk

## Data 

Data can be downloaded from https://nijianmo.github.io/amazon/index.html 

### Data pre-processing 

Run the `pp.sh` script

### Postgres Database setup

#### Linux setup:

- sudo apt update

- sudo apt install postgresql postgresql-contrib

- download pgadmin4 - https://www.pgadmin.org/download/pgadmin-4-python/


- pip install psycopg2

### Change configuration files

- Change the following config files; pg_hba.conf, postgresql.conf.

- To find; 

- `$ psql -U postgres` </br>
`postgres=# SHOW config_file;`

- Alternatively:

`$(ls /etc/postgresql/*/main/pg_hba.conf)`
`$(ls /etc/postgresql/*/main/postgresql.conf)`

### May need to change file permissions to the following...  

- postgresql.conf: add this line listen_addresses = '*'
- In **pg_hba.conf** - first line should go:</br>
#Database administrative </br>
host all postgres 127.0.0.1/32 trust

- Also add this below </br>
#TYPE DATABASE USER CIDR-ADDRESS  METHOD</br>
host  all  all 0.0.0.0/0 md5


### Starting the postgresql database

- `sudo -u postgres -i` ## log into postgres user

- `postgres=# systemctl start postgresql`

- And `postgres=# systemctl stop postgresql` </br> `postgres=# systemctl restart postgresql`. To stop and restart the database.

- Start pgadmin and go to the ip address to create database with the config settings defined in the config dictionary found in ./DBFuncs/dbConfig.py or configure your own!

## Getting the data into the database

### Create the database table
- Run the `python3 create_database.py` found in the root directory.

### Upload reviews into db

- Run `python3 review_upload.py` found in the root directory. Reviews should now be in the db.

## Training the classifier

The algorithms used are based from a thorough Literature review.

- Run `classifier_main.py` found in ./app/ directory. This saves a model to file/database which can be used for iterative learning.</br>

## Running the web application

- To run the web application: python3 SentimentApp/manage.py runserver