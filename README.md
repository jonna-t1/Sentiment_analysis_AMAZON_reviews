# Purpose

The purpose of the project was to gain business intelligence and bolster profit margins through intelligence gained from large scale opinion mining also known as sentiment analysis. To solve a growing need to understand the increasing demands that “big data” trends present. The client uses sentiment analysis models on a regular basis and would like to gain actionable insights regarding data trends. Visualisations are presented to gather information quickly and efficiently on sentiment classification, these are portable visualisations represented in a web application format. In addition, the calibrated models will perform sentiment analysis without labelled data.

The project uses the Sci-kit learn machine learning library to address the research question: what the best performing machine learning classifier and evaluating it with the best suitable method. Experiments with different classifiers identified a Stochastic Gradient Descent iterative learner that regularises the best performing linear model which was Logistic Regression as the most suitable model for sentiment analysis, in agreement with literature. The project is then implemented within a traditional software setting – a web application. 

## Skills gathered

Many skills were adopted on the project some of which encompass; Literature review skills, Sci-learn, Pandas data structures, metrics to validate a classifier, Machine learning technicalities and vocabulary, Heuristic evaluations, Iterative learning model application architecture, Automation skills, Unit/integration testing, Database design, the Django web framework, python data structures, Pagination with search features, JavaScript and Chartjs.

The final project relied on drawing together concepts from machine learning, natural language processing and applying them to a software product. The results and outcomes of the experiment and were a highly accurate, precise machine learning classifier that enables simple and interpretable visualisations of results in a user-friendly display validated through client feedback, heuristic evaluations and user questionnaires. Future investigation would include real time feedback on sentiment this would cascade change to the training architecture. Deep Neural Network investigations for model improvement would also be considered.

## Requirements (Windows)
* Python 3.12.3 
* Git bash (preferred) / Cygwin / MSYS/MinGW / WSL Ubuntu.
* `python -m venv venv`
* `venv\Scripts\activate`
* `python -m pip install --upgrade pip`
* `pip install --upgrade pip setuptools wheel`
* `pip install -r requirements.txt --upgrade`

## Data 

* Data can be downloaded from https://nijianmo.github.io/amazon/index.html
* Scroll down to "Small" subsets for experimentation. I chose Electronics	5-core (6,739,590 reviews)
* cd Sentiment_analysis_AMAZON_reviews
* Create directory DATA/

### Data pre-processing steps (Windows/Unix)

* Install Git bash/Cygwin/MSYS/MinGW/WSL Ubuntu
* From top level run `./reduce_file_size.sh` bash script. To generate more manageable files for ML processing.

## set up admin user
* `cd SentimentApp/`
* `python manage.py createsuperuser {your name}`

## Running the web application

- To run the web application: python SentimentApp/manage.py runserver