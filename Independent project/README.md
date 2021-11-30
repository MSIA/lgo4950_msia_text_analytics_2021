# NLP Independent Project Submission

<!-- toc -->

## Assessing the importance of user-defined product dimensions in  reviews

Louis-Charles Généreux

Northwestern University, MS in Analytics:

<!-- toc -->

## Note for reader

The below file includes step-by-step instructions to running this framework and launching a 
Dockerized product for exploration.

Please note that some data files such as GLOVE embeddings and raw Amazon reviews (too large to be hosted on github) 
should be added to the data folder prior to running.

Some explanatory files (including a full report and literature review notes) 
are available in the ``` deliverables``` folder

## Structure of repository

```
├── README.md                         <- You are here
├── app
│   ├── static/                       <- PNG and GIF files that remain static
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs
│   ├── boot.sh                       <- Start up script for launching app in Docker container.
│   ├── Dockerfile_app                <- Dockerfile for building image to run app  
│
├── config                            <- Directory for configuration files 
│   ├── flaskconfig.py                <- Configurations for Flask API 
│
├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git. 
│   ├── Amazon reviews                <- External data sources
│   ├── intermediate                  <- Artifacts from model training
│
├── deliverables/                     <- Final presentation, writeup, literature review
│
├── notebooks /                       <- Jupyter notebooks (including LDA details)
│
├── src/                              <- Source code for the project 
│
├── test/                             <- Files necessary for running model tests (see documentation below) 
│
├── app.py                            <- Flask wrapper for running the model 
├── run_logit.py                      <- Simplifies the execution of model training src scripts  
├── run_topic_models.py               <- Simplifies the execution of topic modeling src scripts  
├── requirements.txt                  <- Python package dependencies 
├── Makefile                          <- Aggregating commands
```

## Additional details 

All source code is stored in the ‘src’ folder. 

The execution of commands occurs from the root of the directory in the ``` run_logit.py```  and 
```run_topic_model.py```  files. 

All commands (including creation of a Docker instance to run a Flask app) have been aggregated to simple commands in a ```Makefile```.

To ensure reproducibility, tests can be run from the root directory using the ``` python -m pytest command```

## Running the project pipeline

Ensure that you are located at the __root of the project directory__ when 
running ```make``` commands.

#### Reading in review data and creating logistic regression classifiers

These classifiers are used to extract important parameters in determining whether 
product reviews are positive or negative, across different categories. 

``` make train_models ```

#### Evaluating features against user defined dimensions

The following command estimates the importance of various dimension in determining
users' experience across different product categories

``` make topic_models ```

#### Creating a Docker instance to run a Flask app

``` make app_image ```

#### Launching the app within docker

``` make run_app ```

The app can be accessed from http://127.0.0.1:5000/

Users of the app are prompted to specify 2 product categories, which they want to compare side by side.
The resulting 'radar chart' allows user to compare the relative importance of different product 
dimensions for users

#### Killing the app

```make kill-app```
