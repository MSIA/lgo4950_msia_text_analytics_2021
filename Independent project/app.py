import traceback
import pandas as pd
import time
import os
import kaleido
import logging.config
from flask import Flask
from flask import render_template, request
from src.visualization import plot_product_comparaison_png


app = Flask(__name__)

# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates", static_folder="app/static")

# Read in relative dimension importance file in data
relative_dimension_importance = pd.read_csv('data/relative_dimension_importance.csv')
dimension_names = list(relative_dimension_importance.columns[1:])


@app.route('/')
def form():
    return render_template('form.html')


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':

        # dict name needs to be the same as the one used in index and form
        user_input_1 = request.form.to_dict()['product_name_1']
        user_input_1 = str(user_input_1)

        user_input_2 = request.form.to_dict()['product_name_2']
        user_input_2 = str(user_input_2)

        product_list = ['magazine_subscriptions', 'fashion', 'musical_instruments', 'gourmet_food',
                        'toys_and_games', 'appliances']

        if (user_input_1 in product_list) & (user_input_2 in product_list):

            # Create picture and store path
            path_str = plot_product_comparaison_png(user_input_1, user_input_2,
                                                    dimension_names, relative_dimension_importance)
            path_str = str(path_str)
            path_str = path_str.split('app/static/')[1] # keep only picture specific details

            # Wait till image is in folder
            current_files_list = list(os.listdir(str(os.getcwd()) + '/app/static'))

            time_to_wait = 60
            time_counter = 0
            while path_str not in current_files_list:
                time.sleep(2)
                time_counter += 2
                current_files_list = list(os.listdir(str(os.getcwd()) + '/app/static'))
                if time_counter > time_to_wait:
                    break

            if path_str in current_files_list:
                full_path_str = str('static/'+path_str)
                return render_template('success_new.html', full_path_str=full_path_str,
                                       product_one=user_input_1, product_two=user_input_2)
            else:
                traceback.print_exc()
                return render_template('error.html')
        else:
            traceback.print_exc()
            return render_template('error.html')


@app.route('/return_to_home')
def return_opening_page():
    return render_template('form.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
