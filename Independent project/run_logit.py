import os
import pandas as pd
from src.logit import generate_results_for_category


if __name__ == '__main__':

    # Data folder path
    data_path = str(os.getcwd()) +'/data'

    # Read in data, preprocess and run classification models
    results_musical_instruments = generate_results_for_category('musical_instruments',
                                                                data_path, '/Musical_Instruments.json')
    results_magazine_subscriptions = generate_results_for_category('magazine_subscriptions',
                                                                   data_path, '/Magazine_Subscriptions.json')
    results_fashion = generate_results_for_category('fashion', data_path, '/AMAZON_FASHION.json')
    results_appliances = generate_results_for_category('appliances', data_path, '/Appliances.json')
    results_grocery_and_food = generate_results_for_category('gourmet_food',
                                                             data_path, '/Grocery_and_Gourmet_Food_sub.json')
    results_toys_and_games = generate_results_for_category('toys_and_games',
                                                             data_path, '/Toys_and_Games_sub.json')

    # Create list of all results
    results_df_list = [results_musical_instruments, results_magazine_subscriptions, results_fashion,
                       results_appliances, results_grocery_and_food, results_toys_and_games]

    # Create df will all results
    final_results = pd.DataFrame()
    for dataset in results_df_list:
        final_results = final_results.append(dataset)

    # Saving to CSV
    final_results.to_csv('data/logistic_regression_parameters.csv')
    print('- Results stored')