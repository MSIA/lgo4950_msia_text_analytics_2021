import os
import pandas as pd
from src.product_evaluation import read_pretrained_embeddings,  return_n_nn_for_all_dimensions, \
                                    evaluate_paremeter_versus_dimension, \
                                    calculate_agg_dim_score_for_each_category, scale_agg_summary, \
                                    plot_sns_heatmap, plot_product_comparaison


if __name__ == '__main__':

    # Read in pretrained embeddings
    path_to_glove_pretrained = str(os.getcwd()) + '/data/glove.6B.100d.txt'
    embeddings_glove = read_pretrained_embeddings(path_to_glove_pretrained)
    glove_words = list(embeddings_glove.keys())
    print('- pretrained glove embeddings read in')

    # Read in logistic regression parameters summary
    parameters_summary = pd.read_csv('data/logistic_regression_parameters.csv')
    print('- logistic regression parameters summary read in')

    # Define dimensions for product evaluation
    dim_aesthetic = ['beautiful', 'ugly', 'appearance']
    dim_appeal_to_senses = ['sensory', 'smelly', 'aromatic']
    dim_price = ['money', 'expensive', 'affordable']
    dim_size = ['fit', 'bulky', 'weight']
    dim_punctuality = ['delivery', 'delay', 'timely']
    dim_entertainment = ['boring', 'fun', 'literature']
    dim_ease_of_use = ['demanding', 'straightforward', 'intuitive']
    dim_product_quality = ['functional', 'operational', 'defective']

    # Create list of all words used to define dimensions
    list_dimensions = [dim_aesthetic, dim_appeal_to_senses, dim_price, dim_size, dim_punctuality, dim_entertainment,
                       dim_ease_of_use, dim_product_quality]
    dimension_names = ['Aesthetic', 'Smell/Touch/Taste', 'Price', 'Fit / Size',
                       'Delivery', 'Entertainment', 'Ease of use', 'Performance']

    # Extract the top 5 synonyms for each word used to describe dimensions
    nn_for_dimensions = return_n_nn_for_all_dimensions(list_dimensions, embeddings_glove,dimension_names, 5)
    print('- Synonym of dimension descriptors identified, embeddings stored ')

    # Evaluate similarity of each logit parameter versus dimension descriptors
    results = evaluate_paremeter_versus_dimension(parameters_summary, nn_for_dimensions,
                                                  embeddings_glove, glove_words, dimension_names)
    results['category'] = list(map(lambda cat_word: cat_word.split('-')[0], results['category_word']))
    print('- All logit parameters evaluated (proximity to dimensions)')

    # List of all the categories considered
    unique_categories = list(results.category.value_counts().index)

    # Aggregate logit parameter similarity to dimension for each product category (one row per product type)
    results_agg = calculate_agg_dim_score_for_each_category(unique_categories, results, dimension_names)
    print('- Parameter proximity to dimension aggregated at product level (weighted by parameter importance')

    # Scale aggregate results
    results_agg_scaled = scale_agg_summary(results_agg, 1, dimension_names)
    print('+ Scaled aggregated proximity:')
    print(results_agg_scaled)

    # Evaluate relative importance
    relative_dimension_importance = pd.DataFrame()
    relative_dimension_importance['product_category'] = results_agg['product_category']
    for i in dimension_names:
        relative_dimension_importance[i] = results_agg[i].rank() / 6

    dimension_importance_with_index = relative_dimension_importance
    dimension_importance_with_index = dimension_importance_with_index.set_index('product_category')
    dimension_importance_with_index.to_csv('data/relative_dimension_importance.csv')

    # Visualize heatmap
    print('+ Heatmap created')
    plot_sns_heatmap(dimension_importance_with_index)

    # Compare 2 product categories
    print('+ Product comparison created')
    plot_product_comparaison('appliances', 'musical_instruments', dimension_names,
                             relative_dimension_importance)


