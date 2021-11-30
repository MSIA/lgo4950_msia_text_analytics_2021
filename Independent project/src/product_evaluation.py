import numpy as np
import pandas as pd
import seaborn as sns
import dataframe_image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.graph_objects as go

from scipy import spatial
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings("ignore")


def read_pretrained_embeddings(path):
    ''' Takes file path to embeddings json file, returns full embeddings in np array format'''

    embeddings_glove = {}
    # with open(path) as f:
    with open(path, encoding="utf-8") as f:

        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_glove[word] = coefs

    return embeddings_glove


def euclidean_distance_word_vectors(vector_1, vector_2):
    ''' Takes 2 word embedding vectors, returns their euclidean distance'''
    distance = spatial.distance.euclidean(vector_1, vector_2)

    return distance


def find_closest_n_embeddings(target_word, full_embeddings, number_nn):
    '''
    Takes word and embedding matrix, and produces top n nearest words to a chosen word, acroos all embeddings
    Inputs:
        target_word (str): Word for which we want to find nearest neighbors
        full_embeddings (np matrix): matrix of vector encodings for all words
        number_nn (int): number of nearest neighbors
    Returns:
        list of top n words (string)
    '''

    sorted_list = sorted(full_embeddings.keys(),
                         key=lambda iteration_word: euclidean_distance_word_vectors(full_embeddings[iteration_word],
                                                                                    full_embeddings[target_word]))
    top_n_words = sorted_list[0:number_nn]
    return top_n_words


def return_n_nn_for_all_dimensions(list_of_list_dimensions, full_embeddings, list_dimension_names, number_nn):
    '''
    Takes in list of words describing each dimension, returns pd dataframe listing n neareast neighbors for each word
    Inputs:
        list_of_list_dimensions (list of list): list of strings describing each dimension
        full_embeddings (np matrix): matrix of vector encodings for all words
        list_dimension_names (list of strings): name of each dimension
        number_nn (int): number of nearest neighbors
    Returns:
        pd dataframe containing the nearest neighbor to each word describing dimensions
    '''

    # Create list of all dimension nearest neighbor words
    dimension_list = []
    word_list = []
    nn_list = []
    n_dim = 0

    for dimension in list_of_list_dimensions:
        words_in_dimension = []
        for word in dimension:
            nn_all = find_closest_n_embeddings(word, full_embeddings, number_nn)
            for iter_nn in nn_all:
                # Check that nn word is not already among nearest neighbors for this dimension
                if (iter_nn not in words_in_dimension):
                    words_in_dimension.append(iter_nn)
                    dimension_list.append(list_dimension_names[n_dim])
                    word_list.append(word)
                    nn_list.append(iter_nn)
        n_dim = n_dim + 1

    summary_nn_words = pd.DataFrame({'dimension_list': dimension_list,
                                     'word_list': word_list,
                                     'nn_list': nn_list})
    return summary_nn_words


def min_dist_from_dimension(word, dimension_summary, full_embeddings):
    '''
    Takes in a word, a pd object storing the nearest neighbors for each dimension, and a matrix object of embeddings
    Inputs:
        word (str): word which will be scored
        dimension_summary (pd df): contains one row per unique dimension, word describing dimension and nn for word
        full_embeddings (np ): matrix of vector encodings for all words
    Returns:
        pd dataframe containing 1 row per dimension, returning the distance of the dimension to the word
    '''

    current_dimension = dimension_summary.dimension_list[0]
    compared_word_index = -1

    dimension_list = []  # to store the dimension name
    dimension_distance_running_list = []  # to store the distance to each word used for a dimension
    dimension_mean_list = []  # to store the mean distance for a dimension

    for compared_word in dimension_summary.nn_list:

        # For every new word that is compared to a feature, increase the index
        compared_word_index = compared_word_index + 1

        # If the feature tests for the existing dimension, calculate distance, then append to running dist list
        if (current_dimension == dimension_summary.dimension_list[compared_word_index]):
            dist = euclidean_distance_word_vectors(full_embeddings[word], full_embeddings[compared_word])
            dimension_distance_running_list.append(dist)
        else:

            # Calculate mean for old list
            mean_past_dimension = np.mean(dimension_distance_running_list)

            # Store all details for former dimension
            dimension_list.append(current_dimension)
            dimension_mean_list.append(mean_past_dimension)

            # Change current dimension and re-initialize list
            current_dimension = dimension_summary.dimension_list[compared_word_index]
            dimension_distance_running_list = []

            # Append first value to list
            dist = euclidean_distance_word_vectors(full_embeddings[word], full_embeddings[compared_word])
            dimension_distance_running_list.append(dist)

    # Store all details for final former dimension
    mean_past_dimension = np.mean(dimension_distance_running_list)
    dimension_list.append(current_dimension)
    dimension_mean_list.append(mean_past_dimension)

    # Create summary, sorted from shorted distance to least
    summary_for_word = pd.DataFrame({'dimension': dimension_list,
                                     'mean_dist': dimension_mean_list})
    summary_for_word = summary_for_word.sort_values('mean_dist')

    return summary_for_word


def evaluate_paremeter_versus_dimension(summary_parameters, dimension_summary, full_embeddings,
                                        list_word_embeddings, name_of_dimensions):
    '''
    Evaluates the logit parameters and attributes their 'belonging' score to a dimension, weighted by parameter importance
    Inputs:
        summary_parameters (pd df): A pd df which contains the logit weight attached to each word parameter per category
        dimension_summary (pd df): contains one row per unique dimension, word describing dimension and nn for word
        full_embeddings (np matrix): matrix of vector encodings for all words
        list_word_embeddings (list of string): a list of all words contained in parameters
        name_of_dimensions (list of string): a list of the name of dimensions
    Returns:
        pd dataframe containing 1 row per parameter, with weighted
    '''

    # Create new pd dataframe
    new_df = pd.DataFrame()
    new_df['category_word'] = []
    new_df['weight'] = []
    new_df['abs_weight'] = []
    for i in name_of_dimensions:
        new_df[i] = []

    # Keep track of word index
    feature_number = 0

    # for each word in feature list
    for feature in summary_parameters.feature_name:

        # Test if feature appears in pretrained parameters
        if feature in list_word_embeddings:

            # Extract parameter specific info
            category = summary_parameters.category[feature_number]
            weight = summary_parameters.weight_num[feature_number]
            abs_weigth = abs(weight)

            # Store basic information in df
            new_df.at[feature_number, 'category_word'] = str(category) + ('-') + str(feature)
            new_df.at[feature_number, 'weight'] = weight
            new_df.at[feature_number, 'abs_weight'] = abs_weigth

            # Create summary of dimension scores for given feature
            dist_summary_dim = min_dist_from_dimension(str(feature), dimension_summary, full_embeddings)
            # print(dist_summary_dim)
            # For each row of distance summary
            for i in range(0, dist_summary_dim.shape[0]):
                # Extract key information
                dimension_name = dist_summary_dim.dimension[i]
                dimension_summary_dist = dist_summary_dim.mean_dist[i]
                # Avoid inf values when distance = 0
                dimension_summary_dist = np.where(dimension_summary_dist <= 0.001,
                                                  0.5 * np.mean(dist_summary_dim.mean_dist),
                                                  dimension_summary_dist)
                dimension_summary_simil_weighted = 1 / dimension_summary_dist * abs_weigth

                # Append similarity score
                new_df.at[feature_number, dimension_name] = dimension_summary_simil_weighted

        # Increase index when all info is stored for one number
        feature_number = feature_number + 1

    return new_df


def calculate_agg_dim_score_for_each_category(list_of_categories, parameter_dim_scores, name_of_dimensions):
    '''
    Calculates the aggregate similarity score of a category to each dimension
    Inputs:
        list_of_categories (list): list of category names
        parameter_dim_scores (pd df): contains one row per dimension * word and score against each dimension
        name_of_dimensions (list of string): a list of the name of dimensions
    Returns:
        pd dataframe containing 1 row per product category, with a score for each dimension (higher score = higher similarity )
    '''

    # Create dataframe to store results
    new_df = pd.DataFrame()
    new_df['product_category'] = []
    for i in name_of_dimensions:
        new_df[i] = []

    category_index = 0

    # Iterate through every category
    for category in list_of_categories:

        # Store category name in df
        new_df.at[category_index, 'product_category'] = category

        # Create subset of data for just one category by filtering for given category
        subset = parameter_dim_scores[parameter_dim_scores['category'] == category]

        # Select only columns with numeric values
        subset_num = subset[name_of_dimensions]

        # Calculate sum across each dimension / total (giving relative score summing to 100%)
        grouped_summary_sum = pd.DataFrame(subset_num.sum() / sum(subset_num.sum()))

        for i in range(0, grouped_summary_sum.shape[0]):
            # Write data into summary table
            new_df.at[category_index, grouped_summary_sum.index[i]] = grouped_summary_sum.values[i][0]

        category_index = category_index + 1

    return new_df


def scale_agg_summary(agg_results_df, min_max, names_of_dimensions):
    '''
    Scales summary df based on desired min and max values
    Inputs:
        agg_results_df (pd df): pd dataframe containing 1 row per product category, with a score for each dimension
        min_max (int): the absolute value of the min and max value
        names_of_dimensions (list): name of all dimensions that are measured
    Returns:
        Scaled df
    '''

    scaler = MinMaxScaler(feature_range=(-min_max, min_max))
    for dimension in names_of_dimensions:
        agg_results_df[dimension] = scaler.fit_transform(agg_results_df[[dimension]])

    return agg_results_df


def heatmap_product_dimensions(names_of_dimensions, scaled_agg_results):
    cm = sns.light_palette("green", as_cmap=True)
    results = scaled_agg_results.style.background_gradient(cmap=cm, subset=names_of_dimensions)

    return results


def heatmap_product_dimensions_png(names_of_dimensions, scaled_agg_results):
    cm = sns.light_palette("green", as_cmap=True)
    results = scaled_agg_results.style.background_gradient(cmap=cm, subset=names_of_dimensions)
    dataframe_image.export(results, "dimension_heatmap.png")

    img = mpimg.imread('dimension_heatmap.png')
    plt.rcParams['figure.figsize'] = [20, 10]
    imgplot = plt.imshow(img)

    return plt.show()


def plot_sns_heatmap(df):

    sns.set()

    fig, ax1 = matplotlib.pyplot.subplots(figsize=(20,10))
    ax2 = sns.heatmap(df,
                     #annot=True,
                     cmap="YlGnBu", cbar=True)
    ax2.set_xticklabels(ax2.get_xmajorticklabels(),
                        #fontsize = 12,
                        rotation=360)
    ax2.set(title='Impact of customer impact dimension in product reviews')
    ax2.set(xlabel='Customer experience dimension')
    ax2.set(ylabel='Product')

    sns.set(font_scale=1.3)
    # return ax2
    fig.data = df

    plt.show()


def plot_product_comparaison(category_1, category_2, names_of_dimensions, scaled_agg_results):
    ''' Takes in the name of 2 categories and returns plot comparing both categories on a set of dimensions'''

    fig = go.Figure()

    # Extract agg data for dimension
    category_1_data = scaled_agg_results[scaled_agg_results['product_category'] == category_1]
    category_2_data = scaled_agg_results[scaled_agg_results['product_category'] == category_2]

    # Plot for first product
    fig.add_trace(go.Scatterpolar(
        r=list(category_1_data[names_of_dimensions].values[0]),
        theta=names_of_dimensions,
        fill='toself',
        name=category_1
    ))

    # Plot for second product
    fig.add_trace(go.Scatterpolar(
        r=list(category_2_data[names_of_dimensions].values[0]),
        theta=names_of_dimensions,
        fill='toself',
        name=category_2
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    fig.show()