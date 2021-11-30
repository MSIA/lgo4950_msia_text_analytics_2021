import plotly.graph_objects as go
import random
from datetime import datetime


def plot_product_comparaison_png(category_1, category_2, names_of_dimensions, scaled_agg_results):
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

    random_number = random.randint(1, 100)
    day = datetime.now().day
    hour = datetime.now().hour
    minute = datetime.now().minute
    second = datetime.now().second

    path = 'app/static/user_generated_image_' + str(day) + '_' + str(hour)\
            + '_' + str(minute) + '_' + str(second) + str('.png')
    #path= 'app/static/user_generated_image_final.png'
    fig.write_image(path, width=800, height=800, scale=2)
    return path
