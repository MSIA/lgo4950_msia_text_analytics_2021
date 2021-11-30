import pytest
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.product_evaluation import calculate_agg_dim_score_for_each_category, scale_agg_summary


def test_calculate_agg_dimension_score():

    # Define input dataframe
    df_in_values = [['musical_instruments-perfect', 10.869000000000002,
                     10.869000000000002, 2.181380140101074, 1.616320636959409,
                     1.693966503429828, 1.9564571522663272, 1.815773892838748,
                     1.9086077429641157, 1.9977481928306506, 1.7193671215190072,
                     'musical_instruments'],
                    ['musical_instruments-great', 10.640999999999998,
                     10.640999999999998, 1.9058835931444105, 1.4011808147571039,
                     1.5964913804070477, 1.6900641517889303, 1.6037269036333213,
                     1.7753020177511047, 1.6911169285561887, 1.5649209708798548,
                     'musical_instruments'],
                    ['musical_instruments-excellent', 8.986, 8.986, 1.639052129779708,
                     1.3129634297047545, 1.483099704001912, 1.5616729518716823,
                     1.513044102475299, 1.548986687389005, 1.6034993857438458,
                     1.4732354063774653, 'musical_instruments'],
                    ['musical_instruments-perfectly', 8.981, 8.981,
                     1.6233646677909717, 1.3402487877965796, 1.3683734046121274,
                     1.7544159529648478, 1.418641508668432, 1.500307430391178,
                     1.6185845994664163, 1.442323633676698, 'musical_instruments'],
                    ['musical_instruments-love', 8.919, 8.919, 1.6194436590508592,
                     1.2035604445734913, 1.2722217283780743, 1.3511253097269618,
                     1.2517608008135908, 1.533027601416216, 1.416458455635466,
                     1.2327437136810602, 'musical_instruments'],
                    ['musical_instruments-amazing', 8.37, 8.37, 1.7224299324553933,
                     1.2361559273646476, 1.2378168297071177, 1.4141366175953252,
                     1.2778679104326902, 1.5286590077104207, 1.4827021990073233,
                     1.250692233072802, 'musical_instruments'],
                    ['musical_instruments-loves', 7.992000000000001,
                     7.992000000000001, 1.4484632369287984, 1.138153854127192,
                     1.1613240695296176, 1.2698356970583093, 1.134488921112224,
                     1.3870121396521358, 1.2989605965658793, 1.1100506278452418,
                     'musical_instruments'],
                    ['musical_instruments-highly', 7.85, 7.85, 1.3395725439292432,
                     1.13956492692886, 1.3215151532118428, 1.298471533190037,
                     1.2806616409277805, 1.3007853896739858, 1.3667972007639042,
                     1.3227359609461196, 'musical_instruments'],
                    ['musical_instruments-complaints', 7.757000000000001,
                     7.757000000000001, 1.1693121668132376, 1.0534967046954236,
                     1.1435640874897517, 1.1317713227709807, 1.2300310518491802,
                     1.1119823863068936, 1.2394164957581446, 1.181794267965981,
                     'musical_instruments'],
                    ['musical_instruments-pleasantly', 7.746, 7.746,
                     1.3983303246165468, 1.3065731637614324, 1.1537563742437722,
                     1.307159539967219, 1.1982299146974782, 1.3337822417438632,
                     1.3858451930939906, 1.1722000465612457, 'musical_instruments']]

    df_in_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    df_in_columns = ['category_word', 'weight', 'abs_weight', 'Aesthetic',
                     'Smell/Touch/Taste', 'Price', 'Fit / Size', 'Delivery', 'Entertainment',
                     'Ease of use', 'Performance', 'category']

    df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)

    # Define expected output, df_true
    df_true = pd.DataFrame([['musical_instruments', 0.14053520802187347, 0.11164377267965665,
                             0.11763318619682893, 0.12904416975860966, 0.12019125786988352,
                             0.13073738488477632, 0.13224961712802352, 0.11796540346034795]],
                           index=[0],
                           columns=['product_category', 'Aesthetic', 'Smell/Touch/Taste', 'Price',
                                    'Fit / Size', 'Delivery', 'Entertainment', 'Ease of use',
                                    'Performance'])

    dim_names = ['musical_instruments']

    # Compute test output
    df_test = calculate_agg_dim_score_for_each_category(dim_names, df_in,
                                                        ['Aesthetic', 'Smell/Touch/Taste', 'Price', 'Fit / Size',
                                                         'Delivery', 'Entertainment', 'Ease of use', 'Performance'])

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_test, df_true)


def test_scale_agg_summaries():

    # Define input dataframe
    df_in_values = [['magazine_subscriptions', 0.644816464382572, -1.0,
                    0.8576593114993187, -1.0, 0.4665811642583009, 1.0, -1.0,
                    -0.6864324334459866],
                   ['gourmet_food', 1.0, 1.0, -1.0, -0.40459926056405493, -1.0,
                    0.5731726610440546, -0.6931210922939499, -1.0],
                   ['musical_instruments', -0.45218401253085005,
                    -0.11246077172447144, 0.308360214882029, 0.41890327771068314,
                    -0.036430504070210645, -0.5467680873447733, 0.49883214912074436,
                    0.6591169967788915],
                   ['fashion', 0.9440169045595894, -0.01587205079841425,
                    -0.09833699852519828, 1.0, -0.7948317833734251,
                    0.4699239777802404, -0.3124495893363246, -0.6356874864436435],
                   ['toys_and_games', 0.001322833584239902, 0.22303807859408664,
                    -0.30367226789152824, 0.28889005006851676, -0.43989277961895823,
                    0.45798817145110604, -0.45400436576153425, 0.050537410217188494],
                   ['appliances', -1.0, -0.7824475162275988, 1.0,
                    -0.06879036986680376, 1.0, -1.0, 1.0, 1.0]]

    df_in_index = [0, 1, 2, 3, 4, 5]

    df_in_columns = ['product_category', 'Aesthetic', 'Smell/Touch/Taste', 'Price',
                       'Fit / Size', 'Delivery', 'Entertainment', 'Ease of use',
                       'Performance']

    df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)

    # Define expected output, df_true
    df_true = pd.DataFrame([['magazine_subscriptions', 0.644816464382572, -1.0,
                            0.8576593114993187, -1.0, 0.4665811642583009, 1.0, -1.0,
                            -0.6864324334459866],
                           ['gourmet_food', 1.0, 1.0, -1.0, -0.40459926056405493, -1.0,
                            0.5731726610440546, -0.6931210922939499, -1.0],
                           ['musical_instruments', -0.45218401253085005,
                            -0.11246077172447144, 0.308360214882029, 0.41890327771068314,
                            -0.036430504070210645, -0.5467680873447733, 0.49883214912074436,
                            0.6591169967788915],
                           ['fashion', 0.9440169045595894, -0.01587205079841425,
                            -0.09833699852519828, 1.0, -0.7948317833734251,
                            0.4699239777802404, -0.3124495893363246, -0.6356874864436435],
                           ['toys_and_games', 0.001322833584239902, 0.22303807859408664,
                            -0.30367226789152824, 0.28889005006851676, -0.43989277961895823,
                            0.45798817145110604, -0.45400436576153425, 0.050537410217188494],
                           ['appliances', -1.0, -0.7824475162275988, 1.0,
                            -0.06879036986680376, 1.0, -1.0, 1.0, 1.0]],
                           index=[0,1,2,3,4,5],
                           columns=['product_category', 'Aesthetic', 'Smell/Touch/Taste', 'Price',
                                    'Fit / Size', 'Delivery', 'Entertainment', 'Ease of use',
                                    'Performance'])

    dim_names = ['musical_instruments']

    # Compute test output
    df_test = scale_agg_summary(df_in, 1, ['Aesthetic', 'Smell/Touch/Taste', 'Price','Fit / Size',
                                                     'Delivery', 'Entertainment', 'Ease of use','Performance'])

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_test, df_true)