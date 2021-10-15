import pytest
import warnings

warnings.filterwarnings("ignore")

from question_1 import preprocessing_corpus_2


def test_preprocessing():
    # Define input
    string_in = str('From: hm@cs.brown.edu (Harry Mamaysky)\nSubject: Heil Hernlem '
                    '\nIn-Reply-To: hernlem@chess.ncsu.edu\'s message of Wed, 14 Apr '
                    '1993 12:58:13 GMT\nOrganization: Dept. of Computer Science, Brown '
                    'University\nLines: 24\n\nIn article <1993Apr14.125813.21737@ncsu.edu> '
                    'hernlem@chess.ncsu.edu (Brad Hernlem) writes:\n\n   Lebanese resistance '
                    'forces detonated a bomb under an Israeli occupation\n   patrol in Lebanese '
                    'territory two days ago. Three soldiers were killed and\n   two wounded. In '
                    '"retaliation", Israeli and Israeli-backed forces wounded\n   8 civilians by '
                    'bombarding several Lebanese villages.')

    # Define expected output, df_true
    true_out = ['lebanese resistance forces detonated a bomb under an israeli occupation patrol in '
                'lebanese territory two days ago',
                'three soldiers were killed and two wounded',
                'in retaliation israeli and israelibacked forces wounded  civilians '
                'by bombarding several lebanese villages']

    # Compute test output
    test_out = preprocessing_corpus_2(string_in)

    # Test that the true and test are the same
    assert test_out == true_out
