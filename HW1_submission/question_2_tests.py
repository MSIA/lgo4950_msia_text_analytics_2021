import pytest
import pandas as pd

from question_2 import find_function_names, find_dates


def test_function_names():
    # Define input
    string_in = str('def foo(): \n\tprint("foo was called") \n\ndef bar(x): '
                    '\n\tprint("bar was called with an argument %s"%x)')

    # Define expected output, df_true
    true_out = ['foo', 'bar']

    # Compute test output
    test_out = find_function_names(string_in)

    # Test that the true and test are the same
    assert test_out == true_out


def test_add_log_feature():
    # Define input
    string_in = str('Random text here 04/12/2019 and here.'
                    ' Hello...  qwerty: 04/12/19! '
                    ' Random text here 04-12-2019'
                    ' ... 04-12-19...'
                    ' Random text here 04.12.2019!'
                    ' Random text here 04.12.19'
                    ' April 12 2019 is a great date!'
                    ' So is april 12 2019!'
                    ' Another sentence including April 12th 2019 and april 12th 2019, along with...'
                    ' April 12 19. This is starting to be a long list, but is almost over:'
                    ' april 12 19, April 12th 19 and april 12th 19 are finally the last of 14 dates '
                    ' that I have tried')

    # Define expected output, df_true
    true_out = ['04/12/2019', '04/12/19', '04-12-2019', '04-12-19', '04.12.2019', '04.12.19',
                'April 12th 2019', 'april 12th 2019', 'April 12th 19', 'april 12th 19', 'April 12 2019',
                'april 12 2019', 'April 12 19', 'april 12 19']

    # Compute test output
    test_out = find_dates(string_in)

    # Test that the true and test are the same
    assert test_out == true_out
