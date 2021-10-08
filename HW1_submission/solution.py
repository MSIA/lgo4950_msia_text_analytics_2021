import re


# QUESTION 2.1
def find_function_names(code):
    """
    Takes python code and returns the name of functions when they are declared
    Args:
        code: a string of python code
    Returns:
        a list containing the names of functions that are declared in the code
    """
    names = re.findall(r'\bdef (.*?)\(', code)

    return names


# QUESTION 2.2
def find_dates(text):
    """
    Takes a text and extracts all dates from the text, irrespective of format, using regex
    Args:
        text: a string formatted text, containing dates
    Returns:
        a list containing the names dates
    """
    # Identifies dates with numbers only in the format day, month, year (separated by '/' or '.' or '/')
    numeric_dates = re.findall(r'\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}', text)

    # Identifies text dates with 'th' following day number
    text_dates_th = re.findall(r'\b\w*? \d{1,2}th \d{2,4}', text)

    # Identifies text dates without 'th' following day number
    text_dates = re.findall(r'\b\w*? \d{1,2} \d{2,4}', text)

    # Append all date formats into one list
    full_list = numeric_dates + text_dates_th + text_dates

    return full_list


# Testing Q2.1
print("Sample text:")
test_python_code = str('def foo(): \n\tprint("foo was called") \n\ndef bar(x): '
                       '\n\tprint("bar was called with an argument %s"%x)')
print(test_python_code)
print("\nOutput: ")
print(find_function_names(test_python_code))

print('\n------------------------------------------------------')

# Testing Q2.2
print("Sample text:")
test_python_dates = str('Random text here 04/12/2019 and here.'
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

print(test_python_dates)
print("\nOutput: ")
print(find_dates(test_python_dates))

print('\n------------------------------------------------------')
