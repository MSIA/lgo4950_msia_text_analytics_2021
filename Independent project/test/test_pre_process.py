import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from src.logit import prepare_text, preprocess_no_shuffle


def test_prepare_text():
    # Define input dataframe
    df_in_values = ['Bought this for my wife and she is very pleased with it. Great for the beginner or as a refresher '
                    'course for anyone interested in bettering their skills.',
                    'these is my second one.  it just easy and understandable pattern instruction.  it is also written in plain '
                    'English.  I gave it to my sister in law, so she could join the yarn world',
                    'Good condition.  Very helpful book.',
                    "still can't crochet to save my life",
                    'Great way to learn to Crocheting']

    df_in_index = [0, 1, 2, 3, 4]

    df_in_columns = ['reviewText']

    df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)

    # Define expected output, df_true
    df_true = pd.DataFrame([[
                                '  bought this for my wife and she is very pleased with it  great for the beginner or as a refresher course for anyone interested in bettering their skills   '],
                            [
                                '  these is my second one  it just easy and understandable pattern instruction  it is also written in plain english  i gave it to my sister in law  so she could join the yarn world  '],
                            ['  good condition  very helpful book   '],
                            ['  still cannot crochet to save my life  '],
                            ['  great way to learn to crocheting  ']], index=[0, 1, 2, 3, 4], columns=['result'])

    # Compute test output
    df_test = pd.DataFrame(list(map(prepare_text, df_in.values)), index=[0, 1, 2, 3, 4], columns=['result'])

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_test, df_true)


def test_preprocess():

    # Define input dataframe
    df_in_values = [[5.0, '90', False, '08 9, 2004', 'AXHY24HWOF184', '0470536454',
        {'Format:': ' Paperback'}, 'Bendy',
        'Crocheting for Dummies by Karen Manthey & Susan Brittain is a wonderfully thorough and very informative book '
        'for anyone wanting to learn to crochet and or wanting to freshen up their skills.\n\nThe book reads like a '
        'storybook in paragraph form.  Everything is explained in great detail from choosing yarns and hooks, to how '
        'to work a large array of crochet stitches, to how to read a pattern, right down to how to care for ones '
        'crocheted items.\n\nThe stitch drawings are clear and expertly done making learning new stitches so much '
        'easier.\n\nThe book has both a contents page and an index for easy referral.  I especially liked the fact '
        'that an index was included.  So many crochet books do not include this.  The index makes it very easy to '
        'find information on a particular topic quickly.\n\nThe recommendations for people just learning to crochet '
        'are fantastic.  This book wasn\'t out when I learned to crochet and I learned the hard way about many of the'
        ' pit falls this book helps one to avoid.  For instance they recommend one start out with a size H-8 crochet '
        'hook and a light colored worsted weight yarn.  I learned with a B-1 hook and a fingering weight yarn.  After '
        '2 whole days of crocheting it was 36" long and 1.5" tall.  I was trying to make a baby blanket for my doll '
        '(which never got made).\n\nThe book contains humor, not just in the cartoons but in the instructions as '
        'well which makes for very entertaining reading while one learns a new craft.  I always appreciate having '
        'a teacher with a sense of humor!\n\nA good sampling of designs is included so that one can try out their '
        'skills.  These include sweaters, an afghan, doilies, hot pads, pillow, scarves, floral motifs, and '
        'bandanas.\n\nI am a crochet designer and I read the book cover to cover like a storybook while on '
        'vacation this past week.  I thoroughly enjoyed it and learned a few things as well.  I would highly '
        'recommend this book to anyone interested in the art of crochet.',
        'Terrific Book for Learning the Art of Crochet', 1092009600, np.nan],
       [4.0, '2', True, '04 6, 2017', 'A29OWR79AM796H', '0470536454',
        {'Format:': ' Hardcover'}, 'Amazon Customer', 'Very helpful...',
        'Four Stars', 1491436800, np.nan],
       [5.0, np.nan, True, '03 14, 2017', 'AUPWU27A7X5F6', '0470536454',
        {'Format:': ' Paperback'}, 'Amazon Customer',
        'EASY TO UNDERSTAND AND A PROMPT SERVICE TOO', 'Five Stars',
        1489449600, np.nan],
       [4.0, np.nan, True, '02 14, 2017', 'A1N69A47D4JO6K', '0470536454',
        {'Format:': ' Paperback'}, 'Christopher Burnett',
        'My girlfriend use quite often', 'Four Stars', 1487030400, np.nan],
       [5.0, np.nan, True, '01 29, 2017', 'AHTIQUMVCGBFJ', '0470536454',
        {'Format:': ' Paperback'}, 'Amazon Customer',
        'Arrived as described. Very happy.', 'Very happy.', 1485648000,
        np.nan]]

    df_in_index = [0, 1, 2, 3, 4]

    df_in_columns = ['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin',
                     'style', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'image']

    df_in = pd.DataFrame(df_in_values, index=df_in_index, columns=df_in_columns)

    # Define expected output, df_true
    df_true = pd.DataFrame([[5,
        'Crocheting for Dummies by Karen Manthey & Susan Brittain is a wonderfully thorough and very informative book '
        'for anyone wanting to learn to crochet and or wanting to freshen up their skills.\n\nThe book reads like a '
        'storybook in paragraph form.  Everything is explained in great detail from choosing yarns and hooks, to how to '
        'work a large array of crochet stitches, to how to read a pattern, right down to how to care for ones crocheted '
        'items.\n\nThe stitch drawings are clear and expertly done making learning new stitches so much easier.\n\nThe '
        'book has both a contents page and an index for easy referral.  I especially liked the fact that an index was '
        'included.  So many crochet books do not include this.  The index makes it very easy to find information on a '
        'particular topic quickly.\n\nThe recommendations for people just learning to crochet are fantastic.  This book'
        ' wasn\'t out when I learned to crochet and I learned the hard way about many of the pit falls this book helps '
        'one to avoid.  For instance they recommend one start out with a size H-8 crochet hook and a light colored '
        'worsted weight yarn.  I learned with a B-1 hook and a fingering weight yarn.  After 2 whole days of crocheting'
        ' it was 36" long and 1.5" tall.  I was trying to make a baby blanket for my doll (which never got made).\n\nThe '
        'book contains humor, not just in the cartoons but in the instructions as well which makes for very entertaining '
        'reading while one learns a new craft.  I always appreciate having a teacher with a sense of humor!\n\nA good '
        'sampling of designs is included so that one can try out their skills.  These include sweaters, an afghan, '
        'doilies, hot pads, pillow, scarves, floral motifs, and bandanas.\n\nI am a crochet designer and I read the '
        'book cover to cover like a storybook while on vacation this past week.  I thoroughly enjoyed it and learned a '
        'few things as well.  I would highly recommend this book to anyone interested in the art of crochet.',
        'crocheting for dummies by karen manthey   susan brittain is a wonderfully thorough and very informative book '
        'for anyone wanting to learn to crochet and or wanting to freshen up their skills  the book reads like a '
        'storybook in paragraph form  everything is explained in great detail from choosing yarns and hooks  to how to '
        'work a large array of crochet stitches  to how to read a pattern  right down to how to care for ones crocheted '
        'items  the stitch drawings are clear and expertly done making learning new stitches so much easier  the book '
        'has both a contents page and an index for easy referral  i especially liked the fact that an index was included '
        ' so many crochet books do not include this  the index makes it very easy to find information on a particular '
        'topic quickly  the recommendations for people just learning to crochet are fantastic  this book was not out '
        'when i learned to crochet and i learned the hard way about many of the pit falls this book helps one to avoid'
        '  for instance they recommend one start out with a size h  crochet hook and a light colored worsted weight yarn '
        ' i learned with a b  hook and a fingering weight yarn  after  whole days of crocheting it was   long and    '
        'tall  i was trying to make a baby blanket for my doll  which never got made   the book contains humor  not '
        'just in the cartoons but in the instructions as well which makes for very entertaining reading while one learns '
        'a new craft  i always appreciate having a teacher with a sense of humor  a good sampling of designs is included '
        'so that one can try out their skills  these include sweaters  an afghan  doilies  hot pads  pillow  scarves  '
        'floral motifs  and bandanas  i am a crochet designer and i read the book cover to cover like a storybook while '
        'on vacation this past week  i thoroughly enjoyed it and learned a few things as well  i would highly recommend '
        'this book to anyone interested in the art of crochet '],
       [5, 'Very helpful...', 'very helpful   '],
       [5, 'EASY TO UNDERSTAND AND A PROMPT SERVICE TOO',
        'easy to understand and a prompt service too'],
       [5, 'My girlfriend use quite often',
        'my girlfriend use quite often'],
       [5, 'Arrived as described. Very happy.',
        'arrived as described  very happy ']],
        index=[0, 1, 2, 3, 4],
        columns=['overall', 'reviewText', 'clean_text'])

    # Compute test output
    df_test = preprocess_no_shuffle(df_in)

    # Test that the true and test are the same
    pd.testing.assert_frame_equal(df_test, df_true)