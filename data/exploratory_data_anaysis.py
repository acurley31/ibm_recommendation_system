import os
import numpy
import pandas
from matplotlib import pyplot


# Constants
FILE_ARTICLES = './articles_community.csv'
FILE_INTERACTIONS = './user-item-interactions.csv'
EXPLORATORY_PLOTS_DIR = './plots/'

if not os.path.exists(EXPLORATORY_PLOTS_DIR):
    os.makedirs(EXPLORATORY_PLOTS_DIR)


def exploratory_analysis():
    '''Exploratory data analsysis of artcles/user-item interactions'''

    # Load in the data sets
    articles, interactions = load_data()

    # Interactions statistics
    user_interactions_statistics(interactions)

    # Articles statistics
    articles_statistics(articles, interactions)


def load_data():
    '''Return dataframes of the two data sets'''

    # Load the articles data set
    articles = pandas.read_csv(FILE_ARTICLES)
    articles.drop(columns=['Unnamed: 0'], inplace=True)

    # Load the user-article interactions data set
    interactions = pandas.read_csv(FILE_INTERACTIONS)
    interactions.drop(columns=['Unnamed: 0'], inplace=True)
    interactions.article_id = interactions.article_id.astype(int)

    email_id = 0
    email_encoder = dict()
    user_ids = []
    for email in interactions.email:
        if email not in email_encoder:
            email_encoder[email] = email_id
            email_id += 1

        user_ids.append(email_encoder[email])

    interactions.drop(columns=['email'], inplace=True)
    interactions['user_id'] = user_ids

    return articles, interactions


def user_interactions_statistics(df):
    '''Calculate relevant statistics for user-article interactions'''

    # Calculate bulk statistics
    user_interaction_count = df.groupby('user_id').article_id.count()
    median_interactions = user_interaction_count.median().astype(int)
    mean_interactions = user_interaction_count.mean().astype(int)
    max_interactions = user_interaction_count.max().astype(int)
    print('\nUser-Article Interaction Statistics (Part I):')
    print('\tMedian: {}'.format(median_interactions))
    print('\tMean:   {}'.format(mean_interactions))
    print('\tMax:    {}\n'.format(max_interactions))

    # Create a distribution plot (histogram)
    user_interaction_histogram(user_interaction_count)


def user_interaction_histogram(df):
    '''User-article interaction count histogram'''

    # Configure the figure/axes
    fig, ax = pyplot.subplots()
    bins = numpy.sort(df.unique())
    ax.hist(df.values, bins=bins)
    ax.set_xlabel('Number of Interactions')
    ax.set_ylabel('Number of Users')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1500])
    ax.grid(True)
    ax.set_axisbelow(True)

    # Annotate the figure
    mean = df.mean().astype(int)
    median = df.median().astype(int)
    maximum = df.max().astype(int)
    text = '\n'.join([
        'Median = {:.0f}'.format(median),
        'Mean = {:.0f}'.format(mean),
        'Max = {:.0f}'.format(maximum),
    ])
    bbox = dict(boxstyle='square,pad=1', facecolor='white', alpha=1.0)
    ax.text(0.9, 0.9, text, bbox=bbox, ha='right', va='top', transform=ax.transAxes)

    # Save the image to file
    filename = os.path.join(EXPLORATORY_PLOTS_DIR, 
        'user_interaction_histogram.png')
    fig.savefig(filename, bbox_inches='tight', dpi=180)


def articles_statistics(articles, interactions):
    '''Article statistics'''

    # Process the dataframes (join, remove duplicates, etc)
    articles.drop_duplicates(subset=['article_id'], inplace=True)
    df = interactions.join(articles.set_index('article_id'), 
        on='article_id',
        how='left')
    article_popularity = df.article_id.value_counts()

    # Calculate useful statistics
    number_of_unique_articles = articles.shape[0]
    number_of_unique_articles_with_interaction = df.drop_duplicates(
        subset=['article_id'], keep='first').shape[0]
    number_of_unique_users = interactions.user_id.unique().shape[0]
    number_of_user_article_interactions = interactions.shape[0]
    most_viewed_article_id = article_popularity.index.values[0]
    most_viewed_article_views = article_popularity.iloc[0]

    print('\nUser-Article Interaction Statistics (Part II):')
    print('\tNumber of Unique Articles: {}'.format(
        number_of_unique_articles))
    print('\tNumber of Unique Articles (>0 interactions): {}'.format(
        number_of_unique_articles_with_interaction))
    print('\tNumber of Unique Users: {}'.format(
        number_of_unique_users))
    print('\tNumber of User-Article Interactions: {}'.format(
        number_of_user_article_interactions))
    print('\tMost Viewed Article Id: {}'.format(
        most_viewed_article_id))
    print('\tMost Viewed Article Views: {}\n'.format(
        most_viewed_article_views))


# Execute the exploratory data analysis
if __name__ == '__main__':
    exploratory_analysis()

