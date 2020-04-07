import json
import praw
import prawcore
import os


USER_AGENT = 'python:com.example.{username}.txmm_research_project:v1.0 (by /u/{username})'
AMOUNT_OF_POSTS = 500

SUBREDDITS = [
    'advice',
    'amitheasshole',
    'casualconversation',
    'changemyview',
    'confession',
    'copypasta',
    'depression',
    'entitledparents',
    'fantheories',
    'hfy',
    'idontworkherelady',
    'legaladvice',
    'lifeprotips',
    'maliciouscompliance',
    'nosleep',
    'offmychest',
    'outoftheloop',
    'parenting',
    'personalfinance',
    'pettyrevenge',
    'poetry',
    'prorevenge',
    'raisedbynarcissists',
    'rant',
    'relationship_advice',
    'self',
    'stopdrinking',
    'talesfromtechsupport',
    'tifu',
    'trueatheism',
]


def read_settings():
    """
    Reads a file with settings for the crawlers. This file should include the Reddit API client ID, client secret
    and the corresponding username.
    """
    with open('settings.json') as _file:
        return json.load(_file)


def authenticate(settings):
    """
    Authenticates with the Reddit API.
    """
    reddit = praw.Reddit(client_id=settings['client_id'],
                         client_secret=settings['client_secret'],
                         # username=settings['username'],
                         # password=settings['password'],
                         user_agent=USER_AGENT.format(username=settings['username']))
    return reddit


def crawl_top_posts(reddit, subreddit):
    """
    For a given subreddit, collects the top text posts of all time. It does so by retrieving twice the target
    amount of top posts, and filtering out any posts without text body.

    :param reddit: an authenticated praw.Reddit instance
    :param subreddit: subreddit to crawl top posts from
    :return: the title, content, name and url of the top posts from this subreddit
    """
    submissions = reddit.subreddit(subreddit).top(limit=2 * AMOUNT_OF_POSTS)

    results = [{
        'title': submission.title,
        'content': submission.selftext,
        'name': submission.name,
        'url': submission.url,
    } for submission in submissions if submission.is_self and submission.selftext.strip()][:AMOUNT_OF_POSTS]

    return results


def crawl_subreddits(reddit, subreddits):
    """
    For a given list of subreddits, crawl the top posts of each of them and save them.
    Abandons subreddit with too few text posts in order to prevent class imbalance.
    """
    for subreddit in subreddits:
        filename = f'subreddits/{subreddit}.json'

        if os.path.exists(filename):
            print(f'Subreddit "{subreddit}" already exists')
        else:
            try:
                submissions = crawl_top_posts(reddit, subreddit)
            except prawcore.exceptions.ResponseException as e:
                print(f'Skipping subreddit "{subreddit}" due to error "{e}"')
                continue

            if len(submissions) < AMOUNT_OF_POSTS / 2:
                print(f'Abandoning subreddit "{subreddit}" with {len(submissions)} self posts')
            else:
                print(f'Crawled subreddit "{subreddit}" with {len(submissions)} self posts')
                with open(filename, 'w') as _file:
                    json.dump(submissions, _file)


def main():
    reddit = authenticate(read_settings())
    reddit.read_only = True
    crawl_subreddits(reddit, SUBREDDITS)


if __name__ == '__main__':
    main()
