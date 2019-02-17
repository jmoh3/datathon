from TwitterSearch import *
import pandas as pd
from datetime import timedelta, datetime, date
import time

company_to_keywords_dict = {'stock_market':('stock market', 'stocks', 'DOW', 'NASDAQ'), 'hon': ('Honeywell', 'HON'), 'Synchrony':('Synchrony', 'SYF'), '3M':('MMM', 'Minnesota Mining and Manufacturing Company'), 'Bayer':('Bayer', 'BAYRY')}
industry_words = {'Synchrony':('credit cards'), 'Bayer':('pharma', 'pharmaceutical', 'Xarelto', 'Xofigo', 'Eylea')}

try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
    tso.set_language('en') # we want to see German tweets only
    tso.set_include_entities(False) # and don't give us all those entity information

    # it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
        consumer_key = '',
        consumer_secret = '',
        access_token = '',
        access_token_secret = ''
     )

     # this is where the fun actually starts :)

    # def my_callback_closure(current_ts_instance): # accepts ONE argument: an instance of TwitterSearch
    #     queries, tweets_seen = current_ts_instance.get_statistics()
    #     if queries > 0 and (queries % 5) == 0: # trigger delay every 5th query
    #         time.sleep(60) # sleep for 60 seconds


    tweet_list = []

    for key in company_to_keywords_dict.keys():
        tso.set_keywords(company_to_keywords_dict[key], or_operator=True) # let's define all words we would like to have a look for
        time_delta = timedelta(days=30)
        start_time = date(2000, 10, 4)
        end_time = date(2019, 1, 1)
        current_time = start_time
        df_list = []

        sleep_for = 60 # sleep for 60 seconds
        last_amount_of_queries = 0 # used to detect when new queries are done

        while ((end_time - current_time).days > 0):

            tso.set_until(current_time)

            count = 0


            for tweet in ts.search_tweets_iterable(tso):
                if count > 20:
                    break
                tweet_dict = {}
                tweet_dict['user'] = tweet['user']['screen_name']
                tweet_dict['text'] = tweet['text']
                tweet_dict['date'] = tweet['created_at']
                tweet_list.append(tweet_dict)
                print( '@%s tweeted: %s' % ( tweet['user']['screen_name'], tweet['text'] ) )
                count +=1


            df = pd.DataFrame(tweet_list)
            df_list.append(df)
            current_time += time_delta
        
        df = pd.concat(df_list)
        df.to_json(key + '.json')
        time.sleep(sleep_for)


except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print('here')
    print(e)
