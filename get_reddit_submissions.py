from psaw import PushshiftAPI
from datetime ***REMOVED***delta, date, datetime
***REMOVED***
***REMOVED***

def get_reddit_submissions_json_for_keyword(keyword):
	api = PushshiftAPI()

	time_delta = timedelta(days=1)
	start_time = datetime(2009, 1, 1)
	end_time = datetime(2019, 1, 1)

	submission_list = []

	while (end_time - start_time).days > 0:
		gen = api.search_submissions(q=keyword,
									after=start_time.timestamp(),
									before=(start_time+time_delta).timestamp(),
									limit=100)
		print(list(gen))
		submission_list.append(list(gen))
		time.sleep(10)
		start_time += time_delta

	dataframe = pd.DataFrame(submission_list)
	dataframe.to_json(keyword + '.json')

get_reddit_submissions_json_for_keyword('honeywell')