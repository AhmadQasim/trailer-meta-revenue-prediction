import requests  # to make TMDB API calls
import json

# https://www.themoviedb.org/settings/api?language=de
api_key = 'c2f7e6fa01afe60fddf5e3282ee8c5e8'  # where xxxxxxxxxxxxxxx is replaced by your TMDB api_key


# adds Youtube URL ID to our dataset of movies
# https://www.youtube.com/watch?xxxxxxxxxxx  where xxxxxxxxxxx is replaced by the URL ID
def add_youtube_url_to_json_omdb(loadname, savename):
    f = open(loadname, "r")
    titles = json.load(f)

    titles_youtube = dict()

    for idx, title in enumerate(titles):
        url_temp = ''
        response = requests.get(
                'https://api.themoviedb.org/3/movie/' + title + '/videos?api_key=' +
                api_key + '&external_source=imdb_id')
        url_response = response.json()  # store parsed json response

        if 'results' in url_response.keys():
            if len(url_response['results']) != 0:
                url_temp = url_response['results'][0].get('key')

        element = dict()
        element['tconst'] = title
        element['earning'] = titles[title]
        element['YoutubeURL'] = url_temp
        titles_youtube[title] = element
        print(idx, title)

    f = open(savename, "a")
    json.dump(titles_youtube, f)

    return titles_youtube


add_youtube_url_to_json_omdb("crawled_revenue.txt", "crawled_revenue.json")