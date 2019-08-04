from applied_ML_metadata import read_data
import json


def merge(crawl_data, imdb_df):
    """
    Map IMDb tconsts to crawled data
    :param crawl_data: Data that was crawled from BoxOfficeMojo
    :param imdb_df: IMDB data frame
    :return:
        1. res = Dictionary with tconst as key and revenue as value
        2. except = List of crawl data points for which no IMDb entry was found
    """
    results = {}
    excepts = []
    imdb_df['primaryTitle'] = imdb_df['primaryTitle'].str.lower()
    imdb_df['primaryTitle'] = imdb_df['primaryTitle'].str.replace(" ", "")

    for entry in crawl_data:
        try:
            name, release_date, rev_str = entry.split('|')
        except Exception as e:
            print("(Exception) While reading file: ", e)
            continue

        film_name = name.split('(')[0].replace(' ', '').lower().strip()
        revenue = get_revenue(rev_str.strip())
        year = release_date.split(',')[1].strip()
        match = imdb_df.loc[(imdb_df['startYear'] == year) & (imdb_df['primaryTitle'] == film_name)]

        try:
            results[match.iloc[0]['tconst']] = revenue
        except:
            print("(Exception) Empty match: ", name.split('(')[0], release_date)
            excepts.append(name.split('(')[0] + ' | ' + release_date)
            continue

    return results, excepts


def get_revenue(rev_raw):
    revenue_str = rev_raw.replace(',', '').replace('$', '')
    return revenue_str


def save_data(file_name, json_data):
    with open(file_name, 'w') as file:
        file.write(json.dumps(json_data, indent=2))


def read_crawled_data(file_name, is_json=False):
    # read crawled data file
    f = open(file_name, "r")
    if is_json:
        return json.load(f)
    crawled_data = []
    for line in f:
        crawled_data.append(line)
    return crawled_data


def merge_crawl_data_with_imdb_tconsts():
    # read data frame
    titles, _ = read_data('imdb_data/title.basics.tsv')
    titles = titles[titles.titleType == 'movie']
    titles = titles.reset_index(drop=True)

    # read crawled data
    data = read_crawled_data('crawl_data_new.txt')

    # filter out duplicates
    data = list(set(data))

    # join with imdb tconsts
    res, exceptions = merge(data, titles)

    # save merged data to file
    save_data('crawled_revenue.txt', res)

    # save entries which had errors
    save_data('exceptions.txt', exceptions)


def main():
    merge_crawl_data_with_imdb_tconsts()


if __name__ == '__main__':
    main()