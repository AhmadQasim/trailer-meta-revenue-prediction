"""
    Web crawler to get first weekend film revenues from BoxOfficeMojo.com
    usage: (run in CLI) scrapy runspider crawler.py
"""
import scrapy
from applied_ML_metadata import read_data

weekly_url = 'page=weekend&'
base_url = "https://www.boxofficemojo.com"
search_url = 'https://www.boxofficemojo.com/search/?q='

# open file handler
f = open("crawl_data_new.txt", "a")


class FilmSpider(scrapy.Spider):
    name = "film_spider"

    def __init__(self):
        # total films: 518986
        self.download_delay = 0.2
        self.crawl_from = 50000
        self.crawl_to = 100000

    def start_requests(self):
        urls = []
        titles, _ = read_data('imdb_data/title.basics.tsv')
        titles = titles[titles.titleType == 'movie']
        titles = titles.reset_index(drop=True)
        print("Total films: {}".format(len(titles.index)))

        titles = titles.iloc[self.crawl_from:]

        for idx, row in titles.iterrows():
            try:
                urls.append(search_url + str(row['primaryTitle']))
            except Exception as e:
                print("Exception: {}. Title name: {}.".format(e, row['primaryTitle']))

        for idx, url in enumerate(urls):
            yield scrapy.Request(url)

    def parse(self, response):
        cnt = 0
        for link in response.xpath('//a/@href').getall():
            if '/movies' in link:
                if cnt == 1:
                    film_url = base_url + link
                    tokens = film_url.split('?')
                    film_weekly_url = tokens[0] + '?' + weekly_url + tokens[1]
                    # print("URL: ", film_weekly_url)
                    yield scrapy.Request(url=film_weekly_url, callback=self.store_weekly_rev)
                cnt += 1

    def store_weekly_rev(self, response):
        data = ''
        table_row_selector = 'td'
        # search film name
        data += self.search_film_name(response, table_row_selector)
        # search film release date
        data += self.search_film_release_date(response, table_row_selector)
        # search revenue
        data += self.search_film_revenue(response, table_row_selector)
        # write data to file
        f.write(data+"\n")

    def search_film_name(self, response, table_row_selector):
        cnt = 0
        data = ''
        for table_row in response.css(table_row_selector):
            name_container = 'font b::text'
            bold_text = table_row.css(name_container).extract()
            for item in bold_text:
                if 1 < cnt < 4 and '$' not in item:
                    # print("BOLD: ", item)
                    data += item + ' '
                cnt += 1
        return data

    def search_film_release_date(self, response, table_row_selector):
        for table_row in response.css(table_row_selector):
            date_container = 'b nobr a::text'
            link_text = table_row.css(date_container).extract_first()
            if link_text:
                # print("DATE: ", link_text)
                data = ' | ' + link_text
                return data

    def search_film_revenue(self, response, table_row_selector):
        for table_row in response.css(table_row_selector):
            revenue_container = 'font ::text'
            revenue_text = table_row.css(revenue_container).extract_first()
            if revenue_text is not None and '$' in revenue_text:
                    # print("REVENUE: ", revenue_text)
                    data = ' | ' + revenue_text
                    return data