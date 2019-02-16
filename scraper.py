import scrapy

class EarningsCallSpider(scrapy.Spider):
    name = "earnings_call_spider"
    start_urls = ['https://seekingalpha.com/article/4235000-synchrony-financial-syf-ceo-margaret-keane-q4-2018-results-earnings-call-transcript']

    rules = (Rule(LinkExtractor(allow=(), restrict_css=('.pageNextPrev',)),
             callback="parse_item",
             follow=True),)

    def parse(self, response):
        print('Processing..' + response.url)