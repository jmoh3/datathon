# -*- coding: utf-8 -*-
import scrapy


class EarningsCallsScraperSpider(scrapy.Spider):
    name = 'earnings_calls'

    symbol = ''
    allowed_domains = ['seekingalpha.com']
    start_urls = ['https://seekingalpha.com/symbol/MMM/earnings/transcripts']

    # ,
    # 				'https://seekingalpha.com/symbol/BAYZF/earnings/transcripts',
    # 				'https://seekingalpha.com/symbol/HON/earnings/transcripts',
    # 				'https://seekingalpha.com/symbol/SYF/earnings/transcripts'

    # This method must be in the spider, 
    # and will be automatically called by the crawl command.
    def start_requests(self):
        self.index = 0

        headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'}

        for url in self.start_urls:
            # We make a request to each url and call the parse function on the http response.
            yield scrapy.Request(url=url, headers=headers, callback=self.parse)

    def parse_link(self, response):
    	self.index += 1
    	final = ''
    	time = ''
    	for i in response.xpath("//div"):
    		for p in i.xpath('.//p'):
    			final += p.extract()
    		print(i.css("a-info").xpath('./@class').extract())
    	filename = self.symbol+str(self.index)
    	with open(filename, 'w') as f:
    		#All we'll do is save the whole response as a huge text file.
            f.write(final)
            self.log('Saved file %s' % filename)

    def parse(self, response):
    	for i in response.xpath("//div"):
    		link = i.xpath('a/@href').extract()
    		if len(link) > 0 and link[0][0:8] == "/article":
    			print(link[0])
    			request = "https://seekingalpha.com" + link[0]
    			headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0'}
    			yield scrapy.Request(url=request, headers=headers, callback=self.parse_link)





