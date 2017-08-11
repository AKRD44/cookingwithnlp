#http://www.skinnytaste.com/recipe-index/

# activate and save as json with # scrapy crawl recipes -o recipes.json

#In this case I'm going through a main list on page 1. don't have to do anything recursive, which is why I don't call parse
import scrapy  #https://doc.scrapy.org/en/latest/intro/tutorial.html
from scrapy.selector import HtmlXPathSelector
import re
#scrapy crawl recipes

class RecipeSpider(scrapy.Spider):
	name = 'recipes'
	start_urls = ['http://www.skinnytaste.com/recipe-index/']
	def parse(self, response): 
		# follow links to author pages
		#scrapy shell 'http://www.skinnytaste.com/recipe-index/'
		for href in response.css("li.cat-item a::attr(href)"):
			yield response.follow(href, self.parse_filtered_list)

		# follow pagination links
		#for href in response.css('div.next a::attr(href)'):
		#	yield response.follow(href, self.parse)

	def parse_filtered_list(self,response):
		# follow pagination links
		#scrapy shell 'http://www.skinnytaste.com/21dayfix/'

		#when you want to select something with a particular attribute			
		for href in response.css('a[rel*=bookmark]::attr(href)'):
			yield response.follow(href, self.parse_recipe)

	def parse_recipe(self, response):
		#scrapy shell  'http://www.skinnytaste.com/easy-roasted-lemon-garlic-shrimp/'

		hxs=HtmlXPathSelector(response)
		
		no_ingredient_class=False
		def extract_with_css(query):
			return ''.join(response.css(query).extract()).strip()# .strip()  #you can strip later on, for now just extract
		
		
		dictionary={}
		dictionary['name']=extract_with_css('div.post-title h1::text')
		
		ingredients=extract_with_css('li[class*=ingredient]::text').strip()
		
		#ingredients = ''.join(hxs.select("//li[@class='ingredient']//text()").extract())	
		
		
		if len(ingredients)==0:
			no_ingredient_class=True
			ingredients=extract_with_css('li::text').strip()
			#ingredients = ''.join(hxs.select("//li//text()").extract())
		
		dictionary["ingredients"]=ingredients
		
		if 	no_ingredient_class==True:
			#MUCH EASIER TO SELECT WHAT YOU WANT WITH XPATH. The // means everything in between your thing's opening and closing bracquets
			text = ''.join(hxs.select("//div[@id='content']//text()").extract()).strip()		
			instructions=re.split('irections', text)[1]

		else:
			#instructions=extract_with_css('div[class*=instructions] span::text')
			instructions = ''.join(hxs.select("//div[@class='instructions']//text()").extract())	
		
		dictionary["instructions"]=instructions
		
		yield dictionary
		