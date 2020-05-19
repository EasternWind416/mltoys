import feedparser

from test.NaiveBayes.AdStat.function import *


feed1 = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
feed0 = feedparser.parse('http://www.cppblog.com/kevinlynx/category/6337.html/rss')

# ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
# sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')

vocList, phix, phiy = bayesTest(feed1, feed0)
getTopWords(vocList, phix)
vocList, phix, phiy = bayesTest(feed1, feed0)
getTopWords(vocList, phix)
vocList, phix, phiy = bayesTest(feed1, feed0)
getTopWords(vocList, phix)
