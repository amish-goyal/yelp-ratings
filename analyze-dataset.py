import json
import numpy as np 
import matplotlib.pyplot as plt 

jsonfile='yelp_academic_dataset_business.json'

def businessGenerator(jsonfile):
    with open(jsonfile) as fil:
        businesses=fil.read().split('\n')
        for business in businesses:
            try:
                info = json.loads(business)
                yield info
            except:
                print "Read all Reviews"
                continue

def analyze_total_reviews(generator):
    x=[]
    y=[]
    count=0
    for info in generator:
        count+=1
        x.append(count)
        y.append(info['review_count'])
    plt.scatter(x,y)
    plt.show()


def main():
    pass

generator=businessGenerator(jsonfile)
analyze_total_reviews(generator)

"""
'Bagels','Bakeries','Barbeque','Bars','Beer', 'Wine & Spirits','Breakfast & Brunch','Buffets','Bubble Tea','Burgers','Cafes','Cafeteria','Chicken Wings','Cheesesteaks','Coffee & Tea','Comfort Food','Creperies','Cupcakes','Delis','Desserts','Dim Sum','Diners','Donuts','Ethnic Food','Fast Food','Fish & Chips','Food','Food Court','Food Trucks','Food Stands','Gelato','Gluten-Free','Halal','Ice Cream & Frozen Yogurt','Hot Dogs','Juice Bars & Smoothies','Live/Raw Food','Pizza','Pita','Pretzels','Restaurants','Salad','Sandwiches','Seafood','Soul Food','Soup','Steakhouses','Sushi Bars','Szechuan','Tapas Bars','Tapas/Small Plates','Tex-Mex','Vegan','Vegetarian','Wine Bars'
"""
