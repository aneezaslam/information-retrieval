import re
import requests
from bs4 import BeautifulSoup
import json

link = "https://pureportal.coventry.ac.uk/en/organisations/school-of-economics-finance-and-accounting/publications"

content = requests.get(link)
soup = BeautifulSoup(content.content, "html.parser")
#print(soup)

#Finding page length
page_length = soup.find("nav", class_="pages").find_all("li")
page_length = len(page_length)
print(page_length)
print_list = []  
print(type(page_length))

#Looping through pages and storing data into dictionary
for i in range(1,page_length+1):
    url = link+"/?page={}".format(i)
    article = requests.get(url)
    soup = BeautifulSoup(article.content, "html.parser")
    publications_ = soup.find("ul", class_="list-results")
    publication_ = publications_.find_all("div", class_="result-container")
    print(publication_)
    
    for i in publication_:
        publication__name = i.find("h3", class_="title" ).find("a", href = True).find("span").text
        publication__link = i.find("h3", class_="title" ).find("a", href = True).get("href")
        publication_date = i.find("span", class_="date" ).text
        #print(publication__name)
        #print(publication__link)
        #print(publication_date)
        
        next_article = requests.get(publication__link)
        soup1 = BeautifulSoup(next_article.content, "html.parser")
        publication_author = soup1.find("p", class_="relations persons").text
        if soup1.find("div", class_="textblock"):
            publication_description = soup1.find("div", class_="textblock").text
        print (publication_author)
        #print (publication_description)
        
        print_dict = {
        'publication__name': publication__name,
        'publication__link': publication__link,
        'publication_date': publication_date,
        'publication_author': publication_author,
        'publication_description': publication_description
        }

        print_list.append(print_dict)
    
with open('data.json', 'w') as f:     
    json.dump(print_list, f)
    


    