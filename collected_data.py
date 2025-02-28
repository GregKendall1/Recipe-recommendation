# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
import time
import pandas as pd 
import requests 
import os
import numpy as np
from numpy import random 
from time import sleep

headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'}

def create_url(category,page_num):
    """Takes a category and page number and creates the base url"""
    base_url = f'https://www.budgetbytes.com/category/recipes/{category}/page/{page_num}/'
    return base_url

def get_last_page(soup,not_last_page):
    """Finds if page is last page of recipes, returns true if it is not"""
    lastpage= soup.find('li',class_='pagination-next')
    if not lastpage:
        not_last_page = False
        print('Last page for this category')
    else:
        print('Next page')
    return not_last_page

def get_recipe_url(category):
    """
    Goes through all the pages of a given category and finds urls for recipe pages
    """
    page_num = '1'
    url = create_url(category,page_num)
    page = requests.get(url,headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    not_last_page = True
    url_list = []
    common_url_words = ['category','about','faq','contact','welcome-to-budget-bytes','terms-conditions','privacy-policy','random','recipes','accessibility','kitchen-pantry','videos','kitchen-basics']
    while not_last_page:                    #keeps looping until last page is found
        for urls in soup.find_all('a'):
            if 'https://www.budgetbytes.com/' in urls.get('href') and not any(x in urls.get('href') for x in common_url_words):
                url_list.append(urls.get('href'))
        not_last_page = get_last_page(soup,not_last_page)
        page_num = str(int(page_num)+1)
        time.sleep(np.random.randint(1,5))
        url = create_url(category,page_num)
        page = requests.get(url,headers=headers)
        soup = BeautifulSoup(page.content, 'html.parser')
    return url_list

def get_recipe_info(url,category):
    """Given a recipe page, gets information and return as a dictionary"""
    page = requests.get(url,headers=headers)
    soup = BeautifulSoup(page.content,'html.parser')
    info_dic = get_basic_info(soup)
    ingredient_list = []
    for ingredient in soup.find_all('span',class_='wprm-recipe-ingredient-name'):
        ingredient_list.append(ingredient.text)
    info_dic['ingredients'] = ingredient_list
    info_dic['url'] = url
    info_dic['category'] = category
    return info_dic

def get_basic_info(soup):
    """Take a recipe page and returns dictionary of information about recipe; returns np.nan if information not found """
    try:
        name = soup.find('h1').text
    except:
        name = np.nan
    try:
        time = soup.find('span',class_='wprm-recipe-time wprm-block-text-normal').text
    except:
        time = np.nan
    try:
        price = soup.find('span',class_='cost-per').text
    except:
        price = np.nan
    info_dic = {'name':name,'time':time,'price':price}
    return info_dic

def create_df(list_of_info):
    """All the information about recipes and returns dataframe"""
    df = pd.DataFrame(list_of_info)
    return df

def get_data_frame(category):
    """Gets all recipes from a given category and makes a dataframe"""
    list_of_recipes = []
    url_list = get_recipe_url(category)
    for url in url_list:
        info_dic = get_recipe_info(url,category)
        list_of_recipes.append(info_dic)
        time.sleep(np.random.randint(1,5))
    df = create_df(list_of_recipes)
    return df

def save_dataframe(category,df):
    directory = os.getcwd()
    os.makedirs(directory+'\\data\\' ,exist_ok=True)
    df.to_csv(directory+'\\data\\'+category,index = False)
    print('Loaded data for '+category)
    
    
categories = [
    'appetizers',
    'beansandgrains',
    'breakfast',
    'dairy-free',
    'dessert',
    'egg-free',
    'gluten-free',
    'main-dish',
    'pasta',
    'sandwich',
    'side-dish',
    'vegetarian'
    ]
if __name__ == "__main__":
    main_df = pd.DataFrame(columns = ['name','time','price','ingredients','url','category'])
    for category in categories:
        df = get_data_frame(category)
        save_dataframe(category,df)
        df_main = pd.concat([df_main,df])
    save_dataframe('all_recipes',df_main)

