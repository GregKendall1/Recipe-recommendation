from dash import Dash, html, dcc, callback, Output, Input, State, dash_table,ctx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords

#need to remove words from input in order to use w2v model
words_to_remove = ['diced','chopped','fresh','crumbled','peeled','approx','bonein','jar','roughly','sifted','chilled','shaved','frozen','cut','thawed','seeded','room','temperature','softened','melted','one','total','taste','breast','thigh','dried','rings','each','teaspoon','bag','drained','plus','needed','cooked','trimmed','piece','boneless','skinless','small','medium','large','sliced','about','cup','tbsp','freshly','cracked','can','strained','rinsed','crushed','tsp','lb','oz','ground','minced','lightly','whisked','chopped','optional','garnish','divided','uncooked','bunch','finely','shredded','cold','quartered']

def get_data():
    """Gets dataframe with recipe data and model"""
    directory = os.getcwd()
    df = pd.read_json(directory+'\\data\\all_data_json')
    w2v = KeyedVectors.load(directory+"\\model_w2v.bin")
    df = df.drop_duplicates(subset = ['name','ingredients'],ignore_index = True)
    df['average']=[np.array(x) for x in df['average']]
    return df,w2v


def create_suggestions(df):
    """Creates the suggestions to show up in dropdown box"""
    ingredient_list = []
    for i in range(df.shape[0]):
        ingredients = eval(df.loc[i,'ingredients'])
        ingredients = [clean_ingredients(ing,alphabetical=False) for ing in ingredients]
        ingredient_list.extend(ingredients)
    ingredient_list = list(set(ingredient_list))    #remove duplicates
    return ingredient_list

def clean_ingredients(ingredient,alphabetical=False):
    """input string of ingredient and puts it in a standard form """
    lemmatizer = WordNetLemmatizer()
    ingredient_list = ingredient.split()    
    ingredient_list = [''.join(filter(str.isalpha,ingred))for ingred in ingredient_list]
    ingredient_list = [ingred.lower() for ingred in ingredient_list if ingred]
    ingredient_list = [lemmatizer.lemmatize(ingred) for ingred in ingredient_list]
    ingredient_list = [ingred for ingred in ingredient_list if ingred not in words_to_remove]
    if alphabetical:        #only put alphabetical to input in gensim model
        ingredient_list = [remove_stopwords(ingred) for ingred in ingredient_list]
        ingredient_list = sorted(ingredient_list)
    cleaned_ingredient = ' '.join(ingredient_list)
    return cleaned_ingredient

def get_average(ingredients):
    """Get average embedding of ingredient which is inputted by user"""
    value_list = []
    ingredients = [clean_ingredients(ing,alphabetical=True) for ing in ingredients]
    for ing in ingredients:
        if ing in w2v.wv.index_to_key:
            value_list.append(w2v.wv[ing])
    if value_list:
        ave = np.array(value_list).mean(axis=0)
    else:
        ave = np.zeros((w2v.wv.vector_size,1))
    return ave


def find_recommendations(df,input_vec):
    """Takes user input and outputs top 5 most similar recipes"""
    input_vec = get_average(input_vec)
    cossim = [cosine_similarity(input_vec.reshape(1,-1),x.reshape(1,-1))[0][0] for x in df['average']]
    top = [x[0] for x in sorted(enumerate(cossim), key=lambda x: x[1])[-5:]][::-1]
    return top 


app = Dash()


app.layout = html.Div([
    html.H1(children='Recipe recommendations', style={'textAlign':'center'}),
    dcc.Dropdown(
        id='dropdown',
        options=
            sorted(ingredient_list)
        ,
        multi = True,
        placeholder = 'Type ingredients here...'
    ),
    html.Button('Find recipes', id='submit-val'),
    html.Button('Surprise me!',id='surprise-button'),
    dash_table.DataTable(df[['name','ingredients','url']].to_dict('records'),
                         id='rec_table',
                         columns = [{'name': 'name', 'id': 'name'},{'name': 'ingredients', 'id': 'ingredients'},{'name': 'url', 'id': 'url'}],
                         page_size = 10,
                         style_cell={'textAlign': 'left'},
                         style_data={'whiteSpace': 'normal'},
                         css=[{
                         'selector': '.dash-cell div.dash-cell-value',
                         'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                         }])
])
@app.callback(
    Output(component_id='rec_table', component_property='data'),
    [Input(component_id='submit-val', component_property='n_clicks'),Input(component_id='surprise-button', component_property='n_clicks')],
    [State(component_id='dropdown', component_property='value')],
    prevent_initial_call=True
)

def display_recipes(n_clicks1,n_clicks2, input_ingredients):
    """Displays recommendations when submit button is clicked"""
    if "submit-val" == ctx.triggered_id:
        top = find_recommendations(df, input_ingredients)
        data = df.loc[top,['name','ingredients','url']].to_dict('records')
    elif 'surprise-button' == ctx.triggered_id:
        number_choices = df.shape[0]
        choices = np.random.randint(0,number_choices,5)
        data = df.loc[choices,['name','ingredients','url']].to_dict('records')
    return data
    
if __name__ == '__main__':
    df,w2v = get_data()
    ingredient_list = create_suggestions(df)
    app.run(debug=True)
