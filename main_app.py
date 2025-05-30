from dash import Dash, html, dcc, callback, Output, Input, State, dash_table,ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords

def get_data():
    """Gets dataframe with recipe data and model"""
    directory = os.getcwd()
    df = pd.read_json(directory+"\\data\\all_data_improved_2")
    w2v = KeyedVectors.load(directory+"\\model_w2v_update_2.bin")
    df = df.drop_duplicates(subset = ['name','ingredients'],ignore_index = True)
    df = df.loc[df['ingredients']!= "[]"]   #remove empty ingredients 
    df = df.reset_index(drop=True)
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
    one_word_ingredients = [x for x in ingredient_list if len(x.split()) == 1]
    return one_word_ingredients

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

def load_tfidf_model(df):
    """Loads tfidf model and creates dictionary of values for each word in corpus"""
    words = []
    for i in range(df.shape[0]):
        if type(df.loc[i,'ingredients']) == str:    
            ing_list = eval(df.loc[i,'ingredients'])    #needs to make it into list 
        else:
            ing_list = df.loc[i,'ingredients']
        ing_list = [clean_ingredients(ingredient) for ingredient in ing_list]
        words.extend(ing_list)
    directory = os.getcwd()
    with open(directory+"\\vectorizer.pickle", 'rb') as handle:
        tfidf = pickle.load(handle)
    max_idf = max(tfidf.idf_) 
    values = defaultdict(lambda: max_idf,
    [(word, tfidf.idf_[i]) for word, i in tfidf.vocabulary_.items()])
    return values  
    
def get_weighted_average(ingredients,values,weighted = True):
    """Gets the weighted average embedding of ingredient, weighted by tfidf if weighted is true"""
    value_list = []
    user_input = ingredients        
    ingredients = [clean_ingredients(ing,alphabetical=True) for ing in ingredients]
    for ing in user_input:
        ingredients.extend([x for x in w2v.wv.key_to_index if ing in x.split()])        #includes all appearances of word in ingredients 
    for ing in ingredients:
        if ing in w2v.wv.index_to_key:
            if weighted:
                for x in ing:
                    value_list.append(w2v.wv[ing]*values[x])
            else:
                value_list.append(w2v.wv[ing])
    if value_list:
        ave = np.array(value_list).mean(axis=0)
    else:                                       #if none of ingredients appear in corpus
        ave = np.zeros((w2v.wv.vector_size,1))
    return ave

def find_recommendations(df,input_vec,no_of_recs,checked):
    """Takes user input and outputs next 5 most similar recipes (or least similar if appropriate button clicked)"""
    named_ingredients = input_vec.copy()
    input_vec = get_weighted_average(input_vec,values,weighted = True)
    cossim = [cosine_similarity(input_vec.reshape(1,-1),x.reshape(1,-1))[0][0] for x in df['average']]
    if checked == ['Include ingredients listed']:
        for i,y in enumerate(df['ingredients']):        #prioritises recipes which include inputted ingredients
            if set([clean_ingredients(ing,alphabetical=False) for ing in named_ingredients]).issubset([clean_ingredients(x,False) for x in eval(y)]):
                cossim[i] += 10 
    if no_of_recs == 1:             #shows first five most similar
        top = [x[0] for x in sorted(enumerate(cossim), key=lambda x: x[1])[-5:]][::-1]
    elif no_of_recs > 1:            #shows next five most similar 
        top = [x[0] for x in sorted(enumerate(cossim), key=lambda x: x[1])[-(5*no_of_recs):-(5*(no_of_recs-1))]][::-1]
    elif no_of_recs == -1:          #shows five least similar
        top = [x[0] for x in sorted(enumerate(cossim), key=lambda x: x[1])[:5]]
    elif no_of_recs < -1:           #shows next five least similar
        top = [x[0] for x in sorted(enumerate(cossim), key=lambda x: x[1])[5*(-no_of_recs-1):5*(-no_of_recs)]]
    return top 

def display_ingredients(unformatted_ingredients):
    """Displays ingredients as a string"""
    unformatted_ingredients = eval(unformatted_ingredients)
    unformatted_ingredients = [ingred.replace('*','').strip() for ingred in unformatted_ingredients]
    cleaned_ingredient = ', '.join(unformatted_ingredients)
    return cleaned_ingredient

def display_url(url):
    """Changes url to html so it is clickable"""
    url = "<a href='{}' target='_blank'>{}</a>".format(url,url)
    return url

#need to remove words from input in order to use w2v model
words_to_remove = ['diced','chopped','fresh','crumbled','peeled','approx','bonein','jar','roughly','sifted','chilled','shaved','frozen','cut','thawed','seeded','room','temperature','softened','melted','one','total','taste','breast','thigh','dried','rings','each','teaspoon','bag','drained','plus','needed','cooked','trimmed','piece','boneless','skinless','small','medium','large','sliced','about','cup','tbsp','freshly','cracked','can','strained','rinsed','crushed','tsp','lb','oz','ground','minced','lightly','whisked','chopped','optional','garnish','divided','uncooked','bunch','finely','shredded','cold','quartered']


df,w2v = get_data()
df['url'] = df['url'].apply(display_url)
ingredient_list = create_suggestions(df)   
values = load_tfidf_model(df)
    
app = Dash(external_stylesheets=[dbc.themes.LUX])


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
    dcc.Checklist(options = ['Include ingredients listed'],id = 'checkbox',value = []),
    dash_table.DataTable(df[['name','ingredients','url']].to_dict('records'),
                         id='rec_table',
                         columns = [{'name': 'name', 'id': 'name'},{'name': 'ingredients', 'id': 'ingredients'},{'name': 'url', 'id': 'url','presentation': 'markdown'}],
                         page_size = 5,
                          style_cell={'textAlign': 'left'},
                          style_data={'whiteSpace': 'normal'},
                          css=[{
                          'selector': '.dash-cell div.dash-cell-value',
                          'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
                          }],
                         markdown_options={"html": True}),
    dbc.Button('Find recipes!', id='submit-val', color="dark", className="me-1",outline=True),
    dbc.Button('More recipes!', id='more-button',color="dark" ,className="me-1",outline=True),
    dbc.Button('Surprise me!',id='surprise-button',color="dark",className="me-1",outline=True),
    dbc.Button('Least similar!',id = 'bad-button',color="dark",className="me-1",outline=True)
])

@app.callback(
    [Output(component_id='rec_table', component_property='data'),Output(component_id='more-button', component_property='n_clicks'),Output(component_id='bad-button', component_property='n_clicks')],
    [Input(component_id='submit-val', component_property='n_clicks'),Input(component_id='more-button', component_property='n_clicks'),Input(component_id='surprise-button', component_property='n_clicks'),Input(component_id='bad-button', component_property='n_clicks')],
    [State(component_id='dropdown', component_property='value'),State(component_id = 'checkbox',component_property = "value")],
    prevent_initial_call=True
)

def display_recipes(n_clicks1,n_clicks2, n_clicks3,n_clicks4,input_ingredients,checked):
    """Displays recommendations when submit button is clicked"""
    if "submit-val" == ctx.triggered_id:        #finds top five recommendations
        top = find_recommendations(df, input_ingredients,1,checked)
        data = df.loc[top,['name','ingredients','url']]
        data['ingredients'] = data['ingredients'].apply(display_ingredients)
        data = data.to_dict('records')
        n_clicks2= 0             
        n_clicks4 = 0
    elif 'surprise-button' == ctx.triggered_id:
        number_choices = df.shape[0]
        choices = np.random.randint(0,number_choices,5)
        data = df.loc[choices,['name','ingredients','url']]
        data['ingredients'] = data['ingredients'].apply(display_ingredients)
        data = data.to_dict('records')
    elif 'more-button'==ctx.triggered_id:       #gives next five recommendations
        top = find_recommendations(df, input_ingredients,n_clicks2,checked)
        data = df.loc[top,['name','ingredients','url']]
        data['ingredients'] = data['ingredients'].apply(display_ingredients)
        data = data.to_dict('records')
    elif 'bad-button'==ctx.triggered_id:        #least five similar recommendations (for checking and/or fun!)
        top = find_recommendations(df, input_ingredients,-n_clicks4,checked)
        data = df.loc[top,['name','ingredients','url']]
        data['ingredients'] = data['ingredients'].apply(display_ingredients)
        data = data.to_dict('records')
    return data, n_clicks2,n_clicks4

if __name__ == '__main__':
    app.run(debug=True)
