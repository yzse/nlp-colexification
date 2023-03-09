# !pip install fasttext
import pandas as pd
import numpy as np
import fasttext
import fasttext.util
import nltk
import sys
import requests
import urllib
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap
import random
sns.set_style('whitegrid')
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')

# create a dictionary to store the models
def load_models(wiki_top_langs):

    # for lang_id in wiki_top_langs:
    #     fasttext.util.download_model(lang_id, if_exists='ignore')

    ft_models = {}

    # load the FastText models for each language
    for lang_code in wiki_top_langs:
        model_path = f'cc.{lang_code}.300.bin'
        ft_models[lang_code] = fasttext.load_model(model_path)
    
    return ft_models

# merge iso
def merge_iso(df):
    link = 'https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes'
    wiki_ISO = pd.read_html(link, header=0)[1]

    wiki_ISO = wiki_ISO[['ISO language name','639-1']]
    wiki_ISO['ISO language name'] = wiki_ISO['ISO language name'].str.split(',').str[0] #removing multiple alternatives for a language label to improve matching
    replacements = {'Gaelic': 'Scottish Gaelic', #manually replacing matches not found
                    'Bihari languages': 'Maithili',
                    'Sinhala': 'Sinhalese',
                    'Chinese': 'Mandarin Chinese',
                    'Central Khmer': 'Khmer',
                    'Kurdish': 'Northern Kurdish',
                    }

    wiki_ISO['lang_id'] = wiki_ISO['ISO language name'].replace(replacements) #changing names to match

    # merge iso code
    tuples_df = df.drop_duplicates(['Concepticon_ID', 'clics_form'])
    df_langs = pd.merge(tuples_df, wiki_ISO, left_on='variety', right_on='lang_id', how='left') #left joining ISO codes
    df_langs = df_langs.rename(columns={'639-1': 'iso_code'})
    df_langs = df_langs.drop(['ISO language name', 'lang_id'],axis=1)

    # lowercase
    df_langs['Concepticon_Gloss'] = df_langs['Concepticon_Gloss'].str.lower()

    # clean clics_form special characters
    df_langs = df_langs.dropna(subset=['clics_form'])
    df_langs = df_langs[~df_langs['clics_form'].str.contains(r'[^a-zA-Z\s]')]

    return df_langs

# get merged dataframes
def get_lang_pairs_df(df, langs, word1, word2):
    lang_pairs_dict = {}
    
    for lang in langs:
        # filter dataframe to specific language
        lang_df = df[df.iso_code == lang]

        # tokenize variety: "Germanic English" -> "[Germanic, English]"
        for index, row in lang_df.iterrows():
            tokens = nltk.tokenize.word_tokenize(row['iso_code'])
            lang_df.loc[index, 'tk_variety'] = tokens

        # merge dataframe to itself to get all possible pairings
        pairs_df = pd.merge(lang_df, lang_df, on='iso_code', how='outer')
        print(f"{lang} pairs before dropping duplicates:", pairs_df.shape)

        # remove duplicated word pairs
        pairs_df = pairs_df[pairs_df[word1] != pairs_df[word2]]
        print(f"{lang} pairs after dropping duplicates:", pairs_df.shape)

        lang_pairs_dict[lang] = pairs_df
    
    return lang_pairs_dict


def log(w):
  with open('oov_log.txt', 'a+') as file:
    file.write(w)

def get_cos_similarity(df, col1, col2, model):
    # add word pair names
    df['word_pair'] = list(zip(df[col1], df[col2]))
    df['concepticon_id_pair'] = list(zip(df['Concepticon_ID_x'], df['Concepticon_ID_y']))

    # sort words within the tuple to look for duplicates
    df['word_pair'] = df['word_pair'].apply(lambda x: tuple(sorted(x)))
    df = df.drop_duplicates('word_pair')

    # use ft model to get word vector for each word
    try:
        word_vector_1 = df[col1].apply(lambda x: model.get_sentence_vector(x))
        word_vector_2 = df[col2].apply(lambda x: model.get_sentence_vector(x))
    except: # word is out of vocab -> log error
        log(word_vector_1, word_vector_2)

    # get cosine similarity btwn vectors
    cosine_sim = cosine_similarity(np.vstack(word_vector_1), np.vstack(word_vector_2)).diagonal()

    # append col back to dataframe
    df['cosine_similarity'] = cosine_sim

    return df

def save_cos_similarity(lang_pairs_dict, ft_models):

    cos_sim_dict = {}

    for lang_code in lang_pairs_dict.keys():
        model = ft_models[lang_code]

        df_sim = get_cos_similarity(lang_pairs_dict.get(lang_code), 'clics_form_x', 'clics_form_y', model)
        # Save df_sim as a pickle file
        filename = f'df_sim_{lang_code}.pkl'
        df_sim.to_pickle(filename)

        print(f'{lang_code}:', df_sim.shape)

        cos_sim_dict[lang_code] = df_sim

    return cos_sim_dict    

# text wrap for plot
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


def plot_word_pair_similarities(cos_sim_dict, lang_to_test):

    df = pd.merge(cos_sim_dict.get('en'), cos_sim_dict.get(lang_to_test), on='concepticon_id_pair')

    df = df[['concepticon_id_pair',
                    'word_pair_x', 'word_pair_y',
                    'iso_code_x', 'cosine_similarity_x',
                    'iso_code_y', 'cosine_similarity_y']]

    df.to_pickle('word_pair_sim_tl_df.pkl')

    # get top 8 to show
    df = pd.concat([df.sort_values('cosine_similarity_x').head(7), df.sort_values('cosine_similarity_x').tail(7)])
    
    df['word_pair_x'] = df['word_pair_x'].astype(str)

    # Define the x-axis as concepticon_id_pair
    x = 'word_pair_x'

    # Define the y-axis values and hue as iso_code and cosine_similarity
    y_vars = ['cosine_similarity_x', 'cosine_similarity_y']
    hue_var = 'variable'

    # Melt the dataframe to long format
    df_melted = df.melt(id_vars=x, value_vars=y_vars, var_name=hue_var)


    plt.figure(figsize=(18,6))
    # Create the bar plot using seaborn's barplot function
    ax = sns.barplot(x=x, y='value', hue=hue_var, data=df_melted)

    # Rotate x-axis labels 90 degrees and wrap them
    ax.set_xticklabels(ax.get_xticklabels())
    wrap_labels(ax, 10)

    # Add y-axis label
    plt.xlabel('Word Pair')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarities between Word Pairs, EN & {}'.format(lang_to_test.upper()))

    # Add legend to the plot
    hands, labs = ax.get_legend_handles_labels()
    labels=['EN', lang_to_test.upper()]
    plt.legend(handles=hands, labels=labels, title='ISO Code')

    # save dataframe and plot
    plt.savefig('word_pair_sim_{}_plot.png'.format(lang_to_test))
    plt.show()

    return None

# wiki_top_langs = {'en', 'zh', 'hi', 'es', 'fr', 'ar', 'bn', 'ru', 'pt', 'ur'}
wiki_top_langs = {'en', 'tl'}
ft_models = load_models(wiki_top_langs)

df = pd.read_csv('df_all_raw.csv')
df_langs = merge_iso(df)

# dataframe with top 10 languages' deduped word pairs
lang_pairs_dict = get_lang_pairs_df(df_langs, wiki_top_langs, 'Concepticon_ID_x', 'Concepticon_ID_y')

######## sample for testing purposes. remove in production
en_sample = lang_pairs_dict.get('en').sample(n=20000, replace=False, random_state=2)
lang_pairs_dict['en'] = lang_pairs_dict.get('en').loc[en_sample.index]

cos_sim_dict = save_cos_similarity(lang_pairs_dict, ft_models)

plot_word_pair_similarities(cos_sim_dict, 'tl')
