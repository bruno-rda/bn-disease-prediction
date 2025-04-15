from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def get_df_with_text(df: pd.DataFrame, x_col: str, y_col: str):
    '''
    df[x_col] : Iterable of symptoms
    df[y_col] : Disease name
    '''
    df_text = df[[y_col, x_col]]

    # Join the symptoms into a single string
    df_text[x_col] = df_text[x_col].apply(lambda x: ' '.join(x))

    # Group by disease name and join al the symptoms into a single string
    df_text = df_text.groupby(y_col).agg({x_col: ' '.join})
    return df_text


def get_wordcloud(text: str):
    return WordCloud(
        width=800,
        height=400,
        background_color='white',
        collocations=False
    ).generate(text)

def plot_wordcloud(wordcloud: WordCloud, name: str):
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation='gaussian')
    plt.title(name)
    plt.axis('off')
    plt.show()

def save_wordcloud(wordcloud: WordCloud, path: str):
    wordcloud.to_file(path)