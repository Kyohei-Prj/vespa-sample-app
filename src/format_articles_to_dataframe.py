import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from loguru import logger
import click

from glob import glob

model_name = 'intfloat/multilingual-e5-base'
model = SentenceTransformer(model_name)

tqdm.pandas()


@click.command()
@click.option('--save_path', type=str)
def main(save_path: str):

    article_path = './data/text/*/*.txt'
    title_article = [extract_title_and_body(fn) for fn in glob(article_path)]

    df = format_to_dataframe(title_article)
    df['vector'] = df['articles'].progress_apply(generate_embedding)

    df.to_parquet(save_path, index=False)


def extract_title_and_body(file_path: str) -> tuple[str, str]:

    with open(file_path, 'r') as fn:
        text = fn.readlines()[2:]

    text = [t.replace('\n', '') for t in text]
    title = text[0]
    body = ''.join(text[1:])

    return title, body


def format_to_dataframe(title_article: list[tuple[str, str]]) -> pd.DataFrame:

    cols = ['titles', 'articles']
    return pd.DataFrame(title_article, columns=cols)


def generate_embedding(article: str):

    return model.encode(article, normalize_embeddings=True)


if __name__ == '__main__':
    main()
