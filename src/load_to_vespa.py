from vespa.application import Vespa
from tqdm import tqdm
import pandas as pd
import click

tqdm.pandas()

vespa_conn = Vespa(url="http://localhost", port=8080)


@click.command()
@click.option('--data', type=str)
def main(data: str):

    df = pd.read_parquet(data)
    df.progress_apply(load_to_vespa, axis=1)


def load_to_vespa(x):

    vespa_json = {
        "id": x.__dict__['_name'],
        "title": x['titles'],
        "article": x['articles'],
        "embedding": x['vector'].tolist()
    }
    vespa_conn.feed_data_point(
        schema='doc',
        data_id=x.__dict__['_name'],
        fields=vespa_json
    )


if __name__ == '__main__':
    main()
