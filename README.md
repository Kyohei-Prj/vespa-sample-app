# vespa-sample-app
Simple search engine app using vespa.

## download text data
cd data
wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
tar -zxvf ldcc-20140209.tar.gz

## format text data and calculate embeddings
poetry run python ./src/format_articles_to_dataframe.py --save_path ./data/articles_vector.parquet

## create vespa configs
poetry run python ./src/create_vespa_configs.py --app ./vespa_config

## build kuromoji for vespa
Follow steps in https://github.com/yahoojapan/vespa-kuromoji-linguistics

mkdir ./vespa_config/components

cp kuromoji-linguistics-*-deploy.jar ./vespa_config/components

edit services.xml to include kuromoji

## run vespa docker container
docker run --detach --name vespa --hostname vespa-container --publish 8080:8080 --publish 19071:19071 vespaengine/vespa

## install vespa cli
https://github.com/vespa-engine/vespa/releases

## deploy
cd ./vespa_config
vespa deploy --wait 300

## load data to vespa
poetry run python .src/load_to_vespa.py --data ./data/articles_vector.parquet

