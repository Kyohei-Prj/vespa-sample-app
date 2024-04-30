from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    RankProfile,
    FieldSet,
    GlobalPhaseRanking,
    Function,
)
import click


@click.command()
@click.option('--app', type=str)
def main(app: str):

    package = define_app()
    package.to_files(root=app)


def define_app():

    package = ApplicationPackage(
        name="hybridsearch",
        schema=[
            Schema(
                name="doc",
                document=Document(
                    fields=[
                        Field(name='language', type="string",
                              indexing=['"ja"', 'set_language']),
                        Field(name="id", type="string", indexing=["summary"]),
                        Field(
                            name="title",
                            type="string",
                            indexing=["index", "summary"],
                            index="enable-bm25",
                        ),
                        Field(
                            name="article",
                            type="string",
                            indexing=["index", "summary"],
                            index="enable-bm25",
                            bolding=True,
                        ),
                        Field(
                            name="embedding",
                            type="tensor<float>(x[768])",
                            indexing=["index", "attribute"],
                            attribute=[
                                'distance-metric: prenormalized-angular']
                        ),
                    ]
                ),
                fieldsets=[FieldSet(name="default", fields=[
                                    "title", "article"])],
                rank_profiles=[
                    RankProfile(
                        name="bm25",
                        inputs=[("query(q)", "tensor<float>(x[768])")],
                        functions=[
                            Function(name="bm25sum",
                                     expression="bm25(title) + bm25(article)")
                        ],
                        first_phase="bm25sum",
                    ),
                    RankProfile(
                        name="semantic",
                        inputs=[("query(q)", "tensor<float>(x[768])")],
                        first_phase="closeness(field, embedding)",
                    ),
                    RankProfile(
                        name="fusion",
                        inherits="bm25",
                        inputs=[("query(q)", "tensor<float>(x[768])")],
                        first_phase="closeness(field, embedding)",
                        global_phase=GlobalPhaseRanking(
                            expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))",
                            rerank_count=1000,
                        ),
                    ),
                ],
            )
        ],
    )

    return package


if __name__ == '__main__':
    main()
