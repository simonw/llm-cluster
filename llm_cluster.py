import click
import json
import llm
import numpy as np
import sklearn.cluster
import sqlite_utils
import textwrap

DEFAULT_SUMMARY_PROMPT = """
Short, concise title for this cluster of related documents.
""".strip()


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("collection")
    @click.argument("n", type=int)
    @click.option(
        "-d",
        "--database",
        type=click.Path(
            file_okay=True, allow_dash=False, dir_okay=False, writable=True
        ),
        envvar="LLM_EMBEDDINGS_DB",
    )
    @click.option(
        "--summary", is_flag=True, help="Generate summary title for each cluster"
    )
    @click.option("-m", "--model", help="LLM model to use for the summary")
    @click.option("--prompt", help="Custom prompt to use for the summary")
    def cluster(collection, n, database, summary, model, prompt):
        """
        Generate clusters from embeddings in a collection

        Example usage, to create 10 clusters:

        \b
            llm cluster my_collection 10

        Outputs a JSON array of {"id": "cluster_id", "items": [list of items]}

        Pass --summary to generate a summary for each cluster, using the default
        language model or the model you specify with --model.
        """
        from llm.cli import get_default_model, get_key

        clustering_model = sklearn.cluster.MiniBatchKMeans(n_clusters=n, n_init="auto")
        if database:
            db = sqlite_utils.Database(database)
        else:
            db = sqlite_utils.Database(llm.user_dir() / "embeddings.db")
        rows = [
            (row[0], llm.decode(row[1]), row[2])
            for row in db.execute(
                """
            select id, embedding, content from embeddings
            where collection_id = (
                select id from collections where name = ?
            )
        """,
                [collection],
            ).fetchall()
        ]
        to_cluster = np.array([item[1] for item in rows])
        clustering_model.fit(to_cluster)
        assignments = clustering_model.labels_
        # Each one corresponds to an ID
        clusters = {}
        for (id, _, content), cluster in zip(rows, assignments):
            clusters.setdefault(str(cluster), []).append(
                {
                    "id": str(id),
                    "content": content[:100] if content else None,
                }
            )
        # Re-arrange into a list
        output_clusters = [{"id": k, "items": v} for k, v in clusters.items()]

        # Do we need to generate summaries?
        if summary:
            model = llm.get_model(model or get_default_model())
            if model.needs_key:
                model.key = get_key("", model.needs_key, model.key_env_var)
            prompt = prompt or DEFAULT_SUMMARY_PROMPT
            click.echo("[")
            for cluster, is_last in zip(
                output_clusters, [False] * (len(output_clusters) - 1) + [True]
            ):
                click.echo("  {")
                click.echo('    "id": {},'.format(json.dumps(cluster["id"])))
                click.echo(
                    '    "items": '
                    + textwrap.indent(
                        json.dumps(cluster["items"], indent=2), "    "
                    ).lstrip()
                    + ","
                )
                prompt_content = "\n".join(
                    [item["content"] for item in cluster["items"] if item["content"]]
                )
                if prompt_content.strip():
                    summary = model.prompt(
                        prompt_content,
                        system=prompt,
                    ).text()
                else:
                    summary = None
                click.echo('    "summary": {}'.format(json.dumps(summary)))
                click.echo("  }" + ("," if not is_last else ""))
            click.echo("]")
        else:
            click.echo(json.dumps(output_clusters, indent=4))
