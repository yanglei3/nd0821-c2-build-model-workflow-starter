#!/usr/bin/env python
"""
Performs basic cleaning on the data and save the results in Weights & Biases
"""
import argparse
import logging
import wandb
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path, index_col='id')
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    logger.info('Removed price outliers, range info %s-%s', args.min_price, args.max_price)

    df['last_review'] = pd.to_datetime(df['last_review'])
    logger.info("Fixed last_review data type ")

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    tmp_artifact_path = os.path.join(args.tmp_directory, args.output_artifact)
    df.to_csv(tmp_artifact_path)
    logger.info("Temporary artifact saved to %s", tmp_artifact_path)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )

    artifact.add_file(tmp_artifact_path)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Uploaded artifact the cleaned dataset to wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="These steps clean the data")

    parser.add_argument(
        "--tmp_directory",
        type=str,
        help="Temporary directory for dataset storage",
        required=True
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact name",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=int,
        help="Minimum price limit",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help="Maximum price limit",
        required=True
    )

    args = parser.parse_args()

    go(args)
