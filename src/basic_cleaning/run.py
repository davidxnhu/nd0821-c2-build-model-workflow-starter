#!/usr/bin/env python
"""
[An example of a step using MLflow and Weights & Biases]: Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
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

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, index_col="id")
    
    #Remove outliers for price
    min_price = args.min_price
    max_price = args.max_price

    logger.info("Dataset price outliers removal outside range: %s-%s",
                 args.min_price, args.max_price)
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    #Fix data type for last_reviews that have string in it
    logger.info("Dataset fix last_reviews that have string in it")
    df['last_review'] = pd.to_datetime(df['last_review'])

    #Remove out of range longitude and latitude
    logger.info("Dataset removing out of range longitude: (-74.25, -73.50) and latitude: (40.5, 41.2)")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    output_artifact_path = os.path.join(".", args.output_artifact)
    df.to_csv(output_artifact_path)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )

    
    artifact.add_file(output_artifact_path)
    run.log_artifact(artifact)

    artifact.wait()
    logger.info("Cleaned dataset stored at wandb")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


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
        help="Min price limit for outlier",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=int,
        help="Max price limit for outlier",
        required=True
    )


    args = parser.parse_args()

    go(args)
