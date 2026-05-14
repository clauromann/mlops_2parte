from dotenv import load_dotenv
load_dotenv()

from prefect import serve

from f01_process_data import process_data_pipeline
from f02_build_training_image import build_training_image_pipeline
from f04_deploy_serverless import deploy_serverless_pipeline


if __name__ == "__main__":
    serve(
        process_data_pipeline.to_deployment(name="Titanic RunPod - Process data"),
        build_training_image_pipeline.to_deployment(name="Titanic RunPod - Build training image"),
        deploy_serverless_pipeline.to_deployment(name="Titanic RunPod - Serverless deployment"),
    )
