from dotenv import load_dotenv
load_dotenv()

from prefect import serve

from f01_monolith import full_pipeline
from f02_decoupled import process_data_pipeline, train_model_pipeline


if __name__ == "__main__":
    serve(
        full_pipeline.to_deployment(name="Titanic Monolith - Full pipeline"),
        process_data_pipeline.to_deployment(name="Titanic Decoupled - Process data"),
        train_model_pipeline.to_deployment(name="Titanic Decoupled - Train model"),
    )
