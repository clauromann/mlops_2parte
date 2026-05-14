import os

import requests


def update_endpoint(endpoint_id: str, template_id: str) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY environment variable is not set")
    if endpoint_id.startswith("REPLACE_WITH"):
        raise RuntimeError("Set deployment.endpoint_id in config/runpod.json")

    url = f"https://rest.runpod.io/v1/endpoints/{endpoint_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "templateId": template_id,
    }

    response = requests.patch(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()
