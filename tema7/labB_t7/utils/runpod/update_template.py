import os

import requests


def update_template(template_id: str, image_name: str, container_disk_in_gb: int = 5) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY environment variable is not set")
    if template_id.startswith("REPLACE_WITH"):
        raise RuntimeError("Set deployment.template_id in config/runpod.json")

    url = f"https://rest.runpod.io/v1/templates/{template_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "containerDiskInGb": container_disk_in_gb,
        "imageName": image_name,
    }

    response = requests.patch(url, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()
