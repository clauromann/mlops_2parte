import argparse
import os

import requests


def update_template(template_id: str, image_name: str) -> dict:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY environment variable is not set")

    url = f"https://rest.runpod.io/v1/templates/{template_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "containerDiskInGb": 5,
        "imageName": image_name,
    }

    response = requests.patch(url, json=payload, headers=headers)
    response.raise_for_status()
    print(response.text)
    return response.json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update a RunPod template image")
    parser.add_argument("--template-id", required=True)
    parser.add_argument("--image-name", required=True)
    args = parser.parse_args()
    update_template(args.template_id, args.image_name)
