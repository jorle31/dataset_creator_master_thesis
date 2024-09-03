
import json
from typing import List, Any

def remove_bounding_boxes(data: Any, names: List[str]) -> Any:
    """Function that takes a json file and removes all bounding boxes which at image_id feature a name of an image in a list of names.
    
    Args:
        data: Json file content
        names: List of names to remove

    Returns:
        Json file content with bounding boxes removed
    """
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] not in names]

    with open("1234567.json", "w") as f:
        json.dump(data, f, indent=4)
    return data

import os

def get_image_names(folder_path):
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_name = os.path.splitext(filename)[0]
            image_names.append(image_name)
    return image_names

folder_path = './wildcamera_dataset_random/test/images'

image_names = get_image_names(folder_path)
print(image_names)

with open("1234567.json") as f:
    data = json.load(f)
remove_bounding_boxes(data, image_names)
