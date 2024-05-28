# Create a function that receives a list of bounding box sizes as input and shows them on a scatter plot using matplotlib
# The function should have the following signature:

from typing import List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import json

def plot_bounding_boxes(data: Any, data_size: Any) -> None:
    """Function that takes a json fiels content which also features bounding boxes in 
    the form of x, y, width, height, calculates the bounding boxa rea and plotes them 
    on a scatter plot using matplotlib. The x axis should be the size and the y axis should be the image index.
    
    Args:
        boxes: List of bounding boxes in the form of x, y, width, height
        
    Returns:
        None
    """
    image_bb_areas = []
    for image in data["images"]:
        image_id = image["id"]
        annotations = [a for a in data["annotations"] if a["image_id"] == image_id]
        if annotations:
            for annotation in annotations:
                bbox = annotation["bbox"]
                area = bbox[2] * bbox[3]
                image_bb_areas.append((image_id, area, annotation["category_id"]))
    bb_areas_dataset1 = np.array([value[1] for value in image_bb_areas])
    print(len(bb_areas_dataset1))
    image_bb_areas = []
    for image in data_size["images"]:
        image_id = image["id"]
        annotations = [a for a in data_size["annotations"] if a["image_id"] == image_id]
        if annotations:
            for annotation in annotations:
                bbox = annotation["bbox"]
                area = bbox[2] * bbox[3]
                image_bb_areas.append((image_id, area, annotation["category_id"]))
    bb_areas_dataset2 = np.array([value[1] for value in image_bb_areas])
    print(len(bb_areas_dataset2))
    # Calculate overall min and max for bounding box sizes
    min_value = min(min(bb_areas_dataset1), min(bb_areas_dataset2))
    max_value = max(max(bb_areas_dataset1), max(bb_areas_dataset2))

    print(len(bb_areas_dataset1), len(bb_areas_dataset2))

    # Define bins based on overall range
    bins = np.linspace(min_value, max_value, num=30)  # Adjust num for desired resolution

    # Plot histograms with adjusted bins
    plt.hist(bb_areas_dataset1, bins=bins, alpha=0.5, label="Dataset Random", density=False)
    plt.hist(bb_areas_dataset2, bins=bins, alpha=0.5, label="Dataset Homogenous", density=False)

    # Customize the plot
    plt.xlabel("Bounding Box Size (pixels)")
    plt.ylabel("Frequency")
    plt.title("Dataset Comparison (Normalized)")
    plt.legend()
    plt.show()

# Test the function
# with open('./wildcamera_dataset_random/wcs_20220205_bboxes_with_classes 2_1_20240402212650_Mq6SLJWc.json') as f:
#     data = json.load(f)
# with open('./wildcamera_dataset_bb_size/wcs_20220205_bboxes_with_classes 2_1_20240401214607_QGQGLFWf.json') as f:
#     data_size = json.load(f)
# plot_bounding_boxes(data, data_size)

#print(len(data["annotations"]), len(data_size["annotations"]))

import logging
import os
import cv2

# categories: - mitu tuberosum, unknown bird, equus quagga, crax rubra, pecari tajacu, leopardus pardalis, dasyprocta punctata, madoqua guentheri, loxodonta africana, aepyceros melampus
categories = ["mitu tuberosum", "unknown bird", "equus quagga", "crax rubra", "pecari tajacu", "leopardus pardalis", "dasyprocta punctata", "madoqua guentheri", "loxodonta africana", "aepyceros melampus"]

json_filee = "1234567.json"

def create_label_files() -> None:
        """Creates the label files for the dataset.

        Args:
            None

        Returns:
            None
        """
        try:
            #with open("12345678.json", 'r') as json_file:
            #    data = json.load(json_file)
            with open(json_filee, 'r') as json_file:
                data = json.load(json_file)
            # first empty the folder
            label_folder = "wildcamera_dataset_random/full_dataset/labels"
            for file in os.listdir(label_folder):
                os.remove(os.path.join(label_folder, file))
            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                bbox = annotation["bbox"]
                category_id = annotation["category_id"]
                category = next((c for c in data["categories"] if c["id"] == category_id), None)
                category_name = category["name"]
                category_index = categories.index(category_name)
                label_path = os.path.join("wildcamera_dataset_random/full_dataset/labels", image_id + ".txt")
                print(label_path)
                image_path = os.path.join("wildcamera_dataset_random/full_dataset/images", image_id + ".jpg")
                print(image_path)
                x_center, y_center, width, height = normalize_single_bb(image_path, label_path, bbox)
                print(x_center, y_center, width, height)
                with open(label_path, "a") as label_file:
                    x_center = x_center
                    y_center = y_center
                    width = width
                    height = height
                    label_file.write(f"{category_index} {x_center} {y_center} {width} {height}\n")
            logging.info("Created label files for dataset.")
        except Exception as e:
            logging.error(f"Error creating label files: {e}")
            return
        
        
def normalize_single_bb(image_path, label_path, bbox) -> list[float]:
        """
        Normalizes a bounding box from absolute coordinates to relative coordinates.

        Args:
            image_path: Path to the image
            label_path: Path to the label file

        Returns:
            List containing four relative coordinates:
                [x_center_relative, y_center_relative, box_w_relative, box_h_relative]
        """
        if not image_path or not isinstance(image_path, str):
            ValueError("Image path must be a non-empty string.")
        if not label_path or not isinstance(label_path, str):
            ValueError("Label path must be a non-empty string.")
        try:
            cv2_image = cv2.imread(image_path)
            image_height, image_width, channels = cv2_image.shape       
            normalized_bbox = normalize_bbox(bbox, image_width, image_height)           
            logging.info(f"Normalized bounding box for image: {image_path}")
            return normalized_bbox
        except Exception as e:
            logging.error(f"Error normalizing bounding box: {e}")
            return
        
def normalize_bbox(bbox, image_width, image_height) -> list[float]:
    """
    Normalizes a bounding box from absolute coordinates to relative coordinates.

    Args:
        bbox: List containing four absolute coordinates of the bounding box:
            [x_min, y_min, width, height]
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        List containing four relative coordinates:
            [x_center_relative, y_center_relative, box_w_relative, box_h_relative]
    """
    x_center, y_center, box_w, box_h = bbox
    x_centre = (x_center + (x_center+box_w))/2
    y_centre = (y_center + (y_center+box_h))/2
    x_center_relative = x_centre / image_width
    y_center_relative = y_centre / image_height
    box_w_relative = box_w / image_width
    box_h_relative = box_h / image_height
    return [x_center_relative, y_center_relative, box_w_relative, box_h_relative]


def print_non_matching():
    """function that prints the names of images that do not have a corresponding label file and vice versa."""
    try:
        # this image 8fd0a444-92d5-11e9-9988-000d3a74c7de2 should not have a labell file 
        image_files = os.listdir("wildcamera_dataset_random/full_dataset/images")
        label_files = os.listdir("wildcamera_dataset_random/full_dataset/labels")
        image_ids = [f.split(".")[0] for f in image_files]
        label_ids = [f.split(".")[0] for f in label_files]
        missing_labels = [i for i in image_ids if i not in label_ids]
        missing_images = [i for i in label_ids if i not in image_ids]
        print(f"Images without labels: {missing_labels}")
        print(f"Labels without images: {missing_images}")
    except Exception as e:
        logging.error(f"Error printing non-matching images and labels: {e}")
        return

def remove_image_without_annotations_from_json():
    """function that removes images that do not have annotations"""
    try:
        with open(json_filee, 'r') as json_file:
            data = json.load(json_file)
        image_ids = [image["id"] for image in data["images"]]
        annotation_image_ids = [annotation["image_id"] for annotation in data["annotations"]]
        image_ids_to_remove = [i for i in image_ids if i not in annotation_image_ids]
        data["images"] = [image for image in data["images"] if image["id"] not in image_ids_to_remove]
        with open(json_filee, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        logging.error(f"Error removing images without annotations: {e}")
        return
    
def update_cetagory_count():
    """function that updates the category count in the json file"""
    try:
        with open(json_filee, 'r') as json_file:
            data = json.load(json_file)
        for category in data["categories"]:
            category_id = category["id"]
            category_name = category["name"]
            category_count = len([a for a in data["annotations"] if a["category_id"] == category_id])
            category["count"] = category_count
        with open(json_filee, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        logging.error(f"Error updating category count: {e}")
        return

def remove_image_annotation_from_json():
    """function that removes images and their annotations from the json file if they cant be found in folder"""
    try:
        with open(json_filee, 'r') as json_file:
            data = json.load(json_file)
        image_files = os.listdir("wildcamera_dataset_random/full_dataset/images")
        image_ids = [f.split(".")[0] for f in image_files]
        annotation_image_ids = [annotation["image_id"] for annotation in data["annotations"]]
        image_ids_to_remove = [i for i in annotation_image_ids if i not in image_ids]
        data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] not in image_ids_to_remove]
        with open(json_filee, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        logging.error(f"Error removing images and annotations: {e}")
        return
    
def remove_image_without_label():
    """function that removes images that do not have a corresponding label file"""
    try:
        image_files = os.listdir("wildcamera_dataset_random/full_dataset/images")
        label_files = os.listdir("wildcamera_dataset_random/full_dataset/labels")
        image_ids = [f.split(".")[0] for f in image_files]
        label_ids = [f.split(".")[0] for f in label_files]
        image_ids_to_remove = [i for i in image_ids if i not in label_ids]
        for image_id in image_ids_to_remove:
            os.remove(os.path.join("wildcamera_dataset_random/full_dataset/images", f"{image_id}.jpg"))
        with open(json_filee, 'r') as json_file:
            data = json.load(json_file)
        data["images"] = [image for image in data["images"] if image["id"] not in image_ids_to_remove]
        with open(json_filee, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        logging.error(f"Error removing images without labels: {e}")
        return

def find_file_not_image():
    """function that finds fiels in the fodler that are not images e.g. ending with.jpg"""
    try:
        image_files = os.listdir("wildcamera_dataset_random/full_dataset/images")
        for image in image_files:
            if not image.endswith(".jpg"):
                print(f"File not image: {image}")
    except Exception as e:
        logging.error(f"Error finding files that are not images: {e}")
        return
    
def remove_ds_store_from_folder():
    """function that removes .DS_Store files from the folder"""
    try:
        image_files = os.listdir("wildcamera_dataset_random/full_dataset/images")
        for image in image_files:
            if image == ".DS_Store":
                os.remove(os.path.join("wildcamera_dataset_random/full_dataset/images", image))
    except Exception as e:
        logging.error(f"Error removing .DS_Store files: {e}")
        return
    
def get_lenght():
    """function that prints length of images and labels"""
    try:
        image_files = os.listdir("wildcamera_dataset_bb_size/full_dataset/images")
        label_files = os.listdir("wildcamera_dataset_bb_size/full_dataset/labels")
        print(f"Images: {len(image_files)}")
        print(f"Labels: {len(label_files)}")
    except Exception as e:
        logging.error(f"Error printing length of images and labels: {e}")
        return
    
# create_label_files()

# print_non_matching()
# get_lenght()
# find_file_not_image()
# remove_ds_store_from_folder()
# get_lenght()

# remove_image_annotation_from_json()
# remove_image_without_annotations_from_json()
# update_cetagory_count()
get_lenght()