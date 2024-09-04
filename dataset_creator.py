"""File that contaisn the logic for creating a custom dataset from the 
wcs_20220205_bboxes_with_classes json file."""


import random
import json
import logging
import datetime
import os
import string
from typing import Union
import aiohttp
import asyncio
from certifi import where
import ssl
import aiofiles
import numpy as np
import collections
import cv2
import itertools
import requests
import shutil
import random
from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold


class DatasetCreator():
    """Class to create a custom dataset from a JSON file."""


    def __init__(self, copied_filename:str=None, filter_classes:list[str]=None, categories:list[str]=None, dataset_folder:str=None, failed_images:list[str]=None):
        self.blacklist:list[str] = ["group",  "vehicle", "motorcycle", "start", "end", "#ref!", "unknown", "human", "bos taurus", "meleagris ocellata"]
        self.copied_filename:str = copied_filename
        self.filter_classes:list = []
        self.categories:list[str] = []
        self.dataset_folder:str = dataset_folder
        self.failed_images:list[str] = []


    def update_blacklist(self, blacklist:list[str]) -> None:
        """Updates the blacklist of categories. 

        Args:
            blacklist: A list of categories to exclude from the dataset.

        Returns:
            None
        """
        if not blacklist or not isinstance(blacklist, list[str]):
            ValueError("Blacklist must be a non-empty list of strings.")
        try:
            self.blacklist = blacklist
            logging.info(f"Updated blacklist: {self.blacklist}")
        except Exception as e:
            logging.error(f"Error updating blacklist: {e}")
            return


    def generate_unique_filename(self, original_filename:str) -> Union[str, None]:
        """Generates a unique filename for the copied JSON file.

        Args:
            original_filename: The original JSON file to copy.

        Returns:
            new_filename: The newly generated unique filename.
        """
        if not os.path.exists(original_filename):
            ValueError(f"File {original_filename} does not exist.")
        try:
            base, ext = map(str,os.path.splitext(original_filename))
            random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            new_filename = f"{base}_{timestamp}_{random_string}.json"
            self.copied_filename = new_filename
            logging.info(f"Generated unique filename: {new_filename}")
            return new_filename
        except Exception as e:
            logging.error(f"Error generating unique filename: {e}")
            return None


    def copy_json_with_unique_name(self, original_filename:str) -> None:
        """Copies a JSON file with a unique filename.

        Args:
            original_filename: The original JSON file to copy.

        Returns:
            None
        """
        if not os.path.exists(original_filename):
            ValueError(f"File {original_filename} does not exist.")
        try:
            new_filename = self.generate_unique_filename(original_filename=original_filename)
            with open(original_filename, 'r') as original_file:
                data = json.load(original_file)
            with open(new_filename, 'w') as new_file:
                json.dump(data, new_file, indent=4)
            logging.info(f"Created new JSON file: {new_filename}")
        except Exception as e:
            logging.error(f"Error copying JSON file: {e}")
            return


    def filter_categories_min_max_count_and_blacklist(self, data:dict, min_count:int, max_count:int) -> None:
        """Filters categories based on minimum/maximum count and blacklist. This is to ensure
        that only categories with a minimum count are included in the dataset making it more
        likely, that they are featured within the images with bounding boxes.

        Args:
            data: The JSON data.
            min_count: The minimum count for a category to be included.
            max_count: The maximum count for a category to be included.

        Returns:
            None
        """
        if not data or not isinstance(data, dict):
            ValueError("Data must be a non-empty dictionary.")
        if not min_count or not isinstance(min_count, int):
            ValueError("Minimum count must be a non-empty integer.")
        if not max_count or not isinstance(max_count, int) or max_count < min_count:
            ValueError("Minimum count must be of type None or a non-empty integer bigger than min_count.")
        try:
            filtered_categories = []
            for category in data["categories"]:
                category_name = category["name"]
                if category_name not in self.blacklist:
                    category_count = category.get("count", 0)  # In case count is not present
                    if min_count <= category_count <= max_count if max_count else min_count <= category_count:
                        filtered_categories.append(category)
                    elif category_name in self.filter_classes:
                        logging.info(f"Excluded category {category_name} with count {category_count}")
                        self.filter_classes.remove(category_name)
            data["categories"] = filtered_categories
            with open(self.copied_filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            logging.info(f"Filtered categories with min count and blacklist and updated JSON file: {self.copied_filename}")
        except Exception as e:
            logging.error(f"Error filtering categories: {e}")
            return


    def pick_n_categories_and_clean_dataset(self, data:dict, max_attempts:int, class_count:int, annotation_amount:int, num_bb_per_class:int) -> Union[list[str], None]:
        """Selects n classes from the JSON data and cleans the dataset. Provided a class_count the
        function will attempt to select n classes from the JSON data. The function will also
        remove any classes that were not selected and their annotations and images. If the function
        is unable to provide n classes with the needed number of bounding boxes, the function will
        attempt to select more than n classes that meet the criteria. If the iterations run out
        the function will just return the currently selected classes, even if they do not meet the
        criteria.

        Args:
            data: The JSON data.
            max_attempts: The number of iterations to attempt to select classes.
            class_count: The number of classes to select.
            annotation_amount: The number of annotations that the images need to provide. 
                It could be that some images have no annotations. They are removed.
            num_bb_per_class: The minimum number of bounding boxes per class.

        Returns:
            selected_classes: The list of selected class names.
        """
        if not data or not isinstance(data, dict):
            ValueError("Data must be a non-empty dictionary.")
        if not max_attempts or not isinstance(max_attempts, int):
            ValueError("Max attempts must be a non-empty integer.")
        if not class_count or not isinstance(class_count, int):
            ValueError("Class count must be a non-empty integer.")
        if not annotation_amount or not isinstance(annotation_amount, int):
            ValueError("Image amount must be a non-empty integer.")
        if not num_bb_per_class or not isinstance(num_bb_per_class, int):
            ValueError("Number of bounding boxes per class must be a non-empty integer.")
        try:
            annotation_amount = annotation_amount
            selected_classes = set()
            total_bounding_boxes = 0
            explicitly_provided_classes = set(self.filter_classes) if self.filter_classes else set()
            iterations = 0
            if len(explicitly_provided_classes) == class_count:
                remaining_category_ids = {c["id"] for c in data["categories"] if c["name"] in explicitly_provided_classes}
                data["categories"] = [c for c in data["categories"] if c["id"] in remaining_category_ids]
                data["annotations"] = [a for a in data["annotations"] if a["category_id"] in remaining_category_ids]
                with open(self.copied_filename, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                logging.info(f"Selected classes: {explicitly_provided_classes}")
                return list(explicitly_provided_classes)
            else: 
                for class_name in explicitly_provided_classes:
                    category = next((c for c in data["categories"] if c["name"] == class_name), None)
                    if category:
                        class_bounding_boxes = sum(1 for annotation in data["annotations"] if annotation["category_id"] == category["id"])
                        if class_bounding_boxes >= num_bb_per_class:
                            selected_classes.add(class_name)
                            total_bounding_boxes += class_bounding_boxes
                        else:
                            logging.info(f"Class '{class_name}' removed due to insufficient bounding boxes.")
                while len(selected_classes) < class_count and iterations < max_attempts:
                    if len(data["categories"]) < class_count - len(selected_classes):
                        logging.warning("Not enough classes available to meet the specified count.")
                        break
                    random_class = random.choice(data["categories"])
                    class_name = random_class["name"]
                    if class_name not in selected_classes and class_name not in self.blacklist and class_name not in explicitly_provided_classes:
                        class_bounding_boxes = sum(1 for annotation in data["annotations"] if annotation["category_id"] == random_class["id"])
                        if class_bounding_boxes >= num_bb_per_class:
                            selected_classes.add(class_name)
                            total_bounding_boxes += class_bounding_boxes
                            logging.info(f"Added class '{class_name}' with {class_bounding_boxes} bounding boxes.")
                    iterations += 1
                while total_bounding_boxes < annotation_amount and iterations < max_attempts:
                    class_to_remove = random.choice(list(selected_classes - explicitly_provided_classes))
                    selected_classes.remove(class_to_remove)
                    total_bounding_boxes -= sum(1 for annotation in data["annotations"] if annotation["category_id"] in [c["id"] for c in data["categories"] if c["name"] == class_to_remove])
                    new_class = None
                    for category in data["categories"]:
                        if category["name"] not in selected_classes and category["name"] not in self.blacklist and category["name"] not in explicitly_provided_classes:
                            class_bounding_boxes = sum(1 for annotation in data["annotations"] if annotation["category_id"] == category["id"])
                            if class_bounding_boxes >= num_bb_per_class and (new_class is None or class_bounding_boxes > new_class[1]):
                                new_class = (category["name"], class_bounding_boxes)
                                logging.info(f"Added class '{category['name']}' with {class_bounding_boxes} bounding boxes.")
                    if new_class:
                        selected_classes.add(new_class[0])
                        total_bounding_boxes += new_class[1]
                    iterations += 1
                remaining_category_ids = {c["id"] for c in data["categories"] if c["name"] in selected_classes}
                data["categories"] = [c for c in data["categories"] if c["id"] in remaining_category_ids]
                data["annotations"] = [a for a in data["annotations"] if a["category_id"] in remaining_category_ids]
                with open(self.copied_filename, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                with open(self.copied_filename, 'r') as json_file:
                    data = json.load(json_file)
                    annotated_image_ids = {annotation["image_id"] for annotation in data["annotations"]}
                    data["images"] = [img for img in data["images"] if img["id"] in annotated_image_ids]
                with open(self.copied_filename, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                logging.info(f"Selected classes: {selected_classes}, Total images: {len(data['images'])}, Total categories: {len(data['categories'])}, Total annotations: {len(data['annotations'])}")
                return list(selected_classes)
        except Exception as e:
            logging.error(f"Error selecting classes: {e}")
            return None

        """Deletes images with multiple bounding boxes.

        Args:
            data: The JSON data.

        Returns:
            None
        """
        if not data or not isinstance(data, dict):
            ValueError("Data must be a non-empty dictionary.")
        try:
            image_bb_count = {}
            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                if image_id in image_bb_count:
                    image_bb_count[image_id] += 1
                else:
                    image_bb_count[image_id] = 1
            images_to_remove = [image_id for image_id, count in image_bb_count.items() if count > 1]
            data["images"] = [img for img in data["images"] if img["id"] not in images_to_remove]
            data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] not in images_to_remove]
            with open(self.copied_filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            logging.info(f"Deleted images with multiple bounding boxes.")
            logging.info(f"Remaining images: {len(data['images'])}")
        except Exception as e:
            logging.error(f"Error deleting images with multiple bounding boxes: {e}")
            return
        

    def random_filter(self, data: dict, target_count: int, class_count: int) -> None:
        """Filters the dataset based on the selected classes and class limit.

        Args:
            data: The JSON data.
            target_count: The target count of images to keep.

        Returns:
            None
        """
        if not data or not isinstance(data, dict):
            ValueError("Data must be a non-empty dictionary.")
        if not target_count or not isinstance(target_count, int):
            ValueError("Target count must be a non-empty integer.")
        try:
            filtered_annotations = [
                annotation for annotation in data["annotations"]
                if any(c["id"] == annotation["category_id"] and c["name"] in self.categories for c in data["categories"])
            ]
            filtered_image_ids = {annotation["image_id"] for annotation in filtered_annotations}
            filtered_images = [img for img in data["images"] if img["id"] in filtered_image_ids]
            data["annotations"] = filtered_annotations
            data["images"] = filtered_images
            with open(self.copied_filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            logging.info(f"Filtered dataset with target count {target_count}")
            data = json.load(open(self.copied_filename, 'r'))
            class_limit = target_count // class_count * 2.5
            images = data["images"]
            category_counts = collections.Counter()
            selected_image_ids = set()
            while len(selected_image_ids) < target_count:
                print(len(selected_image_ids))
                image_id = random.choice([img["id"] for img in images])
                category_id = next((a["category_id"] for a in data["annotations"] if a["image_id"] == image_id), None)

                if category_id and category_counts[category_id] < class_limit:
                    selected_image_ids.add(image_id)
                    category_counts[category_id] += 1

            removed_image_ids = set(img["id"] for img in images if img["id"] not in selected_image_ids)
            data["images"] = [img for img in images if img["id"] in selected_image_ids]
            data["annotations"] = [annotation for annotation in data["annotations"] if annotation["image_id"] not in removed_image_ids]
            # update image count in categories
            for category in data["categories"]:
                category["count"] = category_counts[category["id"]]
            with open(self.copied_filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)
            logging.info(f"Filtered dataset with target count {target_count}")
        except Exception as e:
            logging.error(f"Error filtering dataset: {e}")
            return


    def create_folders(self, parent_folder:str) -> None:
        """Creates the folders for the dataset.

        Args:
            None

        Returns:
            None
        """
        try:
            os.makedirs(parent_folder, exist_ok=True)
            subfolders = ["test", "valid", "train"]
            for folder in subfolders:
                folder_path = os.path.join(parent_folder, folder)
                os.makedirs(os.path.join(folder_path, "images"), exist_ok=True)
                os.makedirs(os.path.join(folder_path, "labels"), exist_ok=True)
            data_yaml_path = os.path.join(parent_folder, "data.yaml")
            with open(data_yaml_path, "w") as f:
                f.write("names:\n")
                for category in self.categories:
                    f.write(f"- {category}\n")
                f.write(f"nc: {len(self.categories)}\n")
                f.write("test: ../test/images\n")
                f.write("train: ../train/images\n")
                f.write("val: ../valid/images\n")
            self.dataset_folder = parent_folder
            logging.info(f"Created folders for dataset: {parent_folder}")
        except Exception as e:
            logging.error(f"Error creating folders: {e}")
            return


    def create_label_files(self) -> None:
        """Creates the label files for the dataset.

        Args:
            None

        Returns:
            None
        """
        try:
            with open(self.copied_filename, 'r') as json_file:
                data = json.load(json_file)
            for annotation in data["annotations"]:
                image_id = annotation["image_id"]
                bbox = annotation["bbox"]
                category_id = annotation["category_id"]
                category = next((c for c in data["categories"] if c["id"] == category_id), None)
                category_name = category["name"]
                category_index = self.categories.index(category_name)
                label_filename = f"{image_id}.txt"
                label_path = os.path.join(self.dataset_folder, "train", "labels", label_filename)
                with open(label_path, "a") as label_file:
                    x_center = bbox[0] + (bbox[2] / 2)
                    y_center = bbox[1] + (bbox[3] / 2)
                    width = bbox[2] 
                    height = bbox[3] 
                    label_file.write(f"{category_index} {x_center} {y_center} {width} {height}\n")
            logging.info("Created label files for dataset.")
        except Exception as e:
            logging.error(f"Error creating label files: {e}")
            return


    def check_class_distribution(self, data:dict) -> None:
        """Checks the class distribution of the dataset.

        Args:
            data: The JSON data.

        Returns:
            None
        """
        if not data or not isinstance(data, dict):
            ValueError("Data must be a non-empty dictionary.")
        try:
            print(data["categories"])
            class_distribution = {}
            for annotation in data["annotations"]:
                category_id = annotation["category_id"]
                category = next((c for c in data["categories"] if c["id"] == category_id), None)
                category_name = category["name"]
                if category_name not in class_distribution:
                    class_distribution[category_name] = 0
                class_distribution[category_name] += 1
            logging.info(f"Annotation distribution: {class_distribution}")
        except Exception as e:
            logging.error(f"Error checking class distribution: {e}")
            return


    def normalize_bbox(self, bbox, image_width, image_height) -> list[float]:
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
        x_center_relative = (x_center / image_width)
        y_center_relative = (y_center / image_height)
        box_w_relative = box_w / image_width
        box_h_relative = box_h / image_height
        return [x_center_relative, y_center_relative, box_w_relative, box_h_relative]
        

    def normalize_single_bb(self, image_path, label_path) -> list[float]:
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
            with open(label_path, "r+") as label_file:
                lines = label_file.readlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    class_name = parts[0]
                    bbox = list(map(float, parts[1:]))
                    normalized_bbox = self.normalize_bbox(bbox, image_width, image_height)
                    normalized_bbox_str = " ".join([str(b) for b in normalized_bbox])
                    new_line = f"{class_name} {normalized_bbox_str}\n"
                    new_lines.append(new_line)
                label_file.seek(0)
                label_file.truncate()
                label_file.writelines(new_lines)
            logging.info(f"Normalized bounding box for image: {image_path}")
        except Exception as e:
            logging.error(f"Error normalizing bounding box: {e}")
            return


    async def download_image(self, session, image_url:str, parent_folder:str, folder:str, image_id:str) -> None:
        """Downloads an image from a URL.
        
        Args:
            session: The aiohttp session.
            image_url: The URL of the image to download.
            parent_folder: The parent folder to save the image.
            folder: The folder to save the image.
            image_id: The ID of the image.
            
        Returns:
            None
        """
        if not session:
            ValueError("Session must be a aiohttp session.")
        if not image_url or not isinstance(image_url, str):
            ValueError("Image URL must be a non-empty string.")
        if not parent_folder or not isinstance(parent_folder, str):
            ValueError("Parent folder must be a non-empty string.")
        if not folder or not isinstance(folder, str):
            ValueError("Folder must be a non-empty string.")
        if not image_id or not isinstance(image_id, str):
            ValueError("Image ID must be a non-empty string.")
        try:
            image_path = os.path.join(parent_folder, folder, "images", image_id + ".jpg")
            if os.path.exists(image_path):
                logging.info(f"Image already downloaded: {image_id}")
                return
            async with session.get(image_url, timeout=720) as response:
                response.raise_for_status()
                image_data = await response.read()
                async with aiofiles.open(image_path, "wb") as f:
                    await f.write(image_data)
            logging.info(f"Successfully downloaded image: {image_url}")
        except Exception as e:
            logging.error(f"Error downloading image: {e}")   
            logging.info(f"Blob not found: {image_url} (Skipping)")
            self.failed_images.append(image_id)
            with open("failed_images.txt", "w") as f:
                for image_id in self.failed_images:
                    f.write(image_id + "\n")


    async def download_images(self, train_split:int, test_split:int, dataset_name:str) -> list[str]:
        """Downloads images from a JSON file.
        
        Args:
            train_split: The percentage of images to use for training.
            test_split: The percentage of images to use for testing.
            dataset_name: The name of the dataset.
            
        Returns:
            None
        """
        if not train_split or not isinstance(train_split, float):
            ValueError("Train split must be a non-empty float.")
        if not dataset_name or not isinstance(dataset_name, str):
            ValueError("Dataset name must be a non-empty string.")
        if not test_split or not isinstance(test_split, float):
            ValueError("Test split must be a non-empty float.")
        try:
            logging.getLogger().setLevel(logging.INFO)
            with open(self.copied_filename, 'r') as json_file:
                data = json.load(json_file)
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    ssl=ssl.create_default_context(
                        cafile=where()))) as session:
                tasks = []
                iter = 0
                for image_chunk in itertools.batched(data["images"], 1000):
                    logging.info(f"Entering new chunk. Chunk {iter}")
                    for image in image_chunk:
                        image_id = image["id"]
                        file_name = image["file_name"]
                        image_url = f"https://lilawildlife.blob.core.windows.net/lila-wildlife/wcs-unzipped/{file_name}"
                        folder = "train"
                        tasks.append(self.download_image(session, image_url, self.dataset_folder, folder, image_id))
                    await asyncio.gather(*tasks)
                    logging.info(f"Completed chunk {iter}")
                    iter += 1
                    tasks = []
            all_images = [image["id"] for image in data["images"]]
            for image in all_images:
                image_path = os.path.join(self.dataset_folder, "train/images", image + ".jpg")
                label_path = os.path.join(self.dataset_folder, "train/labels", image + ".txt")
                self.normalize_single_bb(image_path, label_path)
            if self.failed_images:
                for image_id in self.failed_images:
                    logging.info(f"Image could not be downlaoded: {image_id}")
            self.split_dataset(data=data, train_split=train_split, test_split=test_split, dataset_name=dataset_name)
        except Exception as e:
            logging.error(f"Error downloading images: {e}")
        return self.failed_images


    def split_dataset(self, data: dict, train_split: float, test_split: float, dataset_name: str) -> None:
        """Splits the dataset into training, validation, and test sets."""
        if not data or not isinstance(data, dict):
            raise ValueError("Data must be a non-empty dictionary.")
        if not isinstance(train_split, float) or not (0 < train_split < 1):
            raise ValueError("Train split must be a float between 0 and 1.")
        if not isinstance(test_split, float) or not (0 < test_split < 1):
            raise ValueError("Test split must be a float between 0 and 1.")
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError("Dataset name must be a non-empty string.")
        try:
            logging.getLogger().setLevel(logging.INFO)
            all_images = [image["id"] for image in data["images"]]
            images = {}
            for image in all_images:
                folder, filename = os.path.split(image)
                if folder not in images:
                    images[folder] = []
                images[folder].append(image)
            total_split = train_split + test_split
            epsilon = 1e-6
            if abs(total_split - 1) > epsilon:
                raise ValueError("Train and test split ratios must add up to 1.")
            for name, images in images.items():
                num_images = len(images)
                logging.info(f"Total images: {num_images}")
            all_images = list(images)
            test_size = int(num_images * test_split)
            print(f"Test size: {test_size}")
            test_images = random.sample(all_images, test_size)
            for image in test_images:
                os.rename(os.path.join(dataset_name, "train/images", image + ".jpg"), os.path.join(dataset_name, "test/images", image + ".jpg"))
                os.rename(os.path.join(dataset_name, "train/labels", image + ".txt"), os.path.join(dataset_name, "test/labels", image + ".txt"))
        except Exception as e:
            logging.info(f"Error splitting dataset: {e}")
            return


    def create_dataset(self, filename:str, min_count:int, max_count:int, parent_folder:str, classes_count:int, image_amount:int, classes:list[str]=None, max_attempts:int=100) -> None:
        """Creates a custom dataset from the JSON file.

        Args:
            filename: The JSON file to create the dataset from. Default is wcs_20220205_bboxes_with_classes.json when downloaded.
            min_count: The minimum image count for a category to be included.
            max_count: The maximum image count for a category to be included.
            parent_folder: The parent folder to save the dataset to. Default is wildcamera_dataset.
            classes_count: The number of classes to be featured in the new dataset.
            image_amount: The number of images to be included in the new dataset.
            classes: A list of class names that have to be featured in the resulting dataset (optional).
            iterations: The number of iterations to attempt to select classes (optional).

        Returns:
            None
        """
        if not filename or not isinstance(filename, str):
            ValueError("Filename must be a non-empty string.")
        if not min_count or not isinstance(min_count, int):
            ValueError("Minimum count must be a non-empty integer.")
        if not max_count or not isinstance(max_count, int):
            ValueError("Maximum count must be a non-empty integer.")
        if not classes_count or not isinstance(classes_count, int) or classes_count < 1:
            ValueError("Classes count must be a non-empty integer bigger than 0.")
        if not image_amount or not isinstance(image_amount, int):
            ValueError("Image amount must be a non-empty integer.")
        if not max_attempts or not isinstance(max_attempts, int):
            ValueError("Max attempts must be a non-empty integer.")
        if not parent_folder or not isinstance(parent_folder, str):
            ValueError("Parent folder must be a non-empty string.")
        try:
            logging.getLogger().setLevel(logging.INFO)
            self.copy_json_with_unique_name(original_filename=filename)
            with open(self.copied_filename, 'r') as json_file:
                data = json.load(json_file)
            self.filter_classes = classes if classes else []
            self.filter_categories_min_max_count_and_blacklist(
                data=data,
                min_count=min_count,
                max_count=max_count
            )
            with open(self.copied_filename, 'r') as json_file:
                data = json.load(json_file)
            self.categories = self.pick_n_categories_and_clean_dataset(data=data, max_attempts=max_attempts, class_count=classes_count, annotation_amount=image_amount, num_bb_per_class=min_count)
            with open(self.copied_filename, 'r') as json_file:
                data = json.load(json_file)
            print("Cleaning dataset...")
            self.random_filter(data=data, target_count=image_amount, class_count=classes_count)
            logging.info(f"Created dataset target count {image_amount}")
            with open(self.copied_filename, 'r') as json_file:
                data = json.load(json_file)
            logging.info(f"Dataset created with {len(data['images'])} images.")
            logging.info(f"Dataset created with {len(data['categories'])} categories.")
            logging.info(f"Dataset created with {len(data['annotations'])} annotations.")
            self.check_class_distribution(data=data)
            self.create_folders(parent_folder=parent_folder)
            self.create_label_files()
        except Exception as e:
            logging.error(f"Error creating dataset: {e}")
            return
    

    def create_k_fold_split(self, dataset_name:str, k_splits:int) -> None:
        """Creates a k-fold split of the dataset.

        Args:
            dataset_name: The name of the dataset.
            k_splits: The number of splits to create.

        Returns:
            None
        """
        if not dataset_name or not isinstance(dataset_name, str):
            ValueError("Dataset name must be a non-empty string.")
        if not k_splits or not isinstance(k_splits, int):
            ValueError("K splits must be a non-empty integer.")
        try:
            logging.getLogger().setLevel(logging.INFO)
            logging.info(f"Length of train folder (before): {len(os.listdir(os.path.join(dataset_name, 'train', 'images')))}")
            logging.info(f"Length of train folder (before): {len(os.listdir(os.path.join(dataset_name, 'train', 'labels')))}")
            logging.info(f"Length of val folder (before): {len(os.listdir(os.path.join(dataset_name, 'valid', 'images')))}")
            logging.info(f"Length of val folder (before): {len(os.listdir(os.path.join(dataset_name, 'valid', 'labels')))}")
            self.organize_dataset(dataset_path=dataset_name, keep_percent=0.0)
            logging.info(f"Length of train folder images (after): {len(os.listdir(os.path.join(dataset_name, 'train', 'images')))}")
            logging.info(f"Length of train folder labels(after): {len(os.listdir(os.path.join(dataset_name, 'train', 'labels')))}")
            logging.info(f"Length of val folder images (after): {len(os.listdir(os.path.join(dataset_name, 'valid', 'images')))}")
            logging.info(f"Length of val folder labels (after): {len(os.listdir(os.path.join(dataset_name, 'valid', 'labels')))}")
            logging.info(f"Length of full dataset folder images (after): {len(os.listdir(os.path.join(dataset_name, 'full_dataset', 'images')))}")
            logging.info(f"Length of full dataset folder labels (after): {len(os.listdir(os.path.join(dataset_name, 'full_dataset', 'labels')))}")
            labels_df, classes, cls_idx, indx, labels  = self.get_labels_df(dataset_path=dataset_name)
            logging.info(f"Done creating labels dataframe.")
            fold_lbl_distrb, folds_df, save_path = self.create_k_folds(k_splits=k_splits, dataset_path=dataset_name, labels_df=labels_df, classes=classes, cls_idx=cls_idx, indx=indx, labels=labels)
            logging.info(f"Done creating k-fold split.")
            self.split_to_csv(fold_lbl_distrb=fold_lbl_distrb, folds_df=folds_df, save_path=save_path, dataset_path=dataset_name)
            logging.info(f"Done saving k-fold split to CSV.")
        except Exception as e:
            logging.error(f"Error creating k-fold split: {e}")
            return


    def move_train_files_to_dataset(self, dataset_path, destination_dir, image_paths, transfer_count) -> None:
        """Moves specified images and their corresponding labels to a new dataset directory.
        Args:
            dataset_path (str): Path to the root dataset directory.
            destination_dir (str): Name of the destination dataset directory.
            image_paths (list): List of image paths to move.
            transfer_count (int): Number of images to move.
        """

        image_destination_dir = os.path.join(dataset_path, destination_dir, 'images')
        os.makedirs(image_destination_dir, exist_ok=True)
        label_destination_dir = os.path.join(dataset_path, destination_dir, 'labels')
        os.makedirs(label_destination_dir, exist_ok=True)
        for image_path in image_paths[:transfer_count]:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(dataset_path, 'train', 'labels', f"{base_filename}.txt")
            shutil.move(image_path, image_destination_dir)
            shutil.move(label_path, label_destination_dir)
        

    def move_valid_files_to_dataset(self, dataset_path, destination_dir, image_paths) -> None:
        """Moves specified images and their corresponding labels to a new dataset directory.
        Args:
            dataset_path (str): Path to the root dataset directory.
            destination_dir (str): Name of the destination dataset directory.
            image_paths (list): List of image paths to move.
            transfer_count (int): Number of images to move.
        """
        image_destination_dir = os.path.join(dataset_path, destination_dir, 'images')
        os.makedirs(image_destination_dir, exist_ok=True)
        label_destination_dir = os.path.join(dataset_path, destination_dir, 'labels')
        os.makedirs(label_destination_dir, exist_ok=True)
        for image_path in image_paths:
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(dataset_path, 'valid', 'labels', f"{base_filename}.txt")
            shutil.move(image_path, image_destination_dir)
            shutil.move(label_path, label_destination_dir)


    def organize_dataset(self, dataset_path, keep_percent):
        """Organizes the dataset by moving a portion of images to a separate 'full_dataset' directory.
        Args:
            dataset_path (str): Path to the root dataset directory.
            keep_percent (float): Percentage of images to keep in the 'train' directory.
        """
        image_count = len(os.listdir(os.path.join(dataset_path, 'train', 'images')))
        transfer_count = int(image_count * (1 - keep_percent))
        image_paths = [
            os.path.join(dataset_path, 'train', 'images', filename)
            for filename in os.listdir(os.path.join(dataset_path, 'train', 'images'))
        ]
        random.shuffle(image_paths)
        self.move_train_files_to_dataset(dataset_path, 'full_dataset', image_paths, transfer_count)
        image_paths = [
            os.path.join(dataset_path, 'valid', 'images', filename)
            for filename in os.listdir(os.path.join(dataset_path, 'valid', 'images'))
        ]
        self.move_valid_files_to_dataset(dataset_path, 'full_dataset', image_paths)


    def get_labels_df(self, dataset_path) -> Union[pd.DataFrame, list[str], list[int], list[str], list[str]]:
        """Creates a pandas DataFrame from the labels in the dataset.
        Returns:
            pd.DataFrame: DataFrame containing the labels.
        """
        labels = sorted(Path(dataset_path).rglob("*labels/*.txt"))
        labels = [label for label in labels if "test" not in label.parts]
        yaml_file = dataset_path + '/data.yaml'
        with open(yaml_file, 'r', encoding="utf8") as y:
            classes = yaml.safe_load(y)['names']
        cls_idx = list(range(len(classes)))
        indx = [l.stem for l in labels]
        labels_df = pd.DataFrame([], columns=cls_idx, index=indx)
        for label in labels:
            lbl_counter = Counter()
            with open(label,'r') as lf:
                lines = lf.readlines()
            for l in lines:
                lbl_counter[int(l.split(' ')[0])] += 1
            labels_df.loc[label.stem] = lbl_counter
        labels_df = labels_df.fillna(0.0)
        return labels_df, classes, cls_idx, indx, labels

 
    def create_k_folds(self, k_splits:int, dataset_path:str, labels_df:pd.DataFrame, classes:list[str], cls_idx:list[int], indx:list[str], labels:list[str]) -> Union[pd.DataFrame, pd.DataFrame, Path]:
        """Creates K-Fold cross-validation splits of the dataset.

        Args:
            dataset_path (str): Path to the root dataset directory.
            labels_df (pd.DataFrame): DataFrame containing the labels.
            classes (list): List of class names.
            cls_idx (list): List of class indices.
            indx (list): List of image indices.
            folds (int): Number of folds to create.
            labels (list): List of class names.

        Returns:
            pd.DataFrame: DataFrame containing the label distribution for each fold.
        """
        ksplit = k_splits
        supported_extensions = ['.jpg', '.jpeg', '.png']
        kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)
        kfolds = list(kf.split(labels_df))
        folds = [f'split_{n}' for n in range(1, ksplit + 1)]
        folds_df = pd.DataFrame(index=indx, columns=folds)
        for idx, (train, val) in enumerate(kfolds, start=1):
            folds_df[f'split_{idx}'].loc[labels_df.iloc[train].index] = 'train'
            folds_df[f'split_{idx}'].loc[labels_df.iloc[val].index] = 'val'
        fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)
        for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
            train_totals = labels_df.iloc[train_indices].sum()
            val_totals = labels_df.iloc[val_indices].sum()
            ratio = val_totals / (train_totals + 1E-7)
            fold_lbl_distrb.loc[f'split_{n}'] = ratio
        images = []
        dataset_path = Path(dataset_path)
        for ext in supported_extensions:
            images.extend(sorted((dataset_path / 'full_dataset' / 'images').rglob(f"*{ext}")))
        save_path = Path(dataset_path / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val')
        save_path.mkdir(parents=True, exist_ok=True)
        ds_yamls = []
        for split in folds_df.columns:
            split_dir = save_path / split
            split_dir.mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
            (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
            dataset_yaml = split_dir / f'{split}_dataset.yaml'
            ds_yamls.append(dataset_yaml)
            with open(dataset_yaml, 'w') as ds_y:
                yaml.safe_dump({
                'path': f'{split_dir.as_posix()}',
                'train': 'train',
                'val': 'val',
                'names': classes
            }, ds_y)
        for image, label in zip(images, labels):
            for split, k_split in folds_df.loc[image.stem].items():
                img_to_path = save_path / split / k_split / 'images'
                lbl_to_path = save_path / split / k_split / 'labels'
                shutil.copy(label, lbl_to_path / label.name)
                if k_split == 'train':
                    full_path = save_path / split / 'full'
                    full_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy(image, full_path / image.name)
                else:
                    shutil.copy(image, img_to_path / image.name)
        return fold_lbl_distrb, folds_df, save_path


    def split_to_csv(self, fold_lbl_distrb, folds_df, save_path, dataset_path) -> None:
        with open(f"{dataset_path}/data.yaml", "r") as f:
            data = yaml.safe_load(f)
        data["train"] = "../train/images"
        data["val"] = "../valid/images"
        data["test"] = "../test/images"
        with open(f"{dataset_path}/data.yaml", "w") as f:
            yaml.dump(data, f)
        folds_df.to_csv(save_path / "kfold_datasplit.csv")
        fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")


## Create the dataset pipeline
dataset_creator = DatasetCreator()
dataset_name = "wcs_camera_dataset"
dataset = dataset_creator.create_dataset(
    filename="wcs_20220205_bboxes_with_classes 2_1.json",
    min_count=100,
    max_count=50000,
    classes={},
    classes_count=6,
    image_amount=500,
    max_attempts=100,
    parent_folder=dataset_name
)
## Check for images that couldn't be downloaded via failed_images.txt and consider how to proceed
asyncio.run(dataset_creator.download_images(train_split=0.8, test_split=0.2, dataset_name=dataset_name))
dataset_creator.create_k_fold_split(dataset_name=dataset_name, k_splits=3)
