from PIL import Image, ImageDraw


def read_bounding_boxes(file_path: str) -> list:
    """
    Reads a file containing bounding box information and returns a list of bounding boxes.
    Args:
        file_path (str): The path to the file containing bounding box information.
    Returns:
        list: A list of bounding boxes, where each bounding box is represented as a tuple of (class_id, x_center, y_center, box_w, box_h).
    """
    try:
        bounding_boxes = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, box_w, box_h = map(float, parts)
                    bounding_boxes.append((class_id, x_center, y_center, box_w, box_h))
        return bounding_boxes
    except Exception as e:
        raise ValueError("Invalid file format") from e


def visualize_bounding_boxes(image_path: str, bounding_boxes: list) -> None:
    """
    Visualizes the bounding boxes on the image.
    Args:
        image_path (str): The path to the image file.
        bounding_boxes (list): A list of bounding boxes, where each bounding box is represented as a tuple of (class_id, x_center, y_center, box_w, box_h).
    """
    try:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            img_width, img_height = img.size
            for class_id, x_center, y_center, box_w, box_h in bounding_boxes:
                left = (x_center - box_w / 2) * img_width
                upper = (y_center - box_h / 2) * img_height
                right = (x_center + box_w / 2) * img_width
                lower = (y_center + box_h / 2) * img_height
                draw.rectangle([left, upper, right, lower], outline='red', width=3)
                draw.text((left, upper), str(class_id), fill='red')
            img.show()
    except Exception as e:
        raise ValueError("Invalid image format") from e


image_path = './wcs_camera_dataset/full_dataset/images/3c122090-92d5-11e9-90d1-000d3a74c7de.jpg'
bounding_box_file_path = './wcs_camera_dataset/full_dataset/labels/3c122090-92d5-11e9-90d1-000d3a74c7de.txt'
bounding_boxes = read_bounding_boxes(bounding_box_file_path)
visualize_bounding_boxes(image_path, bounding_boxes)
