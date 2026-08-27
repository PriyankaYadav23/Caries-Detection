import json
import os
import shutil
from sklearn.model_selection import train_test_split

with open("training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json") as f:
    data = json.load(f)
print(f"images: {len(data['images'])}")
print(data["images"][0])
print(f"annotations: {len(data['annotations'])}")
print(data["annotations"][0])

# for yolo we need images- id, file name, width, height , bbox, all categories data together in txt file
# making dictionary of images
img_find = {img["id"]: img for img in data["images"]}
print(len(img_find))

#  with annotation - we will work with bbox first as yolo needs normalized x_centre , y_centre and width and height
def bbox_convert(i_width, i_height, b_box):
    x, y, w, h = b_box
    x_centre = (x + w / 2) / i_width
    y_centre = (y + h / 2) / i_height
    width_norm = w / i_width
    height_norm = h / i_height
    return (x_centre, y_centre, width_norm, height_norm)

# checking for first image
ann = data["annotations"][0]
image_0 = img_find[ann["image_id"]]
result_0 = bbox_convert(image_0["width"], image_0["height"], ann["bbox"])
print(result_0)

#  will make dictionary of  images _id with annotation
img_annotations = {}
for ann in data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in img_annotations:
        img_annotations[img_id] = []
    img_annotations[img_id].append(ann)
print(len(img_annotations))
# making empty list if img_id not in annotation and then appending rest of ann values in img_annotation


# train test split for images_ids

#  for yolo will split train val img_ids
all_ids = list(img_find.keys())
train_ids, val_ids = train_test_split(all_ids, test_size=0.15, random_state=42)
print(f"train_ids length: {len(train_ids)}")
print(f"val_ids length: {len(val_ids)}")


shutil.rmtree("yolo_dataset", ignore_errors=True)

# making txt label file for all images
os.makedirs("yolo_dataset/labels/train", exist_ok=True)
os.makedirs("yolo_dataset/labels/val", exist_ok=True)

# putting all the yolo required together [img-find info, img_annotation, convert_bbox]

def make_labels(image_ids, split_name):
    for img_id in image_ids:
        # taking out data-images info based on id
        img_info = img_find[img_id]
        # taking out img -annotation based on that id
        img_ann_info = img_annotations.get(img_id, [])
        
        # taking out image_other_info seperately
        img_ann_items = []
        for ann in img_ann_info:
            disease_id = ann["category_id_3"]
            x, y, w, h = bbox_convert(img_info["width"], img_info["height"], ann["bbox"])
            img_ann_items.append(f"{disease_id} {x} {y} {w} {h}")
            
        base_name = os.path.splitext(img_info["file_name"])[0]
        label_path = os.path.join("yolo_dataset", "labels", split_name, base_name + ".txt")
        
        with open(label_path, "w") as f:
            f.write("\n".join(img_ann_items))


make_labels(train_ids, "train")
make_labels(val_ids, "val")
print("done")


# now copy original xray images into these folders
os.makedirs("yolo_dataset/images/train", exist_ok=True)
os.makedirs("yolo_dataset/images/val", exist_ok=True)
xrays_folder = os.path.join("training_data", "quadrant-enumeration-disease", "xrays")

def copy_images(image_ids, split_name):
    for img_id in image_ids:
        img_info = img_find[img_id]
        file_name = img_info["file_name"]

        source = os.path.join(xrays_folder, file_name)
        destination = os.path.join("yolo_dataset", "images", split_name, file_name)

        shutil.copy(source, destination)

copy_images(train_ids, "train")
copy_images(val_ids, "val")
print("copy done")
        