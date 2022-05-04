

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import pandas as pd


class create_DeepFashion:

    def __init__(self, dataset_path):


        img_folder_name = "dataset/img"
        eval_folder_name = "dataset/Eval"
        anno_folder_name = "dataset/Anno"
        list_eval_partition_file = "list_eval_partition.txt"
        list_attr_img_file = "list_attr_img.txt"
        list_category_img_file = "list_category_img.txt"
        list_category_cloth_file = "list_category_cloth.txt"
        list_bbox_file = "list_bbox.txt"

        self.train = pd.DataFrame(columns=["img_path", "bbox", "category", "attributes"])
        self.val = pd.DataFrame(columns=["img_path", "bbox", "category", "attributes"])
        self.test = pd.DataFrame(columns=["img_path", "bbox", "category", "attributes"])

        self.all = pd.DataFrame(columns=["img_path", "bbox", "category", "attributes"])


        self.path = dataset_path
        self.img_dir = os.path.join(self.path, img_folder_name)
        self.eval_dir = os.path.join(self.path, eval_folder_name)
        self.anno_dir = os.path.join(self.path, anno_folder_name)

        self.list_eval_partition = os.path.join(self.eval_dir, list_eval_partition_file)
        self.list_attr_img = os.path.join(self.anno_dir, list_attr_img_file)
        self.list_category_img = os.path.join(self.anno_dir, list_category_img_file)
        self.list_category_cloth = os.path.join(self.anno_dir, list_category_cloth_file)
        self.list_bbox = os.path.join(self.anno_dir, list_bbox_file)

    def read_imgs_and_split(self):

        category_to_name = {}

        with open(self.list_category_cloth) as f:
            count = int(f.readline().strip())  # Read the first line
            _ = f.readline().strip()  # read and throw away the header

            i = 0
            for line in f:
                words = line.split()
                category_to_name[i] = str(words[0])
                i = i + 1

        assert (count == 50)


        image_to_category = {}
        with open(self.list_category_img) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()
            for line in f:
                words = line.split()

                image_to_category[words[0].strip()] = int(words[1].strip())

        assert (imgs_count == len(image_to_category))


        image_to_bbox = {}
        with open(self.list_bbox) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()

            for line in f:
                words = line.split()

                data = (words[1], words[2], words[3], words[4])
                image_to_bbox[words[0]] = data

        assert (imgs_count == len(image_to_bbox))


        with open(self.list_eval_partition) as f:
            imgs_count = int(f.readline().strip())
            _ = f.readline().strip()  # read and throw away the header

            for line in f:
                words = line.split()
                img = words[0].strip()
                category_idx = image_to_category[img]
                category = str(category_to_name[category_idx - 1])
                bbox = np.asarray(image_to_bbox[img], dtype=np.int16)

                if words[1].strip() == "train":
                    self.all = self.all.append({"img_path": img, "bbox": bbox, "category": category},
                                                   ignore_index=True)
                if words[1].strip() == "val":
                    self.all = self.all.append({"img_path": img, "bbox": bbox, "category": category}, ignore_index=True)
                if words[1].strip() == "test":
                    self.all = self.all.append({"img_path": img, "bbox": bbox, "category": category},
                                                 ignore_index=True)

        print("all images", int(self.all.shape[0]))
        #assert (imgs_count == int(self.train.shape[0]) + int(self.test.shape[0]) + int(self.val.shape[0]))
        assert (imgs_count == int(self.all.shape[0]))

        self.all.to_csv(self.path + "split-data/all_data.csv", index=False)

        print("Storage done")


if __name__ == "__main__":

    df = create_DeepFashion("")
    df.read_imgs_and_split()