from db.database import DB
import os
from collections import namedtuple

Patch = namedtuple("Patch", "ground_truth x y width height patch_path full_image_path")


def populate_db_with_patches(db_path, patch_path, patch_ground_truth):
    patches = get_patches(patch_path, patch_ground_truth)
    insert_statement = generate_sql_insert(patches)
    db = DB(db_path)
    conn = db.get_connection()
    conn.execute(insert_statement)
    conn.commit()


def get_patches(patch_path, patch_ground_truth):
    patches = []
    for root, dirs, files in os.walk(patch_path, followlinks=True):  # not lazy, could be replaced with recursive scandir()-calls
        for patch_path in files:
            patch_info = patch_path.split("_")

            ground_truth = patch_ground_truth[patch_path]
            if len(patch_info) == 10:
                offset = 0
            else:
                offset = 2
            x = patch_info[4 + offset].split("x")[1]
            y = patch_info[5 + offset].split("y")[1]
            width = patch_info[6 + offset].split("w")[1]
            height = patch_info[7 + offset].split("h")[1]
            full_image_path = "-".join(patch_path.split("-")[0:2]) + ".jpg"
            patch = Patch(ground_truth, x, y, width, height, patch_path, full_image_path)
            patches.append(patch)
    return patches


def generate_sql_insert(patches):
    num_patches = len(patches)
    if num_patches == 0:
        raise IndexError("No patches were given")

    statement = "INSERT INTO patch (x, y, width, height, patch_path, ground_truth, image_id) VALUES "
    for i, patch in enumerate(patches):
        select_stmt = "(SELECT id FROM image WHERE image_path = '{}')".format(patch.full_image_path)

        if (i + 1) < num_patches:
            statement += "({}, {}, {}, {}, '{}', {}, {}), " \
                .format(patch.x, patch.y, patch.width, patch.height, patch.patch_path, patch.ground_truth,
                        patch.full_image_path, select_stmt)
        else:
            statement += "({}, {}, {}, {}, '{}', {}, {});"\
                .format(patch.x, patch.y, patch.width, patch.height, patch.patch_path, patch.ground_truth, patch.full_image_path, select_stmt)
    return statement


def _process_label_line(line):
    image_name, label = line.strip().split(' ')
    label = int(label)
    return image_name, label


def _get_patch_ground_truth_map(label_path):
    with open(label_path, 'r') as f:
        image_list = map(_process_label_line, f.readlines())

    cache = {}
    for image_path, label in image_list:
        _, image_name = os.path.split(image_path)
        cache[image_name] = label
    return cache


def get_patches_ground_truth(label_path_test, label_path_val, label_path_train):
    test_map = _get_patch_ground_truth_map(label_path_test)
    val_map = _get_patch_ground_truth_map(label_path_val)
    train_map = _get_patch_ground_truth_map(label_path_train)
    test_val_map = {**test_map, **val_map}  # merge test and val
    all_map = {**test_val_map, **train_map}  # merge combined test and val with train
    return all_map



if __name__ == "__main__":
    PATCH_PATH = "../data/ddsm_patches/"
    DB_PATH = "test.db"
    GROUND_TRUTH_TEST = "../data/ddsm_3class/test.txt"
    GROUND_TRUTH_VAL = "../data/ddsm_3class/val.txt"
    GROUND_TRUTH_TRAIN = "../data/ddsm_3class/train.txt"
    patch_ground_truth = get_patches_ground_truth(GROUND_TRUTH_TEST, GROUND_TRUTH_VAL, GROUND_TRUTH_TRAIN)

    populate_db_with_patches(DB_PATH, PATCH_PATH, patch_ground_truth)
