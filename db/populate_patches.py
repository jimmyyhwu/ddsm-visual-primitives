from db.database import DB
import os
from collections import namedtuple

Patch = namedtuple("Patch", "ground_truth x y width height full_image")


def populate_db_with_patches(db_path, patch_path):
    patches = get_patches(patch_path)
    insert_statement = generate_sql_insert(patches)
    db = DB(db_path)
    conn = db.get_connection()
    conn.execute(insert_statement)
    conn.commit()


def get_patches(patch_path):
    patches = []
    for root, dirs, files in os.walk(patch_path, followlinks=True):  # not lazy, coud be replaced with recursive scandir()-calls
        for patch_path in files:
            patch_info = patch_path.split("_")

            ground_truth = patch_info[0]
            if len(patch_info) == 10:
                offset = 0
            else:
                offset = 2
            x = patch_info[4 + offset].split("x")[1]
            y = patch_info[5 + offset].split("y")[1]
            width = patch_info[6 + offset].split("w")[1]
            height = patch_info[7 + offset].split("h")[1]
            full_image = "-".join(patch_path.split("-")[0:2]) + ".jpg"
            patch = Patch(ground_truth, x, y, width, height, full_image)
            patches.append(patch)
    return patches


def generate_sql_insert(patches):
    num_patches = len(patches)
    if num_patches == 0:
        raise IndexError("No patches were given")

    statement = "INSERT INTO patch (x, y, full_image, ground_truth) VALUES "
    for i, patch in enumerate(patches):
        if (i + 1) < num_patches:
            statement += "({}, {}, '{}', '{}'), ".format(patch.x, patch.y, patch.full_image, patch.ground_truth)
        else:
            statement += "({}, {}, '{}', '{}');".format(patch.x, patch.y, patch.full_image, patch.ground_truth)
    return statement


if __name__ == "__main__":
    PATCH_PATH = "../data/ddsm_patches/"
    DB_PATH = "test.db"
    populate_db_with_patches(DB_PATH, PATCH_PATH)
