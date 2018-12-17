import os
from collections import namedtuple

Patch = namedtuple("Patch", "ground_truth x y width height patch_path full_image_path")


def populate_db_with_patches(conn, patch_path, patch_list_path):
    patch_ground_truth = _get_patches_ground_truth(patch_list_path)
    patches = _get_patches(patch_path, patch_ground_truth)

    _insert_patches_batchwise(patches, conn)


def _get_patches(patch_path, patch_ground_truth):
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


def _insert_patches_batchwise(patches, conn, batch_size=500):
    patch_batches = [patches[i:i+batch_size] for i in range(0, len(patches), batch_size)]
    #print("{} batches total".format(len(patch_batches)))
    count = 0
    for batch in patch_batches:
        count += 1
        #print(count)
        statement = "INSERT INTO patch (x, y, width, height, patch_path, ground_truth, image_id) VALUES "
        for (i, patch) in enumerate(batch):
            select_stmt = "(SELECT id FROM image WHERE image_path = '{}')".format(patch.full_image_path)
            if (i + 1) < len(batch):
                statement += "({}, {}, {}, {}, '{}', {}, {}),".format(patch.x, patch.y, patch.width, patch.height,
                                                                      patch.patch_path, patch.ground_truth, select_stmt)
            else:
                statement += "({}, {}, {}, {}, '{}', {}, {});".format(patch.x, patch.y, patch.width, patch.height,
                                                                      patch.patch_path, patch.ground_truth, select_stmt)
        conn.execute(statement)
    conn.commit()


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


def _get_patches_ground_truth(image_lists_path):
    ground_truth_map = {}
    for filename in [e for e in os.listdir(image_lists_path) if e.endswith(".txt")]:
        file_path = os.path.join(image_lists_path, filename)
        ground_truth_map = {**ground_truth_map, **_get_patch_ground_truth_map(file_path)}
    return ground_truth_map

