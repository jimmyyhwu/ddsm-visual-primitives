from db.database import DB
import os
from collections import namedtuple

Image = namedtuple("Image", "image_path ground_truth split")

ground_truth_2_idx = {
    "normal": 0,
    "benign": 1,
    "cancer": 2
}

def check_that_image_table_empty(conn):
    c = conn.cursor()
    c.execute("SELECT * FROM image;")
    if c.fetchone() is not None:
        raise FileExistsError("The `image` table is already populated.")

def populate_db_with_images(db_path, image_lists_path):
    db = DB(db_path)
    conn = db.get_connection()
    check_that_image_table_empty(conn)
    images = get_images(image_lists_path)
    insert_statement = generate_sql_insert(images)
    conn.execute(insert_statement)
    conn.commit()

def get_images(image_lists_path):
    '''
    :param image_lists_path:
    :return:
    '''
    files = []

    for filename in os.listdir(image_lists_path):
        if filename.endswith(".txt"):
            files.append((filename[:-4], filename))

    images = []

    for split, filename in files:
        with open(os.path.join(image_lists_path, filename)) as file:
            for line in file.readlines():
                image_path = line.rstrip()
                ground_truth = ground_truth_2_idx[line.split("_")[0]]
                image = Image(image_path, ground_truth, split.rstrip())
                images.append(image)

    return images

def generate_sql_insert(images):
    '''

    :param images:
    :return:
    '''
    num_images = len(images)
    if num_images == 0:
        raise IndexError("No images were given")

    sql = "INSERT INTO image (image_path, ground_truth, split) VALUES "

    for i, image in enumerate(images):
        if i > 0:
            sql += ","
        sql += f" (\"{image.image_path}\", {image.ground_truth}, \"{image.split}\")"

    sql += ";"

    return sql

if __name__ == "__main__":
    DB_PATH = "test.db"
    IMAGE_LISTS_PATH = os.path.join("..", "data", "ddsm_raw_image_lists")
    populate_db_with_images(DB_PATH, IMAGE_LISTS_PATH)
