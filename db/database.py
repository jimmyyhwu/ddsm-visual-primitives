import sqlite3 as lite
import os

import sys
sys.path.insert(0,'../db')
from populate_images import populate_db_with_images
from populate_patches import populate_db_with_patches


class DB:
    def __init__(self, filename, db_root="../db/"):
        self._db_root = db_root
        db_file_path = os.path.join(self._db_root, filename)
        if os.path.isfile(db_file_path):
            self.__conn = lite.connect(db_file_path)
        else:
            self.__conn = lite.connect(db_file_path)
            self.__generate_tables()
            self.__populate_tables()

    def get_connection(self):
        return self.__conn

    def __generate_tables(self):
        with open(os.path.join(self._db_root, "init.sql"), "r") as generation_script:
            self.__conn.execute("PRAGMA foreign_keys=on;")
            self.__conn.commit()
            self.__conn.executescript(generation_script.read())
            self.__conn.commit()

    def __populate_tables(self):
        # populate images
        image_list_path = os.path.join(self._db_root, "..", "data", "ddsm_raw_image_lists")
        populate_db_with_images(self.__conn, image_list_path)
        # populate patches
        patch_path = os.path.join(self._db_root, "..", "data", "ddsm_patches")
        patch_list_path = os.path.join(self._db_root, "..", "data", "ddsm_3class")
        populate_db_with_patches(self.__conn, patch_path, patch_list_path)
