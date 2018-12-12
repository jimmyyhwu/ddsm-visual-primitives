import sqlite3 as lite
import os


class DB:
    def __init__(self, filename, db_root="../db/"):
        self._db_root = db_root
        db_file_path = os.path.join(self._db_root, filename)
        if os.path.isfile(db_file_path):
            self.__conn = lite.connect(db_file_path)
        else:
            self.__conn = lite.connect(db_file_path)
            self.__generate_tables()

    def get_connection(self):
        return self.__conn

    def __generate_tables(self):
        with open(os.path.join(self._db_root, "init.sql"), "r") as generation_script:
            self.__conn.execute("PRAGMA foreign_keys=on;")
            self.__conn.commit()
            self.__conn.executescript(generation_script.read())
            self.__conn.commit()
