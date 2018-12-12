import sqlite3 as lite
import os


class DB:
    def __init__(self, filename):
        if os.path.isfile(filename):
            self.__conn = lite.connect(filename)
        else:
            self.__conn = lite.connect(filename)
            self.__generate_tables()

    def get_connection(self):
        return self.__conn

    def __generate_tables(self):
        with open("init.sql", "r") as generation_script:
            self.__conn.execute("PRAGMA foreign_keys=on;")
            self.__conn.commit()
            self.__conn.executescript(generation_script.read())
            self.__conn.commit()


if __name__ == "__main__":
    DB_FILE = os.environ['DB_FILE'] or 'test.db'
    db = DB(DB_FILE)
