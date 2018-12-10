import sqlite3 as lite
import os

class DB:
    def __init__(self, filename):
        '''

        :param filename:
        :return:
        '''
        if os.path.isfile(filename):
            self.__conn = lite.connect(filename)
        else:
            self.__conn = lite.connect(filename)
            self.__generate_tables(self.__conn)

    def get_connection(self):
        '''

        :return:
        '''
        return self.__conn

    def __generate_tables(self, conn):
        with open("init.sql", "r") as generation_script:
            conn.execute("PRAGMA foreign_keys=on;")
            conn.commit()
            conn.executescript(generation_script.read())
            conn.commit()


if __name__ == "__main__":
    DB_FILE = "test.db"
    db = DB(DB_FILE)
