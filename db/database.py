import sqlite3 as lite

def create_connection(filename):
    '''

    :param filename:
    :return:
    '''
    conn = lite.connect(filename)
    __generate_tables(conn)
    return (conn.cursor(), conn)

def __generate_tables(conn):
    '''

    :param conn:
    :return:
    '''
    with open("init.sql", "r") as generation_script:
        conn.execute("PRAGMA foreign_keys=on;")
        conn.commit()
        conn.executescript(generation_script.read())
        conn.commit()


if __name__ == "__main__":
    conn = create_connection("test.db")