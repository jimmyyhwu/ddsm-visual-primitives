from db.database import DB


def _is_doctor_existing(username, db_path):
    db = DB(db_path)
    conn = db.get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM doctor WHERE name='{}'".format(username))
    return c.fetchone() is not None


def _insert_doctor(username, db_path):
    insert_statement = "INSERT INTO doctor(name) VALUES('{}');".format(username)
    db = DB(db_path)
    conn = db.get_connection()
    conn.execute(insert_statement)
    conn.commit()


def insert_doctor_into_db_if_not_exists(username, db_path):
    if _is_doctor_existing(username, db_path):
        return
    _insert_doctor(username, db_path)


if __name__ == "__main__":
    insert_doctor_into_db_if_not_exists('doctorA', 'test.db')
