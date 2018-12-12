from db.database import DB


def _is_doctor_existing(username, db_filename, db_root):
    db = DB(db_filename, db_root)
    conn = db.get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM doctor WHERE name='{}'".format(username))
    return c.fetchone() is not None


def _insert_doctor(username, db_filename, db_root):
    insert_statement = "INSERT INTO doctor(name) VALUES('{}');".format(username)
    db = DB(db_filename, db_root)
    conn = db.get_connection()
    conn.execute(insert_statement)
    conn.commit()


def insert_doctor_into_db_if_not_exists(username, db_filename, db_root):
    if _is_doctor_existing(username, db_filename, db_root):
        return
    _insert_doctor(username, db_filename, db_root)
