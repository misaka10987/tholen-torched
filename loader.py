import sqlite3


class Database:
    def __init__(self):
        self.conn = sqlite3.connect("asteroid.db")
        self.cur = self.conn.cursor()

    def __del__(self):
        self.conn.close()

    def sql(self, command: str):
        return self.cur.execute(command).fetchall()

    def commit(self):
        return self.conn.commit()

    def fetch(self):
        return self.sql("SELECT * FROM main WHERE spec_T IS NOT NULL")

    def all(self):
        return self.sql("SELECT * FROM main")
