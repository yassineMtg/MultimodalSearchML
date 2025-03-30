import sqlite3
conn = sqlite3.connect("tfx_metadata/metadata.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())
conn.close()
