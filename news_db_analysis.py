import sqlite3
import config

# Path to the SQLite database


# Connect to the database
connection = sqlite3.connect(config.DB_PATH)
cursor = connection.cursor()

# Fetch the table names
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

# Print headers for each table
for table in tables:
    table_name = table[0]
    print(f"Headers for table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name});")
    headers = cursor.fetchall()
    for header in headers:
        print(header[1])  # Print column name

# Fetch a random sample of 5 rows from the table
cursor.execute(f"SELECT Article_title, Stock_symbol, Article FROM {config.NEWS_TABLE_NAME} ORDER BY RANDOM() LIMIT 5;")
rows = cursor.fetchall()
print("Random sample of 5 rows:")
for row in rows:
    print(f"Article_title: {row[0]}")
    print(f"Stock_symbol: {row[1]}")
    print(f"Article: {row[2]}")
    print("---")

# Close the connection
connection.close()