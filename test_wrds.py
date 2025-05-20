import wrds

db = wrds.Connection()

libs = db.list_libraries()
tables = db.list_tables(library='ibessamp_kpi')
print(tables)

db.describe_table('ibessamp_kpi', 'statsum_kpius')






# Example query
data = db.raw_sql("""
SELECT *
FROM ibessamp_kpi.statsum_kpius
LIMIT 10
""")
print(data.columns)


query = db.raw_sql("""
SELECT COUNT(DISTINCT ticker) AS num_unique_tickers
FROM ibessamp_kpi.statsum_kpius
""")
print(query)


