import psycopg2
import dbConfig
## basic search function

query_str = "SELECT * FROM reviews"

def postgres_test():
    try:
        conn = psycopg2.connect(
          database="Review", user='postgres', password='test', host='127.0.0.1', port= '5432'
        )
        cur = conn.cursor()
        cur.execute(query_str)
        tings = cur.fetchmany(10)
        print(tings)
        # print(len(tings))
        cur.close()
        return True
    except:
        return False

print(dbConfig.config)