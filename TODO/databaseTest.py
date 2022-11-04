import psycopg2

def postgres_test():

    try:
        conn = psycopg2.connect(
          database="postgres", user='postgres', password='test', host='127.0.0.1', port= '5432'
        )
        conn.close()
        return True
    except:
        return False

print(postgres_test())