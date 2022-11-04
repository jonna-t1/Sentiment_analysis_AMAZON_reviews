import psycopg2
import pandas as pd

### removing duplicates from the table

drop_dup_rows = "DELETE FROM reviews\
    WHERE id IN\
    (SELECT id\
    FROM \
    (SELECT id,\
    ROW_NUMBER() OVER( PARTITION BY reviewtext\
    ORDER BY  id ) AS row_num\
    FROM reviews ) t\
    WHERE t.row_num > 1)"

def postgres_remove_dups():
    try:
        conn = psycopg2.connect(
          database="Review", user='postgres', password='test', host='127.0.0.1', port= '5432'
        )
        cur = conn.cursor()
        cur.execute(drop_dup_rows)
        cur.close()
        conn.commit()
        return True
    except Exception as e:
        return f"{e}"


def str2float(df):
    df['overall'] = df['overall'].astype(float)
    return df


def addSenimentColumn(train_set, option):

    if option == 'd1':
        positive_ratings_indices = train_set.loc[(train_set.overall > 2.0)].index
        negative_ratings_indices = train_set.loc[(train_set.overall <= 2.0)].index
        
        train_set['sentiment'] = 'positive'
        train_set.loc[negative_ratings_indices, 'sentiment'] = 'negative'

    if option == 'd2':
        positive_rating_indices = train_set[train_set.reviewsrating > 2].index
        negative_rating_indices = train_set[train_set.reviewsrating <= 2].index

        train_set['sentiment'] = 'positive'
        train_set['sentiment'].loc[negative_rating_indices] = 'negative'

    return train_set