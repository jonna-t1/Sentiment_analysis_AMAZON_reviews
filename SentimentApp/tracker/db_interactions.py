from .models import Review, WeightedAvg, PosScores, NegScores
from datetime import datetime

global pos_score_instance
global neg_score_instance
global weighted_avg_instance

def WeightedAvgCreate(weighted_average):
    # Save to the database
    weighted_avg_instance = WeightedAvg.objects.create(
        precision=weighted_average['precision'],
        recall=weighted_average['recall'],
        f1=weighted_average['f1-score'],
        support=weighted_average['support']
    )
    print(f"Saved Classification Report with Weighted Precision: {weighted_average['precision']}")
    # first_10_reviews = Review.objects.all()[:10]
    # for review in first_10_reviews:
    #   print(review.predictSentiment)
def WeightedAvgDelete():
    WeightedAvg.objects.latest('id').delete()
    print("The last entry has been deleted.")

def PosScoresCreate(pos_Score):
    # Save to the database
    pos_score_instance = PosScores.objects.create(
        precision=pos_Score['precision'],
        recall=pos_Score['recall'],
        f1=pos_Score['f1-score'],
        support=pos_Score['support']
    )
    print(f"Saved Classification Report with Weighted Precision: {pos_Score['precision']}")

def PosScoreDelete():
    PosScores.objects.latest('id').delete()
    print("The last entry has been deleted.")

def NegScoresCreate(neg_Score):
    # Save to the database
    neg_score_instance = NegScores.objects.create(
        precision=neg_Score['precision'],
        recall=neg_Score['recall'],
        f1=neg_Score['f1-score'],
        support=neg_Score['support']
    )
    print(f"Saved Classification Report with Weighted Precision: {neg_Score['precision']}")

def NegScoreDelete():
    NegScores.objects.latest('id').delete()
    print("The last entry has been deleted.")

def ReviewCreate(df):
    reviews = []

    for _, row in df.iterrows():
        try:
            pos_score = PosScores.objects.last().id
            neg_score = NegScores.objects.last().id
            avg_score = WeightedAvg.objects.last().id
            pos_score_instance = PosScores.objects.get(id=pos_score)
            neg_score_instance = NegScores.objects.get(id=neg_score)
            avg_score_instance = WeightedAvg.objects.get(id=avg_score)

            review = Review(
                reviewText=row['reviewTextRaw'],
                predictSentiment=row['predictSentiment'],
                actualSentiment=row['actualSentiment'],
                pos_batch_no=pos_score_instance,
                neg_batch_no=neg_score_instance,
                avg_batch_no=avg_score_instance,
            )
            reviews.append(review)
        except PosScores.DoesNotExist:
            print(f"PosScore with ID {row['pos_score_id']} not found!")
        except NegScores.DoesNotExist:
            print(f"NegScore with ID {row['neg_score_id']} not found!")
        except WeightedAvg.DoesNotExist:
            print(f"WeightedAvg with ID {row['avg_score_id']} not found!")

    # Bulk insert into the database
    Review.objects.bulk_create(reviews)

    print(f"Inserted {len(reviews)} reviews into the database.")