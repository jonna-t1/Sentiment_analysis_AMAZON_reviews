import datetime

from .models import Review, PosScores, NegScores, WeightedAvg


def subtract_one_month(t):
    """Return a `datetime.date` or `datetime.datetime` (as given) that is
    one month later.

    Note that the resultant day of the month might change if the following
    month has fewer days:

        >>> subtract_one_month(datetime.date(2010, 3, 31))
        datetime.date(2010, 2, 28)
    """
    import datetime
    one_day = datetime.timedelta(days=1)
    one_month_earlier = t - one_day
    while one_month_earlier.month == t.month or one_month_earlier.day > t.day:
        one_month_earlier -= one_day
    return one_month_earlier


def getMonthLabels():
    import pandas as pd

    # reviews = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()
    months = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()

    final = []
    array = []
    for month in months:
        mon = month.strftime("%Y-%b")
        array.append(mon)

    start = months.first()
    start = subtract_one_month(start)

    new = start.strftime('%Y-%m-%d')
    finish = months.last().strftime('%Y-%m-%d')

    dateLabels = pd.date_range(new, finish,
                  freq='MS').strftime("%Y-%b").tolist()
    dateLabels = [months.first().strftime("%Y-%b")]+dateLabels

    return dateLabels

def test():
    all = Review.objects.all()


def getLatestBatchPerMonth():
    import pandas as pd
    import calendar

    pos = []
    neg = []
    avg = []
    print(test())

    dates = Review.objects.order_by('batch_date').values_list('batch_date', flat=True).distinct()
    first = dates.first()
    last = dates.last()

    first = subtract_one_month(first)
    new = first.strftime('%Y-%m-%d')
    first = first.strftime('%Y-%m-%d')
    finish = last.strftime('%Y-%m-%d')
    dateRange = pd.date_range(first, finish,
                               freq='MS').tolist()
    hash = {}
    count = 0
    for date in dateRange:
        hash[date.month] = date.year
        count+=1

    for key, value in hash.items():
        month = key
        # nextMonth = key + 1
        year = value
        arr = calendar.monthrange(year, month)
        start_date = datetime.date(year, month, 1)
        end_date = datetime.date(year, month, arr[1])
        print(start_date)
        print(end_date)
        lastReview = Review.objects.filter(batch_date__range=(start_date, end_date)).last()

        print(lastReview)

        # skip if there arent any batches in that month
        if lastReview == None:
            continue

        print(lastReview.pos_batch_no_id)
        print(lastReview.neg_batch_no_id)
        print(lastReview.avg_batch_no_id)

        pos_score = PosScores.objects.filter(pk=str(lastReview.pos_batch_no_id))
        neg_score = NegScores.objects.filter(pk=str(lastReview.neg_batch_no_id))
        avg_score = WeightedAvg.objects.filter(pk=str(lastReview.pos_batch_no_id))

        pos.append(pos_score)
        neg.append(neg_score)
        avg.append(avg_score)

    for avg in avg_score:
        print(avg.id)

    print(len(pos))
    print(len(neg))
    print(len(avg))

    return pos, len(pos), neg, len(neg), avg, len(avg)


