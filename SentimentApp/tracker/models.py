from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from datetime import date
from django.core.exceptions import ValidationError

def no_future(value):
    today = date.today()
    if value > today:
        raise ValidationError('Date cannot be in the future.')

# composite key use
# class ClassificationScores(models.Model):
#     class Meta:
#         unique_together = (('positive', 'negative', 'weightedAvg'),)
#
#     positive = models.AutoField(primary_key=True)
#     negative = models.IntegerField()
#     weightedAvg = models.IntegerField()
#     key1 = models.IntegerField(primary_key=True)
#     key2 = models.IntegerField()

# class Batch(models.Model):
#     batch_date = models.DateTimeField(default=timezone.now)
#
# ## classification table
# class Classification(models.Model):
#     precision = models.DecimalField(max_digits=5, decimal_places=3)
#     recall = models.DecimalField(max_digits=5, decimal_places=3)
#     f1 = models.DecimalField(max_digits=5, decimal_places=3)
#     support = models.IntegerField()
#     batch_ID = models.ForeignKey(Batch, on_delete=models.CASCADE)
#
#     def __str__(self):
#         val = 'Batch Number - ' + str(self.pk)
#         return val



class PosScores(models.Model):

    precision = models.DecimalField(max_digits=5, decimal_places=3)
    recall = models.DecimalField(max_digits=5, decimal_places=3)
    f1 = models.DecimalField(max_digits=5, decimal_places=3)
    support = models.IntegerField()

    def __str__(self):
        val = 'Batch Number - ' + str(self.pk)
        return val


class NegScores(models.Model):

    precision = models.DecimalField(max_digits=5, decimal_places=3)
    recall = models.DecimalField(max_digits=5, decimal_places=3)
    f1 = models.DecimalField(max_digits=5, decimal_places=3)
    support = models.IntegerField()

    def __str__(self):
        val = 'Batch Number - ' + str(self.pk)
        return val


class WeightedAvg(models.Model):

    precision = models.DecimalField(max_digits=5, decimal_places=3)
    recall = models.DecimalField(max_digits=5, decimal_places=3)
    f1 = models.DecimalField(max_digits=5, decimal_places=3)
    support = models.IntegerField()

    def __str__(self):
        val = 'Batch Number - ' + str(self.pk)
        return val

class Review(models.Model):

    SENTIMENT = (
        ('POSITIVE', 'positive'),
        ('NEGATIVE', 'negative'),
    )

    reviewText = models.TextField()
    predictSentiment = models.CharField(max_length=10, choices=SENTIMENT)
    actualSentiment = models.CharField(max_length=10, choices=SENTIMENT)
    batch_date = models.DateTimeField(default=timezone.now)
    pos_batch_no = models.ForeignKey(PosScores, on_delete=models.CASCADE)
    neg_batch_no = models.ForeignKey(NegScores, on_delete=models.CASCADE)
    avg_batch_no = models.ForeignKey(WeightedAvg, on_delete=models.CASCADE)

# class Scores(models.Model):
#
#     precision = models.DecimalField(max_digits=5, decimal_places=3)
#     recall = models.DecimalField(max_digits=5, decimal_places=3)
#     f1 = models.DecimalField(max_digits=5, decimal_places=3)
#     support = models.IntegerField()
#     batch_no = models.IntegerField(default=1, primary_key=True)
#
#     def __str__(self):
#         val = 'Batch Number - ' + self.batch_no
#         return val



class Request(models.Model):

    REASON = (
        ('REQUEST ACCESS', 'REQUEST ACCESS'),
        ('CHANGE OF PRIVILEGE', 'CHANGE OF PRIVILEGE'),
        ('CONTACT', 'CONTACT'),
    )

    name = models.CharField(max_length=100)
    surname = models.CharField(max_length=100)
    email = models.EmailField(max_length=70)
    contactNo = models.BigIntegerField()
    date_posted = models.DateTimeField(auto_now_add=True)
    reason = models.CharField(max_length=20, choices=REASON, default='REQUEST ACCESS')
    other = models.TextField(blank=True, default='')

    def __str__(self):
        val = self.date_posted.strftime("%B %d, %Y") + ' - ' + self.reason
        return val

    def get_absolute_url(self):
        return reverse('request-detail', kwargs={'pk': self.pk})


class Event(models.Model):

    PRIORITY = (
        ('P1', 'P1'),
        ('P2', 'P2'),
        ('P3', 'P3'),
        ('P4', 'P4'),
        ('P5', 'P5'),
    )
    STATUS = (
        (1, '1'),
        (2, '2'),
        (3, '3'),
    )
    TYPE = (
        ('PROBLEM', 'Problem'),
        ('INCIDENT', 'Incident'),
    )
    TEAM = (
        ('NISPI/DWP', 'NISPI/DWP'),
    )

    type = models.CharField(
        choices=TYPE,
        max_length=10,
        default='Problem'
    )
    reference = models.CharField(max_length=10)
    status = models.IntegerField(choices=STATUS, default='1')
    resolution_date = models.DateTimeField()
    priority = models.CharField(
        max_length=2,
        choices=PRIORITY,
        default='P5',
    )
    assigned_team = models.CharField(choices=TEAM, default='NISPI/DWP', max_length=20)
    assigned_person = models.ForeignKey(User, default=User, on_delete=models.CASCADE)
    summary = models.TextField()

    def __str__(self):
        return self.reference

    def get_absolute_url(self):
        return reverse('event-detail', kwargs={'pk': self.pk})
