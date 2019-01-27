from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse

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
