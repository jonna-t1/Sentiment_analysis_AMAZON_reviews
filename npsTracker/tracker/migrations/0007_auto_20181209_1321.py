# Generated by Django 2.1.3 on 2018-12-09 13:21

from django.conf import settings
import django.contrib.auth.models
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('tracker', '0006_auto_20181207_2300'),
    ]

    operations = [
        migrations.AlterField(
            model_name='event',
            name='assigned_person',
            field=models.ForeignKey(default=django.contrib.auth.models.User, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='event',
            name='assigned_team',
            field=models.CharField(choices=[('NISPI/DWP', 'NISPI/DWP')], default='NISPI/DWP', max_length=20),
        ),
        migrations.AlterField(
            model_name='event',
            name='status',
            field=models.IntegerField(choices=[(1, '1'), (2, '2'), (3, '3')], default='1'),
        ),
    ]
