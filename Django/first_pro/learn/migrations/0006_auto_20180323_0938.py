# Generated by Django 2.0.3 on 2018-03-23 09:38

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learn', '0005_auto_20180323_0936'),
    ]

    operations = [
        migrations.AlterField(
            model_name='article',
            name='create_time',
            field=models.DateTimeField(default=datetime.datetime(2018, 3, 23, 9, 38, 16, 458938)),
        ),
    ]
