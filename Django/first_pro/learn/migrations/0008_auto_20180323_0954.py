# Generated by Django 2.0.3 on 2018-03-23 09:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('learn', '0007_auto_20180323_0945'),
    ]

    operations = [
        migrations.AddField(
            model_name='article',
            name='is_deleted',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='article',
            name='readed_num',
            field=models.IntegerField(default=0),
        ),
    ]
