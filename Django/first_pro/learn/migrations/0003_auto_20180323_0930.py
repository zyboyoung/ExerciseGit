# Generated by Django 2.0.3 on 2018-03-23 09:30

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('learn', '0002_article_crete_time'),
    ]

    operations = [
        migrations.RenameField(
            model_name='article',
            old_name='crete_time',
            new_name='create_time',
        ),
    ]