from django.db import models

# Create your models here.
from django.utils import timezone
from django.contrib.auth.models import User

class Article(models.Model):
	# 标题为字符型字段
	title = models.CharField(max_length=30)
	# 内容为文本字段
	content = models.TextField()
	create_time = models.DateTimeField(auto_now_add=True)
	last_updated_time = models.DateTimeField(auto_now=True)
	author = models.ForeignKey(User, on_delete=models.DO_NOTHING, default=1)
	is_deleted = models.BooleanField(default=False)
	read_num = models.IntegerField(default=0)


	def __str__(self):
		return '<Article: %s>' % self.title