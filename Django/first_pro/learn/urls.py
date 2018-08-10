from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name = 'home'),
    path('article/<int:article_id>', views.article_detail, name='article_id'),
    path('article/', views.article_list, name='article_list')
]
