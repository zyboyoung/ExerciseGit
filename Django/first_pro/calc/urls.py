from django.urls import path
from . import views

urlpatterns = [
    path('add/', views.add, name = 'add'),
    path('add/<int:a>/<int:b>/', views.add2, name = 'add2'),
    path('', views.index, name = 'home'),
    path('old/', views.old_url_redirect),
    path('new/', views.old_url, name = 'old_new')
]
