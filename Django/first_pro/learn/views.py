from django.shortcuts import render, get_object_or_404, render_to_response

# Create your views here.
from django.http import HttpResponse, Http404
from .models import Article

def article_detail(request, article_id):
	article = get_object_or_404(Article, id=article_id)
	context = {}
	context['article'] = article
	return render(request, 'learn/detail.html', context)


def index(request):
	return HttpResponse('Welcome')

def article_list(request):
	articles = Article.objects.filter(is_deleted=False)
	context = {}
	context['articles'] = articles
	return render_to_response('learn/article_list.html', context)