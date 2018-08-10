from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.urls import reverse

def add(request):
	a = request.GET.get('a', 1)
	b = request.GET.get('b', 2)
	c = int(a) + int(b)
	return HttpResponse(str(c))

def add2(request, a, b):
	c = int(a) + int(b)
	return HttpResponse(str(c))

def index(request):
	return render(request, 'calc/home.html')

def old_url(request):
	return HttpResponse('这是个网页')

def old_url_redirect(request):
	return HttpResponseRedirect(reverse('old_new'))