import os

from django.http import HttpResponse
from django.shortcuts import render

from classes.model import process

current_dir = os.path.dirname(os.path.realpath(__file__))


def index(request):
    return render(request, 'index/index.html')


def handle(request):
    if request.method == 'POST':
        if not os.path.isdir(os.path.join(current_dir, 'uploads')):
            os.mkdir(os.path.join(current_dir, '..', 'uploads'))

        path = os.path.join(current_dir, 'uploads', str(request.FILES['image']))

        with open(path, 'wb+') as destination:
            for chunk in request.FILES['image'].chunks():
                destination.write(chunk)

        pred = process([path])

        os.remove(path)

        return HttpResponse(pred)
        # return render(request, 'index/result.html')

    return HttpResponse("Failed")