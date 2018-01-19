import os

from django.http import HttpResponse
from django.shortcuts import render

current_dir = os.path.dirname(os.path.realpath(__file__))


def index(request):
    return render(request, 'index/index.html')


def handle(request):
    if request.method == 'POST':
        handle_uploaded_file(request.FILES['image'], str(request.FILES['image']))
        return HttpResponse("Successful")

    return HttpResponse("Failed")


def handle_uploaded_file(file, filename):
    if not os.path.join(current_dir, '..', 'uploads'):
        os.mkdir(os.path.join(current_dir, '..', 'uploads'))

    path = os.path.join(current_dir, '..', 'uploads', filename)

    with open(path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)



    os.remove(path)
