from .models import Request



def add_variable_to_context(request):
    return {
        'requests': Request.objects.all(),
        'request_count': Request.objects.all().count(),
}