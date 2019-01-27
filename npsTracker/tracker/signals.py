from django.contrib.auth import user_logged_in
from django.dispatch import receiver


@receiver(user_logged_in)
def userLoggedIn(sender, **kwargs):
    request = kwargs['request']
    request.session["toggleTime"]
    print("Hello world")