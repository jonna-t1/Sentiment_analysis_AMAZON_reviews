from django.contrib import admin
from .models import Review, NegScores, PosScores, WeightedAvg
from .models import Request


class ReviewAdmin(admin.ModelAdmin):
    search_fields = ('batch_date', 'reviewText')

admin.site.register(Review, ReviewAdmin)
admin.site.register(Request)

#classification
admin.site.register(PosScores)
admin.site.register(NegScores)
admin.site.register(WeightedAvg)
