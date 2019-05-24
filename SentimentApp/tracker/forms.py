from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, ButtonHolder, Submit, Fieldset, Column, Row
from .models import Request
from crispy_forms.layout import Field

from django import forms


class RequestForm(forms.ModelForm):

    class Meta:
        model = Request
        fields = [
            'name',
            'surname',
            'email',
            'contactNo',
            'reason',
            'other',
        ]


