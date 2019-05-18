from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, ButtonHolder, Submit, Fieldset, Column, Row
from .models import Event, Request
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
        ]


class EventForm(forms.ModelForm):
    resolution_date = forms.DateField(
        widget=forms.TextInput(
            attrs={'type': 'date'}
        )
    )

    class Meta:
        model = Event
        fields = [
            'type',
            'reference',
            'status',
            'resolution_date',
            'priority',
            'assigned_team',
            'assigned_person',
            'summary'
        ]

class CustomTimeDate(Field):
    template = 'custom_dateTimeField.html'

class CustomFieldForm(EventForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.layout = Layout(
            Row(
                Column('type', css_class='form-group col-md-6 mb-0'),
                Column('reference', css_class='form-group col-md-6 mb-0'),
                css_class='form-row'
            ),
            'status',
            'priority',
            Row(
                Column('assigned_team', css_class='form-group col-md-6 mb-0'),
                Column('assigned_person', css_class='form-group col-md-4 mb-0'),
                Column('summary', css_class='form-group col-md-2 mb-0'),
                css_class='form-row'
            ),
            CustomTimeDate('check_me_out'),  # <-- Here
            Submit('submit', 'Sign in')
        )