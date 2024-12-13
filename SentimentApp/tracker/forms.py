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


class FileUploadForm(forms.Form):
    file = forms.FileField(label='Upload a .json.gz file')

    def clean_file(self):
        uploaded_file = self.cleaned_data['file']
        # Check the file extension
        if not uploaded_file.name.endswith('.json.gz'):
            raise forms.ValidationError("The file must be a .json.gz file.")
        return uploaded_file
