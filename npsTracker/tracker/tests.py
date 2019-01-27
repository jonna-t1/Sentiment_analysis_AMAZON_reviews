from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from .models import Event

import datetime


class HomePageTests(TestCase):

    def test_home_page_status_code(self):
        response = self.client.get('/')
        self.assertEquals(response.status_code, 200)

    def test_view_url_by_name(self):
        response = self.client.get(reverse('tracker-home'))
        self.assertEquals(response.status_code, 200)

    def test_view_uses_correct_template(self):
        response = self.client.get(reverse('tracker-home'))
        self.assertEquals(response.status_code, 200)
        self.assertTemplateUsed(response, 'tracker/home.html')


class Setup_Class(TestCase):

    def setUp(self):
        self.user = User.objects.create_user(username='jtur', email='jtur@accenture.com', password='onion')
        user = User.objects.first()
        cr_date = datetime.datetime(2019, 5, 5, 18, 23, 29)
        Event.objects.create(type = 'Problem', reference = 'P1063', status = 1, resolution_date=cr_date, priority = 'P4', assigned_team = 'NISPI/DWP', assigned_person = user, summary = 'this is an example summary')

class EventTests(Setup_Class):
    def test_content(self):
        event = Event.objects.first()
        expected_event_reference = f'{event.reference}'
        self.assertEquals(expected_event_reference, 'P1063')

    def test_event_list_view(self):
        response = self.client.get(reverse('tracker-home'))
        self.assertEqual(response.status_code, 200)
        # self.assertContains(response, 'P1063')
        self.assertTemplateUsed(response, 'tracker/home.html')

class LogInTest(TestCase):
    def setUp(self):
        self.credentials = {
            'username': 'jtur',
            'password': 'onion'}
        User.objects.create_user(**self.credentials)

    def test_login(self):
        # send login data
        response = self.client.post('/login/', self.credentials, follow=True)
        # should be logged in now
        self.assertTrue(response.context['user'].is_active)



class TableTest(TestCase):
    def setUp(self):
        self.credentials = {
            'username': 'jtur',
            'password': 'onion'}
        User.objects.create_user(**self.credentials)
        user = User.objects.first()
        cr_date = datetime.datetime(2019, 5, 5, 18, 23, 29)
        Event.objects.create(type='Problem', reference='P1063', status=1, resolution_date=cr_date, priority='P4',
                             assigned_team='NISPI/DWP',assigned_person=user, summary='this is an example summary')
        Event.objects.create(type='Problem', reference='P1039', status=1, resolution_date=cr_date, priority='P4',
                             assigned_team='NISPI/DWP', assigned_person=user, summary='this is an example summary')

    def test_table_layout(self):
        # send login data
        response = self.client.post('/login/', self.credentials, follow=True)
        # should be logged in now
        self.assertTrue(response.context['user'].is_active)
        self.assertContains(response, '<th>Type</th>')
        self.assertContains(response, '<th>Reference</th>')
        self.assertContains(response, '<th>Status </th>')
        self.assertContains(response, '<th>Resolution Date</th>')
        self.assertContains(response, 'Resolution Countdown')
        self.assertContains(response, '<th>Priority</th>')
        self.assertContains(response, '<th>Assigned Team</th>')
        self.assertContains(response, '<th>Assigned Person</th>')
        self.assertContains(response, '<th>Summary</th>')
        self.assertContains(response, 'Incidents for jtur')
        self.assertContains(response, 'P1063')
        self.assertContains(response, 'P1039')