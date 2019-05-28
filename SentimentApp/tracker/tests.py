from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
import datetime

from .models import PosScores, WeightedAvg, NegScores, Review


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
        PosScores.objects.create(precision=0.122, recall=0.988, f1=0.266, support=24423)
        NegScores.objects.create(precision=0.122, recall=0.988, f1=0.242, support=24423)
        WeightedAvg.objects.create(precision=0.122, recall=0.988, f1=0.234, support=24423)
        lpos = PosScores.objects.first()
        lneg = NegScores.objects.first()
        lavg = WeightedAvg.objects.first()
        # cr_date = datetime.date.today()
        for i in range(100):
            Review.objects.create(reviewText = 'This is a test', predictSentiment = 'positive', actualSentiment = 'negative',
                                  pos_batch_no = lpos, neg_batch_no = lneg, avg_batch_no = lavg)

class EventTests(Setup_Class):
    def test_content(self):
        review = Review.objects.first()
        expected_event_reference = f'{review.reviewText}'
        self.assertEquals(expected_event_reference, 'This is a test')
#
    def test_event_list_view(self):
        response = self.client.get(reverse('tracker-home'))
        self.assertEqual(response.status_code, 200)
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



class TableTest(Setup_Class):
    def setUp(self):
        super().setUp()

    def test_table_layout(self):
        # send login data
        self.credentials = {
            'username': 'jtur1',
            'password': 'onion'}
        User.objects.create_user(**self.credentials)
        response = self.client.post('/login/', self.credentials, follow=True)
        self.assertTrue(response.context['user'].is_active)

        response = self.client.get(reverse('tracker-classTable'))
        self.assertEquals(response.status_code, 200)
        self.assertContains(response, 'Precision')
        self.assertContains(response, 'Recall')
        self.assertContains(response, 'F1-score')
        self.assertContains(response, '0.122')
        self.assertContains(response, '0.988')
        self.assertContains(response, '0.266')

class CountTest(Setup_Class):
    def setUp(self):
        super().setUp()

    def test_count_display(self):
        # send login data
        self.credentials = {
            'username': 'jtur1',
            'password': 'onion'}
        User.objects.create_user(**self.credentials)
        response = self.client.post('/login/', self.credentials, follow=True)
        self.assertTrue(response.context['user'].is_active)






