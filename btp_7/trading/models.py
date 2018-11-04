from django.db import models

# Create your models here.
class Adj_close(models.Model):
	adj_open=models.FloatField(max_length=20)
	moving_avg=models.FloatField(max_length=20)
	adj_high=models.FloatField(max_length=20)
	adj_low=models.FloatField(max_length=20)
	hl_pct=models.FloatField(max_length=20)
	pct_change=models.FloatField(max_length=20)
	adj_vol=models.FloatField(max_length=20)
	avg=models.FloatField(max_length=20)
	runtime=models.FloatField(max_length=20)

class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)