
from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.predict_car_price, name='predict_car_price'),
    path('visualize/', views.visualize_data, name='visualize_data'),
]
