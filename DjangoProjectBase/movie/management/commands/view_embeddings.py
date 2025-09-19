import numpy as np
from movie.models import Movie
import random

from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = "Muestra embeddings de películas (ejemplo)"

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS("Buscando un embedding al azar"))
        movies = Movie.objects.all()

        randomMovie = movies[random.randint(0, movies.count())]
        embedding_vector = np.frombuffer(randomMovie.emb , dtype=np.float32)
        print('El embedding de la pelicula ' , randomMovie.title , ' es: ' , embedding_vector)

        #Listar todos los embedding de las peliculas
        # for movie in Movie.objects.all():
        #     embedding_vector = np.frombuffer(movie.emb, dtype=np.float32)
        #     print(movie.title, embedding_vector[:5])  # Muestra los primeros valores

