from django.shortcuts import render

from movie.models import Movie


from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

def recommendation(request):
    searchTerm = request.GET.get('movieDescription') # GET se usa para solicitar recursos de un servidor
    
    load_dotenv('../.env')
    client = OpenAI(api_key=os.environ.get('openai_apikey'))

    def get_embedding(text):
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    if searchTerm:
        userEmbedding = get_embedding(searchTerm)
        # ✅ Store embedding as binary in the database
        allMovies = Movie.objects.all()
        mayorSimilarity = 0
        recommendedMovie = Movie.objects.none()
        for movie in allMovies: #Look in all movies
            movieEmbedding = movie.emb #Transform user input to embedding array
            movieEmbedding_vector = np.frombuffer(movieEmbedding , dtype=np.float32) #Get movie embedding array
            similarityBetweenSearch = cosine_similarity(userEmbedding , movieEmbedding_vector) #Compere user and movie description's embedding

            if similarityBetweenSearch > mayorSimilarity: #If the similarity is higher, stablish a new recommended movie
                mayorSimilarity = similarityBetweenSearch
                recommendedMovie = movie

    else:
        recommendedMovie = Movie.objects.none()
    return render(request, 'recommendations.html', {'searchTerm':searchTerm, 'recommendedMovie':recommendedMovie})