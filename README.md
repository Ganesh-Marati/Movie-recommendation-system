# MovieRecommendationSystem

##### Providing the Recommendation for Movies based on Geners, Cast, reviews and description.

The details of the movies(title, genre, runtime, rating, poster, etc) are fetched using an API by TMDB, https://www.themoviedb.org/documentation/api, and using the IMDB id of the movie in the API, I did web scraping to get the reviews given by the user in the IMDB site using beautifulsoup4 and performed sentiment analysis on those reviews.


# How to get the api key?
Create an account in https://www.themoviedb.org/, click on the API link from the left hand sidebar in your account settings and fill all the details to apply for API key. If you are asked for the website URL, just give "NA" if you don't have one. You will see the API key in your API sidebar once your request is approved.

# How to run the project?
1)Clone or download this repository to your local machine.
2)Install all the libraries mentioned in the requirements.txt file with the command pip install -r requirements.txt
3)Get your API key from https://www.themoviedb.org/. (Refer the above section on how to get the API key)
4)Replace YOUR_API_KEY in both the places (line no. 15 and 29) of static/recommend.js file and hit save.
5)Open your terminal/command prompt from your project directory and run the file main.py by executing the command python main.py.
6)Go to your browser and type http://127.0.0.1:5000/ in the address bar.
7)Hurray! That's it.

# Architecture
![image](https://user-images.githubusercontent.com/106338826/170865033-b8eaea9b-0328-4888-8a87-be283997a5d3.png)
# Similarity Score :
How does it decide which item is most similar to the item user likes? Here come the similarity scores.

It is a numerical value ranges between zero to one which helps to determine how much two items are similar to each other on a scale of zero to one. This similarity score is obtained measuring the similarity between the text details of both of the items. So, similarity score is the measure of similarity between given text details of two items. This can be done by cosine-similarity.

# How Cosine Similarity works?
Cosine similarity is a metric used to measure how similar the documents are irrespective of their size. Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space. The cosine similarity is advantageous because even if the two similar documents are far apart by the Euclidean distance (due to the size of the document), chances are they may still be oriented closer together. The smaller the angle, higher the cosine similarity.

![image](https://user-images.githubusercontent.com/106338826/170865080-6ac9bdb7-3180-4445-8153-f4c32b9f708b.png)
More about Cosine Similarity : Understanding the Math behind Cosine Similarity
 # Sources of the datasets
 IMDB 5000 Movie Dataset
The Movies Dataset
List of movies in 2018
List of movies in 2019
List of movies in 2020
