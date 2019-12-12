## Generating a Spotify Playlist

<a href="https://wfseaton.github.io/TheDigitalFrontier/">Home Page</a> - 
<a href="https://wfseaton.github.io/TheDigitalFrontier/data_preparation.html"><b>Data Preparation</b></a> - 
<a href="https://wfseaton.github.io/TheDigitalFrontier/data_exploration.html">Data Exploration</a> - 
<a href="https://wfseaton.github.io/TheDigitalFrontier/dimensionality_reduction.html">Dimensionality Reduction</a> - 
<a href="https://wfseaton.github.io/TheDigitalFrontier/clustering_techniques.html">Clustering Techniques</a> - 
<a href="https://wfseaton.github.io/TheDigitalFrontier/playlist_generation.html">Playlist Generation</a> - 
<a href="https://wfseaton.github.io/TheDigitalFrontier/conclusion.html">Conclusion</a> - 
<a href="https://wfseaton.github.io/TheDigitalFrontier/authors_gift.html">Authors' Gift</a>

-------------------------------------------------------------------------------------------------------------------

# Preparing and Enriching the Million Playlist Dataset

## Data Source

We began our work with the Million Playlist Dataset. This data set was prepared as part of the [RecSys Challenge 2018](https://recsys-challenge.spotify.com/) organized by Spotify, University of Massachusetts (Amherst), and Johannes Kepler University (Linz).

The raw data includes:

- 1,000 CSV files, totaling 11.63 GB
- Each CSV file has 1,000 playlists, each with a collection of approximately 50-200 songs
- ~65,000 songs in each CSV file, ~65,000,000 total observations (including duplicates) 

The initial provided data features include:

- **Playlist Number:** integer value ranging from 0 to 999 within each CSV file
- **Track Position:** integer value starting from 0 indicating the position of the song in the playlist
- **Track Name:** text value
- **Track URI:** alphanumeric identifier from Spotify
- **Artist Name:** text value
- **Artist URI:** alphanumeric identifier from Spotify
- **Album Name:** text value
- **Album URI:** alphanumeric identifier from Spotify
- **Duration:** integer value in milliseconds

Here is a sample of the provided data.

    Shape of data in CSV file: (64928, 9)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pid</th>
      <th>pos</th>
      <th>artist_name</th>
      <th>track_uri</th>
      <th>artist_uri</th>
      <th>track_name</th>
      <th>album_uri</th>
      <th>duration_ms</th>
      <th>album_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>Deftones</td>
      <td>spotify:track:4rEGJ9KirDlKiOHxqVwcVg</td>
      <td>spotify:artist:6Ghvu1VvMGScGpOUJBAHNH</td>
      <td>Sextape</td>
      <td>spotify:album:4RQnFSkkZlA65Xxchhnaha</td>
      <td>241533</td>
      <td>Diamond Eyes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>Muse</td>
      <td>spotify:track:0It6VJoMAare1zdV2wxqZq</td>
      <td>spotify:artist:12Chz98pHFMPJEknJQMWvI</td>
      <td>Undisclosed Desires</td>
      <td>spotify:album:0eFHYz8NmK75zSplL5qlfM</td>
      <td>235000</td>
      <td>The Resistance</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>Pearl Jam</td>
      <td>spotify:track:0LBmvPJYmtEJ7kkWvc3kbT</td>
      <td>spotify:artist:1w5Kfo2jwwIPruYS2UWh56</td>
      <td>Oceans</td>
      <td>spotify:album:5B4PYA7wNN4WdEXdIJu58a</td>
      <td>161893</td>
      <td>Ten</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>My Chemical Romance</td>
      <td>spotify:track:0uukw2CgEIApv4IWAjXrBC</td>
      <td>spotify:artist:7FBcuc1gsnv6Y1nwFtNRCb</td>
      <td>Dead!</td>
      <td>spotify:album:0FZK97MXMm5mUQ8mtudjuK</td>
      <td>195520</td>
      <td>The Black Parade</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>Red Hot Chili Peppers</td>
      <td>spotify:track:1iFIZUVDBCCkWe705FLXto</td>
      <td>spotify:artist:0L8ExT028jH3ddEcZwqJJ5</td>
      <td>Dosed</td>
      <td>spotify:album:6deiaArbeoqp1xPEGdEKp1</td>
      <td>311866</td>
      <td>By The Way</td>
    </tr>
  </tbody>
</table>
</div>


## Data Structuring

For songs that appear in multiple playlists, data in each of the columns are repeated. This produces a total of 11.63 GB of data, a significant computational challenge. A reasonable first step to slim down the size of the dataset without losing information or fidelity is to parse through all the files to create a reference table of all songs and their metadata. Each playlist can then be stored as a simple named object, where the name is the ID of the playlist and its value a vector of song IDs.

From this exercise, we output two data frames:

- **Songs table:** Pandas dataframe with all songs as rows and all data from individual CSV files as columns
- **Playlists series:** Pandas series with playlist IDs as indices and vectors of song IDs as items

Running this over the entire dataset has a run-time of 2.5 hours. This method of storing songs in a dedicated table reduces 65 million song observations to just 2.5 million unique songs. To make this computationally tractable, we leveraged Pandas data frames and the fact that their indices, if sorted and maintained properly, leverage hash tables for quick lookups. This is in contrast to using a base Python method like loops or list comprehension, which would require searching the full table for the song each time. This reduced the overall data scale from 11.63 GB to just 0.42 GB for the songs master table and 0.54 GB for the playlist vectors, a total that is less than 10% the original size with no information loss. Our efforts in complexity reduction enabled us to perform our modeling at a significantly larger scale and use more data to generate better recommendations.

Once we have our master dataframe with all unique songs, we can assign an ID to each song, which we do as a new column at the end of the below dataframe labeled `song_id`.

    --- 7.567456960678101 seconds ---
    (1003760, 8)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist_name</th>
      <th>artist_uri</th>
      <th>track_name</th>
      <th>album_uri</th>
      <th>duration_ms</th>
      <th>album_name</th>
      <th>count</th>
      <th>song_id</th>
    </tr>
    <tr>
      <th>track_uri</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>spotify:track:0002yNGLtYSYtc0X6ZnFvp</td>
      <td>Sidney Bechet's Blue Note Jazzmen</td>
      <td>spotify:artist:2XouUSO0EAJ9gMMoHiXqMt</td>
      <td>Muskrat Ramble</td>
      <td>spotify:album:04hQBJ7YSuNnZ0nbuXNYbY</td>
      <td>220293</td>
      <td>Jazz Classics</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <td>spotify:track:00039MgrmLoIzSpuYKurn9</td>
      <td>Zach Farlow</td>
      <td>spotify:artist:2jTojc4rAsOMx6200a8Ah1</td>
      <td>Thas What I Do</td>
      <td>spotify:album:0UHfgx3ITlxePDXLaN5Y6x</td>
      <td>222727</td>
      <td>The Great Escape 2</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>spotify:track:0006Rv1e2Xfh6QooyKJqKS</td>
      <td>Two Steps from Hell</td>
      <td>spotify:artist:2qvP9yerCZCS0U1gZU8wYp</td>
      <td>Nightwood</td>
      <td>spotify:album:1BD29pKydSXe1EsHFj0GrQ</td>
      <td>189638</td>
      <td>Colin Frake On Fire Mountain</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <td>spotify:track:0007AYhg2UQbEm88mxu7js</td>
      <td>Little Simz</td>
      <td>spotify:artist:6eXZu6O7nAUA5z6vLV8NKI</td>
      <td>Mandarin Oranges Part 2</td>
      <td>spotify:album:32RJzqlapfiU0fr2l4SSW9</td>
      <td>198000</td>
      <td>E.D.G.E</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <td>spotify:track:0009mEWM7HILVo4VZYtqwc</td>
      <td>Slam</td>
      <td>spotify:artist:0Y0Kj7BOR5DM0UevuY7IvO</td>
      <td>Movement</td>
      <td>spotify:album:62VkRE2ucNvZDnYMCsnNDh</td>
      <td>447534</td>
      <td>Movement</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>


With our generated `song_id`, we replace the `track_uri` in each playlist and switch `song_id` to become the new index column. This allows us to make faster lookups using `song_id` in our future work.

    10000/200000 -- 38.7440550327301 s
    20000/200000 -- 43.11882281303406 s
    30000/200000 -- 44.18085217475891 s
    40000/200000 -- 48.200636863708496 s
    50000/200000 -- 52.817174196243286 s
    60000/200000 -- 50.61113619804382 s
    70000/200000 -- 57.99031400680542 s
    80000/200000 -- 64.7493839263916 s
    90000/200000 -- 67.53792810440063 s
    100000/200000 -- 67.3009626865387 s
    110000/200000 -- 68.4447910785675 s
    120000/200000 -- 71.28671312332153 s
    130000/200000 -- 72.44740080833435 s
    140000/200000 -- 77.3995201587677 s
    150000/200000 -- 80.82779884338379 s
    160000/200000 -- 358.95413088798523 s
    170000/200000 -- 90.39383912086487 s
    180000/200000 -- 89.62128067016602 s
    190000/200000 -- 100.55844020843506 s
    200000/200000 -- 97.31215310096741 s
    --- 1642.5336339473724 seconds ---
    (200000,)
    284_0    [340039, 125250, 881533, 653897, 49614, 356319...
    284_1    [738782, 7646, 142078, 900881, 533258, 429837,...
    284_2    [552361, 135177, 507876, 865927, 638474, 55164...
    284_3    [214695, 27387, 700562, 448130, 1000188, 37723...
    284_4    [576080, 600, 170841, 842370, 450149, 8624, 89...
    dtype: object





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist_name</th>
      <th>artist_uri</th>
      <th>track_name</th>
      <th>album_uri</th>
      <th>duration_ms</th>
      <th>album_name</th>
      <th>count</th>
      <th>track_uri</th>
    </tr>
    <tr>
      <th>song_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Sidney Bechet's Blue Note Jazzmen</td>
      <td>spotify:artist:2XouUSO0EAJ9gMMoHiXqMt</td>
      <td>Muskrat Ramble</td>
      <td>spotify:album:04hQBJ7YSuNnZ0nbuXNYbY</td>
      <td>220293</td>
      <td>Jazz Classics</td>
      <td>1</td>
      <td>spotify:track:0002yNGLtYSYtc0X6ZnFvp</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Zach Farlow</td>
      <td>spotify:artist:2jTojc4rAsOMx6200a8Ah1</td>
      <td>Thas What I Do</td>
      <td>spotify:album:0UHfgx3ITlxePDXLaN5Y6x</td>
      <td>222727</td>
      <td>The Great Escape 2</td>
      <td>2</td>
      <td>spotify:track:00039MgrmLoIzSpuYKurn9</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Two Steps from Hell</td>
      <td>spotify:artist:2qvP9yerCZCS0U1gZU8wYp</td>
      <td>Nightwood</td>
      <td>spotify:album:1BD29pKydSXe1EsHFj0GrQ</td>
      <td>189638</td>
      <td>Colin Frake On Fire Mountain</td>
      <td>4</td>
      <td>spotify:track:0006Rv1e2Xfh6QooyKJqKS</td>
    </tr>
    <tr>
      <td>3</td>
      <td>Little Simz</td>
      <td>spotify:artist:6eXZu6O7nAUA5z6vLV8NKI</td>
      <td>Mandarin Oranges Part 2</td>
      <td>spotify:album:32RJzqlapfiU0fr2l4SSW9</td>
      <td>198000</td>
      <td>E.D.G.E</td>
      <td>1</td>
      <td>spotify:track:0007AYhg2UQbEm88mxu7js</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Slam</td>
      <td>spotify:artist:0Y0Kj7BOR5DM0UevuY7IvO</td>
      <td>Movement</td>
      <td>spotify:album:62VkRE2ucNvZDnYMCsnNDh</td>
      <td>447534</td>
      <td>Movement</td>
      <td>1</td>
      <td>spotify:track:0009mEWM7HILVo4VZYtqwc</td>
    </tr>
  </tbody>
</table>
</div>



## Enriching a Song's Musical Features

Our recommendation hypothesis is that song features provide a data-based way of determining similarity and thus good matches to our seed songs. We leverage Spotify's open API to retrieve these musical features and add it into our dataset using Spotipy, a lightweight Python library that allows us to authenticate to Spotify and easily query features on the song, artist, and album. 

To keep Spotify API requests reasonable, we randomly selected 200,000 playlists (out of the total 1,000,000). Across these 200,000 playlists, we have 1,003,760 unique songs. We pull the data listed below for songs, artists, and albums from the Spotify API. Descriptions here are directly from [Spotify API reference documentation](https://developer.spotify.com/documentation/web-api/reference/).

#### Song features
- **acousticness:** A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic.
- **danceability:** Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable.
- **energy:** Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.
- **instrumentalness:** Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0.
- **key:** The key the track is in. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on.
- **liveness:** Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live.
- **loudness:** The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db.
- **mode:** Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.
- **speechiness:** Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.
- **tempo:** The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.
- **time_signature:** An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).
- **valence:** A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

#### Artist features
- **artist_genres:** A list of the genres the artist is associated with. For example: "Prog Rock" , "Post-Grunge". (If not yet classified, the array is empty.)
- **artist_popularity:** The popularity of the artist. The value will be between 0 and 100, with 100 being the most popular. The artist’s popularity is calculated from the popularity of all the artist’s tracks.

#### Album features
- **album_genres:** A list of the genres used to classify the album. For example: "Prog Rock" , "Post-Grunge". (If not yet classified, the array is empty.)
- **album_popularity:** The popularity of the album. The value will be between 0 and 100, with 100 being the most popular. The popularity is calculated from the popularity of the album’s individual tracks.
- **release_date:** The date the album was first released. 

After pulling data from the Spotify API, we have a dataset of 999,950 unique songs and their associated metadata. We join our retrieved features back into the master tables to produce our final data frames below. We conclude by saving our transformed data into pickle files, which provides faster and more compact files for checkpoint saving.

#### Enriched Tracks Dataframe




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>track_uri</th>
      <th>time_signature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.455</td>
      <td>0.623</td>
      <td>8</td>
      <td>-11.572</td>
      <td>1</td>
      <td>0.0523</td>
      <td>0.797000</td>
      <td>0.903000</td>
      <td>0.6340</td>
      <td>0.9510</td>
      <td>182.345</td>
      <td>spotify:track:0002yNGLtYSYtc0X6ZnFvp</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.742</td>
      <td>0.753</td>
      <td>1</td>
      <td>-5.632</td>
      <td>1</td>
      <td>0.0364</td>
      <td>0.017800</td>
      <td>0.000000</td>
      <td>0.1330</td>
      <td>0.2630</td>
      <td>132.064</td>
      <td>spotify:track:00039MgrmLoIzSpuYKurn9</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.295</td>
      <td>0.498</td>
      <td>2</td>
      <td>-9.190</td>
      <td>0</td>
      <td>0.0301</td>
      <td>0.795000</td>
      <td>0.944000</td>
      <td>0.1070</td>
      <td>0.0445</td>
      <td>89.048</td>
      <td>spotify:track:0006Rv1e2Xfh6QooyKJqKS</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.648</td>
      <td>0.598</td>
      <td>7</td>
      <td>-11.845</td>
      <td>1</td>
      <td>0.3260</td>
      <td>0.164000</td>
      <td>0.000046</td>
      <td>0.1230</td>
      <td>0.4000</td>
      <td>138.883</td>
      <td>spotify:track:0007AYhg2UQbEm88mxu7js</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.695</td>
      <td>0.828</td>
      <td>1</td>
      <td>-6.818</td>
      <td>1</td>
      <td>0.0457</td>
      <td>0.000007</td>
      <td>0.851000</td>
      <td>0.1090</td>
      <td>0.0372</td>
      <td>126.014</td>
      <td>spotify:track:0009mEWM7HILVo4VZYtqwc</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1003729</th>
      <td>0.447</td>
      <td>0.724</td>
      <td>1</td>
      <td>-6.398</td>
      <td>0</td>
      <td>0.0372</td>
      <td>0.788000</td>
      <td>0.202000</td>
      <td>0.2420</td>
      <td>0.9400</td>
      <td>81.071</td>
      <td>spotify:track:7zzptITgTKf4HpJM8ye47v</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1003730</th>
      <td>0.497</td>
      <td>0.698</td>
      <td>7</td>
      <td>-2.558</td>
      <td>1</td>
      <td>0.0317</td>
      <td>0.127000</td>
      <td>0.000000</td>
      <td>0.1160</td>
      <td>0.5520</td>
      <td>129.996</td>
      <td>spotify:track:7zzpwV2lgKsLke68yFoZdp</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1003731</th>
      <td>0.314</td>
      <td>0.359</td>
      <td>0</td>
      <td>-14.035</td>
      <td>1</td>
      <td>0.0378</td>
      <td>0.745000</td>
      <td>0.866000</td>
      <td>0.6700</td>
      <td>0.1030</td>
      <td>131.911</td>
      <td>spotify:track:7zzrUgpSu2MSZF4FecBN3D</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1003732</th>
      <td>0.769</td>
      <td>0.644</td>
      <td>11</td>
      <td>-5.568</td>
      <td>0</td>
      <td>0.0997</td>
      <td>0.242000</td>
      <td>0.010200</td>
      <td>0.0509</td>
      <td>0.3960</td>
      <td>103.014</td>
      <td>spotify:track:7zzuTn6PnJ1DVfAiGsd4N0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1003733</th>
      <td>0.624</td>
      <td>0.851</td>
      <td>9</td>
      <td>-4.254</td>
      <td>1</td>
      <td>0.0723</td>
      <td>0.095800</td>
      <td>0.000000</td>
      <td>0.0766</td>
      <td>0.6930</td>
      <td>128.030</td>
      <td>spotify:track:7zzxEH0xUl5k3p6IxUfgAO</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>1003734 rows × 13 columns</p>
</div>



#### Enriched Artists Dataframe




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist_genres</th>
      <th>artist_popularity</th>
      <th>artist_uri</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[]</td>
      <td>18</td>
      <td>spotify:artist:2XouUSO0EAJ9gMMoHiXqMt</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[]</td>
      <td>27</td>
      <td>spotify:artist:2jTojc4rAsOMx6200a8Ah1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>['epicore', 'scorecore', 'soundtrack', 'video ...</td>
      <td>70</td>
      <td>spotify:artist:2qvP9yerCZCS0U1gZU8wYp</td>
    </tr>
    <tr>
      <th>3</th>
      <td>['alternative r&amp;b', 'escape room', 'indie r&amp;b'...</td>
      <td>63</td>
      <td>spotify:artist:6eXZu6O7nAUA5z6vLV8NKI</td>
    </tr>
    <tr>
      <th>4</th>
      <td>['acid techno', 'minimal dub', 'minimal techno...</td>
      <td>39</td>
      <td>spotify:artist:0Y0Kj7BOR5DM0UevuY7IvO</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>149667</th>
      <td>[]</td>
      <td>12</td>
      <td>spotify:artist:2mRZOivAjsqp7VbLrjfL5g</td>
    </tr>
    <tr>
      <th>149668</th>
      <td>[]</td>
      <td>15</td>
      <td>spotify:artist:4Cx5NofT2E2Ypu32HP0Mot</td>
    </tr>
    <tr>
      <th>149669</th>
      <td>['dub reggae', 'uk dub']</td>
      <td>15</td>
      <td>spotify:artist:3RQFJaSRrhXQ2E9T8vZEe3</td>
    </tr>
    <tr>
      <th>149670</th>
      <td>[]</td>
      <td>17</td>
      <td>spotify:artist:7iihQll6y9O8Iee7D1uEcb</td>
    </tr>
    <tr>
      <th>149671</th>
      <td>[]</td>
      <td>9</td>
      <td>spotify:artist:3uCW660nT9zh4oF4WhlBCl</td>
    </tr>
  </tbody>
</table>
<p>149133 rows × 3 columns</p>
</div>



#### Enriched Albums Dataframe




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>album_genres</th>
      <th>album_popularity</th>
      <th>album_release_date</th>
      <th>album_uri</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[]</td>
      <td>37</td>
      <td>1993-01-01</td>
      <td>spotify:album:04hQBJ7YSuNnZ0nbuXNYbY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[]</td>
      <td>0</td>
      <td>2016-01-08</td>
      <td>spotify:album:0UHfgx3ITlxePDXLaN5Y6x</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[]</td>
      <td>41</td>
      <td>2014-07-01</td>
      <td>spotify:album:1BD29pKydSXe1EsHFj0GrQ</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[]</td>
      <td>33</td>
      <td>2014-10-03</td>
      <td>spotify:album:32RJzqlapfiU0fr2l4SSW9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[]</td>
      <td>8</td>
      <td>2013-08-26</td>
      <td>spotify:album:62VkRE2ucNvZDnYMCsnNDh</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>375441</th>
      <td>[]</td>
      <td>33</td>
      <td>2017-08-25</td>
      <td>spotify:album:6Fnj40x1kkxtHK3icGVsqg</td>
    </tr>
    <tr>
      <th>375442</th>
      <td>[]</td>
      <td>0</td>
      <td>2014-09-16</td>
      <td>spotify:album:5lkHAkEZsMj2bvcAzzrJYz</td>
    </tr>
    <tr>
      <th>375443</th>
      <td>[]</td>
      <td>25</td>
      <td>2016-06-29</td>
      <td>spotify:album:2OhSv0hAHVqo5zIDaUDcQA</td>
    </tr>
    <tr>
      <th>375444</th>
      <td>[]</td>
      <td>22</td>
      <td>1994-09-09</td>
      <td>spotify:album:4WmaRhaV8rs1GOk1GX26j5</td>
    </tr>
    <tr>
      <th>375445</th>
      <td>[]</td>
      <td>0</td>
      <td>2015-03-09</td>
      <td>spotify:album:5BTzgzUsUtB55FsLXLpPaV</td>
    </tr>
  </tbody>
</table>
<p>375446 rows × 4 columns</p>
</div>



#### Total and Complete Master Songs Data Frame




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>artist_name</th>
      <th>artist_uri</th>
      <th>track_name</th>
      <th>album_uri</th>
      <th>duration_ms</th>
      <th>album_name</th>
      <th>count</th>
      <th>track_uri</th>
      <th>danceability</th>
      <th>energy</th>
      <th>...</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>artist_genres</th>
      <th>artist_popularity</th>
      <th>album_genres</th>
      <th>album_popularity</th>
      <th>album_release_date</th>
    </tr>
    <tr>
      <th>song_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sidney Bechet's Blue Note Jazzmen</td>
      <td>spotify:artist:2XouUSO0EAJ9gMMoHiXqMt</td>
      <td>Muskrat Ramble</td>
      <td>spotify:album:04hQBJ7YSuNnZ0nbuXNYbY</td>
      <td>220293</td>
      <td>Jazz Classics</td>
      <td>1</td>
      <td>spotify:track:0002yNGLtYSYtc0X6ZnFvp</td>
      <td>0.455</td>
      <td>0.623</td>
      <td>...</td>
      <td>0.90300</td>
      <td>0.6340</td>
      <td>0.9510</td>
      <td>182.345</td>
      <td>4</td>
      <td>[]</td>
      <td>18</td>
      <td>[]</td>
      <td>37</td>
      <td>1993-01-01</td>
    </tr>
    <tr>
      <th>159583</th>
      <td>Sidney Bechet</td>
      <td>spotify:artist:1RsmXc1ZqW3WBs9iwxiSwk</td>
      <td>Blue Horizon</td>
      <td>spotify:album:04hQBJ7YSuNnZ0nbuXNYbY</td>
      <td>264933</td>
      <td>Jazz Classics</td>
      <td>5</td>
      <td>spotify:track:1EWPMNHfdVNJwBpG9BcxXB</td>
      <td>0.327</td>
      <td>0.372</td>
      <td>...</td>
      <td>0.83500</td>
      <td>0.1530</td>
      <td>0.3800</td>
      <td>66.036</td>
      <td>4</td>
      <td>['bebop', 'big band', 'cool jazz', 'dixieland'...</td>
      <td>52</td>
      <td>[]</td>
      <td>37</td>
      <td>1993-01-01</td>
    </tr>
    <tr>
      <th>271702</th>
      <td>Sidney Bechet</td>
      <td>spotify:artist:1RsmXc1ZqW3WBs9iwxiSwk</td>
      <td>Blame It On The Blues - Alternate Take</td>
      <td>spotify:album:04hQBJ7YSuNnZ0nbuXNYbY</td>
      <td>175893</td>
      <td>Jazz Classics</td>
      <td>1</td>
      <td>spotify:track:26N4Y48EjprAtvlY6yWZTA</td>
      <td>0.574</td>
      <td>0.606</td>
      <td>...</td>
      <td>0.94800</td>
      <td>0.3490</td>
      <td>0.9650</td>
      <td>101.361</td>
      <td>4</td>
      <td>['bebop', 'big band', 'cool jazz', 'dixieland'...</td>
      <td>52</td>
      <td>[]</td>
      <td>37</td>
      <td>1993-01-01</td>
    </tr>
    <tr>
      <th>445190</th>
      <td>Sidney Bechet</td>
      <td>spotify:artist:1RsmXc1ZqW3WBs9iwxiSwk</td>
      <td>Summertime</td>
      <td>spotify:album:04hQBJ7YSuNnZ0nbuXNYbY</td>
      <td>251906</td>
      <td>Jazz Classics</td>
      <td>16</td>
      <td>spotify:track:3RlJx8xwZEyToSuGrygilr</td>
      <td>0.608</td>
      <td>0.138</td>
      <td>...</td>
      <td>0.90800</td>
      <td>0.0853</td>
      <td>0.3180</td>
      <td>83.124</td>
      <td>4</td>
      <td>['bebop', 'big band', 'cool jazz', 'dixieland'...</td>
      <td>52</td>
      <td>[]</td>
      <td>37</td>
      <td>1993-01-01</td>
    </tr>
    <tr>
      <th>626275</th>
      <td>Sidney Bechet</td>
      <td>spotify:artist:1RsmXc1ZqW3WBs9iwxiSwk</td>
      <td>Dear Old Southland</td>
      <td>spotify:album:04hQBJ7YSuNnZ0nbuXNYbY</td>
      <td>243693</td>
      <td>Jazz Classics</td>
      <td>1</td>
      <td>spotify:track:4qwAa1rOm8iaegHzoM1b31</td>
      <td>0.400</td>
      <td>0.320</td>
      <td>...</td>
      <td>0.84200</td>
      <td>0.1950</td>
      <td>0.6130</td>
      <td>86.186</td>
      <td>4</td>
      <td>['bebop', 'big band', 'cool jazz', 'dixieland'...</td>
      <td>52</td>
      <td>[]</td>
      <td>37</td>
      <td>1993-01-01</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1003684</th>
      <td>Royal Rizow</td>
      <td>spotify:artist:5eXba1Axr3YJgg5j8Hn7a8</td>
      <td>I'll Find a Way (feat. Ty Reynolds)</td>
      <td>spotify:album:3XWD2ACwxS3fnGGgu298eS</td>
      <td>205346</td>
      <td>I'll Find a Way (feat. Ty Reynolds)</td>
      <td>11</td>
      <td>spotify:track:7zy1Ck7lyk9oruK1xozs7e</td>
      <td>0.733</td>
      <td>0.876</td>
      <td>...</td>
      <td>0.00000</td>
      <td>0.0830</td>
      <td>0.3980</td>
      <td>104.947</td>
      <td>4</td>
      <td>[]</td>
      <td>11</td>
      <td>[]</td>
      <td>13</td>
      <td>2016-08-05</td>
    </tr>
    <tr>
      <th>1003707</th>
      <td>Moa Felicia</td>
      <td>spotify:artist:2uB6FyYwjUENL072rdwu5B</td>
      <td>Later That Night - Original Version</td>
      <td>spotify:album:2ugoxa1ZsUKtbziD0SonBA</td>
      <td>329032</td>
      <td>Later That Night - Single</td>
      <td>1</td>
      <td>spotify:track:7zyYROflk5MgxcMjpNrCqT</td>
      <td>0.804</td>
      <td>0.513</td>
      <td>...</td>
      <td>0.75200</td>
      <td>0.0686</td>
      <td>0.0394</td>
      <td>124.003</td>
      <td>4</td>
      <td>[]</td>
      <td>0</td>
      <td>[]</td>
      <td>0</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>1003733</th>
      <td>The Commercial Hippies</td>
      <td>spotify:artist:4Cx5NofT2E2Ypu32HP0Mot</td>
      <td>The Antidote - Original Mix</td>
      <td>spotify:album:2UnJeq5hGlBFCcnQZZ2Va9</td>
      <td>511639</td>
      <td>The Antidote</td>
      <td>1</td>
      <td>spotify:track:7zzGEjXJF5HYT5hghmGDXs</td>
      <td>0.627</td>
      <td>0.908</td>
      <td>...</td>
      <td>0.60300</td>
      <td>0.2350</td>
      <td>0.1660</td>
      <td>140.007</td>
      <td>4</td>
      <td>[]</td>
      <td>15</td>
      <td>[]</td>
      <td>8</td>
      <td>2016-09-19</td>
    </tr>
    <tr>
      <th>1003746</th>
      <td>Murray Man</td>
      <td>spotify:artist:3RQFJaSRrhXQ2E9T8vZEe3</td>
      <td>Tell Me What a Gwan</td>
      <td>spotify:album:5p8thDxso3h80HbO5fTnL2</td>
      <td>226813</td>
      <td>The Early Releases</td>
      <td>1</td>
      <td>spotify:track:7zzYmYrt16PNK0ukHCZdkV</td>
      <td>0.787</td>
      <td>0.480</td>
      <td>...</td>
      <td>0.00289</td>
      <td>0.0786</td>
      <td>0.7140</td>
      <td>146.036</td>
      <td>4</td>
      <td>['dub reggae', 'uk dub']</td>
      <td>15</td>
      <td>[]</td>
      <td>5</td>
      <td>2010-10-01</td>
    </tr>
    <tr>
      <th>1003759</th>
      <td>D&amp;B</td>
      <td>spotify:artist:3uCW660nT9zh4oF4WhlBCl</td>
      <td>Princesa De Mis Sueños</td>
      <td>spotify:album:2Qv89jKrJtXf5SjThoJvHE</td>
      <td>185577</td>
      <td>Princesa De Mis Sueños</td>
      <td>1</td>
      <td>spotify:track:7zzxEH0xUl5k3p6IxUfgAO</td>
      <td>0.624</td>
      <td>0.851</td>
      <td>...</td>
      <td>0.00000</td>
      <td>0.0766</td>
      <td>0.6930</td>
      <td>128.030</td>
      <td>4</td>
      <td>[]</td>
      <td>9</td>
      <td>[]</td>
      <td>0</td>
      <td>2014-08-26</td>
    </tr>
  </tbody>
</table>
<p>999950 rows × 25 columns</p>
</div>


