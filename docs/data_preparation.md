## Generating a Spotify Playlist

<a href="https://thedigitalfrontier.github.io/spotify-playlist-generation/">Home Page</a> - 
<a href="https://thedigitalfrontier.github.io/spotify-playlist-generation/data_preparation"><b>Data Preparation</b></a> - 
<a href="https://thedigitalfrontier.github.io/spotify-playlist-generation/data_exploration">Data Exploration</a> - 
<a href="https://thedigitalfrontier.github.io/spotify-playlist-generation/dimensionality_reduction.">Dimensionality Reduction</a> - 
<a href="https://thedigitalfrontier.github.io/spotify-playlist-generation/clustering_techniques">Clustering Techniques</a> - 
<a href="https://thedigitalfrontier.github.io/spotify-playlist-generation/playlist_generation">Playlist Generation</a> - 
<a href="https://thedigitalfrontier.github.io/spotify-playlist-generation/conclusion">Conclusion</a> - 
<a href="https://thedigitalfrontier.github.io/spotify-playlist-generation/authors_gift">Authors' Gift</a>

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

## Data Structuring

For songs that appear in multiple playlists, data in each of the columns are repeated. This produces a total of 11.63 GB of data, a significant computational challenge. A reasonable first step to slim down the size of the dataset without losing information or fidelity is to parse through all the files to create a reference table of all songs and their metadata. Each playlist can then be stored as a simple named object, where the name is the ID of the playlist and its value a vector of song IDs.

From this exercise, we output two data frames:

- **Songs table:** Pandas dataframe with all songs as rows and all data from individual CSV files as columns
- **Playlists series:** Pandas series with playlist IDs as indices and vectors of song IDs as items

Running this over the entire dataset has a run-time of 2.5 hours. This method of storing songs in a dedicated table reduces 65 million song observations to just 2.5 million unique songs. To make this computationally tractable, we leveraged Pandas data frames and the fact that their indices, if sorted and maintained properly, leverage hash tables for quick lookups. This is in contrast to using a base Python method like loops or list comprehension, which would require searching the full table for the song each time. This reduced the overall data scale from 11.63 GB to just 0.42 GB for the songs master table and 0.54 GB for the playlist vectors, a total that is less than 10% the original size with no information loss. Our efforts in complexity reduction enabled us to perform our modeling at a significantly larger scale and use more data to generate better recommendations.

Once we have our master dataframe with all unique songs, we can assign an ID to each song, which we do as a new column at the end of the below dataframe labeled `song_id`.

With our generated `song_id`, we replace the `track_uri` in each playlist and switch `song_id` to become the new index column. This allows us to make faster lookups using `song_id` in our future work.

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


