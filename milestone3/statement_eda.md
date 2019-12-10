# Final Project - Milestone 3
## Revised Project Statement & EDA

#### Instructions
Wed, November 20: EDA and Revised Project Statement (15 points)

On Canvas in the ’Final Project - Milestone 2’ assignment, submit a 2 - 3 page revised project statement and EDA (can be created using Latex, word processing software, etc.) and an accompanying Jupyter notebook (that was used to create the visuals). Your 2 - 3 page submission should include:
- A description of the data: what type of data are you dealing with? What methods have you used to explore the data (initial explorations, data cleaning and reconciliation, etc)?
- Visualizations and captions that summarize the noteworthy findings of the EDA.
- A revised project question based on the insights you gained through EDA.
- A baseline model.


#### Data Description

We begin with 1,000 csv files, each containing about 65,000 songs and associated playlists, possibly totaling 65 million observations. A song appears every time it is added to a playlist. The playlists vary in terms of the number of songs added to each, with the largest playlist comprising 341 songs and the smallest 3. From these files, we get information about the song title, song length, artist name, album name and associated 'uri', or Spotify unique identifier, for each. To handle the scale, we first create a reference table of all unique songs and their metadata found across all 1,000 files. From there, we create a smaller file representing playlists, which contains just the playlist id and a vector of associated song ids. We do this in a For loop, by enumerating the list of file names, reading them in one at a time, using Pandas groupby to calculate count appearances and unique track_uri values and finally appending those to a DataFrame 'songs' and a Series 'playlists, respectively. This took 2.2 hours to run. Over these optimized files, we perform replacement of some values, such as track_uri with song_id and select specific index columns. This took 7.7 hours to run.

Once our unique songs and playlists files are ready, we go to Spotify to enrich our data with information from the Spotify API. We used a package called Spotipy, a lightweight Python library for the Spotify Web API that allows us to authenticate and query a large number of features on the song, artist, and album. We join in new features such as album and artist popularity, album release date, artist genres, and exciting song features like danceability, energy, loudness and more. After enriching our song data with additional features directly from Spotify, we produce our final master pickle files for analysis.


#### Noteworthy Visualisations
Summarize the noteworthy findings 

We explored a number of dimensions of songs and playlist data. For songs, only a few songs appear in an incredibly high number of playlists with rapid and significant drop off. Playlists have a peak at around 20 songs per playlist with a steady decline in distribution. In looking at feature relationships, the majority appear unrelated with a few interesting exceptions. As danceability increases, we see a slight linear relationship with playlist inclusion count. Loudness has a narrow band of inclusion around -7, showing that overly loud songs are not welcome. As expected, popular artists are included in more playlists. Perhaps surprisingly, higher energy songs appear to have no relationship with playlist inclusion, though this likely speaks more to the popularity of playlists focused on "Sleep" or "Classical", which have lower energy. Lastly, if your song is too long, above 5.5 minutes, it is going to be included in very few playlists.

We identified the preferred tempo for dance music, which appears as a "tempo bump" on the plot of danceability against tempo.

There is an interesting split of album popularity, where songs from top quartile popular albums and bottom quartile unpopular albums appear in playlists to a greater extent while songs from middle quartiles average albums appear in very few. This may speak to the presence of "one hit wonders" on unpopular albums still being included across playlists.

There are distinct peaks in tempo popularities, 

In looking at the distribution of album release year for songs in playlists, people primarily care about recently released music. Songs released even five years ago are included at a dramatically lower rate than songs from recent years.

We explored song features and how they have changed over time. The most dramatic cross over occurs with acousticness and energy, as the former declines several starting in YEAR and the latter rises steadily over time. This represents the most dramatic change in the type of music being produced over the last 70 years being included in playlists.

We wanted to see if there were certain musical keys that were more popular and found that First Key and Seventh Key are clear preferences for the top 100 most included songs in our dataset.


#### Baseline model

For our baseline model, we deployed K-Means Clustering as an unsupervised method to cluster songs into clusters, or families. To populate a playlist, we take a new song, predict its cluster and pull other songs from that cluster on the assumption that like songs have been grouped together and a user wants to hear similar songs within their playlist. If several songs have been provided, we pull songs from the mode cluster ## WHICH DO WE DO?

A downside to using an unsupervised model in this way is that we have no objective measure of song or subsequent playlist quality. We are likely adding songs that might be similar but without a sense

Our initial model has been run over a representative sample of 20\% of all songs and playlists due to memory and computational costs. One driving factor is the high dimensionality of our data structure, where we maintain a binary indicator matrix saying whether each song (row) is in each playlist (column). We intend to pursue a reduction in dimensions by a method such as applying K-Means Clustering to the playlists as well, deriving a much smaller number of playlist families that can be used as features in our binary indicator matrix instead.

#### Revised project question based on insights in EDA

We will focus on creating a playlist from a largely cold start with minor user input, such as the first five songs or a genre selection. 

We see from our feature exploration that song attributes have some relationship to playlist inclusion rate. Our current baseline method has no indication of song quality but is clustering like songs by their attributes. Incorporating playlist inclusion as a measure of quality and weighting song selection from our derived clusters can serve as a form of quality by treating playlist inclusion as user-stated listening preferences.


-----------------------

Ideas
- Cold start with user input, e.g. five songs to start with and/or a specification of genre etc.
- Cold start with only starter songs, using information of song-playlist relationships for a better model (hard to measure "better")

- K-Means Clustering: cluster s songs into k clusters, each of which is a grouping or family of songs. Given a new song, predict its cluster, and pull songs from that cluster (perhaps based on proximity to the new song) to populate the new playlist. If several songs are provided, pull from mode cluster or all predicted clusters with probabilities that reflect the frequency of times each cluster was predicted among the cold start songs.

____

Notes
- This is an unsupervised (or semi-supervised) problem. We have no objective measure of playlist quality, and therefore both no way to fit supervised model and no really systematic way to compare methods.
- Limited to 200 first files (/1000), i.e. first 200,000 playlists (/1,000,000). Will (try to) re-run on all the playlists for final implementation. This is a random subset of the files, so like any sample/population relationship, we reasonably expect our EDA on this sample to be representative of the population.
- High dimensionality in terms of songs across playlists. E.g. a binary indicator matrix saying whether each song (row) is in each playlist(column) would have dimensions ca. 2,000,000 * 1,000,000, meaning 2,000,000,000,000 cells, which is computationally intractable. Need some way of condensing playlist-song relationships to a lower-dimensional space. One option is to run clustering on the playlists to create k clusters of playlist types/families/groups (unsupervised), and for each song, given the playlists it's part of, get the playlist cluster as a feature.
