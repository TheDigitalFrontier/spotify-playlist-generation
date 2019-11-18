# Final Project - Milestone 3
## Revised Project Statement & EDA

#### Description of the data

#### Visualisations and captions
Summarise the noteworthy findings 

#### Revised project question based on insights in EDA
Ideas
- Cold start with user input, e.g. five songs to start with and/or a specification of genre etc.
- Cold start with only starter songs, using information of song-playlist relationships for a better model (hard to measure "better")

#### Baseline model
- K-Means Clustering: cluster s songs into k clusters, each of which is a grouping or family of songs. Given a new song, predict its cluster, and pull songs from that cluster (perhaps based on proximity to the new song) to populate the new playlist. If several songs are provided, pull from mode cluster or all predicted clusters with probabilities that reflect the frequency of times each cluster was predicted among the cold start songs.

____

Notes
- This is an unsupervised (or semi-supervised) problem. We have no objective measure of playlist quality, and therefore both no way to fit supervised model and no really systematic way to compare methods.
- Limited to 200 first files (/1000), i.e. first 200,000 playlists (/1,000,000). Will (try to) re-run on all the playlists for final implementation. This is a random subset of the files, so like any sample/population relationship, we reasonably expect our EDA on this sample to be representative of the population.
- High dimensionality in terms of songs across playlists. E.g. a binary indicator matrix saying whether each song (row) is in each playlist(column) would have dimensions ca. 2,000,000 * 1,000,000, meaning 2,000,000,000,000 cells, which is computationally intractable. Need some way of condensing playlist-song relationships to a lower-dimensional space. One option is to run clustering on the playlists to create k clusters of playlist types/families/groups (unsupervised), and for each song, given the playlists it's part of, get the playlist cluster as a feature.
