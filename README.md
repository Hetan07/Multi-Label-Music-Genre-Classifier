# Multi Label Music Genre Classifier

An extension to my project of: [Single Label Music Genre Classifier](https://github.com/Hetan07/Single-Label-Music-Classifier)

I was primarily interested on how prediction (or classification) 
would work in case if a sample can be classified into multiple classes at the same time
which is what most of the time is the case with music

Majority of the work done went into actually making the proper dataset to work and train upon.
What I wanted was a dataset similar to the one I worked upon during single label classification.
GTZAN has no dataset for mulit-label purposes, and so I had to make one

A dataset, [MuMU](https://www.upf.edu/web/mtg/mumu) has the mulit-label tags but works on different components or features.
So I decided to create a GTZAN-like dataset but with multi-labels

---

### Dataset Creation

The work done for creating the dataset were:

- Downloading the appropriate songs taken randomly from the MuMu dataset in sampled manner from ~80 genres (tags)
- Data Cleaning which included to clean and replace the download songs as many of them were things such as album intros, interludes or skits
- There were also issues where the song required was not available on any platform and so had to appropriately replaced for another proper track or I had to manually search and download
- Each file had to properly checked to prevent any distortion or disturbances
- Applying feature extraction on each downloaded song using the *librosa* library
- Reducing the labels from ~80 to around ~15

There was also an issue: MuMu dataset has no Classical Genre and thus it had to be added manually.

In the end I decided to have feature extraction work on 3 second samples and thus have around ~24000 samples.

I have linked the actual dataset created from all the steps if anyone wishes to work upon it

---

For this task I decided to primarily work with neural networks and experimented with various architecture

The models I trained are:

- ANN
- ANN with Batch Normalization
- CNN
- CRNN


The genres classifed are the following:

- Metal
- Jazz
- Blues
- R&B
- Classical
- Reggae
- Rap & Hip-Hop
- Punk
- Rock 
- Country
- Bebop
- Pop
- Soul
- Dance & Electronic
- Folk

I have also deployed this on Hugging Face using streamlit. If one wishes, he can test and play around with different music tracks.

---

All of this took me around 3-4 days but in retrospect, I realize that some parts have been slightly rushed. 
An in-depth analysis of data is further required along with more data. This task as a lot of potential for beginners and experts alike
