## Stack Overflow Question and Answer Analysis

The script in this directory, totalStatsAnalysisPreEmbedded.py, is used to perform a few different analyses on Stack Overflow Question and Answer data using pre-computed embeddings.   

The script takes in a couple data sources that are downloadable from this project's content folder on the internet archive.  

Those files are:
- stackoverflow\_embeddings\_full\_USE.tar.gz (USE embeddings)
- stackoverflow\_questions\_with\_answers\_1000000.json (stack overflow question and answer data)

The program will, upon being run, ask for user input to specify the path to the second file above. It reads from standard input, so the input can be safely scripted by redirecting input.  

By default, the program will perform MRR, NDCG, and paired T test analyses given the pre-computed USE embedding above. The path to the data is hardcoded in the fetchEmbeddingDict() function, along with paths to other possible types of data that can be used.  

In order to add new data sources with which to perform analyses, or to adjust the paths to existing data, it is necessary to make changes to the fetchEmbeddingDict() function to include the new data. Additionally, the calls to the analysis functions in begin\_analysis() must be uncommented or in the case of new data must be added with appropriate function parameters. Comments in the code exist to help guide where the appropriate changes need to be made.
