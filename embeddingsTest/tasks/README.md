Running the code with Docker:

1) Choose a place to hold the data (~20GB), called DATA here.  From `tasks` run `bash download.sh DATA`.  Warning this step takes a lot of data.  If you want to try a small sample instead, use test-data for the next step instead which has miniature datasets just to ensure the code and environment is set up properly.  None of the results are what is reported in the paper though.  For that one does need to get data by running download.sh.

2) Build the docker container;  in this directory, run the following:

       docker build -f docker/blanca.docker -t blanca . 

3) Run and mount DATA:

       docker run -it -v DATA:/data blanca:latest /bin/bash

4) Inside the container, run tests

 - Usage Analysis:

     ```
    cd usage_analysis
    bash run.sh /data
    ```
    
 - Linked Stackoverflow Posts:
        
        ```
        cd stackLinkedPosts
        bash run.sh /data
         ```

 - Answer Prediction:
 
         ```
        cd stack_answer_pred
        bash run.sh /data
         ```
 - Hierarchy Analysis:

     ```
    cd hierarchyAnalysis
    bash run_all.sh 
    ```

 - Search:
 
    ```
    cd searchForums
    bash run_all.sh 
    ```
    
 - Forum posts to docstrings:
     ```
    cd ForumToClassDocumentation
     bash run_all.sh /data
    ```
 
