Running the code with Docker:

1) Choose a place to hold the data (~20GB), called DATA here.  From `tasks` run `bash download.sh DATA`.

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
 
