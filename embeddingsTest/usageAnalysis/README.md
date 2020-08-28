## Usage Analysis

The script in this directory, usageAnalysis.py, is used to run the usage analysis task.  

The script takes in a variety of data sources that are downloadable from this project's content folder.  

Those files are:
- classes.map (old to new class conversion)
- merge-15-22.2.format.json (docstrings dataset)
- usage.txt (usage data)
- classes2superclass.out (old to new class conversion)

The program will, upon being run, ask for user input to specify the paths to the above files. It reads from standard input, so the input of the same paths each time can be safely scripted by redirecting input.  

By default, the program also reports overlap analysis between the class hierarchy and the usage data. The types of analyses it performs can be adjusted via commenting out function calls in the script's main function.  

Additionally, most of the analysis functions contain extra information that can be printed alongside program execution, this too can be adjusted simply by commenting out the relevant print statements.
