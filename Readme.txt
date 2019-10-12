CS 7641 Assignment 2: Randomized Optimization


The code and the dataset I used for this analysis can be download from:
https://github.com/HascoetGwenole/CS7641-Assignment2

This github repository contains the jupyter notebook that I used to run the toys optimization problems, and to train
a neural network. 
The neural Network Training was made on the Phishing websites dataset that I used for the first assignment. 

To be run, Assignment2.ipynb can be opened with Jupyter Notebook, and then each cell
can be run individualy. Enventually rename the variable "file" (in the third cell) so that it points toward the file 
'dataset.csv' that you can find in this repository. Note that this code was written using the mlrose library:
https://github.com/gkhayes/mlrose corrected by hiive: git://github.com/hiive/mlrose
I advise to run the cells in order to avoid any problem of undefined variables.



Also, I had a problem while running the GA algorithm for the neural network training, so I used the GA algorithm
proposed in the ABAGAIL library to train my neural network. This library is aviable at (https://github.com/pushkar/ABAGAIL)
The class PhisingWebsitesGA.java I wrote was adapted from the example AbaloneTest.java that I presented in the abagail 
library. To run this class, one can follow the following steps:

- Java Development Kit: https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html
- install the ABAGAIL library from https://github.com/pushkar/ABAGAIL
- install Apache Ant from  https://ant.apache.org/bindownload.cgi (Installation Instructions: http://ant.apache.org/manual/index.html)
- Add Java and Ant to your windows environment and path variables. More instructions are aviable at https://www.mkyong.com/ant/how-to-install-apache-ant-on-windows/


Once this is done, one can run PhisingWebsitesGA.java by using an IDE (I used the Eclipse IDE: https://www.eclipse.org/ide/):

-First, we must import the ABAGAIL library:

    Open Eclipse, select File > New > Project
    Select "Java Project from Existing Ant Build File"
    Show the ABAGAIL library build and write a project name

-Then move the class PhisingWebsitesGA.java in src/opt/test
-Finaly, just run the class PhisingWebsitesGA.java (eventually modify the variable "filename" if it points toward your location of the dataset 
"Preprocessed_dataset_abagail.csv")