Install:
- Java JDK 1.8 or later
- Maven 3.1.0 or later
- Git

Build Metanome:
- checkout the repository from GitHub and follow the build instructions:
  https://github.com/HPI-Information-Systems/Metanome

Build the shared libraries:
  mvn -f utils/pom.xml clean install
  mvn -f dao/pom.xml clean install
  mvn -f HyFD/pom.xml clean install

Configure the TestRunner:
- Note:
  - The TestRunner emulates what the Metanome framework is usually doing for the algorithms so that we can run HyFD from an IDE of choice.
  - The TestRunner contains a configuration class that we can use to configure our executions.
- Open the file HyFDTestRunner/src/main/java/de/metanome/algorithms/hyfd/config/Config.ava
- Add a new Label for your dataset to the enum at the top, e.g. MY_DATASET
- Add a switch clause for your dataset in the setDataset() method with the filename, separator char, and header info according to the dataset that you want to profile, e.g.
  case MY_DATASET:
    this.inputDatasetName = "mydataset";
    this.inputFileSeparator = ',';
    this.inputFileHasHeader = false;
    break;
- Set the defaultDataset variable to the dataset you just created, e.g.
  private Dataset defaultDataset = Dataset.MY_DATASET;
- Review and change the all the other global variables that are defined in this file. You probably want to change inputFolderPath to the location of your dataset, inputFileEnding to the type of file you read, and measurementsFolderPath to the path you want to find your results in. Note that the TestRunner will probably use a different output format than the Metanome tool.
  
Run the TestRunner:
- Build the TestRunner:
  mvn -f HyFDTestRunner/pom.xml clean package
- The jar will be created in the target-folder
- Run the jar:
  java -jar HyFDTestRunner-1.2-SNAPSHOT.jar

Alternatives:
- You can load the projects into a Java IDE and start the TestRunner from the IDE.
- You can edit the Main.java and HyFDTestRunner.java files to read the dataset name from the algorithm arguments. Then, you can compile the TestRunner once and start the jar file with different parameters from a script file to process multiple tables one after another. It works, but I have not done this with this algorithm for a while now.



