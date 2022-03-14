# cmpt-353-final-project

There are two part of code in this project:

1. For analyzing data (in the data_analysis folder): movement_analysis_spark.py (application analyze the walking data sets consist of symmetry and asymmetry gait data and give the result of two-sample t-test for the variance sets)

    Required libraries: sys, pyspark.sql, matplotlib, scipy, pandas, numpy
    
    input: folder that contains symmetry data as "R_sensor_normal_001" like pattern, and asymmetry data as "R_sensor_injury_001" like pattern (please follow the instructions in "user_application/user guide.pdf" to collect 
            the data)
            
    sample input: Default_data
    
    commands: spark-submit movement_analysis_spark.py Default_data 
    
        or: Python3 movement_analysis_spark.py Default_data 

    
    output: 2 graphs (filtered data for normal and injury) and 1 csv file (contains p-values)

2. For user (in the user_application folder): walking_symmetry_detector (application analyze a user walking data and generate a report of symmetry of user's gait)

    Required libraries: sys, pyspark.sql, matplotlib, scipy, pandas, numpy
    
    input: user's data as "R_sensor_01" and "L_sensor_01" like pattern (please follow the instructions in "user guide.pdf" to collect the data)
    
    sample input: user_data-1, user_data-2
    
    commands: spark-submit movement_analysis_spark.py user_data-1
    
        or:  Python3 movement_analysis_spark.py user_data-1
    
    output: 1 graph for filtered data and 1 txt file for report (asymmetry content)
 
    commands: spark-submit movement_analysis_spark.py user_data-2
    
        or: Python3 movement_analysis_spark.py user_data-2
        
    output: 1 graph for filtered data and 1 txt file for report (symmetry content)
    
    
If you have any problem with running the code, please contact me as soon as possible 