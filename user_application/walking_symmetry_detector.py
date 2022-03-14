# Walking movement analysis 
# Input: the collected walking movement data set
# Output: the frequency for Butterworth filter and the p-value test of two variance set
#         more explanation please see project report
#
#

import sys
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, dense_rank
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from scipy import stats
import numpy as np


spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

pages_schema = types.StructType([
    types.StructField('time', types.StringType()),
    types.StructField('X_axis', types.FloatType()),
    types.StructField('Y_axis', types.FloatType()),
    types.StructField('Z_axis', types.FloatType())])


#this function apply the filter on the dataframe for one axis to get filtered data
#input: dataframe for one axis, frequency use for Butterworth filter
#output: filtered dataframe
def getFilterAxis(df, freq):
    df2 = df.copy(deep=True)#use copy of df so the original data will not change
    df2 = df2.drop(["group_index"], axis=1)
    
    for col in df2:
        column = df2[col].dropna()#since the data may have different size, the tail of df may contains NaN
        column = column.values
        column = column[:len(column)-80]
        b, a = signal.butter(3, freq, btype='lowpass', analog=False)
        templist = signal.filtfilt(b, a, column)
        df2[col] = pd.Series(templist)
    df2 = df2.dropna(how='all')
    return df2


#this function is for spliting a input dataframe into axis
#input: pandas dataframe contains 3 axis
#output: 3 pandas dataframe for each axis
def splitAxis(dataframe): 
    #for x axis
    X_df = dataframe.pivot(index = "group_index",columns = "filename", values = "X_axis")
    X_df = X_df.reset_index()

    #for y axis
    Y_df = dataframe.pivot(index = "group_index",columns = "filename", values = "Y_axis")
    Y_df = Y_df.reset_index()

    #for z axis
    Z_df = dataframe.pivot(index = "group_index",columns = "filename", values = "Z_axis")
    Z_df = Z_df.reset_index()
        
    return X_df, Y_df, Z_df

#the function is for add a group number that divide the whole walking data into slice to calculate variance for each 20 data
#input: walking dataframe
#output: add a column of [0,0,0,0...0,1,1,1...] with each number repeat 20 times 
def getSliceGroups(df):
    arr = np.arange(0,(len(df)//100)+1,1)
    rep = np.repeat(arr,20)
    df["slices"] = pd.Series(rep)
    return df


#this function is for ploting the original and filtered data to help us identify the frequency we used in filter 
#input: pandas dataframes for left foot in 3 axis, pandas dataframes for right foot in 3 axis
#output: 6 plots for filtered data on two feet in each axis
def plotFilterData(L_X_df,L_Y_df,L_Z_df,R_X_df,R_Y_df,R_Z_df):
    fig, axes = plt.subplots(nrows=3, ncols=2)

    #plot left foot data
    L_X_df.plot(ax=axes[0,0],figsize=(50,10))
    axes[0,0].set_title('filtered left foot acceleration in X_axis')
    L_Y_df.plot(ax=axes[1,0],figsize=(50,10))
    axes[1,0].set_title('filtered left foot acceleration in Y_axis')
    L_Z_df.plot(ax=axes[2,0],figsize=(50,10))
    axes[2,0].set_title('filtered left foot acceleration in Z_axis')

    #plot right foot data
    R_X_df.plot(ax=axes[0,1],figsize=(50,10))
    axes[0,1].set_title('filtered right foot acceleration in X_axis')
    R_Y_df.plot(ax=axes[1,1],figsize=(50,10))
    axes[1,1].set_title('filtered right foot acceleration in Y_axis')
    R_Z_df.plot(ax=axes[2,1],figsize=(50,10))
    axes[2,1].set_title('filtered right foot acceleration in Z_axis')

    #save the plot
    plt.savefig('filtered_walking_data.png')
    

#this function is for calculating t-test p value for left and right dataset
#input: left and right dataset for one axis
#output: the t-test p value for the axis
def getPvalueArr(L_df,R_df):
    var_L = L_df.groupby("slices").var()
    var_R = R_df.groupby("slices").var()
    arr = np.array([])
    for i in range(0,len(var_L.columns)):
        L_col = var_L.iloc[:,i].dropna()
        R_col = var_R.iloc[:,i].dropna()
        
        #because the last variance always have data less than 15 second, so if it is variance for very short data, it may not be accurate to present the real variance, so exclude the last vanriance when calculating p-values
        L_col = L_col.drop(index=len(L_col)-1)
        R_col = R_col.drop(index=len(R_col)-1)

        stat,p=stats.ttest_ind(L_col,R_col)
        arr= np.insert(arr,len(arr),p)
    return arr

#this function compare the total variance of two data set
#input: left df, right df
#output: 0 if total variance of left df > right df, 1 otherwise
def compareMean(L, R):
    var_L = L.groupby("slices").var()
    var_R = R.groupby("slices").var()
    if (var_L.sum().sum()>var_R.sum().sum()):
        return 0
    else:
        return 1
    

#this function takes 3 p values from each axis and produce a report based on the p values
#input: 3 p values for each axis
#output: a text file with report
def writeReport(X_p, Y_p, Z_p, L_X, L_Y, L_Z, R_X, R_Y, R_Z):
    
    file = open("report.txt", "w") 
    file.write("Analysis report for the walking movement\n")

    #as the asymmetry data should always have p-value much smaller than alpha, we assume that if the has at least one p-value greater than alpha, we consider it as symmetric
    if (np.max(X_p)>0.05 and np.max(Y_p)>0.01 and np.max(Z_p)>0.0001):
        file.write("Result: Symmetry\n")
        file.write("Congratulations! Your walking movement is relatively symmetry, your body balance looks great.\n")
        file.write("Thank you for using walking symmetry detector.")
        file.close()
    else:
        file.write("Result: Asymmetry\n")
        file.write("We are sorry to tell you that your walking movement may be asymmetry\n")
        if (np.max(X_p)>0.05):
            file.write("forward and backward movement: symmetry\n")
        else:
            if(compareMean(L_X, R_X)==0):
                file.write("your left foot may always move forward or backward faster than your right foot\n")
            if(compareMean(L_X, R_X)==1):
                file.write("your right foot may always move forward or backward faster than your left foot\n")
        if (np.max(Y_p)>0.05):
            file.write("lift up and down movement: symmetry\n")
        else:
            if(compareMean(L_Y, R_Y)==0):
                file.write("your left foot may always lift up and down faster than your right foot\n")
            if(compareMean(L_Y, R_Y)==1):
                file.write("your right foot may always lift up and down faster than your left foot\n")
        if (np.max(Z_p)>0.0001):
            file.write("away and close to body movement: symmetry\n")
        else:
            if(compareMean(L_Z, R_Z)==0):
                file.write("your left foot may always move away and close to body faster than your right foot\n")
            if(compareMean(L_Z, R_Z)==1):
                file.write("your right foot may always move away and close to body faster than your left foot\n")
        file.write("The asymmetry walking movement may cause by the injury, muscle tightness, muscle imbalance, skeletal imbalance or bad walking habit\n")
        file.write("if you feel pain when you walk normally, you would better to find a doctor now to handle your injury\n")
        file.write("Thank you for using walking symmetry detector.")
        file.close()





    
def main(in_directory): 
    
    #read data and add the file name to the dataframe
    Total_walk_data = spark.read.csv(in_directory, schema=pages_schema,sep=",").withColumn('filename', functions.input_file_name().substr(-15,11))
    
    #since the first 5 seconds and last 5 seconds do not contains movement, so subset the data from the 8th second to 8 second before stop
    Total_walk_data = Total_walk_data.withColumn("index",functions.monotonically_increasing_id())
    window = Window.partitionBy(Total_walk_data['filename']).orderBy(Total_walk_data['index'])
    Total_walk_data=Total_walk_data.select('*', rank().over(window).alias('group_index'))
    Total_walk_data = Total_walk_data.filter(Total_walk_data.group_index>80)

    #in order to better use matplotlib and filter, switch the data from spark dataframe to pandas dataframe 
    walk_data_pd = Total_walk_data.toPandas()
 
    #split data into x, y, z axis
    X_data, Y_data, Z_data = splitAxis(walk_data_pd)

    #apply Butterworth filter to each of the df to get filtered data
    X_df_filtered = getFilterAxis(X_data, 0.35)
    Y_df_filtered = getFilterAxis(Y_data, 0.7)
    Z_df_filtered = getFilterAxis(Z_data, 0.9)
    
    #now separate the filtered acceleration for left foot and right foot
    L_X_df_filtered = X_df_filtered.loc[:,X_df_filtered.columns.str.startswith("L")]
    L_Y_df_filtered = Y_df_filtered.loc[:,Y_df_filtered.columns.str.startswith("L")]
    L_Z_df_filtered = Z_df_filtered.loc[:,Z_df_filtered.columns.str.startswith("L")]
    R_X_df_filtered = X_df_filtered.loc[:,X_df_filtered.columns.str.startswith("R")]
    R_Y_df_filtered = Y_df_filtered.loc[:,Y_df_filtered.columns.str.startswith("R")]
    R_Z_df_filtered = Z_df_filtered.loc[:,Z_df_filtered.columns.str.startswith("R")]

    
    #save a plot of users walking data
    plotFilterData(L_X_df_filtered,L_Y_df_filtered,L_Z_df_filtered,R_X_df_filtered,R_Y_df_filtered,R_Z_df_filtered)

    #get sliced groups of the data for each 2 seconds
    L_X_df_filtered = getSliceGroups(L_X_df_filtered)
    L_Y_df_filtered = getSliceGroups(L_Y_df_filtered)
    L_Z_df_filtered = getSliceGroups(L_Z_df_filtered)
    R_X_df_filtered = getSliceGroups(R_X_df_filtered)
    R_Y_df_filtered = getSliceGroups(R_Y_df_filtered)
    R_Z_df_filtered = getSliceGroups(R_Z_df_filtered)
    
    #get p_value for each axis to see if the normal data is different from injury data
    p_X_arr = getPvalueArr(L_X_df_filtered, R_X_df_filtered)
    p_Y_arr = getPvalueArr(L_Y_df_filtered, R_Y_df_filtered)
    p_Z_arr = getPvalueArr(L_Z_df_filtered, R_Z_df_filtered)
    
    #produce a report base on the p values
    writeReport(p_X_arr , p_Y_arr, p_Z_arr, L_X_df_filtered,L_Y_df_filtered, L_Z_df_filtered, R_X_df_filtered, R_Y_df_filtered, R_Z_df_filtered)
    
    
    
    
    
                                              
    #male_walk_data.show()
    
if __name__=='__main__':
    in_directory = sys.argv[1]   
    main(in_directory)
