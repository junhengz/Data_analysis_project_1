# Walking movement analysis 
# Input: the collected walking movement data set
# Output: two filtered graph by Butterworth filter and the p-value of two variance set
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
import numpy as np
from scipy import stats

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
        column = column[:len(column)-80]#drop the data from last 8 seconds
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

#the function is for add a group number that divide the whole walking data into slice to calculate variance for each 20 (2 seconds) data
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
def plotFilterData(L_X_df,L_Y_df,L_Z_df,R_X_df,R_Y_df,R_Z_df,type1):
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
    if (type1==0):
        plt.savefig('filtered_normal_data.png')
    if (type1==1):
        plt.savefig('filtered_injury_data.png')

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
        
        #print(stats.normaltest(L_col)[1])
        #print(stats.normaltest(R_col)[1])
        stat,p=stats.ttest_ind(L_col,R_col)
        arr= np.insert(arr,len(arr),p)
    return arr

def main(in_directory):
    #read data and add the file name to the dataframe
    Total_walk_data = spark.read.csv(in_directory, schema=pages_schema,sep=",").withColumn('filename', functions.input_file_name().substr(-23,19))#if the data is more than 999 sets, make sure to change the substr length to load data
    
    #since the first 8 seconds and last 8 seconds always contain unstable data, so filter out the data in 8 second, and last 8 second will be filter out in getFilterAxis function 
    Total_walk_data = Total_walk_data.withColumn("index",functions.monotonically_increasing_id())
    window = Window.partitionBy(Total_walk_data['filename']).orderBy(Total_walk_data['index'])
    Total_walk_data=Total_walk_data.select('*', rank().over(window).alias('group_index'))
    Total_walk_data = Total_walk_data.filter(Total_walk_data.group_index>80)

    #in order to better use matplotlib and filter, switch the data from spark dataframe to pandas dataframe
    Total_walk_data = Total_walk_data.toPandas()

    #split data into x, y, z axis
    X_data, Y_data, Z_data = splitAxis(Total_walk_data)

    #apply Butterworth filter to each of the df to get filtered data
    X_df_filtered = getFilterAxis(X_data, 0.35)
    Y_df_filtered = getFilterAxis(Y_data, 0.7)
    Z_df_filtered = getFilterAxis(Z_data, 0.9)

    #separate normal and injury data
    normal_X_df = X_df_filtered.loc[:,X_df_filtered.columns.str.contains("normal")]
    normal_Y_df = Y_df_filtered.loc[:,Y_df_filtered.columns.str.contains("normal")]
    normal_Z_df = Z_df_filtered.loc[:,Z_df_filtered.columns.str.contains("normal")]

    injury_X_df = X_df_filtered.loc[:,X_df_filtered.columns.str.contains("injury")]
    injury_Y_df = Y_df_filtered.loc[:,Y_df_filtered.columns.str.contains("injury")]
    injury_Z_df = Z_df_filtered.loc[:,Z_df_filtered.columns.str.contains("injury")]

    #now separate the filtered acceleration for left foot and right foot
    normal_L_X_df_filtered = normal_X_df.loc[:,normal_X_df.columns.str.startswith("L")]
    normal_L_Y_df_filtered = normal_Y_df.loc[:,normal_Y_df.columns.str.startswith("L")]
    normal_L_Z_df_filtered = normal_Z_df.loc[:,normal_Z_df.columns.str.startswith("L")]
    normal_R_X_df_filtered = normal_X_df.loc[:,normal_X_df.columns.str.startswith("R")]
    normal_R_Y_df_filtered = normal_Y_df.loc[:,normal_Y_df.columns.str.startswith("R")]
    normal_R_Z_df_filtered = normal_Z_df.loc[:,normal_Z_df.columns.str.startswith("R")]

    injury_L_X_df_filtered = injury_X_df.loc[:,injury_X_df.columns.str.startswith("L")]
    injury_L_Y_df_filtered = injury_Y_df.loc[:,injury_Y_df.columns.str.startswith("L")]
    injury_L_Z_df_filtered = injury_Z_df.loc[:,injury_Z_df.columns.str.startswith("L")]
    injury_R_X_df_filtered = injury_X_df.loc[:,injury_X_df.columns.str.startswith("R")]
    injury_R_Y_df_filtered = injury_Y_df.loc[:,injury_Y_df.columns.str.startswith("R")]
    injury_R_Z_df_filtered = injury_Z_df.loc[:,injury_Z_df.columns.str.startswith("R")]


    #use plot to adjust the freq used in filter
    plotFilterData(normal_L_X_df_filtered,normal_L_Y_df_filtered,normal_L_Z_df_filtered,normal_R_X_df_filtered,normal_R_Y_df_filtered,normal_R_Z_df_filtered,0)
    plotFilterData(injury_L_X_df_filtered,injury_L_Y_df_filtered,injury_L_Z_df_filtered,injury_R_X_df_filtered,injury_R_Y_df_filtered,injury_R_Z_df_filtered,1)

    
    #get sliced groups of the data for each 2 seconds
    normal_L_X_df_filtered = getSliceGroups(normal_L_X_df_filtered)
    normal_L_Y_df_filtered = getSliceGroups(normal_L_Y_df_filtered)
    normal_L_Z_df_filtered = getSliceGroups(normal_L_Z_df_filtered)
    normal_R_X_df_filtered = getSliceGroups(normal_R_X_df_filtered)
    normal_R_Y_df_filtered = getSliceGroups(normal_R_Y_df_filtered)
    normal_R_Z_df_filtered = getSliceGroups(normal_R_Z_df_filtered)

    injury_L_X_df_filtered = getSliceGroups(injury_L_X_df_filtered)
    injury_L_Y_df_filtered = getSliceGroups(injury_L_Y_df_filtered)
    injury_L_Z_df_filtered = getSliceGroups(injury_L_Z_df_filtered)
    injury_R_X_df_filtered = getSliceGroups(injury_R_X_df_filtered)
    injury_R_Y_df_filtered = getSliceGroups(injury_R_Y_df_filtered)
    injury_R_Z_df_filtered = getSliceGroups(injury_R_Z_df_filtered)

    #get p_value for each axis to see if the normal data is different from injury data
    normal_p_X_arr = getPvalueArr(normal_L_X_df_filtered, normal_R_X_df_filtered)
    normal_p_Y_arr = getPvalueArr(normal_L_Y_df_filtered, normal_R_Y_df_filtered)
    normal_p_Z_arr = getPvalueArr(normal_L_Z_df_filtered, normal_R_Z_df_filtered)
    
    injury_p_X_arr = getPvalueArr(injury_L_X_df_filtered, injury_R_X_df_filtered)
    injury_p_Y_arr = getPvalueArr(injury_L_Y_df_filtered, injury_R_Y_df_filtered)
    injury_p_Z_arr = getPvalueArr(injury_L_Z_df_filtered, injury_R_Z_df_filtered)
    
    #output the value of analyzing the different of normal data is different from injury data
    output = pd.DataFrame(
        columns=['normal_X', 'normal_Y','normal_Z','injury_X', 'injury_Y','injury_Z']
    )
    output["normal_X"] = pd.Series(normal_p_X_arr)
    output["normal_Y"] = pd.Series(normal_p_Y_arr)
    output["normal_Z"] = pd.Series(normal_p_Z_arr)
    output["injury_X"] = pd.Series(injury_p_X_arr)
    output["injury_Y"] = pd.Series(injury_p_Y_arr)
    output["injury_Z"] = pd.Series(injury_p_Z_arr)

    output.to_csv("p-values.csv",index=False)

    
    
    
    
    
                                         
    
if __name__=='__main__':
    in_directory = sys.argv[1]   
    main(in_directory)
