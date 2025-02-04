### LSTM-Research-on-electricity-consumption-data
This is a project that uses electricity consumption datasets to study LSTM!  
Happy Chinese New Year!:)  
We all know that LSTM is very effective in processing temporal logic data, which is why it is widely used in fields such as speech recognition and language translation. However, how to efficiently use this algorithm is a headache for many beginners in the AI field, including myself. Therefore, in this project, we will conduct research on temporal logic algorithms from shallow to deep, including naive model algorithms (simple model algorithms), CNN, CNN-LSTM (which I am currently studying), attention mechanisms, etc. The data used comes from  
https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption  
The data contains 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 (47 months).   
More about the data  
1.(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3.  
2.The dataset contains some missing values in the measurements (nearly 1,25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. 

In this project, I analysis electricity consumption data using seven models.The RMSE of each models are over here.
