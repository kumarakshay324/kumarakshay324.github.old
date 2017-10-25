from __future__ import division

import numpy as np 
import statistics 
import matplotlib.pyplot as plt
import csv

data_lidar_1m, data_ultrasound_1m = [], []
data_lidar_2m, data_ultrasound_2m = [], []

data_1m_fused, data_2m_fused = [], []

ultrasound_1m_bias, ultrasound_2m_bias = 0, 0
lidar_1m_bias, lidar_2m_bias = 0, 0

lidar_1m_var, lidar_2m_var, ultrasound_1m_var, ultrasound_2m_var = 0,0,0,0

def dataread():
	with open("Midterm_F17_LIDAR_Ultrasound_100ms_1m_000_soft.csv", 'r') as datafile1:
		data1 = csv.reader(datafile1)

		for p1 in data1:
			data_lidar_1m.append(int(p1[0]))
			data_ultrasound_1m.append(int(p1[1]))	

	with open("Midterm_F17_LIDAR_Ultrasound_100ms_2m_000_soft.csv", 'r') as datafile2:
		data2 = csv.reader(datafile2)

		for p2 in data2:
			data_lidar_2m.append(int(p2[0]))
			data_ultrasound_2m.append(int(p2[1]))	

def printdata():

	print (data_lidar_1m)
	print ("------------------------------------------------------------------")
	print (len(data_lidar_1m))
	print ("------------------------------------------------------------------")
	print (data_ultrasound_1m)
	print ("------------------------------------------------------------------")
	print (len(data_ultrasound_1m))
	print ("------------------------------------------------------------------")

	print (data_lidar_2m)
	print ("------------------------------------------------------------------")
	print (len(data_lidar_2m))
	print ("------------------------------------------------------------------")
	print (data_ultrasound_2m)
	print ("------------------------------------------------------------------")
	print (len(data_ultrasound_2m))

def lidar_9_bias_std_var():

	#Mean
	lidar_1m_mean = np.mean(data_lidar_1m)
	lidar_2m_mean = np.mean(data_lidar_2m)
	print ("Lidar Mean for 1m range is %f and for 2m range is %f" %(lidar_1m_mean,lidar_2m_mean))\

	#Bias 
	lidar_1m_bias = lidar_1m_mean - 100.0;
	lidar_2m_bias = lidar_2m_mean - 200.0;
	print ("Lidar Bias for 1m range is %f and for 2m range is %f" %(lidar_1m_bias,lidar_2m_bias))

	#Variance
	lidar_1m_var = np.var(data_lidar_1m);
	lidar_2m_var = np.var(data_lidar_2m);
	print ("Lidar Variance for 1m range is %f and for 2m range is %f" %(lidar_1m_var,lidar_2m_var))

	lidar_1m_std = np.std(data_lidar_1m)
	lidar_2m_std = np.std(data_lidar_2m)
	print ("Lidar Standard Deviation for 1m range is %f and for 2m range is %f\n" %(lidar_1m_std,lidar_2m_std))

	return lidar_1m_var, lidar_2m_var, lidar_1m_bias, lidar_2m_bias

def ultrasound_9_bias_std_var():

	#Mean
	ultrasound_1m_mean = np.mean(data_ultrasound_1m)
	ultrasound_2m_mean = np.mean(data_ultrasound_2m)
	print ("Ultrasound Mean for 1m range is %f and for 2m range is %f" %(ultrasound_1m_mean,ultrasound_2m_mean))

	#Bias
	ultrasound_1m_bias = ultrasound_1m_mean - 100.0;
	ultrasound_2m_bias = ultrasound_2m_mean - 200.0;
	print ("Ultrasound Bias for 1m range is %f and for 2m range is %f" %(ultrasound_1m_bias,ultrasound_2m_bias))

	#Variance
	ultrasound_1m_var = np.var(data_ultrasound_1m);
	ultrasound_2m_var = np.var(data_ultrasound_2m);
	print ("Ultrasound Variance for 1m range is %f and for 2m range is %f" %(ultrasound_1m_var,ultrasound_2m_var))

	#Standard Deviation
	ultrasound_1m_std = np.std(data_ultrasound_1m)
	ultrasound_2m_std = np.std(data_ultrasound_2m)
	print ("Ultrasound Standard Deviation for 1m range is %f and for 2m range is %f \n" %(ultrasound_1m_std,ultrasound_2m_std))

	return ultrasound_1m_var, ultrasound_2m_var, ultrasound_1m_bias, ultrasound_2m_bias
	
def sensor_fusion_9_1m_2m(lidar_1m_var, lidar_2m_var, lidar_1m_bias, lidar_2m_bias, ultrasound_1m_var, ultrasound_2m_var, ultrasound_1m_bias, ultrasound_2m_bias):

	#Classical Central Limit Theorem
	var_combined_1m = (lidar_1m_var * ultrasound_1m_var)/(lidar_1m_var + ultrasound_1m_var)
	var_combined_2m = (lidar_2m_var * ultrasound_2m_var)/(lidar_2m_var + ultrasound_2m_var)

	# Remember to subtract any bias from each sensor - This HINT IS TO BE USED - DISCOVER ABOUT THIS

	for d1 in range(len(data_lidar_1m)):


		d1_lidar_bias =  lidar_1m_bias
		d1_ultrasound_bias = ultrasound_1m_bias
	
				
		temp1 = var_combined_1m*(((data_lidar_1m[d1-1]- d1_lidar_bias) /lidar_1m_var) + ((data_ultrasound_1m[d1-1] - d1_ultrasound_bias)/ultrasound_1m_var))
		data_1m_fused.append(temp1)

	for d2 in range(len(data_lidar_2m)):


		d2_lidar_bias = lidar_2m_bias
		d2_ultrasound_bias = ultrasound_2m_bias
		
		temp2 = var_combined_2m*((data_lidar_2m[d2-1]-d2_lidar_bias) /lidar_2m_var + (data_ultrasound_2m[d2-1]-d2_ultrasound_bias)/ultrasound_2m_var)
		data_2m_fused.append(temp2)

	optimal_estimate_1m = np.mean(data_1m_fused)
	optimal_estimate_2m = np.mean(data_2m_fused)

	std_1m = np.std(data_1m_fused)
	std_2m = np.std(data_2m_fused)

	print ("Optimal Estimate and Standard Deviation for the 1m range combined data from the lidar and ultrasound sensor is %f and %f" %(optimal_estimate_1m, std_1m))
	print ("Optimal Estimate and Standard Deviation for the 2m range combined data from the lidar and ultrasound sensor is %f and %f" %(optimal_estimate_2m, std_2m))

	print ("Weights for the lidar and ultrasonic sensors for 1m respectively", var_combined_1m/lidar_1m_var, var_combined_1m/ultrasound_1m_var)
	print ("Weights for the lidar and ultrasonic sensors for 2m respectively", var_combined_2m/lidar_2m_var, var_combined_2m/ultrasound_2m_var)


def plotdata_subplots():

	f, xaxis = plt.subplots(4, sharex=True)

	xaxis[0].plot(data_lidar_1m, 'ro')
	xaxis[0].set_title(' Sensor Data for Lidar and Ultrasonic sensors mounted at 1m and 2m distance from a SOFT surface wall\n\n Lidar Data mounted at 1m' )
	xaxis[1].plot(data_ultrasound_1m, 'g*')
	xaxis[1].set_title('Ultrasound Data mounted at 1m' )
	xaxis[2].plot(data_lidar_2m, 'bo')
	xaxis[2].set_title('Lidar Data mounted at 2m' )
	xaxis[3].plot(data_ultrasound_2m, 'r*')
	xaxis[3].set_title('Ultrasound Data mounted at 2m' )
	xaxis[3].set_xlabel('Time steps count(every 100 ms)\n Y axis shows the distance from wall, measured in centimeters')
	plt.show()


print ("Soft Wall Interactions \n \n")
dataread()
# printdata()
lidar_1m_var, lidar_2m_var, lidar_1m_bias, lidar_2m_bias = lidar_9_bias_std_var()
ultrasound_1m_var, ultrasound_2m_var, ultrasound_1m_bias, ultrasound_2m_bias = ultrasound_9_bias_std_var()
sensor_fusion_9_1m_2m(lidar_1m_var, lidar_2m_var, lidar_1m_bias, lidar_2m_bias, ultrasound_1m_var, ultrasound_2m_var, ultrasound_1m_bias, ultrasound_2m_bias)
plotdata_subplots()
