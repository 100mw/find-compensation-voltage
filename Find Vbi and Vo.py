# python3

"""
Tues 25 Oct 2022

@author: cbogh
@author: 100mW
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import special, interpolate
from scipy.optimize import curve_fit, fmin, fsolve
import pandas as pd
from pathlib import Path 							#for Mac/Windows compatibility
from matplotlib.widgets import Slider, Button 		#for sliders, buttons

# Type path to folder here (Mac/Windows/Unix compatible):
files = Path(
	"/Users/arangolab/Library/Mobile Documents/com~apple~CloudDocs/iCloud Data/Masha Helen/Single layer JV data/20221027-1-4d-light-3_10:31:22.txt"
)

# New folder for results
folderpath = files.parent / (files.stem + ' results')	#path of new folder
folderpath.mkdir(parents=True,exist_ok=True)			#make new folder

# Read file data into DataFrame and sort
jvcurve = pd.read_csv(files,sep='\t')						#tab delimited
jvcurve['delta'] = jvcurve[jvcurve.columns[0]].diff().shift(-1) 	#new column of deltas between voltages
jvcurve_a = jvcurve.loc[jvcurve['delta'] > 0]				#select ascending sweep
jvcurve_a.sort_values(by=[jvcurve_a.columns[0]], ascending=True, inplace=True)	#sort
v_a = jvcurve_a[jvcurve_a.columns[0]].to_numpy()		#turn DataFrame first column into voltage array
l_a = jvcurve_a[jvcurve_a.columns[1]].to_numpy()		#turn DataFrame second column into light current array
d_a = jvcurve_a[jvcurve_a.columns[2]].to_numpy()		#turn DataFrame third column into dark current array
p_a = l_a-d_a											#calculate photocurrent
jvcurve_d = jvcurve.loc[jvcurve['delta'] < 0]			#select decending sweep
jvcurve_d.sort_values(by=[jvcurve_d.columns[0]], ascending=False, inplace=True)	#sort
v_d = jvcurve_d[jvcurve_d.columns[0]].to_numpy()		#turn DataFrame first column into voltage array
l_d = jvcurve_d[jvcurve_d.columns[1]].to_numpy()		#turn DataFrame second column into light current array
d_d = jvcurve_d[jvcurve_d.columns[2]].to_numpy()		#turn DataFrame third column into dark current array
p_d = l_d-d_d											#calculate photocurrent

# Create figure
plt.style.use('dark_background')								#black background
fig, axs = plt.subplots(2,1,sharex=True,figsize=(5,7.5))		#create figure

# Create scatter plots
axs[1].set_xlabel("Voltage [V]")
axs[0].set_ylabel("Current density [mA/cm$^2$]")
axs[1].set_ylabel("Current density [mA/cm$^2$]")
axs[0].set_yscale("log")
axs[1].set_ylim(			#set axis limits
	1.2*l_a.min(),			#grab min from current data and adjust spacing
	-1.2*l_a.min()			#grab -min from current data and adjust spacing
)
fig.subplots_adjust(top=0.98,right=0.97,bottom=0.06,left=0.15,hspace=0) 		#adjust plot

# Interpolate light current and photocurrent
fint_l_a = interpolate.interp1d(v_a, l_a, kind='cubic')
Voc_a = fsolve(fint_l_a,[1])
fint_p_a = interpolate.interp1d(v_a, p_a, kind='cubic')
Vo_a = fsolve(fint_p_a,[1])
fint_l_d = interpolate.interp1d(v_d, l_d, kind='cubic')
Voc_d = fsolve(fint_l_d,[1])
fint_p_d = interpolate.interp1d(v_d, p_d, kind='cubic')
Vo_d = fsolve(fint_p_d,[1])

# Create pandas of Voc and Vo values
results = pd.DataFrame(index=['Ascending','Descending'])
results['Voc'] = [np.round(Voc_a, 2),np.round(Voc_d, 2)]
results['Vo'] = [np.round(Vo_a, 2),np.round(Vo_d, 2)]

# Add data to plots
axs[0].scatter(v_d,np.absolute(d_d),s=1.3,color='#4d96f9', label = files.name)			#plot abs value of dark current density
axs[0].scatter(v_d,np.absolute(l_d),s=1.3,color='orange')			#plot abs value of light current density
axs[0].scatter(v_d,np.absolute(p_d),s=1.3,color='lightgreen')		#plot abs value of photocurrent current density
axs[1].scatter(v_d,d_d,s=1.3,color='#4d96f9')
axs[1].scatter(v_d,l_d,s=1.3,color='orange')
axs[1].plot(np.linspace(-v_d.min(),v_d.max(),num=500),fint_l_d(np.linspace(-v_d.min(),v_d.max(),num=500)),linewidth=0.5,color='orange')
axs[1].scatter(v_d,p_d,s=1.3,color='lightgreen')
axs[1].plot(np.linspace(-v_d.min(),v_d.max(),num=500),fint_p_d(np.linspace(-v_d.min(),v_d.max(),num=500)),linewidth=0.5,color='lightgreen')
axs[0].axvline(Voc_d, linewidth=0.5, color='orange', linestyle='dashed')
axs[0].axvline(Vo_d, linewidth=0.5, color='lightgreen', linestyle='dashed')
axs[1].axvline(Voc_d, linewidth=0.5, color='orange', linestyle='dashed')
axs[1].axvline(Vo_d, linewidth=0.5, color='lightgreen', linestyle='dashed')
axs[0].scatter(v_a,np.absolute(d_a),s=1.3,color='#4d96f9')			#plot abs value of dark current density
axs[0].scatter(v_a,np.absolute(l_a),s=1.3,color='orange')			#plot abs value of light current density
axs[0].scatter(v_a,np.absolute(p_a),s=1.3,color='lightgreen')		#plot abs value of photocurrent current density
axs[1].scatter(v_a,d_a,s=1.3,color='#4d96f9')
axs[1].scatter(v_a,l_a,s=1.3,color='orange')
axs[1].plot(np.linspace(-v_a.min(),v_a.max(),num=500),fint_l_a(np.linspace(-v_a.min(),v_a.max(),num=500)),linewidth=0.5,color='orange')
axs[1].scatter(v_a,p_a,s=1.3,color='lightgreen')
axs[1].plot(np.linspace(-v_a.min(),v_a.max(),num=500),fint_p_a(np.linspace(-v_a.min(),v_a.max(),num=500)),linewidth=0.5,color='lightgreen')
axs[0].axvline(Voc_a, linewidth=0.5, color='orange', linestyle='dashed')
axs[0].axvline(Vo_a, linewidth=0.5, color='lightgreen', linestyle='dashed')
axs[1].axvline(Voc_a, linewidth=0.5, color='orange', linestyle='dashed')
axs[1].axvline(Vo_a, linewidth=0.5, color='lightgreen', linestyle='dashed')
axs[1].axhline(0, linewidth=0.5)

# Add legend and text to plot
axs[0].legend(loc='upper left',frameon=False)
axs[0].text(0.5,0.05,results.to_string(),transform=axs[0].transAxes)			#text box for model version

# Create a `matplotlib.widgets.Button` to save
saveax = fig.add_axes([0.8, 0.1, 0.1, 0.04])
button = Button(saveax, 'Save', color='orange')

# Function that activates when Save button is pressed
def save(event):
	folderpath = files.parent / (files.stem + ' results')			#path of new folder for results
	folderpath.mkdir(parents=True,exist_ok=True)					#make new folder
	plt.savefig(folderpath/'Plot.pdf')								#save plot
	print('Results Saved')
	
button.on_clicked(save)

print(results)
plt.show()
