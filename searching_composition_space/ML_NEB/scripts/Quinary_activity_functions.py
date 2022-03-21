import numpy as np
import pandas as pd
import plotting_ternary_functions as pt
from scipy import interpolate as ip


df=pd.read_csv("grid_activities.csv",sep=",",skiprows=1,header=None,names=["Ag","Ir","Pd","Pt","Ru","Activity"])

matrix=df.to_numpy()


#stop
mask1= df["Pt"].to_numpy()==0
mask2=df["Ru"].to_numpy()==0
mask=mask1*mask2
AgIrPd_comp=matrix[:,:3][mask]
act=matrix[:,-1][mask]
grid=pt.molar_fractions_to_cartesians(AgIrPd_comp)
grid_points = np.array([list(grid[:,i]) for i in range(len(grid[0]))])





def interpolation(xy):
    dist_list = [np.linalg.norm(grid_points[i]-xy) for i in range(len(grid_points))]
 
    nearest_index = [x for _,x in sorted(zip(dist_list,np.arange(0,len(grid_points))))]
    nearest_points = grid_points[nearest_index[:3]]
    nearest_mf = pt.cartesians_to_molar_fractions(nearest_points)
    nearest_act=act[nearest_index[:3]]
    nearest_dist=np.array(dist_list)[nearest_index[:3]]
    interp=ip.LinearNDInterpolator(list(zip(nearest_points[:,0],nearest_points[:,1])),nearest_act)
    z=interp(*xy)
    return z

def R_correction(xy):
    molar_f = np.around(pt.cartesians_to_molar_fractions(np.array([xy]))[0],decimals=6)
    while True:
        if molar_f[0]<0 and molar_f[1]<0:
            xy=pt.molar_fractions_to_cartesians(np.array([0,0,1])).T[0]
        elif molar_f[0]<0 and molar_f[2]<0:
            xy=np.array([1,0])
        elif molar_f[1]<0 and molar_f[2]<0:
            xy=np.array([0,0])
    
        
        elif molar_f[0]<0:
            P1=np.array([1,0])
            P2 = pt.molar_fractions_to_cartesians(np.array([0,0,1])).T[0]
            n_vec=pt.molar_fractions_to_cartesians(np.array([2,-1,-1])).T[0]
            n_vec=n_vec/np.linalg.norm(n_vec)
            d=abs((P2[0]-P1[0])*(P1[1]-xy[1]) - (P1[0]-xy[0])*(P2[1]-P1[1]))/np.sqrt((P2[0]-P1[0])**2 + (P2[1]-P1[1])**2)
            xy = xy + n_vec*d
    
        elif molar_f[1]<0:
            P1=np.array([0,0])
            P2 = pt.molar_fractions_to_cartesians(np.array([0,0,1])).T[0]
            n_vec=pt.molar_fractions_to_cartesians(np.array([-1,2,-1])).T[0]
            n_vec=n_vec/np.linalg.norm(n_vec)
            d=abs((P2[0]-P1[0])*(P1[1]-xy[1]) - (P1[0]-xy[0])*(P2[1]-P1[1]))/np.sqrt((P2[0]-P1[0])**2 + (P2[1]-P1[1])**2)
            xy = xy + n_vec*d 
        elif molar_f[2]<0:
            n_vec=np.array([0,1])
            d=abs(xy[1])
            xy = xy + n_vec*d         
        molar_f = np.around(pt.cartesians_to_molar_fractions(np.array([xy]))[0],decimals=6)
        if np.all(molar_f>=0):
            break
    return xy

def activity(xy):
    mf_in_grid=False
    molar_f = np.around(pt.cartesians_to_molar_fractions(np.array([xy]))[0],decimals=6)

    if np.any(molar_f<0):
        xy=R_correction(xy)
        molar_f = np.around(pt.cartesians_to_molar_fractions(np.array([xy]))[0],decimals=6)

    
    for i in range(len(AgIrPd_comp)):
        if np.all(AgIrPd_comp[i]==molar_f):
            act_i = act[i]
            mf_in_grid = True
            #new_row = np.array([np.append(molar_f,act_i)])
            #calc_act=np.concatenate((calc_act,new_row))
            break
        
    if mf_in_grid==False:
        act_i=interpolation(xy)
    return act_i