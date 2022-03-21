from ML_NEB.scripts.bayesian_sampling import BayesianSampler
from ML_NEB.scripts.gaussian_process import GPR
import numpy as np
from ML_NEB.scripts import plotting_ternary_functions as pt
from ML_NEB.scripts import Quinary_activity_functions as af
import matplotlib.pyplot as plt
from celluloid import Camera

n_elems=3

# Set random seed for reproducibility
np.random.seed(42)

# Get point repulsion sampler
sampler = BayesianSampler(n_elems)


def initialize_NEB(r_initial,r_final,n_images):
    connecting_vector = (r_final-r_initial)/n_images
    connecting_images = [list(r_initial + connecting_vector*(i+1)) for i in range(n_images)]

    r_images=np.array([list(r_initial)]+connecting_images)
    return r_images

def gradient(xy,gpr,h=0.01):
    xy=xy.reshape(1,-1)
    gradient=np.array([*(gpr.predict(xy+np.array([h,0]))-gpr.predict(xy-np.array([h,0])))/(2*h),*(gpr.predict(xy+np.array([0,h]))- gpr.predict(xy-np.array([0,h])))/(2*h)])
    return gradient

def Forces(R,n,gpr,E_ref=0.05,k_max=20,Dk=19.9):
    F_tot = []
    G_list=np.array([])

    U_list= [gpr.predict(r.reshape(1,-1)) for r in R]
    #U_sort = np.sort(U_list)[::-1]
    U_max=np.max(U_list)
    for i in range(n_images-1):
        i+=1
        
        U_ip1 = U_list[i+1]
        U_i = U_list[i]
        U_im1 = U_list[i-1]
        
        E_i = np.max([U_im1,U_i])
        E_max=U_max
        if E_i >E_ref:
            k=k_max-Dk*((E_max-E_i)/(E_max-E_ref))
        else:
            k=k_max-Dk
        

        
        if U_ip1 < U_i < U_im1:
            t_i=(R[i+1]-R[i])
        elif U_ip1 > U_i > U_im1:
            t_i=(R[i]-R[i-1])
        else:
            t_i=(R[i+1]-R[i-1])
        

        t_i= t_i/(np.linalg.norm(t_i))
        

        
        Fs_i = k*(R[i+1]-R[i])-k*(R[i]-R[i-1])
        Fs_i = np.dot(Fs_i,t_i) * t_i
        G_i = gradient(R[i],gpr)
        G_ort= G_i - np.dot(G_i,t_i) * t_i
        G_list=np.append(G_list,np.linalg.norm(G_ort))
        
        if n>0 and U_i == U_max:
            F_i = (G_i + 2*np.dot(G_i,t_i)*t_i)*2
            #print(i)
        else:
            F_i=Fs_i + G_ort

        F_tot.append(list(F_i))

    if np.all(G_list<0.1):
        break_neb=True
    else:
        break_neb=False
    return np.array(F_tot)#, break_neb

def NEB(r_images,gpr,max_iterations=200,learn_rate=0.01,force_converge=0.05):
    F_converge=False
    for n in range(max_iterations):
        F=Forces(r_images,n,gpr)
        r_images[1:-1] = r_images[1:-1] + learn_rate*F

        for i in range(len(r_images)):
            mf=pt.cartesians_to_molar_fractions(np.array([r_images[i]]))
            if np.any(mf<0):
                r_images[i]=af.R_correction(r_images[i])
                

        F_norm=np.array([np.linalg.norm(f) for f in F])
        if np.all(F_norm<=force_converge):

            F_converge=True
            
            break

    return r_images, F_converge


def next_sample(fs_train,r_images,gpr):
    predicted_activities,std = gpr.predict(r_images,return_std=True)
    max_uncertainty_image = r_images[np.argmax(std)]
    fs_next=pt.cartesians_to_molar_fractions(max_uncertainty_image.reshape(1,-1))
    fs_train = np.concatenate([fs_train,fs_next])
    return fs_train, np.max(std)
    



gpr=GPR()
fs_train=[]
activities=[]
fs_train=sampler.get_molar_fraction_samples(fs_train, activities, gpr)
rs_train=pt.molar_fractions_to_cartesians(fs_train).T

n_images = 20
fs_initial = (1,0,0)
fs_final = (0,0,1)
r_initial = pt.molar_fractions_to_cartesians(fs_initial).T[0]
r_final = pt.molar_fractions_to_cartesians(fs_final).T[0]
fs_train=np.concatenate([fs_train,np.array([fs_initial,fs_final])])
rs_train=np.concatenate([rs_train,np.array([r_initial,r_final])])

activities=[float(af.activity(r)) for r in rs_train]

r_images= initialize_NEB(r_initial,r_final,n_images)
gpr.fit(rs_train,activities)

max_std=5
cmap = pt.truncate_colormap(plt.get_cmap('viridis'), minval=0.2, maxval=1.0, n=100)
plot_kwargs = dict(cmap=cmap, levels=15, zorder=0)

fig, ax = plt.subplots(figsize=(4,4),dpi=400)
pt.prepare_triangle_plot(ax, ["Ag","Ir","Pd"])
camera=Camera(fig)
grid_activities_list = np.empty((0,len(af.grid[0])), int)
for step in range(1,47):
    r_images, F_converge = NEB(r_images,gpr)
    
    grid_activities = gpr.predict(af.grid_points)
    grid_activities_list = np.append(grid_activities_list,grid_activities.reshape(1,-1),axis=0)
    #fig,ax = pt.make_plot(af.grid,grid_activities , ["Ag","Ir","Pd"])
    pt.prepare_triangle_plot(ax, ["Ag","Ir","Pd"])
    ax.tricontourf(*af.grid, grid_activities, **plot_kwargs)
    ax.scatter(*r_images.T,marker=".", color="red")
    ax.scatter(*rs_train.T,marker="x",color="black")
    #plt.show()
    camera.snap()
    if F_converge and max_std<0.001:
        print("converge at step: ",step)
        break
    else:
        fs_train,max_std=next_sample(fs_train, r_images, gpr)
        rs_train=pt.molar_fractions_to_cartesians(fs_train).T
        activities.append(float(af.activity(rs_train[-1])))
        gpr.fit(rs_train,activities)

print("Number of samples: ",len(fs_train))
print(max_std)


animation = camera.animate(interval=1000)
animation.save("test_AgIrPd_50samples.gif",writer="Pillow")
