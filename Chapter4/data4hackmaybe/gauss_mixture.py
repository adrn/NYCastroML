import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from astroML.plotting import setup_text_plots

#######################################################
######Fit Gaussian mixture using EM algorithm##########
#######################################################
def compute_weights(x,alpha,mu,sigma):
	Nij = alpha*stats.norm.pdf(x,loc=mu,scale=sigma)
	Di = Nij.sum(axis=1)[:,np.newaxis]
	return Nij/Di


def fit_gaussian_mixture(x,Ngauss,Niterations=10):

	#Convert x to column vector
	x = x[:,np.newaxis]
	
	#Guess for the mixture parameters
	if(Ngauss==2):
		mu_guess = np.array([x.min(),x.max()])
		print mu_guess
	elif(Ngauss==1):
		mu_guess = np.array([x.mean()])
	else:
		mu_guess = x[np.random.randint(0,len(x),Ngauss),0]
	
	sigma2_guess = np.ones(Ngauss)*x.var()
	alpha_guess = np.ones(Ngauss)*(1.0/Ngauss)

	print "Guess:"
	print "mean=", mu_guess
	print "sigma2= ", sigma2_guess
	print "prob=", alpha_guess
	print ""

	#Weight tensor
	w_guess = compute_weights(x,alpha_guess,mu_guess,np.sqrt(sigma2_guess))

	#Iterate the computation
	for i in range(Niterations):
		#Iterative step
		mu = (w_guess*x).sum(axis=0)/(w_guess.sum(axis=0))
		sigma2 = (w_guess*(x-mu_guess)**2).sum(axis=0)/(w_guess.sum(axis=0))
		alpha = (1.0/len(x))*w_guess.sum(axis=0)

		print "Iteration %d"%(i+1)
		print "mean=", mu
		print "sigma2= ", sigma2
		print "prob=", alpha
		print ""

		#Update guess values
		mu_guess = mu.copy()
		sigma2_guess = sigma2.copy()
		alpha_guess = alpha.copy()
		w_guess = compute_weights(x,alpha_guess,mu_guess,np.sqrt(sigma2_guess))

	#Return the fitted parameters
	return alpha,mu,sigma2

def mixture_pdf(x,alpha,mu,sigma):
	#Plot the gaussian mixture
	x = x[:,np.newaxis]
	Nij = alpha*stats.norm.pdf(x,loc=mu,scale=sigma)
	return Nij.sum(axis=1)

#############################################################
#############################################################
#############################################################

if(len(sys.argv)<3):
	print "Usage python %s <spectral_line_file> <num_iterations>"%sys.argv[0]
	exit(1)

setup_text_plots(fontsize=12,usetex=True)

#Load spectral line data from sys.argv[1]
sp=np.loadtxt(sys.argv[1], unpack=True)

####################################################################
#Select region of interest (borrowed from emspectrapicks.ipynb)#####
####################################################################

win1=np.where(((sp[0]>5050) & (sp[0]<5100)) | ((sp[0]>5500) & (sp[0]<5600)))
winall1=np.where(((sp[0]>5050) & (sp[0]<5600)))
slope1, intercept1, r_value, p_value, std_err = stats.linregress(sp[0][win1[0]],sp[1][win1[0]])

win2=np.where(((sp[0]>5700) & (sp[0]<5750)) | ((sp[0]>6100) & (sp[0]<6150)))
winall2=np.where(((sp[0]>5700) & (sp[0]<6150)))
slope2, intercept2, r_value, p_value, std_err = stats.linregress(sp[0][win2[0]],sp[1][win2[0]])

feat1=(sp[1][winall1]-(sp[0][winall1]*slope1+intercept1))
tmp= (abs(feat1[1]-feat1[:-1]))
norm1=min(tmp[np.where(tmp > 0)[0]])
feat1=feat1/norm1
feat1=feat1-min(feat1)

feat2=(sp[1][winall2]-(sp[0][winall2]*slope2+intercept2))
tmp= (abs(feat2[1]-feat2[:-1]))
norm2=min(tmp[np.where(tmp > 0)[0]])
feat2=feat2/norm2
feat2=feat2-min(feat2)

pop1=[winall1[0][0]]*feat1[0]
pop2=[winall2[0][0]]*feat2[0]

for i,x in enumerate(winall1[0][1:]):
    pop1 = pop1+[x]*feat1[i]

for i,x in enumerate(winall2[0][1:]):
    pop2 = pop2+[x]*feat2[i]

plt.hist(pop2, bins=winall2[0],weights=np.zeros_like(pop2)+1. / len(pop2), alpha=0.1)

p = np.arange(np.array(pop2).min(),np.array(pop2).max(),1)

a,m,s = fit_gaussian_mixture(np.array(pop2)*1.0,2,Niterations=int(sys.argv[2]))
pdf = mixture_pdf(p,a,m,np.sqrt(s))
plt.plot(p,pdf,label="Two gaussians")

a,m,s = fit_gaussian_mixture(np.array(pop2)*1.0,1,Niterations=int(sys.argv[2]))
pdf = mixture_pdf(p,a,m,np.sqrt(s))
plt.plot(p,pdf,label="One gaussian")

plt.xlabel("$\lambda$")
plt.ylabel("$I_\lambda/I_{tot}$")
plt.legend(loc="upper left")

plt.savefig("spectral2.png")