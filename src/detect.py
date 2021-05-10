import itertools
import random

import numpy as np
import ruptures as rpt
import tqdm

from scipy.optimize import minimize
from ruptures.utils import sanity_check



class Pelt_lambda(rpt.detection.pelt.Pelt):


    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a Pelt instance.
        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        self.d=None
        self.trained=False
        self.features=None
        self.lambd=None
        super().__init__(model=model, custom_cost=custom_cost,min_size= min_size,jump= jump,params= params)


    def fit(self, signal) -> "Pelt":
        """Set params.
        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).
        Returns:
            self
        """
        # update params
        self.cost.fit(signal)
        if signal.ndim == 1:
            (n_samples,) = signal.shape
        else:
            n_samples,d = signal.shape
        self.n_samples = n_samples
        self.d=d
        self.features=self.calcul_features(signal)
        return self

    def calcul_features(self,signal):
        bruit=np.log(np.var(signal))
        longueur=np.log(np.log(len(signal)))
        return [bruit,longueur]

    def exp_pen(self,penalisation,features,d):
        return np.exp(np.sum(np.array(features)*np.array(penalisation)))

    def pen_BIC(self,penalisation,features,d):
        return 2*np.exp(np.sum(features))*(d+1)

    def fonction_cible_un(self,pen,features,signal,bkps,cout,exp_pen,d):

        penexp=exp_pen(pen,features,d)

        algo = rpt.Pelt(custom_cost=cout, min_size=3, jump=1).fit(signal)
        my_bkps = algo.predict(pen=penexp)
        #fig, ax_array = rpt.display(signal,bkps, my_bkps)

        calcul = cout.sum_of_costs(bkps)-cout.sum_of_costs(my_bkps)+penexp*(len(bkps)-len(my_bkps))
        global bkps_pelt
        bkps_pelt=my_bkps
        #print(calcul)
        return calcul

    def fonction_cible(self,pen,signaux,bkps_list,cout,pen_methode):
        cible=0
        global jac
        jac=0
        for i,j in zip(signaux,bkps_list):
            cout.fit(signal=i)
            features=self.calcul_features(i)
            d=i[0].shape[0]
            cible=cible+self.fonction_cible_un(pen,features,i,j,cout,pen_methode,d)
            jac=jac+self.jacob_un(pen,features,i,j,cout,pen_methode,d)

        jac=jac/len(signaux)

        #print("#######################################")
        return cible/len(signaux)

    def jacob_un(self,pen,features,signal,bkps,cout,exp_pen,d):
        #print("Jacob")
        penexp=exp_pen(pen,features,d)
        #algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(signal)
        #my_bkps = algo.predict(pen=penexp)
        my_bkps=bkps_pelt
        #print(my_bkps)
        calcul=np.array(features)*penexp*(len(bkps)-len(my_bkps))
        #print(calcul)
        return calcul

    def jacob(self,pen,signal,bkps,cout,pen_methode):
        return jac


    def calcul_penality(self,signaux,bkps):
        j=np.random.randint(1,len(signaux))
        init=minimize(fun=self.fonction_cible,args=([signaux[j]],[bkps[j]],self.cost,self.exp_pen),x0=[0,0],method="CG",jac=self.jacob,tol=1e-10)
        solution=minimize(fun=self.fonction_cible,args=(signaux,bkps,self.cost,self.exp_pen),x0=init.x,method="CG",jac=self.jacob,tol=1e-1)
        self.trained=True
        self.lambd=solution.x


    def predict(self,pena="exp_pen"):
        """Return the optimal breakpoints.
        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.pelt.Pelt.fit].
        Args:
            pen (float): penalty value (>0)
        Raises:
            BadSegmentationParameters: in case of impossible segmentation
                configuration
        Returns:
            list: sorted list of breakpoints
        """
        """
        if(self.trained):
            pen= self.calcul_penality()
        else:
            pen=pena
        """


        pen=eval("self."+pena)(self.lambd,self.calcul_features(self.cost.signal),self.d)
        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=1,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, pen=None):
        """Fit to the signal and return the optimal breakpoints.
        Helper method to call fit and predict once
        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)
        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)
