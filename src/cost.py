import itertools
import random

import numpy as np
import ruptures as rpt
import tqdm


from scipy.optimize import minimize
from ruptures.utils import sanity_check

class KernelWithPartialAnnotationCost2(rpt.base.BaseCost):
    """"""

    model = "Learned Kernel Partial Annotation"
    min_size = 3

    def pre_fit(
        self,
        signals,
        labels,
        initial_kernel_fct,
        upper_bound_similarity,
        lower_bound_dissimilarity,
        gamma,
    ):
        """computes the parameters (G_hat, G, training_samples) of the learned metrics

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must be integers (>=0). The non-labelled
                samples can be anything below 0.
            upper_bound_similarity: [Bregman param] upper bound for the similarity constrains
            lower_bound_dissimilarity: [Bregman param] lower bound for the dissimilarity constrains
            gamma: [Bregman param] tradeoff between satisfying the constraints and minimizing DKL(G_hat,G)
        """
        self.initial_kernel_fct = initial_kernel_fct
        self.u = upper_bound_similarity
        self.l = lower_bound_dissimilarity
        self.gamma = gamma

        self.training_samples, self.constrains = self.get_training_samples_and_constains(
            signals, labels
        )

        self.G = initial_kernel_fct(self.training_samples, self.training_samples)

        self.G_inv = np.linalg.pinv(self.G)

        self.G_hat = self.compute_bregman()

        self.G_core = self.G_inv @ (self.G_hat - self.G) @ self.G_inv

    def fit(self, signal):
        """Compute params to segment signal.
        Args:
            signal (array): signal to segment.
        """
        self.signal = signal

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost
        """
        # compute equation 8.7) and then 8.8) in Charles Truong. Détection de ruptures multiples –
        # application aux signaux physiologiques.

        subsignal = self.signal[start:end]

        self.inner_product = self.initial_kernel_fct(subsignal, subsignal)

        self.inner_product_with_training_samples = self.initial_kernel_fct(
            subsignal, self.training_samples
        )

        # TODO: optimisation replace np.diag(self.initial_kernel_fct(subsignal, subsignal)) by
        # self.initial_kernel_fct.diag(subsignal)
        inner_product_sum = np.sum(np.diag(self.inner_product)) - 1.0 / (end - start) * np.sum(
            self.inner_product
        )
        second_term = (
            self.inner_product_with_training_samples
            @ self.G_core
            @ self.inner_product_with_training_samples.T
        )
        new_kernel_product = np.sum(np.diag(second_term)) - 1.0 / (end - start) * np.sum(
            second_term
        )
        cost_bis = inner_product_sum + new_kernel_product

        return cost_bis

    def _phi_m_hat_phi(self, i, j):
        ki = self.inner_product_with_training_samples[i, :][np.newaxis, :]
        kj = self.inner_product_with_training_samples[j, :][np.newaxis, :]

        return self.inner_product[i, j] + (ki @ self.G_core @ kj.T)[0][0]

    @staticmethod
    def get_training_samples_and_constains(signals, labels):
        """Derives training samples and constrains dictionnary from labels list.

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must be integers (>=0).The non-labelled
                samples can be anything below 0.

        Returns:
            (list, dict): list containing the training samples and a dictionnary whose keys
                corresponds to indexes in the list and the value to the label 1 (similar) or -1
                (dissimilar)
        """
        training_samples = []
        constrains = {}
        last_idx = -1
        for signal, label in zip(signals, labels):
            begin_signal = True
            idx_max = np.max(label)
            for idx in range(idx_max + 1):

                sub_signal = signal[(label == idx).squeeze()]
                new_idx_iterator = range(last_idx + 1, sub_signal.shape[0] + last_idx + 1)

                for key in itertools.combinations(new_idx_iterator, r=2):
                    constrains[key] = 1  # similar

                if begin_signal:
                    begin_signal = False
                else:
                    for key in itertools.product(last_idx_iterator, new_idx_iterator):
                        constrains[key] = -1  # disimilar

                training_samples.append(sub_signal)
                last_idx_iterator = new_idx_iterator
                *_, last_idx = last_idx_iterator

        training_samples = np.concatenate(training_samples)
        print(f"training_samples.shape: {training_samples.shape}")

        return training_samples, constrains

    def compute_bregman(self):
        # Algorithm 1 in P. Jain, B. Kulis, J. V. Davis, and I. S. Dhillon, “Metric and kernel
        # learning using a linear transformation,” Journal of Machine Learning Research (JMLR), vol.
        #  13, pp.519–547, 2012.

        eps = np.finfo(float).eps

        K = self.G.copy()
        lambdas = dict.fromkeys(self.constrains.keys(), np.array(0))
        xi = dict(
            [
                (key, self.u) if value == 1 else (key, self.l)
                for key, value in self.constrains.items()
            ]
        )

        n = K.shape[0]
        convergence = True

        min_ite = int(len(self.constrains)/16)
        num_ite = 0

        while convergence:
            num_ite += 1
            print(f"Iteration numéro°: {num_ite}")
            convergence = False
            upd=[]
            for _ in range(min_ite):

                (i, j), delta = random.choice(list(self.constrains.items()))

                if i != j:

                    ei = np.zeros((n, 1))
                    ei[i] = 1
                    ej = np.zeros((n, 1))
                    ej[j] = 1

                    p = ((ei - ej).T @ K @ (ei - ej))[0][0]

                    alpha = np.minimum(
                        lambdas[(i, j)],
                        delta
                        * self.gamma
                        * (1.0 / (p + eps) - 1.0 / (xi[(i, j)] + eps))
                        / (self.gamma + 1.0),
                    )
                    beta = delta * alpha / (1 - delta * alpha * p)
                    xi[(i, j)] = self.gamma * xi[(i, j)] / (self.gamma + delta * alpha * xi[(i, j)])
                    lambdas[(i, j)] = lambdas[(i, j)] - alpha

                    update = beta * (K @ (ei - ej) @ (ei - ej).T @ K)

                    K = K + update
                    #print(np.linalg.norm(update))
                    if np.linalg.norm(update) > 1e-3:


                        #print(np.linalg.norm(update))
                        upd.append(np.linalg.norm(update,ord=1))
                        convergence = True
            moyenne=np.mean(upd)
            print("MOYENNE UPD")
            print(moyenne)
            if moyenne<1e-2:
                convergence = False

        return K


class MetricWithPartialAnnotationCost(rpt.base.BaseCost):
    """"""

    model = "Learned Kernel Partial Annotation"
    min_size = 3

    def pre_fit(
        self,
        signals,
        labels,
        upper_bound_similarity,
        lower_bound_dissimilarity,
        gamma,
        init=None,
    ):
        """computes the parameters (G_hat, G, training_samples) of the learned metrics

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must be integers (>=0). The non-labelled
                samples can be anything below 0.
            upper_bound_similarity: [Bregman param] upper bound for the similarity constrains
            lower_bound_dissimilarity: [Bregman param] lower bound for the dissimilarity constrains
            gamma: [Bregman param] tradeoff between satisfying the constraints and minimizing DKL(G_hat,G)
        """
        #self.initial_kernel_fct = initial_kernel_fct
        self.u = upper_bound_similarity
        self.l = lower_bound_dissimilarity
        self.gamma = gamma

        self.training_samples, self.constrains = self.get_training_samples_and_constains(
            signals, labels
        )

        #self.G = initial_kernel_fct(self.training_samples, self.training_samples)

        #self.G_hat = self.compute_bregman()

        #self.G_inv = np.linalg.inv(self.G)

        #self.G_core = self.G_inv @ (self.G_hat - self.G) @ self.G_inv
        self.M = self.compute_bregman(init)

    def fit(self, signal):
        """Compute params to segment signal.
        Args:
            signal (array): signal to segment.
        """
        self.signal = signal

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            segment cost
        """
        # compute equation 8.7) and then 8.8) in Charles Truong. Détection de ruptures multiples –
        # application aux signaux physiologiques.

        subsignal = self.signal[start:end]
        matrice=(subsignal@self.M).dot(subsignal.T)
        error=np.trace(matrice)-(np.sum(matrice))/(end-start)
        """
        cost_bis=0
        cost_bis_2=0
        for i in range(subsignal.shape[0]):
            y=subsignal[i]
            cost_bis = cost_bis+ y@self.M@y
        for i in range(subsignal.shape[0]):
            y=subsignal[i]
            for j in range(subsignal.shape[0]):
                yd=subsignal[j]
                cost_bis_2 = cost_bis_2+ y@self.M@yd

        cost_bis=cost_bis - cost_bis_2/(end-start)"""
        return error

    def _phi_m_hat_phi(self, i, j):
        ki = self.inner_product_with_training_samples[i, :][np.newaxis, :]
        kj = self.inner_product_with_training_samples[j, :][np.newaxis, :]

        return self.inner_product[i, j] + (ki @ self.G_core @ kj.T)[0][0]

    @staticmethod
    def get_training_samples_and_constains(signals, labels):
        """Derives training samples and constrains dictionnary from labels list.

        Args:
            signals (List[array]): signals on which the metric is learned. List of len n_signals of
                array of shape(n_samples, n_features)
            labels (List[array]): corresponding labels of the signals. List of len n_signals of
                array of shape (n_samples, 1). The labels must be integers (>=0).The non-labelled
                samples can be anything below 0.

        Returns:
            (list, dict): list containing the training samples and a dictionnary whose keys
                corresponds to indexes in the list and the value to the label 1 (similar) or -1
                (dissimilar)
        """
        training_samples = []
        constrains = {}
        last_idx = -1
        for signal, label in zip(signals, labels):
            begin_signal = True
            idx_max = np.max(label)
            for idx in range(idx_max + 1):

                sub_signal = signal[(label == idx).squeeze()]
                new_idx_iterator = range(last_idx + 1, sub_signal.shape[0] + last_idx + 1)

                for key in itertools.combinations(new_idx_iterator, r=2):
                    constrains[key] = 1  # similar

                if begin_signal:
                    begin_signal = False
                else:
                    for key in itertools.product(last_idx_iterator, new_idx_iterator):
                        constrains[key] = -1  # disimilar

                training_samples.append(sub_signal)
                last_idx_iterator = new_idx_iterator
                *_, last_idx = last_idx_iterator
        #print(training_samples)
        #training_samples = np.hstack(training_samples)
        tr_samples=[]
        for i in training_samples:
            for j in i:
                tr_samples.append(j)
        #print("NOUVEAU")
        #print(tr_samples)
        tr_samples=np.array(tr_samples)
        #print(f"training_samples.shape: {tr_samples.shape}")

        return tr_samples, constrains

    def compute_bregman(self,init):
        # Algorithm 1 in P. Jain, B. Kulis, J. V. Davis, and I. S. Dhillon, “Metric and kernel
        # learning using a linear transformation,” Journal of Machine Learning Research (JMLR), vol.
        #  13, pp.519–547, 2012.

        eps = np.finfo(float).eps
        taille=self.training_samples[0].shape[0]
        print("taille",taille)
        if init.any()==None:
            K = np.eye(taille)
        else:
            K=init
        lambdas = dict.fromkeys(self.constrains.keys(), np.array(0))
        xi = dict(
            [
                (key, self.u) if value == 1 else (key, self.l)
                for key, value in self.constrains.items()
            ]
        )

        n = self.training_samples.shape[0]
        print("n egale =",n)
        convergence = True
        #print(self.constrains)
        min_ite = int(len(self.constrains)/32)
        num_ite = 0
        #print(list(self.constrains.items()))
        while convergence:
            upd=[]
            num_ite += 1
            print(f"Iteration numéro°: {num_ite}")
            convergence = False
            for _ in range(min_ite):
                (i, j), delta = random.choice(list(self.constrains.items()))

                if i != j:
                    ei=np.array(self.training_samples[i])
                    ej=np.array(self.training_samples[j])

                    #print("ei",ei)
                    #print("ej",ej)
                    #ei = np.zeros((n, 1))
                    #ei[i] = 1
                    #ej = np.zeros((n, 1))
                    #ej[j] = 1
                    #print("p", ((ei - ej).T @ K @ (ei - ej)))
                    p = ((ei - ej).T @ K @ (ei - ej))

                    alpha = np.minimum(
                        lambdas[(i, j)],
                        delta
                        * (1.0 / (p + eps) - self.gamma / (xi[(i, j)] + eps))
                        / (2),
                    )
                    beta = delta * alpha / (1 - delta * alpha * p)
                    xi[(i, j)] = self.gamma * xi[(i, j)] / (self.gamma + delta * alpha * xi[(i, j)])
                    lambdas[(i, j)] = lambdas[(i, j)] - alpha

                    #print("K", K.shape)
                    #print("ei-ej",(ei-ej).shape)
                    #print("trans",((ei-ej).T).shape)
                    temp=np.expand_dims(ei-ej, axis=1)
                    update = beta * (K @ temp @ temp.T @ K)

                    K = K + update
                    #print(np.linalg.norm(update))
                    if np.linalg.norm(update) > 1e-3:
                    # 1e-2:

                        upd.append(np.linalg.norm(update,ord=1))

                        convergence = True
            moyenne =np.linalg.norm(upd)
            print("Moyenne : ", moyenne)
            if moyenne <1e-2:
                convergence= False

        return K
