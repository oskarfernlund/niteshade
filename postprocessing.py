# Written by: Alexandra, Jaime
# Last edited: 2022/04/08
# Description: Postprocessing pipeline. This contains analytics tools that 
# assess the damage of the attacker / impact of the defender.


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.colors as pltc

import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
from utils import save_plot
from utils import get_cmap
import random

class PostProcessor:
    def __init__(self, wrapped_models, batch_size, episode_size,
    base_model, epochs=100):
        
        self.wrapped_models = wrapped_models
        self.batch_size = batch_size
        self.episode_size = episode_size
        self.base_model = base_model
        self.epochs = epochs
        
    # =========================================================================
    #  Analytics Tools
    # =========================================================================
    
    def compute_accuracies(self, X_test, y_test):
        """
        # Returns a dictionary of lists with accuracies
        """
        accuracies = {}
        
        for model_name, list_of_models in tqdm(self.wrapped_models.items()):
            for model_specs in list_of_models:
                model = self.base_model
                model.load_state_dict(model_specs)
                _, test_accuracy = model.evaluate(X_test, y_test, self.batch_size)
                if model_name in accuracies:
                    accuracies[model_name].append(test_accuracy)
                else:
                    accuracies[model_name] = [test_accuracy]
        return accuracies

    def plot_online_learning_accuracies(self, X_test, y_test, save=True):
        """
        # Prints a plot into a console
        """
        accuracies = self.compute_accuracies(X_test, y_test)
        
        x = [e for e in range(len(accuracies['regular']))]

        fig, ax = plt.subplots(1, figsize=(15,10))
        for model_name, accuracy in accuracies.items():
            ax.plot(x, accuracy, label=model_name)
            ax.legend()

        ax.set_title('Test Accuracy Online Learing')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Accuracy')
        plt.show()
        
        if save: 
            save_plot(plt)
    
    #==================================================
    # Plot different decision boundaries relative to baseline
    #==================================================
    def extract_z(self, dataset, perplexity=40, n_iter=300):
        """"""
        """Extract embedded x,y positions for every datapoint in the dataset.
        
        Args:
            dataset {np.ndarray, torch.Tensor}: NumPy array containing data.
        
        Returns:
            z_embedded: embedded x,y positions for every datapoint in the dataset.
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
        
        tsne_results = tsne.fit_transform(dataset.reshape(dataset.shape[0], -1))

        return tsne_results

    def _state_dicts_are_equal(self, state_dict1, state_dict2):
        for ((k_1, v_1), (k_2, v_2)) in zip(state_dict1.items(), state_dict2.items()):
            if k_1 != k_2:
                return False
            # convert both to the same CUDA device
            if str(v_1.device) != "cuda:0":
                v_1 = v_1.to("cuda:0" if torch.cuda.is_available() else "cpu")
            if str(v_2.device) != "cuda:0":
                v_2 = v_2.to("cuda:0" if torch.cuda.is_available() else "cpu")

            if not torch.allclose(v_1, v_2):
                return False
        return True
    
    def _get_predictions(self, X_test, state_dicts):
        """"""
        #convert test set to torch tensor
        if type(X_test) != torch.Tensor:
            X_test = torch.tensor(X_test)

        sim_predictions = {}
        for label, state_dict in state_dicts.items():
            model = self.base_model
            model.load_state_dict(state_dict)
            if isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    predictions = model.forward(X_test.float())
                    predictions = predictions.data.max(1, keepdim=True)[1]
                    sim_predictions[label] = predictions
            else: 
                raise ValueError("""Niteshade only supports models developed 
                                    through PyTorch (i.e of type nn.Module).""")
        
        return sim_predictions

    def plot_decision_boundaries(self, X_test, y_test, num_points = 500, perplexity=50, 
                                 n_iter=2000, reg=10, kernel='poly', figsize=(20,20), 
                                 degree=3, fontsize=10, markersize=20, resolution = 0.1, 
                                 save=False): 
        """"""
        #check the type of input           
        idxs = [random.randint(0, len(X_test)-1) for _ in range(num_points)]             
        if type(X_test) == torch.Tensor:
            X_test = X_test.numpy()[idxs]
            y_test = y_test.numpy()[idxs]
        else: 
            X_test = X_test[idxs]
            y_test = y_test[idxs]

        #retrieve state dictionaries of trained models from the simulations ran
        final_state_dicts = {label:sim_models[-1] for label, sim_models in self.wrapped_models.items()}

        #get model predictions for test data
        model_predictions = self._get_predictions(X_test, final_state_dicts)

        #reduce dimensionality of model embeddings to two so they can be plotted
        z_embedding = self.extract_z(X_test, perplexity, n_iter) 

        # create a mesh to plot in
        x_min, x_max = z_embedding[:, 0].min() - 1, z_embedding[:, 0].max() + 1
        y_min, y_max = z_embedding[:, 1].min() - 1, z_embedding[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                             np.arange(y_min, y_max, resolution))

        #number labels in simulations and data
        num_labels = len(np.unique(y_test))

        cmap = get_cmap(num_labels) #colormap for points and decision boundaries
        for sim_label, predictions in model_predictions.items():
            #check if the predictions are a tensor and convert to numpy array
            if type(predictions) == torch.Tensor:
                predictions = predictions.numpy()
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()

            #plot decision boundaries for final model prediction
            clf = SVC(C=reg, kernel=kernel, degree=degree).fit(z_embedding, predictions)

            #predict on mesh and plot results in contour
            clf_preds = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = clf_preds.reshape(xx.shape)

            vmin,vmax = (fun(np.concatenate([clf_preds, y_test])) for fun in (np.min,np.max))

            fig = plt.figure(figsize=figsize)
            #plot contour of decision boundary from SVM predictions 
            plt.contourf(xx, yy, Z, levels=num_labels+1, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8)
            #scatter plot of points with their true labels
            scatter = plt.scatter(z_embedding[:, 0], z_embedding[:, 1], c=y_test, 
                                    s=markersize, cmap=cmap, vmin=vmin, 
                                    vmax=vmax, edgecolors='black')
    
            #calculate accuracy of simulation on test set
            accuracy = accuracy_score(np.array(y_test), np.array(predictions))
            #labels/title/legend
            plt.title(f"'{sim_label}' Decision Boundaries (Test Accuracy = {accuracy})", fontsize=fontsize)
            plt.ylabel("Embedded Y", fontsize=fontsize)
            plt.xlabel("Embedded X", fontsize=fontsize)
            plt.legend(*scatter.legend_elements(),loc="best", title="Classes", fontsize=fontsize)
            plt.show()

            if save: 
                save_plot(fig, plot_name=f'{sim_label}')