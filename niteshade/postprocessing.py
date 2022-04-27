#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analytics tools for assessing damage inflicted by attacks and damage prevented 
by defences.
"""

# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn") 

import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from tqdm import tqdm
from fpdf import FPDF

from niteshade.simulation import wrap_results
from niteshade.utils import save_plot, get_cmap, get_time_stamp_as_string


# =============================================================================
#  CLASSES
# =============================================================================

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', '', 15)
        w = self.get_string_width(self.title) + 6
        self.set_x((self.w - w) / 2)
        self.set_draw_color(255, 255, 255)
        self.set_fill_color(255, 255, 255)
        self.set_text_color(128)
        self.cell(w, 9, self.title, 1, 1, 'C', 1)
        self.set_line_width(0.25)
        self.set_draw_color(128)
        #self.line(10, 22, self.w-10, 22)
        self.line(10, 23, self.w-10, 23)
        self.ln(20)


    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')


    def add_table(self, df, table_title=None, new_page=True):
        """Add a table to the pdf.
        Args:
            df (pd.core.frame.DataFrame) : pandas data frame.
            table_title (str) : title as necessary.
            new_page (bool) : print table on a new page.
        """
        assert type(df) == pd.core.frame.DataFrame
        
        # Prepare the df to be parsed row by row
        df.reset_index(inplace=True)
        data = [list(df.columns)]
        data.extend(df.values.tolist())
        
        # Add to PDF
        if new_page: self.add_page()
        
        effective_page_width = self.w - 2*self.l_margin
        col_width = effective_page_width/len(df.columns)
        cell_thickness = self.font_size
        
        # Title
        self.set_font('Arial','B', 10) 
        if table_title: self.cell(effective_page_width, 0, 
                                  table_title, align='C')
        
        # Table 
        self.set_font('Arial','',8) 
        self.ln(6)

        for row in data:
            for datum in row:
                self.cell(col_width, 1.5*cell_thickness, str(datum), border=1)
            self.ln(1.5*cell_thickness) 

        self.ln()


    def add_chart(self, file_path, chart_title=None, new_page=True):
        """Add a jpg / png / jpeg to a pdf.
        Args:
            file_path (str) : file path. 
            chart_title (str) : title as necessary.
            new_page (bool) : print table on a new page.
        """
        if new_page: self.add_page()
        self.set_font('Arial','B', 10)
        effective_page_width = self.w - 2*self.l_margin
        if chart_title: 
            self.cell(effective_page_width, 0, chart_title, align='C')
            self.ln(2)
        self.image(file_path, x = 0, y = None, w = 200, h = 0, 
                   type = '', link = '')
        self.ln()


    def add_all_charts_from_directory(self, dir_path, new_page=True):
        """Add all jpg / png / jpeg to a pdf from a directory.
        Args:
            dir_path (str) : directory path.
            new_page (bool) : print table on a new page.
        """
        contents = os.listdir(dir_path)
        for chart in contents: 
            if chart.split('.')[-1] in ['jpeg', 'jpg', 'png']:
                self.add_chart(f'{dir_path}/{chart}', 
                               chart_title=chart.split('.')[0], new_page=True)


class PostProcessor():
    """Class used for a variety of post-processing functionalities common
    in assessing and visualising the effectiveness of different attack
    and defense strategies in simulating data poisoning attacks during 
    online learning. 
    Args: 
        simulators (dict) : Dictionary containing the simulator objects 
                            (presumably after making use of their .run()
                            method) as values and descriptive labels for 
                            each Simulator as keys.
    """
    def __init__(self, simulators: dict) -> None:
        
        self.simulators = simulators
        self.wrapped_data, self.wrapped_models = wrap_results(simulators)
        self.batch_sizes = {label:sim.batch_size for label,sim in simulators.items()}
        self.episode_nums = {label:sim.num_episodes for label,sim in simulators.items()}
        self.final_models = {label:sim.model for label,sim in simulators.items()}

    def get_data_modifications(self):
        """Retrieves for each simulation the following: a) number of poisoned points,
        b) number of points that were unaugmented by the attacker, c) number of points 
        correctly rejected by the defender, d) number of points that were incorrectly
        rejected by the defender, e) total number of points that were originally available
        for training, f) number of points that the model was actually trained on.

        Returns:
            results (pd.core.frame.DataFrame): Dictionary with keys 
                                               corresponding to simulator names, 
                                               values corresponding to dicts 
                                               with keys a, b, c, d per above.
        """
        results_df = {}
        for label, simulator in self.simulators.items():
            simulator_results = {
                'poisoned': simulator.poisoned,
                'not_poisoned': simulator.not_poisoned,
                'correctly_defended': simulator.correctly_defended,
                'incorrectly_defended': simulator.incorrectly_defended,
                'original_points_total': simulator.original_points,
                'training_points_total': simulator.training_points
            }

            results_df[label] = simulator_results

        return pd.DataFrame(results_df)
    

    def evaluate_simulators_metrics(self, X_test, y_test):
        """Returns a dictionary of lists with metrics. Requires the model 
        to have an .evaluate() method with arguments (X_test, y_test) that returns
        any given metric that the user deems appropriate for the task at hand.
        Args:
            X_test (np.ndarray) : NumPy array containing features.
            y_test (np.ndarray) : NumPy array containing labels.
        Returns:
            metrics (dict) : Dictionary where each key is a simulator 
                             and each value is a final evaluation metric.
        """
        metrics = {}
        
        for simulation_label, list_of_models in tqdm(self.wrapped_models.items()):
            final_model_specs = list_of_models[-1]
            model = self.final_models[simulation_label]
            model.load_state_dict(final_model_specs)
            metric = model.evaluate(X_test, y_test) 
            if isinstance(metric, torch.Tensor):
                if metric.is_cuda:
                    metric = metric.cpu().numpy()
                else:
                    metric = metric.numpy()
            metrics[simulation_label] = [metric]
        return metrics


    def compute_online_learning_metrics(self, X_test, y_test):
        """Returns a dictionary of lists with metrics. Requirement: The model 
        must have an evaluate method of the form: Input: X_test, y_test, 
        self.batch_sizes[simulation_label]. Output: metric.
        Args:
            X_test (np.ndarray) : NumPy array containing features.
            y_test (np.ndarray) : NumPy array containing labels.
        Returns:
            metrics (dict) : Dictionary where each key is a simulator and each 
                             value is a list of coresponding metrics throughout 
                             the simulation (each value corresponds to a single 
                             timestep of a simulation).
        """
        metrics = {}
        
        for simulation_label, list_of_models in tqdm(self.wrapped_models.items()):
            for model_specs in list_of_models:
                model = self.final_models[simulation_label]
                model.load_state_dict(model_specs)
                metric = model.evaluate(X_test, y_test)
                if isinstance(metric, torch.Tensor):
                    if metric.is_cuda:
                        metric = metric.cpu().numpy()
                    else:
                        metric = metric.numpy()
                if simulation_label in metrics:
                    metrics[simulation_label].append(metric)
                else:
                    metrics[simulation_label] = [metric]
        return metrics


    def plot_online_learning_metrics(self, metrics, show_plot=True, save=True, 
                                     plotname=None, set_plot_title=True):
        """Prints a plot into a console. Supports supervised learning only.
        
        Args:
            metrics (np.ndarray) : an array of metrics of length equal to the 
                                   number of episodes in a simulation.
            save (bool) : enable saving.
            plotname (str) : if set to None, file name is set to a current 
                             timestamp.
        """
        if plotname is None: plotname = get_time_stamp_as_string()
        
        for _,v in metrics.items(): l = len(v)
        x = [e for e in range(l)]
        #x = [e for e in range((self.num_episodes))]

        fig, ax = plt.subplots(1, figsize=(15,10))
        for model_name, metric in metrics.items():
            ax.plot(x, metric, label=model_name)
            ax.legend()

        if set_plot_title: ax.set_title(plotname)
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Metric')
        if show_plot: plt.show()
        
        if save: save_plot(fig, plotname=plotname)


    def _extract_z(self, dataset, perplexity=50, n_iter=2000):
        """Extract embedded x,y positions for every datapoint in the dataset.
        
        Args:
            dataset (np.ndarray) : NumPy array containing data.
            perplexity (int) : The perplexity is related to the number of 
                                 nearest neighbors that is used in other manifold 
                                 learning algorithms. Larger datasets usually 
                                 require a larger perplexity. Consider selecting 
                                 a value between 5 and 50. Different values can 
                                 result in significantly different results. 
                                 Default = 50.
            n_iter (iter) : Maximum number of iterations for the optimization. 
                              Should be at least 250. SeeDefault = 2000.
        Returns:
            tsne_results : embedded x,y positions for every datapoint 
                             in the dataset.
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
        tsne_results = tsne.fit_transform(dataset.reshape(dataset.shape[0], -1))

        return tsne_results


    def _get_predictions(self, X_test, state_dicts):
        """Get all predictions on a test set of the models inside state_dicts.
        Args:
            X_test (np.ndarray, torch.Tensor) : Test data.
            state_dicts (dict) : Dictionary containing state dictionaries of 
                                   any number of trained models.
        Returns:
            sim_predictions (dict) : Dictionary with same keys as state_dicts 
                                       and values corresponding to the predictions 
                                       of the respective trained models.
        """
        #convert test set to torch tensor
        if type(X_test) != torch.Tensor:
            X_test = torch.tensor(X_test)

        sim_predictions = {}
        for label, state_dict in state_dicts.items():
            model = self.final_models[label]
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


    def plot_decision_boundaries(self, X_test, y_test, num_points=500, 
                                 perplexity=50, n_iter=2000, C=10, 
                                 kernel='poly', degree=3, figsize=(20,20), 
                                 fontsize=10, markersize=20, resolution = 0.2, 
                                 save=False, show_plot=True): 
        """Plot the decision boundaries of the final models inside all the ran 
        Simulator objects passed in the constructor method of the PostProcessor. 
        This method uses sklearn.manifold.TSNE to reduce the dimensionality of 
        X_test to 2D for visualisation purposes. An sklearn C-Support Vector 
        Classifier is then trained using the points in the smaller feature space 
        with the predicted labels of each model to show their decision 
        boundaries in 2D.
        Args: 
            X_test (np.ndarray, torch.Tensor) : Test input data.
            y_test (np.ndarray, torch.Tensor) : Test labels.
            num_points (int) : Number of points within X_test/y_test to plot 
                                 in the figure. Consider selecting a value 
                                 between 300 and 1000.
            perplexity (int) : The perplexity is related to the number of 
                                 nearest neighbors that is used in other manifold 
                                 learning algorithms. Larger datasets usually 
                                 require a larger perplexity. Consider selecting 
                                 a value between 5 and 50. Different values can 
                                 result in significantly different results. 
                                 Default = 50.
            n_iter (iter) : Maximum number of iterations for the optimization. 
                              Should be at least 250. Default = 2000.
            C (float) : Regularization parameter. The strength of the 
                          regularization is inversely proportional to C. Must be 
                          strictly positive. The penalty is a squared l2 penalty.
            kernel (str) : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} 
                             or callable. Specifies the kernel type to be used in 
                             the algorithm. If none is given, 'rbf' will be used. 
                             If a callable is given it is used to pre-compute the 
                             kernel matrix from data matrices; that matrix should 
                             be an array of shape (n_samples, n_samples). 
                             Default='poly'.
            degree (int) : Degree of the polynomial kernel function ('poly'). 
                           Ignored by all other kernels. Default = 3.
            figsize (tuple) : Tuple (W,H) indicating the size of the figures to plot.
                              Default = (20,20).
            fontsize (int) : Size of the font in the plots. Default = 10.
            markersize (int) : Size of the markers representing individual points in 
                               X_test and y_test. Default = 20.
            resolution (float) : Size of the "steps" used to create the meshgrid upon
                                 which the predictions are made to plot the decision
                                 boundaries of the model. The smaller the value the 
                                 more computationally exhaustive the process becomes. 
                                 Values < 0.1 are not recommended for this reason 
                                 (Default = 0.2).
            save (bool) : Boolean indicating wether to save the plots or not. (Default = False).
            show_plot (bool) : Boolean indicating if plot should be showed (Default = True). 
        """
        #make sure number of points is smaller than length of test set
        assert num_points <= len(X_test), 'Number of points to plot must be \
        smaller than len(X_test).'
        #check the type of input           
        idxs = [random.randint(0, len(X_test)-1) for _ in range(num_points)]             
        if type(X_test) == torch.Tensor:
            X_test = X_test.numpy()[idxs]
            y_test = y_test.numpy()[idxs]
        else: 
            X_test = X_test[idxs]
            y_test = y_test[idxs]
        
        #check if one hot encoded 
        if len(y_test.shape) > 1:
            y_test = np.argmax(y_test, axis=1)

        #retrieve state dictionaries of trained models from the simulations ran
        final_state_dicts = {label:sim_models[-1] for label, sim_models \
            in self.wrapped_models.items()}

        #get model predictions for test data
        model_predictions = self._get_predictions(X_test, final_state_dicts)

        #reduce dimensionality of model embeddings to two so they can be plotted
        z_embedding = self._extract_z(X_test, perplexity, n_iter) 

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
                if predictions.is_cuda:
                    predictions = predictions.cpu().numpy()
                else: 
                    predictions = predictions.numpy()
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()

            #plot decision boundaries for final model prediction
            clf = SVC(C=C, kernel=kernel, degree=degree).fit(z_embedding, predictions)

            #predict on mesh and plot results in contour
            clf_preds = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = clf_preds.reshape(xx.shape)

            vmin,vmax = (fun(np.concatenate([clf_preds, y_test])) \
                for fun in (np.min,np.max))

            fig = plt.figure(figsize=figsize)
            #plot contour of decision boundary from SVM predictions 
            plt.contourf(xx, yy, Z, levels=num_labels+1, cmap=cmap, 
                        vmin=vmin, vmax=vmax, alpha=0.8)
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
            plt.legend(*scatter.legend_elements(),loc="best",
                        title="Classes", fontsize=fontsize)
            if show_plot: plt.show()

            if save: 
                save_plot(fig, plotname=f'{sim_label}')


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    pass