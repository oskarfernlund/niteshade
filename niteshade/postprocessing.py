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

import torch
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from tqdm import tqdm
from fpdf import FPDF

from niteshade.utils import save_plot, get_cmap, get_time_stamp_as_string


# =============================================================================
#  CLASSES
# =============================================================================

class PostProcessor:
    def __init__(self, wrapped_data, wrapped_models, 
                 batch_size, num_episodes, base_model):
        
        self.wrapped_data = wrapped_data
        self.wrapped_models = wrapped_models
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.base_model = base_model


    def track_data_modifications(self):
        """Computes for each simulation the following:
            a) the number of points modified by the attacker
            b) the number of points removed by the defender
            c) the number of points modified by the defender

        Returns:
            - results (pd.core.frame.DataFrame): Dictionary with keys 
            corresponding to simulator names, values corresponding to dicts 
            with keys a, b, c, d per above.
        """
        results = {}
        for simulator in self.wrapped_data.keys():
            s = self.wrapped_data[simulator]
            
            original_point_set = set(k for list_item in s['original'] \
                for (k,_) in list_item.items())
            post_attack_point_set = set(k for list_item in s['post_attack'] \
                for (k,_) in list_item.items())
            post_defense_point_set = set(k for list_item in s['post_defense'] \
                for (k,_) in list_item.items())
            
            poisoned_by_attacker = len(post_attack_point_set-original_point_set)
            removed_by_defender = len(post_attack_point_set-post_defense_point_set)
            modified_by_defender = len(post_defense_point_set-post_attack_point_set)

            if len(post_attack_point_set) == 0:
                poisoned_by_attacker = 0
                removed_by_defender = len(original_point_set-post_defense_point_set)
                modified_by_defender = len(post_defense_point_set-original_point_set)

            if len(post_defense_point_set) == 0:
                removed_by_defender, modified_by_defender = 0, 0
            
            simulator_result = {
                'poisoned_by_attacker': poisoned_by_attacker,
                'removed_by_defender': removed_by_defender,
                'modified_by_defender': modified_by_defender
            }
    
            results[simulator] = simulator_result

        return pd.DataFrame(results)
    

    def evaluate_simulators_metrics(self, X_test, y_test):
        """Returns a dictionary of lists with metrics. 
           Requirement: The model must have an evaluate method of the form:
                            Input: X_test, y_test, self.batch_size
                            Output: metric

        Args:
            - X_test (np.ndarray): NumPy array containing features.
            - y_test (np.ndarray): NumPy array containing labels.

        Returns:
            - metrics (dict): Dictionary where each key is a simulator 
                              and each value is a final evaluation metric.
        """
        metrics = {}
        
        for model_name, list_of_models in tqdm(self.wrapped_models.items()):
            final_model_specs = list_of_models[-1]
            model = self.base_model
            model.load_state_dict(final_model_specs)
            metric = model.evaluate(X_test, y_test, self.batch_size)
            metrics[model_name] = [metric]
        return metrics


    def compute_online_learning_metrics(self, X_test, y_test):
        """Returns a dictionary of lists with metrics. 
           Requirement: The model must have an evaluate method of the form:
                            Input: X_test, y_test, self.batch_size
                            Output: metric

        Args:
            - X_test {np.ndarray}: NumPy array containing features.
            - y_test {np.ndarray}: NumPy array containing labels.

        Returns:
            - metrics {dict}: Dictionary where each key is a simulator and each 
            value is a list of coresponding metrics throughout the simulation 
            (each value corresponds to a single timestep of a simulation).
        """
        metrics = {}
        
        for model_name, list_of_models in tqdm(self.wrapped_models.items()):
            for model_specs in list_of_models:
                model = self.base_model
                model.load_state_dict(model_specs)
                metric = model.evaluate(X_test, y_test, self.batch_size)
                if model_name in metrics:
                    metrics[model_name].append(metric)
                else:
                    metrics[model_name] = [metric]
        return metrics


    def plot_online_learning_metrics(self, metrics, show_plot=True, save=True, 
                                     plotname=None, set_plot_title=True):
        """Prints a plot into a console. Supports supervised learning only.
        
        Args:
            - metrics (np.ndarray) : an array of metrics of length equal to the 
            number of episodes in a simulation.
            - save (bool) : enable saving. 
            - plotname (str) : if set to None, file name is set to a current 
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


    def extract_z(self, dataset, perplexity=50, n_iter=2000):
        """Extract embedded x,y positions for every datapoint in the dataset.
        
        Args:
            - dataset (np.ndarray) : NumPy array containing data.
            - perplexity (int) : The perplexity is related to the number of 
            nearest neighbors that is used in other manifold learning algorithms.
            Larger datasets usually require a larger perplexity. Consider 
            selecting a value between 5 and 50. Different values can result 
            in significantly different results. Default = 50.
            - n_iter (iter) : Maximum number of iterations for the optimization. 
            Should be at least 250. SeeDefault = 2000.

        Returns:
            - tsne_results : embedded x,y positions for every datapoint 
            in the dataset.
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter)
        tsne_results = tsne.fit_transform(dataset.reshape(dataset.shape[0], -1))

        return tsne_results


    def _get_predictions(self, X_test, state_dicts):
        """Get all predictions on a test set of the models inside state_dicts.

        Args:
            - X_test (np.ndarray, torch.Tensor) : Test data. 
            - state_dicts (dict) : Dictionary containing state dictionaries of 
            any number of trained models. 

        Returns:
            - sim_predictions (dict) : Dictionary with same keys as state_dicts 
            and values corresponding to the predictions of the respective 
            trained models. 
        """
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


    def plot_decision_boundaries(self, X_test, y_test, num_points = 500, 
                                 perplexity=50, n_iter=2000, C=10, 
                                 kernel='poly', degree=3, figsize=(20,20), 
                                 fontsize=10, markersize=20, resolution = 0.1, 
                                 save=False, show_plot=True): 
        """Plot the decision boundaries of the final models inside all the ran 
        Simulator objects passed in the constructor method of the PostProcessor. 
        This method uses sklearn.manifold.TSNE to reduce the dimensionality of 
        X_test to 2D for visualisation purposes. An sklearn C-Support Vector 
        Classifier is then trained using the points in the smaller feature space 
        with the predicted labels of each model to show their decision 
        boundaries in 2D.

        Args: 
            - X_test (np.ndarray, torch.Tensor) : Test input data.
            - y_test (np.ndarray, torch.Tensor) : Test labels. 
            - num_points (int) : Number of points within X_test/y_test to plot 
            in the figure. Consider selecting a value between 300 and 1000.
            - perplexity (int) : The perplexity is related to the number of 
            nearest neighbors that is used in other manifold learning algorithms. 
            Larger datasets usually require a larger perplexity. Consider 
            selecting a value between 5 and 50. Different values can result 
            in significantly different results. Default = 50.
            - n_iter (iter) : Maximum number of iterations for the optimization. 
            Should be at least 250. Default = 2000.
            - C (float) : Regularization parameter. The strength of the 
            regularization is inversely proportional to C. Must be strictly 
            positive. The penalty is a squared l2 penalty.
            - kernel (str) : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} 
            or callable. Specifies the kernel type to be used in the algorithm.
            If none is given, 'rbf' will be used. If a callable is given it is used 
            to pre-compute the kernel matrix from data matrices; that matrix should 
            be an array of shape (n_samples, n_samples). Default='poly'.
            - degree (int) : Degree of the polynomial kernel function ('poly'). 
            Ignored by all other kernels.
            
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
            plt.title(f"'{sim_label}' Decision Boundaries \
                (Test Accuracy = {accuracy})", fontsize=fontsize)
            plt.ylabel("Embedded Y", fontsize=fontsize)
            plt.xlabel("Embedded X", fontsize=fontsize)
            plt.legend(*scatter.legend_elements(),loc="best",
                        title="Classes", fontsize=fontsize)
            if show_plot: plt.show()

            if save: 
                save_plot(fig, plotname=f'{sim_label}')


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
            - df (pd.core.frame.DataFrame) : pandas data frame.
            - table_title (string) : title as necessary.
            - new_page (bool) : print table on a new page.
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
            - file_path (string) : file path.
            - chart_title (string) : title as necessary.
            - new_page (bool) : print table on a new page.
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
            - dir_path (string) : directory path.
            - new_page (bool) : print table on a new page.
        """
        contents = os.listdir(dir_path)
        for chart in contents: 
            if chart.split('.')[-1] in ['jpeg', 'jpg', 'png']:
                self.add_chart(f'{dir_path}/{chart}', 
                               chart_title=chart.split('.')[0], new_page=True)


# =============================================================================
#  MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    pass