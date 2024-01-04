import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('TKAgg')

class EasyVisualize:
    def __init__(self, data):
        """
        Initialize with a pandas DataFrame.
        """
        self.data = data

    def plot_distribution(self, column, bins=30, kde=True, color='blue', figsize=(8,6)):
        """
        Plot the distribution of a column using a histogram.
        """
        plt.figure(figsize=figsize)
        sns.histplot(self.data[column], bins=bins, kde=kde, color=color)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def plot_count(self, column, palette='viridis', hue=None, figsize=(8,6)):
        """
        Plot the counts of a categorical column using a bar chart.
        """
        plt.figure(figsize=figsize)
        sns.countplot(x=column, data=self.data, palette=palette, hue=hue)
        plt.title(f'Count of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

    def plot_box(self, column, y=None, color='green', figsize=(8,6)):
        """
        Plot a boxplot for one or two variables.
        """
        plt.figure(figsize=figsize)
        sns.boxplot(x=column, y=y, data=self.data, color=color)
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.ylabel('Value')
        plt.show()

    def plot_correlation_matrix(self, figsize=(8,6)):
        """
        Plot the correlation matrix of the DataFrame.
        """
        plt.figure(figsize=figsize)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def plot_pairplot(self, hue=None):
        """
        Plot pairwise relationships in the dataset.
        """
        sns.pairplot(self.data, hue=hue)
        plt.title('Pairwise Relationships')
        plt.show()

    def plot_time_series(self, date_column, value_column, date_format='%Y-%m-%d', color='purple', figsize=(8,6)):
        """
        Plot a time series line plot for a given date and value column.
        """
        # Convert the date column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column], format=date_format)

        plt.figure(figsize=figsize)
        sns.lineplot(x=date_column, y=value_column, data=self.data.sort_values(by=date_column), color=color)
        plt.title(f'Time Series Plot of {value_column} over {date_column}')
        plt.xlabel('Date')
        plt.ylabel(value_column)
        plt.show()
    
    def create_mixed_subplots(self, columns_info, nrows=1, ncols=1, figsize=(15, 8), hue=None):
            """
            Create subplots for the given columns with different types of plots.
            
            Parameters:
            - columns_info: a dictionary mapping column names to plot types
                            e.g., {'column1': 'dist', 'column2': 'count', ...}
            - nrows, ncols: number of rows and columns in the subplot grid
            - figsize: figure size
            - hue: optional hue parameter for all plots
            """
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            if nrows * ncols == 1:
                axes = [axes]  # Wrap it in a list if only one plot
            axes = axes.flatten()  # Flatten axes array if necessary

            for ax, (column, plot_type) in zip(axes, columns_info.items()):
                if plot_type == 'dist':
                    sns.histplot(self.data[column], ax=ax, kde=True, hue=hue)
                elif plot_type == 'count':
                    sns.countplot(x=column, data=self.data, ax=ax, hue=hue)
                elif plot_type == 'box':
                    sns.boxplot(x=column, data=self.data, ax=ax, hue=hue)
                elif plot_type == 'line':
                    sns.lineplot(x=column, y=self.data[column], data=self.data, ax=ax, hue=hue)
                else:
                    raise ValueError(f"Invalid plot type for column {column}: {plot_type}")

                ax.set_title(f'{plot_type.title()} Plot of {column}' + (f' by {hue}' if hue else ''))
                ax.set_xlabel(column)
                ax.set_ylabel('Value')

            plt.tight_layout()
            plt.show()
        
    def plot_bar(self, x_column, y_column=None, hue=None, palette='viridis', figsize=(8,6)):
        """
        Plot a bar plot for a variable, with an optional hue.
        """
        plt.figure(figsize=figsize)
        sns.barplot(x=x_column, y=y_column, hue=hue, data=self.data, palette=palette)
        plt.title(f'Bar Plot of {x_column}' + (f' by {hue}' if hue else ''))
        plt.xlabel(x_column)
        plt.ylabel('Value' if y_column else 'Count')
        plt.show()
    
    def create_mixed_subplots_st(self, columns_info, nrows=1, ncols=1, figsize=(15, 8), hue=None):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if nrows * ncols == 1:
            axes = [axes]  # Wrap it in a list if only one plot
        axes = axes.flatten()  # Flatten axes array if necessary

        for ax, (column, plot_type) in zip(axes, columns_info.items()):
            if plot_type == 'dist':
                sns.histplot(self.data[column], ax=ax, kde=True, hue=hue)
            elif plot_type == 'count':
                sns.countplot(x=column, data=self.data, ax=ax, hue=hue)
            elif plot_type == 'box':
                sns.boxplot(x=column, data=self.data, ax=ax, hue=hue)
            elif plot_type == 'line':
                sns.lineplot(x=column, y=self.data[column], data=self.data, ax=ax, hue=hue)
            else:
                raise ValueError(f"Invalid plot type for column {column}: {plot_type}")

            ax.set_title(f'{plot_type.title()} Plot of {column}' + (f' by {hue}' if hue else ''))
            ax.set_xlabel(column)
            ax.set_ylabel('Value')

        plt.tight_layout()
        return fig  # Return the figure object instead of showing it
    
    def plot_distribution_st(self, column, bins=30, kde=True, color='blue', figsize=(8,6)):
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(self.data[column], bins=bins, kde=kde, color=color, ax=ax)
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        return fig

    def plot_count_st(self, column, palette='viridis', hue=None, figsize=(8,6)):
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(x=column, data=self.data, palette=palette, hue=hue, ax=ax)
        ax.set_title(f'Count of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Count')
        return fig

    def plot_box_st(self, column, y=None, color='green', figsize=(8,6)):
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(x=column, y=y, data=self.data, color=color, ax=ax)
        ax.set_title(f'Boxplot of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Value')
        return fig

    def plot_correlation_matrix_st(self, figsize=(8,6)):
        fig, ax = plt.subplots(figsize=figsize)
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Matrix')
        return fig

    def plot_pairplot_st(self, hue=None):
        pairplot_fig = sns.pairplot(self.data, hue=hue)
        pairplot_fig.fig.suptitle('Pairwise Relationships', y=1.02)  # Adjust title position
        return pairplot_fig.fig

    def plot_time_series_st(self, date_column, value_column, date_format='%Y-%m-%d', color='purple', figsize=(8,6)):
        fig, ax = plt.subplots(figsize=figsize)
        if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
            self.data[date_column] = pd.to_datetime(self.data[date_column], format=date_format)
        sns.lineplot(x=date_column, y=value_column, data=self.data.sort_values(by=date_column), color=color, ax=ax)
        ax.set_title(f'Time Series Plot of {value_column} over {date_column}')
        ax.set_xlabel('Date')
        ax.set_ylabel(value_column)
        return fig

    def plot_bar_st(self, x_column, y_column=None, hue=None, palette='viridis', figsize=(8,6)):
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=x_column, y=y_column, hue=hue, data=self.data, palette=palette, ax=ax)
        ax.set_title(f'Bar Plot of {x_column}' + (f' by {hue}' if hue else ''))
        ax.set_xlabel(x_column)
        ax.set_ylabel('Value' if y_column else 'Count')
        return fig