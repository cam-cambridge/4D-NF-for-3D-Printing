import torch
from termcolor import colored
import matplotlib.pyplot as plt
import csv 
import os 

def latin_hypercube_sampling_1D(N):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    intervals = torch.linspace(-1, 1, N, dtype=torch.float32, device=device)
    points = intervals + torch.rand(N, dtype=torch.float32, device=device) / N
    shuffled_indices = torch.randperm(N, device=device)
    points_shuffled = points[shuffled_indices]
    
    return points_shuffled

class Logger:
    def __init__(self, batch, log_dir=None, v_num=None):
        self.metrics = {}
        self.batch = batch
        self.csv_files = {}
        self.v_num = v_num
        self.log_dir = log_dir or f"logs/train/lightning_logs/version_{self.v_num}"
        self.metrics_dir = os.path.join(self.log_dir, "metrics")

        if not os.path.exists(self.metrics_dir):
            os.makedirs(self.metrics_dir)

    def log(self, key, value):
        if key not in self.metrics:
            self.metrics[key] = []
            csv_file_path = os.path.join(self.metrics_dir, f"{key}.csv")
            self.csv_files[key] = csv_file_path
            if not os.path.exists(csv_file_path):
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Value'])

        # Append the value to the metric list
        self.metrics[key].append(value)

        # Append the value to the CSV file
        with open(self.csv_files[key], mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([value])

    def get_metrics(self):
        return self.metrics

    def report_running_mean(self, plot):
        
        report = ">|"
        for key, values in self.metrics.items():
            mean_value = sum(values[-self.batch:]) / len(values[-self.batch:])
            report += f"|{key}: {colored(f'{mean_value:.4f}','blue')}|"

        if plot:
            self.plot_metrics()

    def plot_metrics(self):
        plt.figure(figsize=(10, 6),dpi=250)
        
        for key, values in self.metrics.items():
            color = "black"
            plt.plot(values, alpha=0.4, color=color)

        plt.xlabel('Batch')
        plt.yscale('log')
        plt.ylabel('Value')
        plt.title('Training Metrics Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "loss_plot.jpg"))
        plt.close()
