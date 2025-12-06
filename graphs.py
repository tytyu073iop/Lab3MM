import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def plot_with_line(x, y, title):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot data
    ax.plot(x, y, label=title, color='blue', linewidth=2)
    plt.savefig(f"{title.replace(' ', '_')}_line.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


def plot_value_occurrences(values, graph_title, num_buckets=None):
    """
    Plots a bar chart showing the number of occurrences of each unique value.
    For numerical data, can group values into specified number of buckets.

    Parameters:
    - values: list of values (can be strings, numbers, etc.)
    - graph_title: title of the graph
    - num_buckets: if specified and values are numerical, groups values into this many buckets
    """
    plt.clf()  # Clears current figure

    # If buckets are requested and values are numerical
    if num_buckets and all(isinstance(x, (int, float)) for x in values):
        values_array = np.array(values)
        
        # Create buckets
        min_val, max_val = np.min(values_array), np.max(values_array)
        bucket_edges = np.linspace(min_val, max_val, num_buckets + 1)
        
        # Assign values to buckets and count
        bucket_counts = np.zeros(num_buckets)
        for value in values_array:
            # Find which bucket this value belongs to
            for i in range(num_buckets):
                if i == num_buckets - 1:  # Last bucket includes both edges
                    if bucket_edges[i] <= value <= bucket_edges[i + 1]:
                        bucket_counts[i] += 1
                        break
                else:  # Other buckets include left edge, exclude right edge
                    if bucket_edges[i] <= value < bucket_edges[i + 1]:
                        bucket_counts[i] += 1
                        break
        
        # Create labels for buckets
        labels = []
        for i in range(num_buckets):
            if i < num_buckets - 1:
                label = f"[{bucket_edges[i]:.1f}-{bucket_edges[i+1]:.1f})"
            else:
                label = f"[{bucket_edges[i]:.1f}-{bucket_edges[i+1]:.1f}]"
            labels.append(label)
        
        counts = bucket_counts
        graph_title += f" ({num_buckets} buckets)"
        
    else:
        # Original behavior for categorical data or no buckets
        value_counts = Counter(values)
        labels = list(value_counts.keys())
        counts = list(value_counts.values())

    # Plotting
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color='skyblue', alpha=0.7, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(counts),
                f'{int(count)}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Values' + (' (Buckets)' if num_buckets else ''))
    plt.ylabel('Occurrences')
    plt.title(graph_title)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save and show
    plt.savefig(f"{graph_title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

def generate_graph(data_2d, graph_name):
    """
    Generate a graph from a 2D array with only matplotlib.
    
    Args:
        data_2d: 2D array/list of numerical values
        graph_name: Name/title for the graph
    """
    
    # Convert to numpy array if it's not already
    data = np.array(data_2d)
    
    # Validate it's a 2D array
    if len(data.shape) != 2:
        raise ValueError("Input must be a 2D array")
    
    rows, cols = data.shape
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each row as a line
    for i in range(rows):
        ax.plot(range(cols), data[i, :], marker='o', label=f'Row {i+1}')
    
    # Set labels and title
    ax.set_xlabel('Column Index')
    ax.set_ylabel('Value')
    ax.set_title(graph_name)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Adjust layout and show
    plt.savefig(f"{graph_name.replace(' ', '_')}_line.png", dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot_with_line(x, y, 'sin(x)')