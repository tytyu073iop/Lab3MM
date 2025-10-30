import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

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

# Example usage
if __name__ == "__main__":
    # Example 1: Categorical data (no buckets)
    categories = ['A', 'B', 'A', 'C', 'B', 'B', 'A', 'A', 'C', 'D']
    plot_value_occurrences(categories, "Categorical Data Example")
    
    # Example 2: Numerical data without buckets
    numbers = [1.2, 2.5, 3.1, 1.8, 2.9, 3.3, 1.1, 2.7, 3.0, 2.2, 1.5, 2.8]
    plot_value_occurrences(numbers, "Numerical Data - No Buckets")
    
    # Example 3: Numerical data with 5 buckets
    plot_value_occurrences(numbers, "Numerical Data", num_buckets=5)
    
    # Example 4: Larger dataset with buckets
    larger_data = np.random.normal(100, 15, 1000)
    plot_value_occurrences(larger_data, "Large Dataset", num_buckets=10)