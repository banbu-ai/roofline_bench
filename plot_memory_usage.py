import matplotlib.pyplot as plt
import pandas as pd

# Define the file name
file_path = 'log/memory_usage_1755599518.log'

try:
    # Use pandas to read the file into a DataFrame.
    df = pd.read_csv(file_path, header=None, names=['Memory Usage'])

    # The monitoring interval is 0.01s. We can create a 'Time' column based on this.
    df['Time (s)'] = df.index * 0.01

    # Convert memory usage from KB to MB for better readability
    df['Memory Usage (MB)'] = df['Memory Usage'] / (2 ** 10)

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time (s)'], df['Memory Usage (MB)'], linewidth=1)

    # Add annotations and a title to explain the plot
    plt.title('Memory Usage During Inference', fontsize=16)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)

    # Add text annotations for monitoring details
    # We place the text relative to the plot's coordinates
    max_memory = df['Memory Usage (MB)'].max()
    plt.text(0.01, max_memory * 0.95,
             'Monitoring interval: 0.01s',
             fontsize=10, color='red')
    plt.text(0.01, max_memory * 0.90,
             'Process starts after sleep(1)',
             fontsize=10, color='red')

    # Add grid lines and display the plot
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('Memory Usage During Inference.png')
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")