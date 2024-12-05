import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import webbrowser
from fpdf import FPDF



# Read CSV files
cpu_data = pd.read_csv('cpu_output.csv')
gpu_data = pd.read_csv('gpu_output.csv')

# Assuming the CSV files have columns 'time' and 'execution_time'
cpu_times = cpu_data['Dim']
cpu_execution_times = cpu_data['CPU Execution Time (ms)']
gpu_times = gpu_data['Dim']
gpu_execution_times = gpu_data['GPU Execution Time (ms)']

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Plot CPU vs GPU Execution Time
axs[0, 0].bar(np.arange(len(cpu_times)), cpu_execution_times, 0.35, label='CPU Execution Time')
axs[0, 0].bar(np.arange(len(cpu_times)) + 0.35, gpu_execution_times, 0.35, label='GPU Execution Time')
axs[0, 0].set_title('CPU vs GPU Execution Time')
axs[0, 0].set_xlabel('Dimension')
axs[0, 0].set_ylabel('Execution Time (ms)')
axs[0, 0].set_xticks(np.arange(len(cpu_times)) + 0.35 / 2)
axs[0, 0].set_xticklabels(cpu_times)
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot CPU vs GPU Performance
axs[0, 1].bar(np.arange(len(cpu_times)), cpu_data['CPU Performance (GFLOP/s)'], 0.35, label='CPU Performance (GFLOP/s)')
axs[0, 1].bar(np.arange(len(cpu_times)) + 0.35, gpu_data['GPU Performance (GFLOP/s)'], 0.35, label='GPU Performance (GFLOP/s)')
axs[0, 1].set_title('CPU vs GPU Performance')
axs[0, 1].set_xlabel('Dimension')
axs[0, 1].set_ylabel('Performance (GFLOP/s)')
axs[0, 1].set_xticks(np.arange(len(cpu_times)) + 0.35 / 2)
axs[0, 1].set_xticklabels(cpu_times)
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot CPU vs GPU Energy Consumption
axs[1, 0].bar(np.arange(len(cpu_times)), cpu_data['CPU Energy (J)'], 0.35, label='CPU Energy (J)')
axs[1, 0].bar(np.arange(len(cpu_times)) + 0.35, gpu_data['GPU Energy (J)'], 0.35, label='GPU Energy (J)')
axs[1, 0].set_title('CPU vs GPU Energy Consumption')
axs[1, 0].set_xlabel('Dimension')
axs[1, 0].set_ylabel('Energy (J)')
axs[1, 0].set_xticks(np.arange(len(cpu_times)) + 0.35 / 2)
axs[1, 0].set_xticklabels(cpu_times)
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot CPU vs GPU Power Efficiency
axs[1, 1].bar(np.arange(len(cpu_times)), cpu_data['Power Efficiency (GFLOP/J)'], 0.35, label='CPU Power Efficiency (GFLOP/J)')
axs[1, 1].bar(np.arange(len(cpu_times)) + 0.35, gpu_data['Power Efficiency (GFLOP/J)'], 0.35, label='GPU Power Efficiency (GFLOP/J)')
axs[1, 1].set_title('CPU vs GPU Power Efficiency')
axs[1, 1].set_xlabel('Dimension')
axs[1, 1].set_ylabel('Power Efficiency (GFLOP/J)')
axs[1, 1].set_xticks(np.arange(len(cpu_times)) + 0.35 / 2)
axs[1, 1].set_xticklabels(cpu_times)
axs[1, 1].legend()
axs[1, 1].grid(True)

# Plot GPU Speedup
speedup = cpu_execution_times / gpu_execution_times
axs[2, 0].bar(np.arange(len(cpu_times)), speedup, 0.35, label='Speedup (x)')
axs[2, 0].set_title('GPU Speedup')
axs[2, 0].set_xlabel('Dimension')
axs[2, 0].set_ylabel('Speedup (x)')
axs[2, 0].set_xticks(np.arange(len(cpu_times)))
axs[2, 0].set_xticklabels(cpu_times)
axs[2, 0].legend()
axs[2, 0].grid(True)


# Hide the empty subplot (bottom right)
fig.delaxes(axs[2, 1])

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.4) 

# Show the plot
plt.show()


# Save the plot as an image file
fig.savefig('plot.png')




# Define the prompt for the report generation
prompt = f"Generate a well-formatted HTML report based on the following CSV data dont put anything before or after the html code, i want the report to have introduction , more explication and summary for each data and comparaison between gpu and cpu in all aspects and a conlusion, make it long and very detailed, and always add just one time a image tag that open 'plot.png' one time , and please please style it well with icons and tables and good colors to make it visualy apealling:\n\nCPU Data:\n{cpu_data.head().to_string()}\n\nGPU Data:\n{gpu_data.head().to_string()}"

# Encode the prompt
encoded_prompt = requests.utils.quote(prompt)

# Define the endpoint URL
url = f"https://a.picoapps.xyz/ask-ai?prompt={encoded_prompt}"

# Make the request to the endpoint
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Get the HTML report from the response
    html_report = response.text
    # Parse the response to get the HTML report
    html_report = response.json().get('response', '')
    
    # Save the HTML report to a file
    with open('report.html', 'w') as file:
        file.write(html_report)
    print("HTML report generated and saved as 'report.html'")

    # Convert HTML report to PDF

    # Create a PDF document
    pdf = FPDF()
    pdf.add_page()

    # Set font
    pdf.set_font("Arial", size=12)

    # Open the HTML file
    with open('report.html', 'r') as file:
        html_content = file.read()

    # Add HTML content to PDF
    pdf.write(5, html_content)

    # Save the PDF to a file
    pdf.output("report.pdf")

    print("PDF report generated and saved as 'report.pdf'")

   

    # Open the generated HTML report in the default web browser
    webbrowser.open('report.html')
else:
    print(f"Failed to generate report. Status code: {response.status_code}")


