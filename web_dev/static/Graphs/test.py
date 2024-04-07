import pandas as pd

dataset = pd.read_csv('ticket_amt_powerbi.csv')


# Paste or type your script code here:
import plotly.express as px


# Generate Plotly graph
fig_distribution = px.histogram(dataset, 
                                x='set_fine_amount', 
                                y='number_of_tickets',
                                nbins=50,
                                marginal='box')

fig_distribution.update_layout(title_text='Distribution of Ticket Amount')

# Save the Plotly graph as an interactive HTML file
fig_distribution.write_html('plotly_graph.html')
