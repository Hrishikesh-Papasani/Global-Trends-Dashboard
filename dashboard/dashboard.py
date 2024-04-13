# Import necessary libraries
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import folium
from geopy.geocoders import Nominatim
import json
import branca.colormap as cm
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.io as pio
import os

# Determine the project root (one level up from the script directory if this script is located in the root)
# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Go up one level from the script to get to the project root
PROJECT_ROOT = os.path.dirname(script_directory)

# Construct paths to the data files
data_path = os.path.join(PROJECT_ROOT, 'data-collection/data')

# Load the datasets
df_population = pd.read_csv(os.path.join(data_path, 'world_population_data.csv'))
df_gdp = pd.read_csv(os.path.join(data_path, 'API_NY_3/GDP-TOTAL.csv'))
df_gdp_growth = pd.read_csv(os.path.join(data_path, 'API_NY_2/GDP-growth-percentage.csv'))
df_life_expectancy = pd.read_csv(os.path.join(data_path, 'world_life_expectancy_data.csv'))
df_literacy_rate = pd.read_csv(os.path.join(data_path, 'world_literacy_rate_data.csv'))
df_unemployment = pd.read_csv(os.path.join(data_path, 'world_unemployment_data.csv'))
df_poverty_ratio = pd.read_csv(os.path.join(data_path, 'world_poverty_headcount_ratio_data.csv'))
df_map = pd.read_csv(os.path.join(data_path, 'API_NY_1/GDP-PPP.csv'))
df_map_growth = pd.read_csv(os.path.join(data_path, 'API_NY_2/GDP-growth-percentage.csv'))





def create_gdp_chart(df_gdp, selected_countries=None):
    """
    Generates a bar chart for total GDP of selected countries (top 7 countries by default)

    Parameters:
    - df_gdp: DataFrame containing GDP data.
    - selected_countries: List of countries to include in the chart. Defaults to top 7 countries if None.

    Returns:
    - Plotly Express figure object.
    """


    # Melt the DataFrame to long format
    df_long = pd.melt(df_gdp, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                      var_name='Year', value_name='GDP')
    
    # Convert 'Year' and 'GDP' to numeric, dropping rows with errors
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long['GDP'] = pd.to_numeric(df_long['GDP'], errors='coerce')
    
    # Keep rows with non-null 'Year' and 'GDP'
    df_clean = df_long.dropna(subset=['Year', 'GDP'])
    
    # Find the most recent year in the data
    most_recent_year = df_clean['Year'].max()
    
    # Filter for data from the most recent year 
    df_recent = df_clean[df_clean['Year'] == most_recent_year]

    # Filter for selected countries or default to top 7 by GDP if none selected
    if not selected_countries:
        df_recent = df_recent.sort_values(by='GDP', ascending=False).head(7)
    else:
        df_recent = df_recent[df_recent['Country Name'].isin(selected_countries)]

    # Formatting the large GDP values into a more understandable format for users 
    df_recent['GDP Trillions'] = (df_recent['GDP'] / 1e12).map('${:,.2f}T'.format)

    # Bar chart
    fig = px.bar(df_recent, x='Country Name', y='GDP',
                 title=f'Countries by GDP in {most_recent_year}',
                 labels={'GDP': 'GDP (current US$)', 'Country Name': 'Country'},
                 text='GDP Trillions')

    # Updating text position and layout for clarity
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'title': 'GDP (current US$)'}, yaxis_range=[0, df_recent['GDP'].max()*1.1])
    
    return fig





def create_gdp_growth_chart(df_gdp_growth, selected_countries=None):
    """
    Generates a bar chart for GDP growth of selected countries or top 7 countries by default.
    This method has similar implementation to create_gdp_start().

    Parameters:
    - df_gdp_growth: DataFrame containing GDP growth data.
    - selected_countries: List of countries to include in the chart. Defaults to top 7 countries if None.

    Returns:
    - Plotly Express figure object.
    """


    # Convert wide data to long format 
    df_long = pd.melt(df_gdp_growth, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                      var_name='Year', value_name='GDP Growth')
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long['GDP Growth'] = pd.to_numeric(df_long['GDP Growth'], errors='coerce')
    df_clean = df_long.dropna(subset=['Year', 'GDP Growth'])
    most_recent_year = df_clean['Year'].max()
    df_recent = df_clean[df_clean['Year'] == most_recent_year]

    if not selected_countries:
        # Default to top 7 by GDP Growth if none selected
        df_recent = df_recent.sort_values(by='GDP Growth', ascending=False).head(7)
    else:
        df_recent = df_recent[df_recent['Country Name'].isin(selected_countries)]

    # Ensure there's always a minimum range for the y-axis to display single bars properly
    y_axis_min = min(0, df_recent['GDP Growth'].min() * 1.1)
    y_axis_max = df_recent['GDP Growth'].max() * 1.1

    # Bar chart
    fig = px.bar(df_recent, x='Country Name', y='GDP Growth',
                 title=f'Countries by GDP Growth in {most_recent_year}',
                 labels={'GDP Growth': 'GDP Growth (annual %)', 'Country Name': 'Country'},
                 text=df_recent['GDP Growth'].apply(lambda x: f"{x:.2f}%"))

    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'title': 'GDP Growth (annual %)', 'range': [y_axis_min, y_axis_max]})

    return fig






## Below is the implementation for FOLIUM WORLD MAP regarding GDP GROWTH RATE (ANNUAL % WISE)

# Initialize the geolocator
geolocator = Nominatim(user_agent="geoapiExercises")

def get_country_coordinates(country_name):
    """
    Dynamically fetches the coordinates for a given country.
    """
    try:
        location = geolocator.geocode(country_name)
        return [location.latitude, location.longitude]
    except:
        return [0, 0]  # Return a default value if country not found or an error occurs


geojson_path = os.path.join(PROJECT_ROOT, 'data-collection/countries.geo.json')
with open(geojson_path, 'r', encoding='utf-8') as f:
    countries_geojson = json.load(f)

def get_years_with_data(df_map, indicator=None):
    """
    Returns a list of years where there is data available for the given indicator.
    """

    # If no indicator is selected. (Only one at present)
    if indicator is None:
        return []

    # Filter the DataFrame for the given indicator
    df_indicator = df_map[df_map['Indicator Code'] == indicator]

    # Drop non-year columns to focus on the data
    df_years = df_indicator.drop(columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'])

    # Find years where all values are not NaN 
    years_with_data = df_years.dropna(how='all', axis=1).columns.tolist()

    return years_with_data

def get_color(value):
    """
    Returns color of the country based on the GDP growth value. (Manually setting in this case)
    """

    try:
        # Convert value to float for comparison
        value = float(value)
    except ValueError:
        # Return a default color if conversion fails
        return '#D3D3D3'  # Light gray 

    if value < 0:
        if value > -2:
            return '#FFCC00'  # Yellow to Orange
        elif value > -4:
            return '#FF8000'  # Orange to Red
        else:
            return '#E60000'  # Dark Red
       
    else:
        if value < 2:
            return '#ADD8E6'  # Light Blue
        elif value < 4:
            return '#87CEEB'  # Sky Blue
        elif value < 6:
            return '#1E90FF'  # Dodger Blue
        else:
            return '#0000FF'  # Dark Blue
        

def generate_folium_map(indicator, year, df_map, countries_geojson):
    """
    Code for generating the required folium map
    """
    # Filter the dataframe for the chosen year and indicator
    data_for_year = df_map[(df_map['Indicator Code'] == indicator) & (df_map[str(year)].notna())]
    data_for_year = data_for_year[['Country Name', 'Country Code', str(year)]]
    data_for_year = data_for_year.rename(columns={str(year): 'Value', 'Country Code': 'id'})

    # Handle null values: Setting them to -1 that will be used to apply a distinct color
    data_for_year['Value'] = data_for_year['Value'].fillna(-1)

    # Merge the data with the GeoJSON file
    for feature in countries_geojson['features']:
        country_id = feature['id']
        value = data_for_year.loc[data_for_year['id'] == country_id, 'Value'].values
        name = data_for_year.loc[data_for_year['id'] == country_id, 'Country Name'].values
        feature['properties']['Value'] = value[0] if len(value) > 0 else 'No data'
        feature['properties']['Name'] = name[0] if len(name) > 0 else 'Unknown'

    # Initialize a folium map with default view
    folium_map = folium.Map(location=[20, 0], zoom_start=2, min_zoom=2, max_bounds=True)

    # Custom choropleth
    folium.GeoJson(
        countries_geojson,
        style_function=lambda feature: {
            'fillColor': get_color(feature['properties']['Value']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['Name', 'Value'],
            aliases=['Country: ', 'Value: '],
            localize=True
        )
    ).add_to(folium_map)

    folium.LayerControl().add_to(folium_map)


    # Adding legend
    colormap = cm.StepColormap(
    colors=[
        '#E60000',  # Dark Red for <= -4
        '#FF8000',  # Orange to Red for -4 to -2
        '#FFCC00',  # Yellow to Orange for -2 to 0
        '#ADD8E6',  # Light Blue for 0 to 2
        '#87CEEB',  # Sky Blue for 2 to 4
        '#1E90FF',  # Dodger Blue for 4 to 6
        '#0000FF',  # Dark Blue for >= 6
        '#D3D3D3'   # Light gray for unknown or non-numeric values
    ],

    index=[
        -100,  # Start of the colormap
        -4,    # Transition to Orange to Red
        -2,    # Transition to Yellow to Orange
        0,     # Transition to Light Blue
        2,     # Transition to Sky Blue
        4,     # Transition to Dodger Blue
        6,     # Transition to Dark Blue
        100    # End of the colormap
    ],
    vmin=-6,  # Adjusted based on the expected range of GDP growth
    vmax=8,   # Adjusted based on the expected range of GDP growth
    caption='GDP Growth (annual %)'
    )


    # Add the legend to map
    colormap.add_to(folium_map)

    return folium_map._repr_html_()






## Variables for DROPDOWN OPTIONS in literacy rate 
filtered_df = df_literacy_rate.dropna(subset=['World'])

# Find the most recent year with data
most_recent_year = df_literacy_rate['Year'].max()

# Find the earliest year with data
earliest_year = df_literacy_rate['Year'].min()

# Calculate the earliest start year for the 10-year ranges, ensuring the most recent year is included
earliest_start_year = most_recent_year - 9  # Ensure the most recent year is included in the range

# Generate a list of start years for 10-year ranges
# Note: Adjust the range to ensure it correctly captures from the earliest_year to the earliest_start_year
start_years = list(range(earliest_start_year, earliest_year - 10, -10))  # Adjusted step to -10 to go backward

# Adjust if the earliest start year goes beyond the earliest year with data
if start_years[-1] < earliest_year:
    start_years[-1] = earliest_year

# Generate dropdown options from the list of start years
year_options = [{'label': 'All Time', 'value': 'all'}]  # Add 'All Time' option
year_options += [{'label': f'{start_year}-{start_year+9}', 'value': f'{start_year}-{start_year+9}'} for start_year in start_years]

## Variables for DROPDOWN OPTIONS in population time period
filtered_df_population = df_population.dropna(subset=['World'])  # Assuming 'World' column exists

most_recent_year_population = df_population['Year'].max()
earliest_year_population = df_population['Year'].min()

earliest_start_year_population = most_recent_year_population - 9

start_years_population = list(range(earliest_start_year_population, earliest_year_population - 10, -10))

if start_years_population[-1] < earliest_year_population:
    start_years_population[-1] = earliest_year_population

year_options_population = [{'label': f'{year}-{year+9}', 'value': f'{year}-{year+9}'} for year in start_years_population]
# Adding an option for cumulative view at the start of the year options list
year_options_population.insert(0, {'label': 'All Time', 'value': 'all'})





# Set the Dash application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.config.suppress_callback_exceptions = True # Written before deployment to avoid any unknown errors in the UI
server = app.server


"""
Utilising Dash Bootstrap components for making the Dash Application Responsive. 
Utilising width = 12 (Grid with rows & columns where each column -> ddc.Col() shall fill the entire row -> ddb.Row()
utilising viewport space efficiently)
"""

# Defining the Dash application layout
app.layout = dbc.Container(fluid=True, children=[

    # App title and introdution
    dbc.Row(dbc.Col(html.H1(children=["Global Trends Visualised"], style={'font-weight': 'bold', 'text-align': 'center'}), width=12)),

   dbc.Row(
    dbc.Col(
        html.P(
            "This dashboard offers an interactive exploration of key global trends, "
            "providing insights into diverse topics such as population dynamics, economic indicators, and literacy rates. "
            "Tailored for a broad audience, from researchers to educators and the general public, "
            "it serves as a gateway to understanding and analyzing critical global issues.",
            className="text-center mt-4",  # Added margin-top for spacing
            # style={'fontSize': 16}  # Adjust font size for better readability
        ),
        width={"size": 10, "offset": 1}  # Center the text with some padding on the sides
    )
)
,

    html.Br(), html.Br(), html.Br(),

    dbc.Row(dbc.Col(html.H4("How to Use"), width=12)), 

    dbc.Row(dbc.Col(html.Ul([
        html.Li("Explore default visualizations reflecting recent and historical global trends."),
        html.Li("Customize charts by selecting different parameters to gain tailored insights."),
        html.Li("Download your generated charts, and utilize other available interactive tools."),
        html.Li("Refresh the page if maps do not load correctly."),
        html.Li("Note: Some specific countries in certain charts may be aggregated into World Bank classified regions.")
    ], className="text-left"), width=12)),


    html.Br(),
    # Population Trends Section
    dbc.Row([
        dbc.Col(html.H2("Population Trends", style={'font-weight': 'bold'}), width=12),
        dbc.Col([
            html.Label("Select Countries (Can be multiple):"),
            dcc.Dropdown(
                id='population-dropdown',
                options=[{'label': i, 'value': i} for i in df_population.columns[2:]],
                value='World',
                multi=True
            ),

            html.Label("Select Time-period:"),
            dcc.Dropdown(
                id='population-year-range-dropdown',
                options=year_options_population,
                value=year_options_population[0]['value'],
            ),

            dcc.Graph(id='population-graph'),

        ], width=12),
    ]),

    html.Br(), html.Br(),

    # GDP Trends Section
    dbc.Row([
        
        dbc.Col(html.H2("GDP Trends", style={'font-weight': 'bold'}), width=12),

        dbc.Col([
            html.Label("Select Countries (Can be multiple):"),

            dcc.Dropdown(
                id='gdp-country-dropdown',
                options=[{'label': i, 'value': i} for i in df_gdp.columns[2:]],
                value=None,
                multi=True,
                placeholder="Top Seven Economies"
            ),

            dcc.Graph(id='gdp-chart'),

            dcc.Graph(id='gdp-growth-chart'),

        ], width=12),
    ]),

    html.Br(), html.Br(),


    # Poverty Ratio Trends Section
    dbc.Row([
        dbc.Col(html.H2("Poverty Ratio Trends",style={'font-weight': 'bold'}), width=12),

        dbc.Col([
            html.Label("Select Region:"),

            dcc.Dropdown(
                id='poverty-ratio-dropdown',
                options=[{'label': col, 'value': col} for col in df_poverty_ratio.columns[2:]],
                value='World',
                multi=False
            ),

            dcc.Graph(id='poverty-ratio-pie-chart'),

        ], width=12),
    ]),

    html.Br(), html.Br(),


    # Unemployment Rate Section
    dbc.Row([
        dbc.Col(html.H2("Unemployment Rate", style={'font-weight': 'bold'}), width=12),

        dbc.Col([
            html.Label("Select Region:"),
            dcc.Dropdown(
                id='unemployment-dropdown',
                options=[{'label': col, 'value': col} for col in df_unemployment.columns[2:]],
                value='World',
                multi=False
            ),

            html.Label("Select Time Period:"),

            dcc.Dropdown(
                id='unemployment-year-range-dropdown',
                value='all',
            ),

            dcc.Graph(id='unemployment-trends-chart'),

        ], width=12),
    ]),

    html.Br(), html.Br(),


    # Literacy Rate Section
    dbc.Row([
        dbc.Col(html.H2("Literacy Rate", style={'font-weight': 'bold'}), width=12),

        dbc.Col([
            html.Label("Select Region:"),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': i, 'value': i} for i in df_literacy_rate.columns[2:]],
                value='World',
                multi=True
            ),

            html.Label("Select Time Period:"),
            dcc.Dropdown(
                id='year-range-dropdown',
                options=year_options,
                value='all',
            ),

            dcc.Graph(id='literacy-rate-line-chart'),

        ], width=12),
    ]),

    html.Br(), html.Br(),



    # World through Maps Section
    dbc.Row([
        dbc.Col(html.H2("World through Maps", style={'font-weight': 'bold'}), width=12),

        dbc.Col([
            html.Label('Select Indicator:'),
            dcc.Dropdown(
                id='your_indicator_dropdown',
                options=[{'label': 'GDP Growth Rate (Annual %)', 'value': 'NY.GDP.MKTP.KD.ZG'}],
                value='NY.GDP.MKTP.KD.ZG',
            ),

            html.Label('Select Year:'),
            dcc.Dropdown(
                id='your_year_dropdown',
                options=[],
                value=2022,
            ),

            html.Iframe(id='folium_map', style={'width': '100%', 'height': '750px'}),

        ], width=12),
    ]),


html.Br(),
html.Hr(),


    dbc.Row(dbc.Col([
    html.H4("Details", className="text-center mb-3"),

    html.P([
        "Data sourced from the World Bank API."
    ], className="text-center"),


    html.P([
        html.A("World Bank API Documentation", href="https://datahelpdesk.worldbank.org/knowledgebase/articles/898581-api-documentation", target="_blank")
    ], className="text-center"),

], width=12)),

html.Hr(),


])


## Callback section (These methods are executed each time user changes a Dropdown Menu option)


# Callback for total GDP bar chart based on selected countries
@app.callback(
    Output('gdp-chart', 'figure'),
    [Input('gdp-country-dropdown', 'value')],
)

def update_gdp_chart(selected_countries=None):   
    return create_gdp_chart(df_gdp, selected_countries)



# Callback for GDP Growth bar chart based on selected countries
@app.callback(
    Output('gdp-growth-chart', 'figure'),  
    [Input('gdp-country-dropdown', 'value')]
)

def update_gdp_growth_chart(selected_countries=None):
    return create_gdp_growth_chart(df_gdp_growth, selected_countries)



# Callback for Folium World Map (year)
@app.callback(
    Output('folium_map', 'srcDoc'),
    [Input('your_indicator_dropdown', 'value'),
     Input('your_year_dropdown', 'value')]
)

def update_map(indicator, year):
    return generate_folium_map(indicator, year, df_gdp_growth, countries_geojson)


# Callback for Folium World Map (indicator)
@app.callback(
    Output('your_year_dropdown', 'options'),
    [Input('your_indicator_dropdown', 'value')]
)

def update_year_dropdown(selected_indicator):
    years_with_data = get_years_with_data(df_gdp_growth, selected_indicator)
    # Convert years to string if they are not, and generate options for the dropdown
    year_options = [{'label': str(year), 'value': str(year)} for year in years_with_data]
    return year_options



# Callback for literacy rate chart
@app.callback(
    Output('literacy-rate-line-chart', 'figure'),
    [Input('year-range-dropdown', 'value'),
     Input('region-dropdown', 'value')]
)

def update_literacy_rate_chart(selected_year_range, selected_regions):
    if not selected_regions:
        raise PreventUpdate

    if not selected_year_range:
        raise PreventUpdate


    # Handle single region selection as a list
    if not isinstance(selected_regions, list):
        selected_regions = [selected_regions]

    # Check if 'All Time' option is selected
    if selected_year_range == "all":
        start_year = df_literacy_rate['Year'].min()
        end_year = df_literacy_rate['Year'].max()
    else:
        start_year, end_year_str = selected_year_range.split('-')
        start_year = int(start_year)
        end_year = int(end_year_str)

    # Filter the dataset for the selected year range
    filtered_df = df_literacy_rate[(df_literacy_rate['Year'] >= start_year) & (df_literacy_rate['Year'] <= end_year)]

    # Ensure the DataFrame contains only selected regions and is not empty
    filtered_df = filtered_df[['Year'] + [region for region in selected_regions if region in filtered_df.columns]]
    melted_df = filtered_df.melt(id_vars='Year', var_name='Region', value_name='Literacy Rate').dropna()

    # Ensure the melted DataFrame is sorted by Year to avoid plotting issues
    melted_df = melted_df.sort_values(by='Year')

    # Line chart for literacy rate for the selected regions
    fig = px.line(
        melted_df,
        x='Year', 
        y='Literacy Rate', 
        color='Region', 
        title=f'Literacy Rate ({start_year}-{end_year})',
        labels={'Literacy Rate': 'Literacy Rate (%)'},
        markers=True,
        line_shape='linear'  # Lines are drawn even if some data points are missing
    )

    fig.update_layout(yaxis_title='Literacy Rate (%)', xaxis_title='Year')
    fig.update_traces(line=dict(width=2), marker=dict(size=7))

    return fig




# Callback function for population line graph
@app.callback(
    Output('population-graph', 'figure'),
    [Input('population-year-range-dropdown', 'value'),
     Input('population-dropdown', 'value')]
)

def update_population_graph(selected_year_range, selected_countries):
    # If no countries selected by the user, don't make any changes to the existing Dashboard
    if selected_year_range is None:
        raise PreventUpdate

    # Check if the 'All Time' option is selected
    if selected_year_range == "all":
        start_year = df_population['Year'].min()
        end_year = df_population['Year'].max()
    else:
        start_year, end_year = map(int, selected_year_range.split('-'))
    
    filtered_df = df_population[(df_population['Year'] >= start_year) & (df_population['Year'] <= end_year)]
    
    # Ensure selected_countries is a list
    if not isinstance(selected_countries, list):
        selected_countries = [selected_countries]
    
    long_df = filtered_df.melt(id_vars=['Year'], var_name='Country', value_name='Population')
    long_df = long_df[long_df['Country'].isin(selected_countries)]  # Filter by selected countries
    

    # Line graph
    fig = px.line(
        long_df,
        x='Year', 
        y='Population', 
        color='Country', 
        title=f"Population Trends ({start_year if start_year != df_population['Year'].min() else 'All Time'}-{end_year})",
        labels={'Population': 'Population', 'Year': 'Year'},
        markers=True
    )
    
    fig.update_layout(yaxis_title='Population', xaxis_title='Year')
    fig.update_traces(line=dict(width=2), marker=dict(size=7))
    
    return fig




# Callback to update the poverty ratio pie chart based on the selected country/region
@app.callback(
    Output('poverty-ratio-pie-chart', 'figure'),
    [Input('poverty-ratio-dropdown', 'value')]
)

def update_poverty_ratio_chart(selected_country):
    # If no countries selected by the user, make no changes to the existing dashboard
    if selected_country is None:
        raise PreventUpdate

    # Extract the most recent year's data for the selected country
    most_recent_year_data = df_poverty_ratio[['Year', selected_country]].dropna().iloc[-1]
    most_recent_year, poverty_ratio = most_recent_year_data
    non_poverty_ratio = 100 - poverty_ratio  # Assuming the poverty ratio is the percentage living below the poverty line

    # Data for the pie chart
    labels = ['Below Poverty Line', 'Above Poverty Line']
    values = [poverty_ratio, non_poverty_ratio]
    
    # Creating the pie chart
    fig = px.pie(values=values, names=labels, title=f'Poverty Ratio in {selected_country} ({most_recent_year})')
    fig.update_traces(textinfo='percent+label')

    return fig


# Callback to update the unemployment trends line chart
@app.callback(
    [Output('unemployment-year-range-dropdown', 'options'),
     Output('unemployment-year-range-dropdown', 'value')],
    [Input('unemployment-dropdown', 'value')]
)

def set_year_range_options(selected_country):
    if selected_country is None:
        PreventUpdate
    # Finding the most recent and earliest years in the dataset
    most_recent_year = df_unemployment['Year'].max()
    earliest_year = df_unemployment['Year'].min()
    # Generating year range options
    year_options = [{'label': 'All Time', 'value': 'all'}]
    year_options += [{'label': f'{year}-{year+9}', 'value': f'{year}-{year+9}'} for year in range(earliest_year, most_recent_year, 10)]
    return year_options, 'all'



# Callback to update the unemployment trends chart 
@app.callback(
    Output('unemployment-trends-chart', 'figure'),
    [Input('unemployment-dropdown', 'value'),
     Input('unemployment-year-range-dropdown', 'value')]
)

def update_unemployment_chart(selected_country, selected_year_range):

    # If any parameter is empty, make no changes to existing dashboard
    if selected_year_range is None or selected_country is None:
        raise PreventUpdate

    current_year = pd.Timestamp.now().year  # Assuming you're using the current year to compare
    is_all_time = selected_year_range == "all"
    if is_all_time:
        filtered_df = df_unemployment[['Year', selected_country]].dropna()
        title_year_range = "All Time"
        start_year = df_unemployment['Year'].min()
        end_year = df_unemployment['Year'].max()
    else:
        start_year, end_year = map(int, selected_year_range.split('-'))
        filtered_df = df_unemployment[(df_unemployment['Year'] >= start_year) & (df_unemployment['Year'] <= end_year)][['Year', selected_country]].dropna()
        title_year_range = f"({start_year}-{end_year})"

    fig = px.bar(
        filtered_df,
        x='Year',
        y=selected_country,
        title=f'Unemployment Trends in {selected_country} {title_year_range}',
        labels={selected_country: 'Unemployment Rate (%)'},
    )

    # Adjust the x-axis based on the year range
    if not is_all_time and end_year >= current_year:
        # For the most recent year range, ensure no decimals and display each year
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                tick0=start_year,
                dtick=1  # Ensure every year is displayed
            )
        )

    fig.update_traces(
        texttemplate='%{y:.1f}%',  # Simplify to 1 decimal point
        hovertemplate='<b>Year: %{x}</b><br>Unemployment Rate: %{y}%<extra></extra>'
    )

    fig.update_yaxes(range=[df_unemployment[selected_country].min() - 1, df_unemployment[selected_country].max() + 1])

    return fig




# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)






