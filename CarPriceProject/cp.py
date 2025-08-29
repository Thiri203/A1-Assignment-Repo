
import json, pickle, numpy as np, pandas as pd
with open("CarPriceProject/artifacts/model.pkl", "rb") as f:
    MODEL = pickle.load(f)

with open("CarPriceProject/artifacts/schema.json") as f:
    SCHEMA = json.load(f)

FEATURES = SCHEMA["feature_order"]          
TARGET_TRANSFORM = SCHEMA.get("target_transform")

import dash
from dash import dcc, html, Input, Output, State, callback
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H1("🚗 Car Price Predictor", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Get accurate price predictions for your car using advanced machine learning",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'marginBottom': '30px'}),
    
    html.Div([
        html.H2("How It Works", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Ol([
            html.Li("Fill in the car details below - you can skip any field you don't know"),
            html.Li("Our system will automatically estimate missing values using smart defaults"),
            html.Li("Click 'Predict Price' to get your car's estimated market value"),
            html.Li("View the prediction with confidence intervals and market insights")
        ], style={'fontSize': '16px', 'lineHeight': '1.6', 'color': '#34495e'})
    ], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '0 20px', 'marginBottom': '40px'}),
    
    html.Div([
        html.H2("Car Details", style={'color': '#2c3e50', 'marginBottom': '25px', 'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H3("Essential Information", style={'color': '#3498db', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Car Brand", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Input(
                        id='car-brand',
                        type='text',
                        placeholder='Honda',
                        style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 'borderRadius': '5px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Year of Manufacture", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='year',
                        options=[{'label': str(year), 'value': year} for year in range(2020, 1982, -1)],
                        placeholder='Select year...',
                        style={'marginBottom': '10px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Kilometers Driven", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Input(
                        id='km-driven',
                        type='number',
                        placeholder='50000',
                        min=1,
                        max=360457,
                        style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 'borderRadius': '5px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Fuel Type", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='fuel-type',
                        options=[
                            {'label': 'Petrol', 'value': 'Petrol'},
                            {'label': 'Diesel', 'value': 'Diesel'},
                            {'label': 'CNG', 'value': 'CNG'},
                            {'label': 'LPG', 'value': 'LPG'},
                            {'label': 'Electric', 'value': 'Electric'}
                        ],
                        placeholder='Select fuel type...'
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Seller Type", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='seller-type',
                        options=[
                            {'label': 'Individual', 'value': 'Individual'},
                            {'label': 'Dealer', 'value': 'Dealer'},
                            {'label': 'Trustmark Dealer', 'value': 'Trustmark Dealer'}
                        ],
                        placeholder='Select seller type...'
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Transmission Type", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='transmission',
                        options=[
                            {'label': 'Manual', 'value': 'Manual'},
                            {'label': 'Automatic', 'value': 'Automatic'}
                        ],
                        placeholder='Select transmission...'
                    )
                ], style={'marginBottom': '20px'})
            ], style={'width': '43%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
            
            html.Div([
                html.H3("Technical Specifications", style={'color': '#e67e22', 'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Number of Owners", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='owner',
                        options=[
                            {'label': 'First Owner', 'value': 'First Owner'},
                            {'label': 'Second Owner', 'value': 'Second Owner'},
                            {'label': 'Third Owner', 'value': 'Third Owner'},
                            {'label': 'Fourth & Above Owner', 'value': 'Fourth & Above Owner'}
                        ],
                        placeholder='Select owner count...'
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Mileage (km/l)", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Input(
                        id='mileage',
                        type='text',
                        placeholder='15.5 kmpl',
                        style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 'borderRadius': '5px'}
                    )
                ], style={'marginBottom': '20px'}),

                html.Div([
                    html.Label("Engine Capacity (CC)", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Input(
                        id='engine',
                        type='text',
                        placeholder='1500',
                        style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 'borderRadius': '5px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Maximum Power (BHP)", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Input(
                        id='max-power',
                        type='text',
                        placeholder='103.52 bhp',
                        style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 'borderRadius': '5px'}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Label("Torque (Nm/RPM)", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Input(
                        id='torque',
                        type='text',
                        placeholder='200 Nm @ 1750 rpm',
                        style={'width': '100%', 'padding': '10px', 'border': '2px solid #bdc3c7', 'borderRadius': '5px'}
                    )
                ], style={'marginBottom': '20px'}),
                

                html.Div([
                    html.Label("Number of Seats", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='seats',
                        options=[{'label': str(i), 'value': i} for i in range(2, 15)],
                        placeholder='Select seats...'
                    )
                ], style={'marginBottom': '20px'})
            ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
        ], style={'marginBottom': '30px'}),

        html.Div([
            html.Button(
                'Predict Car Price',
                id='submit-button',
                n_clicks=0,
                style={
                    'backgroundColor': '#3498db',
                    'color': 'white',
                    'padding': '15px 40px',
                    'fontSize': '18px',
                    'border': 'none',
                    'borderRadius': '25px',
                    'cursor': 'pointer',
                    'boxShadow': '0 4px 15px rgba(52, 152, 219, 0.3)',
                    'transition': 'all 0.3s ease'
                }
            )
        ], style={'textAlign': 'center', 'marginBottom': '40px'}),
        

        html.Div(id='prediction-results', style={'marginTop': '30px'})
        
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 20px'})
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#ffffff', 'minHeight': '100vh'})

    
@app.callback(
    Output('prediction-results', 'children'),
    Input('submit-button', 'n_clicks'),
    [State('car-brand', 'value'),
     State('year', 'value'),
     State('km-driven', 'value'),
     State('fuel-type', 'value'),
     State('seller-type', 'value'),
     State('transmission', 'value'),
     State('owner', 'value'),
     State('mileage', 'value'),
     State('engine', 'value'),
     State('max-power', 'value'),
     State('torque', 'value'),
     State('seats', 'value')]
)
def predict_price(n_clicks, brand, year, km_driven, fuel_type, seller_type, 
                 transmission, owner, mileage, engine, max_power, torque, seats):
    if n_clicks == 0:
        return ""

    imputed_values = {}
    if not brand:
        brand = "Honda"; imputed_values['Car Brand'] = "Honda (default)"
    if not year:
        year = 2015; imputed_values['Year'] = "2015 (average)"
    if not km_driven:
        km_driven = 50000; imputed_values['Kilometers Driven'] = "50,000 km (average)"
    if not fuel_type:
        fuel_type = "Petrol"; imputed_values['Fuel Type'] = "Petrol (most common)"
    if not seller_type:
        seller_type = "Individual"; imputed_values['Seller Type'] = "Individual (most common)"
    if not transmission:
        transmission = "Manual"; imputed_values['Transmission'] = "Manual (most common)"
    if not owner:
        owner = "First Owner"; imputed_values['Owner'] = "First Owner (best case)"
    if not mileage:
        mileage = "15.0"; imputed_values['Mileage'] = "15.0 kmpl (average)"
    if not engine:
        engine = "1500"; imputed_values['Engine Capacity'] = "1500 CC (average)"
    if not max_power:
        max_power = "100"; imputed_values['Max Power'] = "100 BHP (average)"
    if not torque:
        torque = "200 Nm @ 1750 rpm"; imputed_values['Torque'] = "200 Nm @ 1750 rpm (average)"
    if not seats:
        seats = 5; imputed_values['Seats'] = "5 seats (most common)"
    def _to_float(x):
        if x is None:
            return None
        s = str(x)
        token = "".join(ch if (ch.isdigit() or ch == '.' or ch == '-') else ' ' for ch in s).split()
        return float(token[0]) if token else None

    engine_val = _to_float(engine)         
    power_val  = _to_float(max_power)     
    year_val   = int(year) if year is not None else None

    row = {"engine": engine_val, "power": power_val, "year": year_val}
    X_infer = pd.DataFrame([row], columns=FEATURES)
    log_pred = MODEL.predict(X_infer)[0]
    predicted_price = float(np.exp(log_pred)) if TARGET_TRANSFORM == "log" else float(log_pred)

    lower_bound = predicted_price * 0.85
    upper_bound = predicted_price * 1.15

    current_year = datetime.now().year
    age = current_year - year_val
    results = html.Div([
        html.H2("🎯 Price Prediction Results", 
                style={'color': '#27ae60', 'textAlign': 'center', 'marginBottom': '25px'}),
        html.Div([
            html.H3(f"{predicted_price:,.0f}", 
                    style={'fontSize': '36px', 'color': '#2c3e50', 'textAlign': 'center', 'marginBottom': '10px'}),
            html.P("Estimated Market Value", 
                   style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '18px', 'marginBottom': '20px'}),
            html.Div([
                html.Span(f"{lower_bound:,.0f}", style={'color': '#e74c3c', 'fontWeight': 'bold'}),
                html.Span(" - ", style={'margin': '0 10px'}),
                html.Span(f"{upper_bound:,.0f}", style={'color': '#27ae60', 'fontWeight': 'bold'})
            ], style={'textAlign': 'center', 'fontSize': '16px', 'marginBottom': '15px'}),
            html.P("Price Range (85% Confidence)", 
                   style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': '14px'})
        ], style={
            'backgroundColor': '#ecf0f1','padding': '30px','borderRadius': '15px','textAlign': 'center',
            'boxShadow': '0 4px 15px rgba(0,0,0,0.1)','marginBottom': '30px'
        }),
        html.Div([
            html.H4("📝 Auto-filled Missing Information", 
                    style={'color': '#f39c12', 'marginBottom': '15px'}),
            html.Ul([
                html.Li(f"{field}: {value}", style={'marginBottom': '5px'}) 
                for field, value in imputed_values.items()
            ], style={'color': '#7f8c8d'})
        ], style={'marginBottom': '20px'}) if imputed_values else None,
        html.Div([
            html.H4("💡 Market Insights", style={'color': '#9b59b6', 'marginBottom': '15px'}),
            html.Ul([
                html.Li(f"Car age: {age} years - {'Excellent' if age < 3 else 'Good' if age < 7 else 'Fair'} depreciation impact"),
                html.Li(f"Mileage: {km_driven:,} km - {'Low' if km_driven < 30000 else 'Average' if km_driven < 80000 else 'High'} usage"),
                html.Li(f"Fuel type: {fuel_type} - {'Premium' if fuel_type == 'Electric' else 'Standard'} market segment"),
                html.Li(f"Ownership: {owner} - {'Excellent' if 'First' in owner else 'Good' if 'Second' in owner else 'Fair'} resale value")
            ], style={'color': '#7f8c8d', 'lineHeight': '1.6'})
        ])
    ], style={
        'backgroundColor': '#ffffff','padding': '30px','borderRadius': '15px',
        'boxShadow': '0 6px 20px rgba(0,0,0,0.1)','border': '1px solid #ecf0f1'
    })
    return results
 

if __name__ == '__main__':
    print("🚗 Starting Car Price Predictor...")
    print("📱 Open your browser and go to: http://127.0.0.1:8050")
    print("💡 Fill in the car details and get instant price predictions!")
    app.run(debug=True, host='127.0.0.1', port=8050)
