from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.pipline.prediction_pipeline import PredictionPipeline
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define your stock options
STOCK_OPTIONS = ["Apple", "Google", "Microsoft", "Amazon", "Tesla", "Nvidia", "Meta"]
STOCK_MAPPING = {"Apple":'AAPL','Google':'GOOGL',"Microsoft":'MSFT', "Amazon":'AMZN', "Tesla":'TSLA', "Nvidia":'NVDA', "Meta":'META'}

@app.get("/", response_class=HTMLResponse)
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "stock_options": STOCK_OPTIONS})

@app.post("/forecast", response_class=HTMLResponse)
async def forecast(
    request: Request,
    portfolio: float = Form(...),
    stock_name: list[str] = Form(...),
    stock_percent: list[float] = Form(...)
):
    total_percent = sum(stock_percent)

    allocations = []
    for name, percent in zip(stock_name, stock_percent):
        allocations.append({"stock": name, "percent": percent})

    if round(total_percent, 2) != 100.0:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Total allocation must be 100%, got {total_percent:.2f}%",
            "portfolio": portfolio,
            "allocations":allocations,
            "stock_options": STOCK_OPTIONS
        })
    
    stock_options = [STOCK_MAPPING[i] for i in stock_name]
    weights = [i/100 for i in stock_percent]

    pipeline = PredictionPipeline(portfolio,stock_options,weights)
    forecast_df = pipeline.run_pipeline()

     # Daily percent returns
    returns = forecast_df.pct_change().dropna()

    # Portfolio returns (dot product of weights and daily returns)
    portfolio_returns = returns.dot(weights)

    # Cumulative portfolio value (e.g., starting at $1000)
    forecasted_value = (1 + portfolio_returns).cumprod() * portfolio
    forecasted_value = round(forecasted_value.iloc[-1],2)

    # Daily metrics
    #mean_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()

    # Value at Risk (95% confidence)
    #VaR_95 = -np.percentile(portfolio_returns, 5)

    if volatility > 0.02:
        risk_message = "Alert: High portfolio volatility. Consider diversification."
    else:
        risk_message = "Stable portfolio: Volatility is within a manageable range."
    

    return templates.TemplateResponse("index.html", {
        "request": request,
        "forecast": forecasted_value,
        "portfolio": portfolio,
        "stock_options": STOCK_OPTIONS,
         "allocations":allocations,
        "risk_message": risk_message,
         "forecast_data_json": forecast_df.to_json(),  # Send as JSON

    })
