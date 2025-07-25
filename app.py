from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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
    growth_rate = 0.05  # 5% mock monthly growth
    allocations = []
    weighted_growth = 0

    for name, percent in zip(stock_name, stock_percent):
        weighted_growth += (percent / 100) * growth_rate
        allocations.append({"stock": name, "percent": percent})

    if round(total_percent, 2) != 100.0:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Total allocation must be 100%, got {total_percent:.2f}%",
            "portfolio": portfolio,
            "allocations": allocations,
            "stock_options": STOCK_OPTIONS
        })

    forecasted_value = round(portfolio * (1 + weighted_growth), 2)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "forecast": forecasted_value,
        "portfolio": portfolio,
        "allocations": allocations,
        "stock_options": STOCK_OPTIONS
    })
