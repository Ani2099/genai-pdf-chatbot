import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import google.generativeai as genai

df=pd.read_csv('google.csv')

# Convert 'High' to a binary target variable using the mean
mean_high = df['High'].mean()
df['High_Category'] = df['High'].apply(lambda x: 'High' if x > mean_high else 'Low')

x=df[['Open', 'Low', 'Close', 'Adj Close', 'Volume']]
y=df['High_Category'] # Use the new categorical target

lg=LogisticRegression(max_iter=1000) # Increase max_iter
lg.fit(x,y)
def get_stock_input():
  print("\n Stock details:")
  a=float(input("Open:"))
  b=float(input("Low:"))
  c=float(input("Close:"))
  d=float(input("Adj Close:"))
  e=float(input("Volume:"))

  return[[a,b,c,d,e]]
stock_data = get_stock_input()
prediction=lg.predict(stock_data)[0]
# label_reverse={0:"Low",1:"High"} # No longer needed
predicted_class=prediction # Prediction is already the category
print(f"\n prediction: stock {predicted_class}")
os.environ["GOOGLE_API_KEY"]=" "
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
def get_stock_suggestion(condition):
  lg=genai.GenerativeModel("gemini-2.5-flash")
  prompt=f"""
  Suggest similar 5 stocks that are considered {condition} (potentially undervalued), considering the following criteria for value stock analysis:
  - Similar domain/industry

Similar market capitalization

-Similar domain/industry
--
--Similar market capitalization
--
--Favorable Price-to-Earnings (P/E) ratio (lower than industry average)
--
--Low Price-to-Book (P/B) ratio
--
--Healthy Dividend Yield (if applicable)
--
--Strong Free Cash Flow generation
--
--Stable or growing revenue and earnings
--
--Low debt-to-equity ratio
--
--Positive future growth prospects
--
--Strong management team
--
--Return on Equity (ROE)
--
--Return on Assets (ROA)
--
--Gross Profit Margin
--
--Operating Profit Margin
--
--Net Profit Margin
--
--Current Ratio
--
--Quick Ratio
--
--Inventory Turnover
--
--Accounts Receivable Turnover
--
--Price-to-Sales (P/S) ratio
--
--Enterprise Value to EBITDA (EV/EBITDA)
--
--PEG Ratio (Price/Earnings to Growth)
--
--Analyst recommendations and price targets
--
--Competitive Moat (sustainable competitive advantage)
--
--Regulatory and legal environment
--
--Geopolitical risks
--
--Industry trends and outlook
--
--Shareholder friendliness (buybacks, dividends)
--
--Insider ownership and activity
--
--Price/Cash Flow ratio (P/CF)
--
--Price/Free Cash Flow ratio (P/FCF)
--
--Dividend Payout Ratio
--
--Earnings Yield (inverse of P/E)
--
--Book Value per Share (BVPS)
--
--Tangible Book Value per Share (TBVPS)
--
--Enterprise Value (EV)
--
--EV/Revenue ratio
--
--EV/EBIT ratio
--
--EBIT Margin
--
--EBITDA Margin
--
--Pre-Tax Profit Margin
--
--Asset Turnover Ratio
--
--Fixed Asset Turnover
--
--Cash Conversion Cycle (CCC)
--
--Operating Cash Flow to Net Income ratio
--
--Free Cash Flow Margin
--
--Return on Capital Employed (ROCE)
--
--Return on Invested Capital (ROIC)
--
--Economic Value Added (EVA)
--
--Revenue Growth (YoY and 3Y CAGR)
--
--EPS Growth (YoY and 3Y CAGR)
--
--Free Cash Flow Growth
--
--Operating Income Growth
--
--Book Value Growth
--
--Dividend Growth Rate
--
--Capex Growth Trend
--
--R&D Expense Growth
--
--Market Share Trend
--
--Interest Coverage Ratio
--
--Debt/EBITDA ratio
--
--Debt/Free Cash Flow ratio
--
--Total Liabilities/Total Assets
--
--Cash Ratio
--
--Working Capital Ratio
--
--Altman Z-Score (financial stability)
--
--Piotroski F-Score (financial strength)
--
--Beta (systematic risk)
--
--Volatility (standard deviation of returns)
--
--Value-at-Risk (VaR)
--
--Days Sales Outstanding (DSO)
--
--Days Inventory Outstanding (DIO)
--
--Days Payables Outstanding (DPO)
--
--Operating Leverage
--
--R&D-to-Revenue Ratio
--
--SG&A-to-Revenue Ratio
--
--Capital Expenditure to Revenue Ratio
--
--Employee Productivity (Revenue per Employee)
--
--Business Model Resilience
--
--Product/Service Diversification
--
--Brand Strength / Market Position
--
--Innovation Pipeline / Patents / R&D intensity
--
--Customer Retention Rate
--
--Management Tenure and Track Record
--
--Corporate Governance Quality
--
--Transparency & Reporting Standards
--
--Sustainability / ESG Initiatives
--
--Regulatory Exposure
--
--Supply Chain Robustness
--
--Cybersecurity Readiness
--
--Litigation Risk
--
--Geographic Revenue Diversification
--
--Commodity Price Exposure
--
--M&A Activity / Strategic Partnerships
--
--Relative Strength Index (RSI)
--
--Moving Average Convergence Divergence (MACD)
--
--50-day / 200-day Moving Average Trend
--
--Price Momentum (1M / 3M / 6M returns)
--
--Short Interest Ratio
--
--Institutional Ownership Percentage
--
--Insider Transactions (buy/sell trend)
--
--Analyst Recommendation Trend (Upgrades/Downgrades)
--
--Price Target Upside (%)
--
--Volatility Index Correlation (VIX sensitivity)
--
--ESG Composite Score
--
--Carbon Emissions Intensity
--
--Renewable Energy Usage %
--
--Board Diversity
--
--Employee Satisfaction / Retention Rate
--
--Community & Philanthropy Rating  """
  response=lg.generate_content(prompt)
  return response.text
stock_suggestions = get_stock_suggestion(predicted_class.lower())
print("\n🍽️ Suggested stocks:\n")
print(stock_suggestions)

