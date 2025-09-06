import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import google.generativeai as genai
import kagglehub
import os
import io

# --- Optimized Helper Functions with Caching ---

@st.cache_data
def load_and_clean_data(file_path: str, date_column: str) -> pd.DataFrame:
    """Loads and cleans the dataset from a local file path."""
    df = pd.read_csv(file_path)
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
    df.dropna(subset=[date_column], inplace=True)
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

@st.cache_resource
def create_and_run_forecast(df: pd.DataFrame) -> tuple:
    """Creates and returns a sales forecast and the model."""
    prophet_df = df[['Order Date', 'Sales']].rename(columns={'Order Date': 'ds', 'Sales': 'y'})
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return model, forecast

def perform_eda_and_visualize(df: pd.DataFrame) -> tuple:
    sales_by_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
    sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    return sales_by_category, sales_by_region

def get_ai_summary(forecast_df: pd.DataFrame, historical_df: pd.DataFrame, sales_by_category: pd.Series, sales_by_region: pd.Series) -> str:
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    last_historical_date = historical_df['Order Date'].max()
    next_year_forecast = forecast_df[forecast_df['ds'] > last_historical_date]
    total_forecasted_sales = next_year_forecast['yhat'].sum()
    last_full_year = last_historical_date.year - 1
    last_year_sales = historical_df[historical_df['Order Date'].dt.year == last_full_year]['Sales'].sum()
    yoy_growth = ((total_forecasted_sales - last_year_sales) / last_year_sales) * 100 if last_year_sales > 0 else 0
    top_category = sales_by_category.index[0]
    top_region = sales_by_region.index[0]
    prompt = f"""
    You are a professional business analyst summarizing a sales report for an executive.
    Based on the following data, generate a concise 3-paragraph summary:
    Key Performance Indicators:
    - Total Forecasted Sales for the Next 12 Months: ${total_forecasted_sales:,.2f}
    - Predicted Year-over-Year (YoY) Growth: {yoy_growth:.2f}%
    - Top-Performing Product Category: {top_category}
    - Top-Performing Sales Region: {top_region}
    """
    response = model.generate_content(prompt)
    return response.text

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🤖 AI-Powered Sales Forecaster & Insights Engine")
st.markdown("This application automatically downloads the required sales data, forecasts future trends, and generates a business summary using Generative AI.")

st.sidebar.header("⚙️ Configuration")
api_key_input = st.sidebar.text_input(
    "Enter your Google Gemini API Key", 
    type="password",
    help="Get your free API key from Google AI Studio to run the analysis."
)

if api_key_input:
    try:
        genai.configure(api_key=api_key_input)
        st.sidebar.success("✅ Gemini API Key Loaded Successfully!")

        with st.spinner('Downloading dataset from Kaggle...'):
            # Download latest version of the dataset
            path = kagglehub.dataset_download("rohitsahoo/sales-forecasting")
            st.sidebar.info(f"Dataset downloaded to: {path}")
            # Construct the full path to the train.csv file
            file_path = os.path.join(path, "train.csv")

        with st.spinner('Processing data... This may take a minute.'):
            cleaned_df = load_and_clean_data(file_path, 'Order Date')
            prophet_model, forecast = create_and_run_forecast(cleaned_df)
        
        st.sidebar.success("Data processed successfully!")
        st.header("📊 Business Intelligence Dashboard")
        category_sales, region_sales = perform_eda_and_visualize(cleaned_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sales by Region")
            fig, ax = plt.subplots()
            sns.barplot(x=region_sales.index, y=region_sales.values, ax=ax, palette='plasma')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        with col2:
            st.subheader("Sales by Category")
            fig, ax = plt.subplots()
            sns.barplot(x=category_sales.index, y=category_sales.values, ax=ax, palette='viridis')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        st.header("📈 Sales Forecast for the Next 12 Months")
        fig2 = prophet_model.plot(forecast)
        st.pyplot(fig2)
        
        st.header("📝 AI-Generated Business Summary")
        with st.spinner("Generating AI summary..."):
            summary = get_ai_summary(forecast, cleaned_df, category_sales, region_sales)
            st.markdown(summary)

    except Exception as e:
        st.sidebar.error(f"An error occurred: {e}")
        st.error(f"An error occurred during processing. Please check the sidebar for details.")

else:
    st.warning("Please enter your Google Gemini API Key in the sidebar to begin.")

