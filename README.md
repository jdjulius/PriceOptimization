# Price Optimization Project

## Project Overview

This project aims to analyze and optimize pricing strategies using data-driven insights. The analysis is performed on sales data and visualized through various Python-based notebooks.

## Project Goals

The primary goals of this project are:

1. To analyze sales data and identify trends and patterns.
2. To compare store prices with competition prices and evaluate their impact on sales.
3. To visualize the effect of discounts on sales performance.
4. To provide actionable insights for optimizing pricing strategies.

## Folder Structure

- **data/raw/**

  - `Competition_Data.csv`: Contains raw sales data including pricing, quantities sold, and competition prices.

- **analysis/**
  - `DataAnalysis.ipynb`: Notebook for detailed data analysis, including data cleaning, transformation, and statistical summaries.
  - `VisualsationAnalysis.ipynb`: Notebook for visualizing trends, comparisons, and distributions in the sales data.
- `prediction_model.py`: Script to train a sales prediction model and save it as `model.pkl`.

## Data Dictionary

### Columns Description

- **Fiscal_Week_Id**: The fiscal week identifier.
- **Store_Id**: The store identifier.
- **Item_Id**: The item identifier.
- **Price**: The price of the item at our store.
- **Item_Quantity**: The quantity of the item sold.
- **Sales_Amount_No_Discount**: Sales amount without discount.
- **Sales_Amount**: Sales amount after discounts.
- **Competition_Price**: The price of the item at a competing store.

## Libraries Used

This project utilizes the following Python libraries:

- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib**: For creating static, interactive, and animated visualizations.
- **seaborn**: For statistical data visualization.

Ensure these libraries are installed before running the notebooks. You can install them using:

```bash
pip install pandas numpy matplotlib seaborn
```

## Configuration

To use this project, follow these configuration steps:

1. **Python Environment:**
   - Ensure Python 3.8 or higher is installed.
   - Create a virtual environment using the following command:
     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source venv/bin/activate
       ```

2. **Install Dependencies:**
   - Install the required libraries using:
     ```bash
     pip install pandas numpy matplotlib seaborn
     ```

3. **Data File:**
   - Ensure the raw data file `Competition_Data.csv` is located in the `data/raw/` directory.

4. **Jupyter Notebook:**
   - Install Jupyter Notebook if not already installed:
     ```bash
     pip install notebook
     ```
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```

5. **Run Notebooks:**
   - Open the notebooks in the `analysis/` folder.
   - Execute the cells sequentially to reproduce the analysis and visualizations.

## Notebooks Overview

### DataAnalysis.ipynb

- **Imports:**
  - `pandas`: For data manipulation and analysis.
  - `numpy`: For numerical computations.
- **Key Features:**
  - Data cleaning and preprocessing.
  - Statistical summaries and exploratory data analysis.

### VisualsationAnalysis.ipynb

- **Imports:**
  - `pandas`: For data manipulation.
  - `matplotlib.pyplot`: For creating plots.
  - `seaborn`: For advanced visualizations.
- **Key Features:**
  - Visualization of sales trends by fiscal week.
  - Comparison of store prices versus competition prices.
  - Analysis of the impact of discounts on sales.
  - Distribution analysis of item quantities and prices.

## Key Insights

This project provides the following insights:

- **Sales Trends:** Weekly sales trends help identify peak and low-performing weeks.
- **Price Comparison:** Understanding the relationship between store prices and competition prices.
- **Discount Impact:** Evaluating how discounts influence sales volume and revenue.
- **Product Performance:** Identifying top-selling products and their pricing strategies.

## Future Work

Potential future enhancements to the project include:

- Incorporating additional datasets, such as customer demographics or marketing campaigns.
- Developing predictive models to forecast sales based on pricing strategies.
- Automating the analysis pipeline for real-time insights.
- Expanding visualizations to include geographic trends and customer segmentation.

## Prediction Model

The repository includes `prediction_model.py`, which trains a Random Forest model
to predict `Sales_Amount`. After installing the required libraries, run:

```bash
python prediction_model.py
```

This command reads `data/raw/Competition_Data.csv`, trains the model, and saves
the resulting pipeline to `model.pkl`.

## API

Serve the trained model through a FastAPI application.

1. Install the API dependencies:
   ```bash
   pip install fastapi uvicorn joblib pandas scikit-learn
   ```
2. Ensure `model.pkl` exists by running:
   ```bash
   python prediction_model.py
   ```
3. Start the API with:
   ```bash
   uvicorn api:app --reload
   ```
   The server will be available at `http://127.0.0.1:8000`.
4. Send a POST request to `/predict` with JSON containing the following fields:
   - `Store_ID`
   - `Item_ID`
   - `Price`
   - `Item_Quantity`
   - `Competition_Price`

## Contact

For questions or collaboration, please contact Julio.

## Contributors

This project was developed by Julio. Contributions and collaborations are welcome. Please reach out via the contact information provided.
