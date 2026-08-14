import pandas as pd
from pathlib import Path

base = Path(__file__).parent
orders_fp = base / 'Day9_Orders.csv'
customers_fp = base / 'Day9_Customers.csv'
products_fp = base / 'Day9_Products.csv'
output_fp = base / 'processed_ecommerce_dataset.csv'

# Load datasets
orders = pd.read_csv(orders_fp, parse_dates=['Order_Date'])
customers = pd.read_csv(customers_fp)
products = pd.read_csv(products_fp)

# Demonstrate concat: split orders into two parts and rejoin
mid = len(orders) // 2
orders_part1 = orders.iloc[:mid].copy()
orders_part2 = orders.iloc[mid:].copy()
orders_rejoined = pd.concat([orders_part1, orders_part2], ignore_index=True)

# Merge datasets
df = orders_rejoined.merge(customers, on='Customer_ID', how='left')
df = df.merge(products, on='Product_ID', how='left')

# Create computed columns
# Ensure Unit_Price and Quantity are numeric
df['Unit_Price'] = pd.to_numeric(df['Unit_Price'], errors='coerce')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')

# Total price using apply (demonstration)
def compute_total(row):
    return row['Quantity'] * row['Unit_Price']

df['Total_Price'] = df.apply(compute_total, axis=1)

# Date/time operations
# Order_Date was parsed earlier; extract month, day, day name
if not pd.api.types.is_datetime64_any_dtype(df['Order_Date']):
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')

df['Order_Month'] = df['Order_Date'].dt.month_name()
df['Order_Day'] = df['Order_Date'].dt.day
df['Order_Weekday'] = df['Order_Date'].dt.day_name()

# Organize final DataFrame
cols = [
    'Order_ID','Order_Date','Order_Month','Order_Day','Order_Weekday',
    'Customer_ID','Customer_Name','City','Region','Membership_Type',
    'Product_ID','Product_Name','Category','Brand',
    'Quantity','Unit_Price','Total_Price',
    'Payment_Method','Order_Status'
]
final_df = df[cols].sort_values(by='Order_Date').reset_index(drop=True)

# Save to CSV
final_df.to_csv(output_fp, index=False)
print(f"Processed dataset saved to: {output_fp}")
