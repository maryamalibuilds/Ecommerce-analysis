import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load dataset
products_data = pd.read_csv("ecommerce.csv")

# Feature engineering
products_data["discounted_price"] = products_data["price"] * (1 - products_data["discount"] / 100)
products_data["revenue"] = products_data["discounted_price"] * products_data["units_sold"]
products_data["value_score"] = products_data["rating"] / products_data["discounted_price"]

# Core analysis
best_seller = products_data.loc[products_data["units_sold"].idxmax()]
highest_revenue = products_data.loc[products_data["revenue"].idxmax()]
best_value = products_data.loc[products_data["value_score"].idxmax()]

print("Best selling product:")
print(best_seller)

print("\nHighest revenue product:")
print(highest_revenue)

print("\nBest value product:")
print(best_value)

category_summary = (
    products_data.groupby("category")[["units_sold", "revenue"]]
    .sum()
    .sort_values("revenue", ascending=False)
)
print("\nCategory performance:")
print(category_summary)

top_revenue_products = products_data.sort_values("revenue", ascending=False).head(5)
print("\nTop 5 products by revenue:")
print(top_revenue_products[["product", "category", "revenue"]])

correlation_data = products_data[["price", "discount", "rating", "units_sold", "revenue"]].corr()
print("\nCorrelation matrix:")
print(correlation_data)

# Save summary files
category_summary.to_csv("category_summary.csv")
top_revenue_products.to_csv("top_revenue_products.csv", index=False)
correlation_data.to_csv("correlation_matrix.csv")

# Visualizations
plt.figure(figsize=(8, 5))
category_summary["revenue"].plot(kind="bar")
plt.title("Revenue by Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("revenue_by_category.png")
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
products_data.sort_values("units_sold", ascending=False).plot(
    x="product", y="units_sold", kind="bar", legend=False
)
plt.title("Units Sold by Product")
plt.xlabel("Product")
plt.ylabel("Units Sold")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("units_sold_by_product.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(products_data["price"], products_data["units_sold"])
plt.title("Price vs Units Sold")
plt.xlabel("Price")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig("price_vs_units_sold.png")
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
plt.scatter(products_data["rating"], products_data["units_sold"])
plt.title("Rating vs Units Sold")
plt.xlabel("Rating")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig("rating_vs_units_sold.png")
plt.show()
plt.close()

# Prediction model
X = products_data[["price", "rating", "discount", "revenue"]]
y = products_data["units_sold"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

results = pd.DataFrame({
    "actual_units_sold": y_test.values,
    "predicted_units_sold": predictions.round(2)
})
results.to_csv("prediction_results.csv", index=False)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(y_test, predictions)
plt.title("Actual vs Predicted Units Sold")
plt.xlabel("Actual Units Sold")
plt.ylabel("Predicted Units Sold")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()
plt.close()

print("\nProject files generated successfully.")
