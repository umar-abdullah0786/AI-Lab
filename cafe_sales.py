import numpy as np

units = np.array([[30, 20, 10],   # rows: 4 days
                  [25, 30, 15],   # cols: Tea, Coffee, Cake
                  [40, 25, 20],
                  [35, 15, 25]])
price = np.array([100, 150, 200])

# Broadcast price across every row: revenue per item per day
revenue = units * price

# Aggregations
revenue_per_day = revenue.sum(axis=1)      # collapse columns -> one per day
units_per_item  = units.sum(axis=0)        # collapse rows    -> one per item

print("Revenue per day:", revenue_per_day)
print("Units sold per item:", units_per_item)

# Best day by total revenue
print("Best day index:", np.argmax(revenue_per_day))

# Coffee (column index 1), last two days
print("Coffee, last two days:", units[-2:, 1])