# car_analysis.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Đặt seed cho numpy để đảm bảo tái lập được kết quả
np.random.seed(0)

# Số lượng mẫu dữ liệu
n_samples = 1000

# Tạo dữ liệu
data = {
    'Brand': np.random.choice(['Toyota', 'Honda', 'BMW', 'Mercedes', 'Ford'], n_samples),
    'Model': np.random.choice(['Model A', 'Model B', 'Model C', 'Model D'], n_samples),
    'Year': np.random.randint(2000, 2024, n_samples),
    'Engine_Size': np.round(np.random.uniform(1.0, 5.0, n_samples), 2),
    'Horsepower': np.random.randint(100, 400, n_samples),
    'MPG_City': np.round(np.random.uniform(10, 30, n_samples), 2),
    'MPG_Highway': np.round(np.random.uniform(20, 40, n_samples), 2),
    'Price': np.round(np.random.uniform(20000, 100000, n_samples), 2),
    'Mileage': np.random.randint(0, 200000, n_samples)
}

# Tạo DataFrame từ dữ liệu
df = pd.DataFrame(data)

# Hiển thị 5 dòng đầu tiên của DataFrame
print(df.head())

# Thống kê cơ bản
print(df.describe())

# Tương quan giữa giá và năm sản xuất
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year', y='Price', data=df, hue='Brand')
plt.title('Giá xe theo năm sản xuất')
plt.xlabel('Năm sản xuất')
plt.ylabel('Giá (USD)')
plt.show()

# Tương quan giữa kích thước động cơ và sức ngựa
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Engine_Size', y='Horsepower', data=df, hue='Brand')
plt.title('Sức ngựa theo kích thước động cơ')
plt.xlabel('Kích thước động cơ (lít)')
plt.ylabel('Sức ngựa (HP)')
plt.show()

# Tính trung bình giá xe theo từng hãng
avg_price_per_brand = df.groupby('Brand')['Price'].mean().sort_values()

# Trực quan hóa trung bình giá xe theo từng hãng
plt.figure(figsize=(12, 6))
sns.barplot(x=avg_price_per_brand.index, y=avg_price_per_brand.values)
plt.title('Trung bình giá xe theo hãng')
plt.xlabel('Hãng xe')
plt.ylabel('Giá trung bình (USD)')
plt.show()

# Phân tích tỷ lệ tiêu thụ nhiên liệu trong thành phố và trên xa lộ
df['MPG_Difference'] = df['MPG_Highway'] - df['MPG_City']

plt.figure(figsize=(10, 6))
sns.histplot(df['MPG_Difference'], bins=30, kde=True)
plt.title('Phân bố chênh lệch MPG giữa thành phố và xa lộ')
plt.xlabel('Chênh lệch MPG')
plt.ylabel('Số lượng xe')
plt.show()

# Ma trận tương quan
correlation_matrix = df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Ma trận tương quan giữa các biến')
plt.show()

# Biến độc lập
X = df[['Year', 'Engine_Size', 'Horsepower', 'Mileage']]
# Thêm cột hằng số
X = sm.add_constant(X)
# Biến phụ thuộc
y = df['Price']

# Tạo mô hình hồi quy tuyến tính
model = sm.OLS(y, X).fit()
# Tóm tắt kết quả
print(model.summary())

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X = df[['Year', 'Engine_Size', 'Horsepower', 'Mileage']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tạo mô hình Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Dự đoán giá xe
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Trực quan hóa kết quả dự đoán
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Giá thực tế')
plt.ylabel('Giá dự đoán')
plt.title('Giá thực tế vs Giá dự đoán')
plt.show()
