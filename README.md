# 🛒 Simple E-commerce Store

A full-stack **E-commerce Web Application** built with **Django**, **HTML**, **CSS**, and **JavaScript**. The application provides a seamless online shopping experience where users can browse products, view product details, add items to a shopping cart, place orders, and manage their accounts.

---

## 📖 About the Project

This project is a basic e-commerce platform developed as a full-stack web application. It demonstrates essential e-commerce functionalities including user authentication, product management, shopping cart operations, and order processing using Django's robust backend framework.

---

## ✨ Features

### 👤 User Authentication

* User Registration
* Secure Login & Logout
* Password Encryption using Django Authentication
* User Session Management

### 🛍️ Product Management

* Display all available products
* Product detail page
* Product images
* Product descriptions
* Product pricing

### 🛒 Shopping Cart

* Add products to cart
* Remove products from cart
* Update product quantities
* View cart summary
* Calculate total price

### 📦 Order Processing

* Checkout page
* Place orders
* Store order details in the database
* Order confirmation

### 📱 Responsive User Interface

* Responsive layout
* Mobile-friendly design
* Clean and modern interface
* Easy navigation

### 🔐 Admin Dashboard

* Manage products
* Manage users
* Manage customer orders
* Update product information
* View order history

---

## 🛠️ Tech Stack

### Frontend

* HTML5
* CSS3
* JavaScript
* Bootstrap 5 (Optional)

### Backend

* Python
* Django

### Database

* SQLite (Development)
* Compatible with PostgreSQL/MySQL

---

## 📂 Project Structure

```text
ecommerce-store/
│
├── manage.py
├── requirements.txt
├── README.md
├── db.sqlite3
│
├── ecommerce/
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── store/
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   ├── forms.py
│   ├── admin.py
│   └── migrations/
│
├── templates/
│
├── static/
│
├── media/
│
└── screenshots/
```

---

## 💾 Database Models

The application includes the following database models:

* User
* Product
* Cart
* Order
* Order Item

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ecommerce-store.git
```

### 2. Navigate to the Project Directory

```bash
cd ecommerce-store
```

### 3. Create a Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Apply Database Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Create an Admin User

```bash
python manage.py createsuperuser
```

### 7. Start the Development Server

```bash
python manage.py runserver
```

Open your browser and visit:

```
http://127.0.0.1:8000/
```

---

## 🔑 Admin Panel

Access the Django admin dashboard:

```
http://127.0.0.1:8000/admin/
```

Log in using the superuser credentials you created.

---

## 📸 Screenshots

Add screenshots of your application inside the `screenshots/` folder.

Suggested screenshots:

* Home Page
* Product Listing
* Product Details
* Shopping Cart
* Checkout Page
* Login Page
* Registration Page
* Admin Dashboard

---

## 🚀 Future Improvements

* Product Categories
* Product Search
* Product Filters
* Wishlist
* Product Reviews & Ratings
* Online Payment Integration (Stripe/Razorpay)
* Order Tracking
* Order History
* Inventory Management
* Email Notifications
* Discount Coupons
* Responsive Admin Dashboard
* REST API with Django REST Framework

---

## 🧪 Testing Checklist

Verify the following features:

* User registration
* User login and logout
* Browse products
* View product details
* Add products to cart
* Update cart quantity
* Remove products from cart
* Place an order
* View order confirmation
* Access admin dashboard
* Manage products and orders

---

## 📚 Learning Outcomes

This project demonstrates practical knowledge of:

* Django Framework
* Django Authentication
* CRUD Operations
* Database Design
* Model-View-Template (MVT) Architecture
* HTML5 & CSS3
* JavaScript
* Bootstrap
* Responsive Web Design
* Git & GitHub
* Full-Stack Web Development

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## 📄 License

This project is intended for educational and learning purposes.

---

## 👨‍💻 Author

**Your Name**

* GitHub: https://github.com/your-username
* LinkedIn: https://linkedin.com/in/your-profile

---

## ⭐ Acknowledgements

This project was developed as part of the **CodeAlpha Full Stack Development Internship** to demonstrate the implementation of a basic e-commerce web application using Django.

If you found this project helpful, please consider giving it a ⭐ on GitHub!
