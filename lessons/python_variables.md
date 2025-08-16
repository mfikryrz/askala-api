# 🧠 Python Variables

## 📌 Variables

Variables are containers for storing data values.

---

## ✅ Creating Variables

Python has no command for declaring a variable. A variable is created the moment you first assign a value to it.

```python
x = 5
y = "John"
print(x)
print(y)
```

Variables do not need to be declared with any particular type, and can even change type after they have been set.

```python
x = 4       # x is of type int
x = "Sally" # x is now of type str
print(x)
```

---

## 🎯 Casting

If you want to specify the data type of a variable, this can be done with casting:

```python
x = str(3)    # x will be '3'
y = int(3)    # y will be 3
z = float(3)  # z will be 3.0
```

---

## 🔍 Get the Type

You can get the data type of a variable with the `type()` function:

```python
x = 5
y = "John"
print(type(x))
print(type(y))
```

You will learn more about data types and casting later in this tutorial.

---

## 🧵 Single or Double Quotes?

String variables can be declared either by using single or double quotes:

```python
x = "John"
# is the same as
x = 'John'
```

---

## 🆚 Case-Sensitive

Variable names are **case-sensitive**.

```python
a = 4
A = "Sally"
# A will not overwrite a
```

# 🐍 Python Variable Names Guide

A variable can have a short name (like `x` and `y`) or a more descriptive name (`age`, `carname`, `total_volume`).

## 📋 Rules for Python Variables

### ✅ **Valid Variable Rules**
- **Must start with a letter or underscore** (`_`)
- **Cannot start with a number**
- **Only alpha-numeric characters and underscores** (A-z, 0-9, and `_`)
- **Case-sensitive** (`age`, `Age`, and `AGE` are three different variables)
- **Cannot be Python keywords** (reserved words)

---

## ✅ Legal Variable Names

```python
myvar = "John"
my_var = "John"
_my_var = "John"
myVar = "John"
MYVAR = "John"
myvar2 = "John"
```

## ❌ Illegal Variable Names

```python
2myvar = "John"    # Cannot start with number
my-var = "John"    # Cannot contain hyphen
my var = "John"    # Cannot contain space
```

> **💡 Remember:** Variable names are **case-sensitive**!

---

## 🔤 Multi-Word Variable Names

Variable names with more than one word can be difficult to read. Here are several techniques to make them more readable:

### 🐫 **Camel Case**
Each word, except the first, starts with a capital letter:

```python
myVariableName = "John"
firstName = "Alice"
totalAmount = 100
```

### 🏛️ **Pascal Case**
Each word starts with a capital letter:

```python
MyVariableName = "John"
FirstName = "Alice"
TotalAmount = 100
```

### 🐍 **Snake Case** *(Recommended in Python)*
Each word is separated by an underscore character:

```python
my_variable_name = "John"
first_name = "Alice"
total_amount = 100
```

---

## 🎯 Best Practices

- **Use descriptive names** instead of single letters
- **Follow Python's PEP 8 style guide** (snake_case for variables)
- **Keep names concise but meaningful**
- **Use lowercase with underscores** for better readability

### Example of Good Variable Names:
```python
user_age = 25
car_brand = "Toyota"
total_sales_amount = 15000.50
is_valid_email = True
```