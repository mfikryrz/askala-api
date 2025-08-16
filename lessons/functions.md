# 🔧 Python Functions Guide

## 🎯 What is a Function?

A **function** is a **block of code that only runs when it is called**.

- You can **pass data** (parameters) into a function
- A function can **return data** as a result
- Functions help organize code and avoid repetition

---

## 🏗️ Creating a Function

In Python, a function is defined using the `def` keyword:

```python
def my_function():
    print("Hello from a function")
```

## 📞 Calling a Function

To call a function, use the **function name followed by parentheses**:

```python
def my_function():
    print("Hello from a function")

my_function()  # Output: Hello from a function
```

---

## 📥 Arguments & Parameters

### 🎯 **Basic Arguments**
Information can be passed into functions as **arguments**:

```python
def my_function(fname):
    print(fname + " Refsnes")

my_function("Emil")    # Output: Emil Refsnes
my_function("Tobias")  # Output: Tobias Refsnes
my_function("Linus")   # Output: Linus Refsnes
```

### 📋 **Parameters vs Arguments**
| Term | Definition | Example |
|------|------------|---------|
| **Parameter** | Variable in function definition | `def func(name):` ← `name` is parameter |
| **Argument** | Value passed when calling function | `func("John")` ← `"John"` is argument |

> 💡 **Note:** Arguments are often shortened to **args** in Python documentation.

---

## 🔢 Number of Arguments

### ✅ **Correct Number:**
```python
def my_function(fname, lname):
    print(fname + " " + lname)

my_function("Emil", "Refsnes")  # Output: Emil Refsnes
```

### ❌ **Incorrect Number (will cause error):**
```python
def my_function(fname, lname):
    print(fname + " " + lname)

my_function("Emil")  # TypeError: missing 1 required positional argument
```

---

## 🌟 Advanced Argument Types

### 🎒 **Arbitrary Arguments (*args)**
When you don't know how many arguments will be passed, use `*`:

```python
def my_function(*kids):
    print("The youngest child is " + kids[2])

my_function("Emil", "Tobias", "Linus")  # Output: The youngest child is Linus
```

> 💡 **Note:** The function receives a **tuple** of arguments.

### 🗝️ **Keyword Arguments**
Send arguments with `key = value` syntax (order doesn't matter):

```python
def my_function(child3, child2, child1):
    print("The youngest child is " + child3)

my_function(child1="Emil", child2="Tobias", child3="Linus")
# Output: The youngest child is Linus
```

### 🎒🗝️ **Arbitrary Keyword Arguments (**kwargs)**
When you don't know how many keyword arguments will be passed, use `**`:

```python
def my_function(**kid):
    print("His last name is " + kid["lname"])

my_function(fname="Tobias", lname="Refsnes")
# Output: His last name is Refsnes
```

> 💡 **Note:** The function receives a **dictionary** of arguments.

---

## 🔧 Default Parameter Values

Set default values for parameters:

```python
def my_function(country="Norway"):
    print("I am from " + country)

my_function("Sweden")  # Output: I am from Sweden
my_function("India")   # Output: I am from India
my_function()          # Output: I am from Norway (default)
my_function("Brazil")  # Output: I am from Brazil
```

---

## 📦 Passing Different Data Types

You can pass **any data type** to a function:

```python
def my_function(food):
    for x in food:
        print(x)

fruits = ["apple", "banana", "cherry"]
my_function(fruits)
```

**Output:**
```
apple
banana
cherry
```

---

## 🔄 Return Values

Use the `return` statement to return a value:

```python
def my_function(x):
    return 5 * x

print(my_function(3))  # Output: 15
print(my_function(5))  # Output: 25
print(my_function(9))  # Output: 45
```

---

## 🚫 The Pass Statement

Function definitions cannot be empty. Use `pass` to avoid errors:

```python
def myfunction():
    pass  # Empty function, no error
```

---

## 🎯 Argument Restrictions

### 📍 **Positional-Only Arguments**
Use `, /` to specify positional-only arguments:

```python
def my_function(x, /):
    print(x)

my_function(3)        # ✅ Works
# my_function(x=3)    # ❌ Error! Cannot use keyword
```

### 🗝️ **Keyword-Only Arguments**
Use `*, ` to specify keyword-only arguments:

```python
def my_function(*, x):
    print(x)

my_function(x=3)     # ✅ Works
# my_function(3)     # ❌ Error! Must use keyword
```

### 🎭 **Combined Restrictions**
```python
def my_function(a, b, /, *, c, d):
    print(a + b + c + d)

# a, b are positional-only
# c, d are keyword-only
my_function(5, 6, c=7, d=8)  # Output: 26
```

---

## 🔄 Recursion

**Recursion** means a function calls itself:

```python
def tri_recursion(k):
    if k > 0:
        result = k + tri_recursion(k - 1)
        print(result)
    else:
        result = 0
    return result

print("Recursion Example Results:")
tri_recursion(6)
```

**Output:**
```
Recursion Example Results:
1
3
6
10
15
21
```

> ⚠️ **Warning:** Be careful with recursion to avoid infinite loops and excessive memory usage!

---

## 📊 Function Types Summary

| Function Type | Syntax | Purpose |
|---------------|--------|---------|
| **Basic** | `def func():` | Simple function, no parameters |
| **With Parameters** | `def func(a, b):` | Fixed number of parameters |
| **Default Values** | `def func(a="default"):` | Optional parameters with defaults |
| **Variable Args** | `def func(*args):` | Unknown number of positional args |
| **Variable Kwargs** | `def func(**kwargs):` | Unknown number of keyword args |
| **Positional-Only** | `def func(a, /):` | Must use positional arguments |
| **Keyword-Only** | `def func(*, a):` | Must use keyword arguments |
| **Return Value** | `def func(): return x` | Function returns a value |

---

## 💡 Best Practices

1. **Use descriptive function names** (`calculate_area` vs `calc`)
2. **Add docstrings** to document your functions
3. **Keep functions small** and focused on one task
4. **Use default parameters** for optional functionality
5. **Return values** instead of just printing
6. **Handle edge cases** and validate inputs

### 🎯 **Example with Best Practices:**
```python
def calculate_rectangle_area(length, width=1):
    """
    Calculate the area of a rectangle.
    
    Args:
        length (float): The length of the rectangle
        width (float): The width of the rectangle (default: 1)
    
    Returns:
        float: The area of the rectangle
    """
    if length <= 0 or width <= 0:
        raise ValueError("Length and width must be positive")
    
    return length * width

# Usage
area = calculate_rectangle_area(5, 3)
print(f"Area: {area}")  # Output: Area: 15
```