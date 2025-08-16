# ⚖️ Python If...Else Guide

## 🧮 Logical Conditions

Python supports the usual logical conditions from mathematics:

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | **Equals** | `a == b` |
| `!=` | **Not Equals** | `a != b` |
| `<` | **Less than** | `a < b` |
| `<=` | **Less than or equal to** | `a <= b` |
| `>` | **Greater than** | `a > b` |
| `>=` | **Greater than or equal to** | `a >= b` |

These conditions are most commonly used in **"if statements"** and **loops**.

---

## 🔀 Basic If Statement

An **if statement** is written using the `if` keyword:

```python
# If statement
a = 33
b = 200
if b > a:
    print("b is greater than a")
```

In this example, since `a` is 33 and `b` is 200, we know that 200 is greater than 33, so it prints **"b is greater than a"**.

---

## 📏 Indentation

⚠️ **Important:** Python relies on **indentation** (whitespace at the beginning of a line) to define scope in code. Other programming languages often use curly-brackets `{}` for this purpose.

### ❌ **Incorrect (will cause an error):**
```python
a = 33
b = 200
if b > a:
print("b is greater than a")  # IndentationError!
```

### ✅ **Correct:**
```python
a = 33
b = 200
if b > a:
    print("b is greater than a")  # Properly indented
```

---

## 🔄 Elif Statement

The `elif` keyword means **"if the previous conditions were not true, then try this condition"**.

```python
a = 33
b = 33
if b > a:
    print("b is greater than a")
elif a == b:
    print("a and b are equal")
```

Since `a` equals `b`, the first condition is false, but the `elif` condition is true, so it prints **"a and b are equal"**.

---

## 🎯 Else Statement

The `else` keyword catches anything not caught by the preceding conditions:

```python
a = 200
b = 33
if b > a:
    print("b is greater than a")
elif a == b:
    print("a and b are equal")
else:
    print("a is greater than b")
```

Since `a` is greater than `b`, both previous conditions are false, so it executes the `else` block.

### 🔗 **Else without Elif:**
```python
a = 200
b = 33
if b > a:
    print("b is greater than a")
else:
    print("b is not greater than a")
```

---

## ⚡ Short Hand If

For **one statement**, you can put it on the same line as the if statement:

```python
# One line if statement
if a > b: print("a is greater than b")
```

---

## 🎭 Short Hand If...Else (Ternary Operator)

For **one statement each** for if and else, you can put it all on one line:

```python
# One line if else statement
a = 2
b = 330
print("A") if a > b else print("B")
```

This is known as **Ternary Operators** or **Conditional Expressions**.

### 🔢 **Multiple Conditions:**
```python
# One line if else with 3 conditions
a = 330
b = 330
print("A") if a > b else print("=") if a == b else print("B")
```

---

## 🔗 Logical Operators

### 🤝 **And Operator**
The `and` keyword combines conditional statements (both must be true):

```python
# Test if a > b AND c > a
a = 200
b = 33
c = 500
if a > b and c > a:
    print("Both conditions are True")
```

### 🔀 **Or Operator**
The `or` keyword combines conditional statements (at least one must be true):

```python
# Test if a > b OR a > c
a = 200
b = 33
c = 500
if a > b or a > c:
    print("At least one of the conditions is True")
```

### 🚫 **Not Operator**
The `not` keyword reverses the result of the conditional statement:

```python
# Test if a is NOT greater than b
a = 33
b = 200
if not a > b:
    print("a is NOT greater than b")
```

---

## 🪆 Nested If Statements

You can have `if` statements inside `if` statements:

```python
x = 41
if x > 10:
    print("Above ten,")
    if x > 20:
        print("and also above 20!")
    else:
        print("but not above 20.")
```

---

## 🚫 The Pass Statement

`if` statements cannot be empty. If you need an empty `if` statement, use the `pass` statement:

```python
a = 33
b = 200
if b > a:
    pass  # Do nothing, but avoid syntax error
```

> **💡 Pro Tip:** Use `pass` as a placeholder when you're planning to add code later!

---

## 🎯 Quick Reference

| Statement | Purpose | Example |
|-----------|---------|---------|
| `if` | Execute code if condition is true | `if x > 5:` |
| `elif` | Test another condition if previous was false | `elif x == 5:` |
| `else` | Execute code if all conditions are false | `else:` |
| `and` | Both conditions must be true | `if x > 5 and y < 10:` |
| `or` | At least one condition must be true | `if x > 5 or y < 10:` |
| `not` | Reverse the condition | `if not x > 5:` |
| `pass` | Empty placeholder | `if condition: pass` |