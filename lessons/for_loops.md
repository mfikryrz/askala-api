# 🔁 Python For Loops Guide

## 🎯 What is a For Loop?

A **for loop** is used for **iterating over a sequence** (list, tuple, dictionary, set, or string).

Unlike other programming languages, Python's `for` loop works more like an **iterator method** found in object-oriented programming languages.

### 📝 **Basic Syntax:**
```python
for variable in sequence:
    # code to execute for each item
```

---

## 🍎 Basic For Loop Examples

### 📋 **Looping Through a List:**
```python
# Print each fruit in a fruit list
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    print(x)
```

**Output:**
```
apple
banana
cherry
```

> 💡 **Note:** The for loop does **not require an indexing variable** to be set beforehand!

### 🔤 **Looping Through a String:**
Even **strings are iterable objects** - they contain a sequence of characters:

```python
# Loop through the letters in the word "banana"
for x in "banana":
    print(x)
```

**Output:**
```
b
a
n
a
n
a
```

---

## 🛑 The Break Statement

With the `break` statement, we can **stop the loop before it has looped through all items**:

### 🎯 **Break After Print:**
```python
# Exit the loop when x is "banana"
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    print(x)
    if x == "banana":
        break
```

**Output:**
```
apple
banana
```

### 🎯 **Break Before Print:**
```python
# Break before printing "banana"
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    if x == "banana":
        break
    print(x)
```

**Output:**
```
apple
```

---

## ⏭️ The Continue Statement

With the `continue` statement, we can **stop the current iteration and continue with the next**:

```python
# Do not print banana
fruits = ["apple", "banana", "cherry"]
for x in fruits:
    if x == "banana":
        continue
    print(x)
```

**Output:**
```
apple
cherry
```

---

## 🔢 The range() Function

To loop through code a **specified number of times**, use the `range()` function.

### 📊 **Range Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `stop` | End value (exclusive) | `range(6)` → 0,1,2,3,4,5 |
| `start, stop` | Start and end values | `range(2,6)` → 2,3,4,5 |
| `start, stop, step` | Start, end, and increment | `range(2,30,3)` → 2,5,8,11... |

### 🎯 **Basic Range:**
```python
# Using the range() function
for x in range(6):
    print(x)
```

**Output:** `0 1 2 3 4 5`

> ⚠️ **Note:** `range(6)` is values 0 to 5, **not** 0 to 6!

### 🎯 **Range with Start:**
```python
# Using the start parameter
for x in range(2, 6):
    print(x)
```

**Output:** `2 3 4 5`

### 🎯 **Range with Step:**
```python
# Increment the sequence by 3 (default is 1)
for x in range(2, 30, 3):
    print(x)
```

**Output:** `2 5 8 11 14 17 20 23 26 29`

---

## 🏁 Else in For Loop

The `else` keyword specifies a **block of code to execute when the loop finishes**:

### ✅ **Normal Completion:**
```python
# Print message when loop ends
for x in range(6):
    print(x)
else:
    print("Finally finished!")
```

**Output:**
```
0
1
2
3
4
5
Finally finished!
```

### 🚫 **With Break (Else Won't Execute):**
```python
# Break the loop when x is 3
for x in range(6):
    if x == 3:
        break
    print(x)
else:
    print("Finally finished!")
```

**Output:**
```
0
1
2
```

> ⚠️ **Important:** The `else` block will **NOT** execute if the loop is stopped by a `break` statement!

---

## 🪆 Nested Loops

A **nested loop** is a loop inside a loop. The inner loop executes completely for each iteration of the outer loop:

```python
# Print each adjective for every fruit
adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]

for x in adj:
    for y in fruits:
        print(x, y)
```

**Output:**
```
red apple
red banana
red cherry
big apple
big banana
big cherry
tasty apple
tasty banana
tasty cherry
```

---

## 🚫 The Pass Statement

`for` loops cannot be empty. If you need an empty loop, use the `pass` statement:

```python
# Empty loop with pass statement
for x in [0, 1, 2]:
    pass  # Do nothing, avoid syntax error
```

---

## 🎯 Practical Examples

### 📊 **Enumerate for Index and Value:**
```python
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

**Output:**
```
0: apple
1: banana
2: cherry
```

### 📚 **Dictionary Iteration:**
```python
student_grades = {"Alice": 95, "Bob": 87, "Charlie": 92}

# Iterate over keys
for name in student_grades:
    print(f"{name}: {student_grades[name]}")

# Iterate over key-value pairs
for name, grade in student_grades.items():
    print(f"{name} scored {grade}")
```

### 🔄 **List Comprehension (Alternative):**
```python
# Traditional for loop
squares = []
for x in range(5):
    squares.append(x**2)

# List comprehension (more Pythonic)
squares = [x**2 for x in range(5)]
print(squares)  # [0, 1, 4, 9, 16]
```

---

## 📊 For Loop vs While Loop

| Feature | For Loop | While Loop |
|---------|----------|------------|
| **Best for** | Known sequences/iterations | Unknown number of iterations |
| **Syntax** | `for item in sequence:` | `while condition:` |
| **Automatic** | Iterator management | Manual variable management |
| **Safety** | Less prone to infinite loops | Risk of infinite loops |
| **Use cases** | Lists, strings, ranges | Conditional loops |

---

## 🎯 Quick Reference

| Statement | Purpose | Example |
|-----------|---------|---------|
| `for item in sequence:` | Iterate over sequence | `for x in [1,2,3]:` |
| `range(stop)` | Numbers 0 to stop-1 | `range(5)` → 0,1,2,3,4 |
| `range(start, stop)` | Numbers start to stop-1 | `range(2,5)` → 2,3,4 |
| `range(start, stop, step)` | Numbers with custom step | `range(0,10,2)` → 0,2,4,6,8 |
| `break` | Exit loop immediately | `if x == 5: break` |
| `continue` | Skip current iteration | `if x == 3: continue` |
| `else:` | Execute after normal completion | `for x in range(5): ... else: ...` |
| `pass` | Empty placeholder | `for x in range(5): pass` |

---

## 💡 Best Practices

1. **Use descriptive variable names** instead of `x` or `i`
2. **Use `enumerate()`** when you need both index and value
3. **Consider list comprehensions** for simple transformations
4. **Use `break` and `continue` sparingly** for cleaner code
5. **Remember that `range()` is exclusive** of the end value