# 📋 Python Lists Guide

```python
mylist = ["apple", "banana", "cherry"]
```

## 🎯 What Are Lists?

Lists are used to **store multiple items in a single variable**. They are one of 4 built-in data types in Python used to store collections of data, alongside **Tuple**, **Set**, and **Dictionary**.

Lists are created using **square brackets**:

```python
# Create a List
thislist = ["apple", "banana", "cherry"]
print(thislist)
```

---

## 🔍 List Properties

### 📊 **Ordered**
Lists maintain a **defined order** that will not change. When you add new items to a list, they are placed at the end.

> **💡 Note:** Some list methods can change the order, but generally the order remains consistent.

### ✏️ **Changeable (Mutable)**
Lists are **changeable**, meaning you can modify, add, and remove items after creation.

### 🔄 **Allow Duplicates**
Since lists are indexed, they can contain items with the same value:

```python
# Lists allow duplicate values
thislist = ["apple", "banana", "cherry", "apple", "cherry"]
print(thislist)
```

---

## 🏷️ List Items & Indexing

- **List items are indexed** starting from `[0]`
- First item: `[0]`
- Second item: `[1]`
- And so on...

```python
fruits = ["apple", "banana", "cherry"]
print(fruits[0])  # Output: apple
print(fruits[1])  # Output: banana
```

---

## 📏 List Length

Use the `len()` function to determine how many items a list contains:

```python
# Print the number of items in the list
thislist = ["apple", "banana", "cherry"]
print(len(thislist))  # Output: 3
```

---

## 🎨 List Data Types

### 📝 **Single Data Type Lists**

```python
# String list
list1 = ["apple", "banana", "cherry"]

# Integer list
list2 = [1, 5, 7, 9, 3]

# Boolean list
list3 = [True, False, False]
```

### 🎭 **Mixed Data Type Lists**

```python
# A list with strings, integers, and boolean values
mixed_list = ["abc", 34, True, 40, "male"]
```

---

## 🔧 List Type & Constructor

### 📋 **Data Type**
From Python's perspective, lists are objects with the data type `'list'`:

```python
mylist = ["apple", "banana", "cherry"]
print(type(mylist))  # Output: <class 'list'>
```

### 🏗️ **List Constructor**
You can also use the `list()` constructor to create a new list:

```python
# Using the list() constructor
thislist = list(("apple", "banana", "cherry"))  # Note the double round-brackets
print(thislist)
```

---

## 🗃️ Python Collections Overview

There are **four collection data types** in Python:

| Collection | Ordered | Changeable | Duplicates | Symbol |
|------------|---------|------------|------------|---------|
| **📋 List** | ✅ Yes | ✅ Yes | ✅ Allowed | `[ ]` |
| **📦 Tuple** | ✅ Yes | ❌ No | ✅ Allowed | `( )` |
| **🎯 Set** | ❌ No | ❌ No* | ❌ Not Allowed | `{ }` |
| **📚 Dictionary** | ✅ Yes** | ✅ Yes | ❌ Not Allowed | `{ : }` |

### 📌 **Important Notes:**
- ***Set items*** are unchangeable, but you can remove and/or add items
- ****Dictionaries*** are ordered as of Python 3.7+. In Python 3.6 and earlier, they were unordered

---

## 🎯 Choosing the Right Collection

When selecting a collection type, consider these factors:

- **📋 Use Lists** when you need ordered, changeable data with duplicates
- **📦 Use Tuples** when you need ordered, unchangeable data with duplicates  
- **🎯 Use Sets** when you need unique items without duplicates
- **📚 Use Dictionaries** when you need key-value pairs

> **💡 Pro Tip:** Choosing the right collection type can improve code efficiency, security, and maintain data meaning!