# 🔄 Python While Loops Guide

## 🎯 Python Loop Types

Python has **two primitive loop commands**:

| Loop Type | Symbol | Purpose |
|-----------|---------|---------|
| **🔄 While Loops** | `while` | Execute code as long as condition is true |
| **📋 For Loops** | `for` | Iterate over sequences (lists, strings, etc.) |

---

## 🔁 The While Loop

With the `while` loop, we can **execute a set of statements as long as a condition is true**.

### 📝 **Basic Syntax:**
```python
while condition:
    # code to execute
    # remember to update the condition variable!
```

### 🎯 **Example:**
```python
# Print i as long as i is less than 6
i = 1
while i < 6:
    print(i)
    i += 1
```

**Output:**
```
1
2
3
4
5
```

> ⚠️ **Important:** Remember to increment `i`, or else the loop will continue forever (infinite loop)!

---

## 🛑 The Break Statement

With the `break` statement, we can **stop the loop even if the while condition is true**:

```python
# Exit the loop when i is 3
i = 1
while i < 6:
    print(i)
    if i == 3:
        break
    i += 1
```

**Output:**
```
1
2
3
```

> 💡 **What happens:** The loop stops immediately when `i` equals 3, even though the condition `i < 6` is still true.

---

## ⏭️ The Continue Statement

With the `continue` statement, we can **stop the current iteration and continue with the next**:

```python
# Continue to the next iteration if i is 3
i = 0
while i < 6:
    i += 1
    if i == 3:
        continue
    print(i)
```

**Output:**
```
1
2
4
5
6
```

> 💡 **What happens:** When `i` equals 3, the `continue` statement skips the `print(i)` and jumps to the next iteration.

---

## 🏁 The Else Statement

With the `else` statement, we can **run a block of code once when the condition is no longer true**:

```python
# Print a message once the condition is false
i = 1
while i < 6:
    print(i)
    i += 1
else:
    print("i is no longer less than 6")
```

**Output:**
```
1
2
3
4
5
i is no longer less than 6
```

> 📌 **Note:** The `else` block executes only when the loop completes normally (not when broken by `break`).

---

## 🔧 Loop Control Flow

### 🎯 **Complete Example with All Statements:**
```python
i = 1
while i <= 10:
    if i == 3:
        print(f"Skipping {i}")
        i += 1
        continue
    
    if i == 7:
        print(f"Breaking at {i}")
        break
    
    print(f"Processing {i}")
    i += 1
else:
    print("Loop completed normally")
```

**Output:**
```
Processing 1
Processing 2
Skipping 3
Processing 4
Processing 5
Processing 6
Breaking at 7
```

---

## ⚠️ Common Pitfalls

### 🚫 **Infinite Loop:**
```python
# DON'T DO THIS!
i = 1
while i < 6:
    print(i)
    # Missing: i += 1  <- This will cause infinite loop!
```

### ✅ **Correct Version:**
```python
# DO THIS INSTEAD
i = 1
while i < 6:
    print(i)
    i += 1  # Always update the condition variable
```

---

## 📊 While Loop vs For Loop

| Feature | While Loop | For Loop |
|---------|------------|----------|
| **Use Case** | Unknown number of iterations | Known number of iterations |
| **Condition** | Based on boolean expression | Based on sequence/range |
| **Control** | Manual variable management | Automatic iteration |
| **Risk** | Infinite loop if not careful | Safer, automatic termination |

---

## 🎯 Quick Reference

| Statement | Purpose | Example |
|-----------|---------|---------|
| `while condition:` | Execute code while condition is true | `while i < 10:` |
| `break` | Exit the loop immediately | `if i == 5: break` |
| `continue` | Skip current iteration, go to next | `if i == 3: continue` |
| `else:` | Execute after loop completes normally | `while i < 5: ... else: ...` |

---

## 💡 Best Practices

1. **Always update the condition variable** to avoid infinite loops
2. **Use meaningful variable names** instead of just `i`
3. **Consider using `for` loops** when you know the number of iterations
4. **Use `break` and `continue` sparingly** for cleaner code
5. **Test your loops** with different input values

### 🎯 **Example with Best Practices:**
```python
counter = 1
max_attempts = 5

while counter <= max_attempts:
    print(f"Attempt {counter}")
    
    if counter == 3:
        print("Special handling for attempt 3")
    
    counter += 1
else:
    print("All attempts completed successfully")
```