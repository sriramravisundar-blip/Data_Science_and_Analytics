def main():
    print("Simple Calculator - Day 1 Assignment")
    print("Enter two numbers and see the result of each arithmetic operation.")

    first_value = input("Enter the first number: ")
    second_value = input("Enter the second number: ")

    try:
        a = float(first_value)
        b = float(second_value)
    except ValueError:
        print("Error: Please enter valid numeric values.")
        return

    print(f"\nNumbers: a = {a}, b = {b}")
    print(f"Addition (a + b): {a + b}")
    print(f"Subtraction (a - b): {a - b}")
    print(f"Multiplication (a * b): {a * b}")

    if b != 0:
        print(f"Division (a / b): {a / b}")
        print(f"Floor Division (a // b): {a // b}")
        print(f"Modulus (a % b): {a % b}")
    else:
        print("Division (a / b): Error - division by zero")
        print("Floor Division (a // b): Error - division by zero")
        print("Modulus (a % b): Error - division by zero")

    print(f"Exponentiation (a ** b): {a ** b}")


if __name__ == "__main__":
    main()
