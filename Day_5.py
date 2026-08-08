import csv
import os
from datetime import datetime


class MenuItem:
    def __init__(self, code: str, name: str, category: str, price: float):
        self.code = code
        self.name = name
        self.category = category
        self.price = price

    def __repr__(self):
        return f"{self.code}: {self.name} ({self.category}) - ${self.price:.2f}"


class Order:
    def __init__(self):
        self.items = {}

    def add_item(self, item: MenuItem, quantity: int = 1):
        if item.code in self.items:
            self.items[item.code]["quantity"] += quantity
        else:
            self.items[item.code] = {"item": item, "quantity": quantity}

    def remove_item(self, item_code: str):
        if item_code in self.items:
            del self.items[item_code]
            return True
        return False

    def is_empty(self):
        return len(self.items) == 0

    def total(self):
        return sum(entry["item"].price * entry["quantity"] for entry in self.items.values())

    def summary(self):
        lines = []
        for entry in self.items.values():
            item = entry["item"]
            qty = entry["quantity"]
            lines.append(f"{item.name} x{qty} = ${item.price * qty:.2f}")
        lines.append(f"Total: ${self.total():.2f}")
        return "\n".join(lines)

    def to_csv_rows(self, order_id: int):
        rows = []
        for entry in self.items.values():
            item = entry["item"]
            qty = entry["quantity"]
            rows.append({
                "order_id": order_id,
                "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                "item_code": item.code,
                "item_name": item.name,
                "category": item.category,
                "quantity": qty,
                "unit_price": f"{item.price:.2f}",
                "line_total": f"{item.price * qty:.2f}",
                "order_total": f"{self.total():.2f}",
            })
        return rows


class RestaurantOrderingSystem:
    ORDER_FILE = "saved_orders.csv"

    def __init__(self):
        self.menu_items = self._load_menu()
        self.current_order = Order()
        self.order_ids = self._load_existing_order_ids()

    def _load_menu(self):
        categories = ("Starters", "Mains", "Beverages", "Desserts")
        menu = [
            MenuItem("S1", "Garlic Bread", categories[0], 4.50),
            MenuItem("S2", "Veg Spring Rolls", categories[0], 5.25),
            MenuItem("M1", "Margherita Pizza", categories[1], 10.99),
            MenuItem("M2", "Grilled Chicken", categories[1], 13.75),
            MenuItem("B1", "Fresh Lemonade", categories[2], 3.50),
            MenuItem("B2", "Iced Coffee", categories[2], 4.00),
            MenuItem("D1", "Chocolate Brownie", categories[3], 5.99),
            MenuItem("D2", "Fruit Salad", categories[3], 4.99),
        ]
        return {item.code: item for item in menu}

    def _load_existing_order_ids(self):
        if not os.path.exists(self.ORDER_FILE):
            return set()
        order_ids = set()
        try:
            with open(self.ORDER_FILE, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    order_ids.add(int(row["order_id"]))
        except (IOError, ValueError, KeyError):
            return set()
        return order_ids

    def _next_order_id(self):
        return max(self.order_ids, default=0) + 1

    def show_menu(self):
        print("\nRestaurant Menu")
        print("--------------")
        for item in self.menu_items.values():
            print(f"{item.code:>3} | {item.name:<20} | {item.category:<10} | ${item.price:.2f}")

    def place_order(self):
        self.show_menu()
        print("Enter menu item codes to add items to your order.")
        print("Type 'done' when finished.")

        while True:
            user_input = input("Item code (or done): ").strip().upper()
            if user_input == "DONE":
                break
            if user_input not in self.menu_items:
                print("Invalid code. Please enter a valid menu item code.")
                continue

            quantity = self._safe_int_input("Quantity: ", min_value=1)
            if quantity is None:
                continue

            self.current_order.add_item(self.menu_items[user_input], quantity)
            print(f"Added {quantity} x {self.menu_items[user_input].name}.")

    def review_order(self):
        if self.current_order.is_empty():
            print("\nYour order is currently empty.")
            return
        print("\nCurrent Order")
        print("-------------")
        print(self.current_order.summary())

    def remove_order_item(self):
        if self.current_order.is_empty():
            print("\nYour order is currently empty.")
            return
        self.review_order()
        item_code = input("Enter the item code to remove: ").strip().upper()
        if self.current_order.remove_item(item_code):
            print(f"Removed item {item_code} from the order.")
        else:
            print("Item code not found in the current order.")

    def checkout(self):
        if self.current_order.is_empty():
            print("\nCannot checkout an empty order.")
            return

        self.review_order()
        confirm = input("Proceed to checkout? (yes/no): ").strip().lower()
        if confirm not in {"yes", "y"}:
            print("Checkout canceled.")
            return

        order_id = self._next_order_id()
        self._save_order(order_id)
        self.order_ids.add(order_id)
        print(f"Order #{order_id} saved successfully.")
        self.current_order = Order()

    def _save_order(self, order_id: int):
        file_exists = os.path.exists(self.ORDER_FILE)
        rows = self.current_order.to_csv_rows(order_id)
        fieldnames = [
            "order_id",
            "timestamp",
            "item_code",
            "item_name",
            "category",
            "quantity",
            "unit_price",
            "line_total",
            "order_total",
        ]
        try:
            with open(self.ORDER_FILE, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(rows)
        except IOError:
            print("Could not save order. Please check file permissions.")

    def show_saved_orders(self):
        if not os.path.exists(self.ORDER_FILE):
            print("\nNo saved orders yet.")
            return

        print("\nSaved Orders")
        print("------------")
        try:
            with open(self.ORDER_FILE, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                orders = {}
                for row in reader:
                    order_id = int(row["order_id"])
                    orders.setdefault(order_id, []).append(row)
                for order_id, rows in orders.items():
                    print(f"\nOrder #{order_id} - {rows[0]['timestamp']}")
                    for line in rows:
                        print(f"  {line['quantity']} x {line['item_name']} (${line['line_total']})")
                    print(f"  Order total: ${rows[0]['order_total']}")
        except IOError:
            print("Could not read saved orders.")

    def _safe_int_input(self, prompt: str, min_value: int = None):
        user_input = input(prompt).strip()
        try:
            value = int(user_input)
            if min_value is not None and value < min_value:
                print(f"Please enter a number greater than or equal to {min_value}.")
                return None
            return value
        except ValueError:
            print("Invalid number. Please enter an integer.")
            return None

    def run(self):
        print("Welcome to the Python OOP Restaurant Ordering System")
        while True:
            print("\nMain Menu")
            print("1. View menu")
            print("2. Place a new order")
            print("3. Review current order")
            print("4. Remove an item from order")
            print("5. Checkout and save order")
            print("6. Show saved orders")
            print("7. Exit")

            choice = input("Choose an option (1-7): ").strip()
            if choice == "1":
                self.show_menu()
            elif choice == "2":
                self.place_order()
            elif choice == "3":
                self.review_order()
            elif choice == "4":
                self.remove_order_item()
            elif choice == "5":
                self.checkout()
            elif choice == "6":
                self.show_saved_orders()
            elif choice == "7":
                print("Thank you for using the system. Goodbye!")
                break
            else:
                print("Invalid choice. Please choose a number between 1 and 7.")


if __name__ == "__main__":
    RestaurantOrderingSystem().run()
