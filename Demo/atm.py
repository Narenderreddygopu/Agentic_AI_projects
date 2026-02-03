print("\n=== Welcome to Mini ATM ===\n")

class BankAccount:
    def __init__(self, name, balance=0):
        self.name = name
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount
        print(f"{amount} deposited. New balance: {self.balance}")

    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            print(f"{amount} withdrawn. New balance: {self.balance}")
        else:
            print("Insufficient balance!")

    def show_balance(self):
        print(f"{self.name}'s current balance: {self.balance}")

# Create account
name = input("Enter account holder name: ")
account = BankAccount(name)

# Menu-driven loop
while True:
    print("\n--- ATM Menu ---")
    print("1. Deposit")
    print("2. Withdraw")
    print("3. Check Balance")
    print("4. Exit")

    choice = input("Choose an option (1-4): ")

    if choice == '1':
        amount = float(input("Enter amount to deposit: "))
        account.deposit(amount)
    elif choice == '2':
        amount = float(input("Enter amount to withdraw: "))
        account.withdraw(amount)
    elif choice == '3':
        account.show_balance()
    elif choice == '4':
        print("Thank you for using Mini ATM. Goodbye!",name)
        break
    else:
        print("Invalid choice. Please try again",name)
