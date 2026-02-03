def check_pin():
    print("PIN verified")

def check_balance():
    print("Balance displayed")

def withdraw():
    print("Cash withdrawn")

def atm():
    check_pin()
    check_balance()
    withdraw()

atm()

print("\n Using OOP concepts \n")

class ATM:
    def check_balance(self):
        print("Balance checked")

atm = ATM()
atm.check_balance()

print("\n Using OOP2 concepts \n")

def fun():
    print("Hello")
def FunctionName():
    print("Welcome to functions")
    fun()
FunctionName()

