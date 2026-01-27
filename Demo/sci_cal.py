# Python
import tkinter as tk
import math

def on_click(event):
    text = event.widget.cget("text")
    if text == "=":
        try:
            expr = entry.get()
            # Replace function names with math module equivalents
            expr = expr.replace('sin', 'math.sin')
            expr = expr.replace('cos', 'math.cos')
            expr = expr.replace('tan', 'math.tan')
            expr = expr.replace('log', 'math.log10')
            expr = expr.replace('ln', 'math.log')
            expr = expr.replace('sqrt', 'math.sqrt')
            expr = expr.replace('^', '**')
            result = eval(expr)
            entry.delete(0, tk.END)
            entry.insert(tk.END, str(result))
        except Exception:
            entry.delete(0, tk.END)
            entry.insert(tk.END, "Error")
    elif text == "C":
        entry.delete(0, tk.END)
    else:
        entry.insert(tk.END, text)

root = tk.Tk()
root.title("Scientific Calculator")
root.geometry("400x500")
root.resizable(False, False)

entry = tk.Entry(root, font="Arial 20", borderwidth=2, relief=tk.RIDGE, justify='right')
entry.pack(fill=tk.BOTH, ipadx=8, ipady=15, pady=10, padx=10)

button_texts = [
    ['7', '8', '9', '/', 'C'],
    ['4', '5', '6', '*', '('],
    ['1', '2', '3', '-', ')'],
    ['0', '.', '^', '+', '='],
    ['sin', 'cos', 'tan', 'log', 'sqrt'],
    ['ln', 'pi', 'e', '', '']
]

button_frame = tk.Frame(root)
button_frame.pack()

for i, row in enumerate(button_texts):
    for j, text in enumerate(row):
        if text:
            btn = tk.Button(button_frame, text=text, font="Arial 14", width=6, height=2)
            btn.grid(row=i, column=j, padx=3, pady=3)
            btn.bind("<Button-1>", on_click)

# Insert constants handling
def insert_constant(const):
    if const == 'pi':
        entry.insert(tk.END, str(math.pi))
    elif const == 'e':
        entry.insert(tk.END, str(math.e))

for const, row, col in [('pi', 5, 1), ('e', 5, 2)]:
    btn = tk.Button(button_frame, text=const, font="Arial 14", width=6, height=2,
                    command=lambda c=const: insert_constant(c))
    btn.grid(row=row, column=col, padx=3, pady=3)

root.mainloop()