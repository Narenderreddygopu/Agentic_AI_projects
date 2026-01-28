#Hello 



#Agentic traning 1/27/2026

def recursive_search(name_list, target_name, index=0):
    # Base case: if the index is out of bounds
    if index >= len(name_list):
        return False
    # Check if the current element matches the target name
    if name_list[index] == target_name:
        return True
    # Recursive case: check the next element
    return recursive_search(name_list, target_name, index + 1)

# Example usage
names = ["Alice", "Bob", "Narender", "Charlie"]
found = recursive_search(names, "Narender")
print("Name found Narender:", found)





